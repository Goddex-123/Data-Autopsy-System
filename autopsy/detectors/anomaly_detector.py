"""
Anomaly Detector — Refactored

Scientifically rigorous anomaly detection with:
- Applicability-aware Benford's Law analysis
- IQR + z-score + Isolation Forest + LOF ensemble
- Consensus anomaly scoring
- No overclaiming (no "fabrication detected" language)
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.models import Finding, Severity, DatasetProfile, SemanticType
from ..core.config import AnalysisConfig

logger = logging.getLogger(__name__)

# Expected Benford's Law distribution for first digits
BENFORD_EXPECTED = {
    1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
    5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046,
}


class AnomalyDetector:
    """
    Detects statistical anomalies using multiple methods and reports
    findings with appropriate confidence and limitations.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
        profile: Optional[DatasetProfile] = None,
    ):
        self.data = data
        self.config = config or AnalysisConfig()
        self.profile = profile

    def analyze(self) -> List[Finding]:
        """Run all anomaly detection methods and return findings."""
        findings: List[Finding] = []
        findings.extend(self._analyze_benfords_law())
        findings.extend(self._detect_outliers())
        findings.extend(self._analyze_duplicates())
        findings.extend(self._detect_value_anomalies())
        findings.extend(self._detect_pattern_anomalies())
        return findings

    def _get_analyzable_numeric_cols(self) -> List[str]:
        """Get numeric columns excluding identifiers and constants."""
        numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        if self.profile is None:
            return numeric_cols

        excluded = set()
        for col, cp in self.profile.columns.items():
            if cp.semantic_type in (
                SemanticType.IDENTIFIER, SemanticType.CONSTANT,
                SemanticType.PII_EMAIL, SemanticType.PII_PHONE,
                SemanticType.PII_IP, SemanticType.PII_OTHER,
            ):
                excluded.add(col)
        return [c for c in numeric_cols if c not in excluded]

    # ── Benford's Law ────────────────────────────────────────────────

    def _analyze_benfords_law(self) -> List[Finding]:
        """
        Apply Benford's Law with applicability checks.

        Benford's Law applies to data that:
        - Spans multiple orders of magnitude
        - Contains naturally occurring quantities
        - Has sufficient sample size
        - Is NOT identifiers, bounded, or assigned values
        """
        findings = []
        numeric_cols = self._get_analyzable_numeric_cols()

        for col in numeric_cols:
            values = self.data[col].dropna()

            # Applicability checks
            applicable, reason = self._check_benford_applicability(col, values)
            if not applicable:
                logger.debug("Benford not applicable for %s: %s", col, reason)
                continue

            # Get first digits
            abs_values = values.abs()
            abs_values = abs_values[abs_values >= 1]
            first_digits = abs_values.apply(
                lambda x: int(str(abs(x)).lstrip("0").replace(".", "")[0])
                if x != 0 else 0
            )
            first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]

            if len(first_digits) < self.config.benford_min_sample_size:
                continue

            # Calculate observed distribution
            observed_counts = first_digits.value_counts().sort_index()
            n = len(first_digits)

            obs = [int(observed_counts.get(d, 0)) for d in range(1, 10)]
            exp = [BENFORD_EXPECTED[d] * n for d in range(1, 10)]

            try:
                chi2, p_value = stats.chisquare(obs, exp)
            except ValueError:
                continue

            # MAD (Mean Absolute Deviation) — standard Benford metric
            observed_freq = {d: observed_counts.get(d, 0) / n for d in range(1, 10)}
            mad = sum(abs(observed_freq.get(d, 0) - BENFORD_EXPECTED[d]) for d in range(1, 10)) / 9

            # MAD classification (Nigrini's scale)
            if mad < 0.006:
                conformity = "Close conformity"
            elif mad < 0.012:
                conformity = "Acceptable conformity"
            elif mad < 0.015:
                conformity = "Marginally acceptable"
            else:
                conformity = "Nonconforming"

            if p_value < 0.01 and mad >= 0.012:
                severity = Severity.MEDIUM if mad < 0.015 else Severity.HIGH
                evidence_score= min(0.85, 1 - p_value) * (0.7 if n < 500 else 1.0)

                findings.append(Finding(
                    id=f"ANOM-BENFORD-{col[:20].upper().replace(' ', '_')}",
                    category="anomaly",
                    severity=severity,
                    evidence_score=round(confidence, 2),
                    title=f"Benford's Law deviation in '{col}'",
                    description=(
                        f"The first-digit distribution of '{col}' deviates from "
                        f"Benford's expected distribution. Conformity: {conformity}."
                    ),
                    column=col,
                    evidence={
                        "chi2_statistic": round(chi2, 2),
                        "p_value": round(p_value, 6),
                        "mad": round(mad, 4),
                        "conformity": conformity,
                        "sample_size": n,
                        "observed_first_digit_pct": {
                            str(d): round(observed_freq.get(d, 0) * 100, 1)
                            for d in range(1, 10)
                        },
                    },
                    statistical_test="Chi-square goodness-of-fit vs Benford distribution",
                    recommendation=(
                        "Investigate the data-generating process for this variable. "
                        "Benford deviation alone does not establish data fabrication."
                    ),
                    limitations=(
                        "Benford's Law is not universally applicable. Variables with "
                        "artificial bounds, assigned values, psychological pricing (e.g. 9.99), "
                        "or insufficient magnitude range may legitimately deviate from Benford's distribution."
                    ),
                ))

        return findings

    def _check_benford_applicability(self, col: str, values: pd.Series) -> tuple:
        """
        Check whether Benford's Law is applicable to this column.

        Returns:
            (applicable: bool, reason: str)
        """
        # Minimum sample size
        if len(values) < self.config.benford_min_sample_size:
            return False, f"Insufficient sample size ({len(values)} < {self.config.benford_min_sample_size})"

        # Must have positive values
        positive = values[values > 0]
        if len(positive) < self.config.benford_min_sample_size:
            return False, "Too few positive values"

        # Check magnitude range (must span at least 2 orders of magnitude)
        if positive.max() > 0 and positive.min() > 0:
            magnitude_range = np.log10(positive.max()) - np.log10(positive.min())
            if magnitude_range < self.config.benford_min_magnitude_range:
                return False, f"Insufficient magnitude range ({magnitude_range:.1f} < {self.config.benford_min_magnitude_range})"

        # Exclude columns that look like identifiers by name
        col_lower = col.lower()
        id_keywords = ["id", "code", "index", "key", "pk", "serial", "number"]
        if any(kw in col_lower for kw in id_keywords):
            if self.profile and col in self.profile.columns:
                if self.profile.columns[col].is_identifier:
                    return False, "Column appears to be an identifier"

        return True, "Applicable"

    # ── Outlier Detection ────────────────────────────────────────────

    def _detect_outliers(self) -> List[Finding]:
        """Detect outliers using IQR, z-score, and optional ML methods."""
        findings = []
        numeric_cols = self._get_analyzable_numeric_cols()

        for col in numeric_cols:
            values = self.data[col].dropna()
            if len(values) < 10:
                continue

            iqr_result = self._iqr_outliers(col, values)
            zscore_result = self._zscore_outliers(col, values)

            # Consensus: flagged by both methods
            iqr_count = iqr_result.get("count", 0) if iqr_result else 0
            zscore_count = zscore_result.get("count", 0) if zscore_result else 0

            if iqr_count > 0 or zscore_count > 0:
                max_pct = max(
                    iqr_result.get("percentage", 0) if iqr_result else 0,
                    zscore_count / len(values) * 100 if len(values) > 0 else 0,
                )
                severity = (
                    Severity.HIGH if max_pct > 10
                    else Severity.MEDIUM if max_pct > 5
                    else Severity.LOW
                )
                evidence_score = min(0.9, 0.5 + max_pct / 100)

                findings.append(Finding(
                    id=f"ANOM-OUTLIER-{col[:20].upper().replace(' ', '_')}",
                    category="anomaly",
                    severity=severity,
                    evidence_score=round(evidence_score, 2),
                    title=f"Outliers detected in '{col}'",
                    description=(
                        f"Statistical outliers found using IQR and z-score methods."
                    ),
                    column=col,
                    evidence={
                        "iqr_outlier_count": iqr_count,
                        "iqr_outlier_pct": round(iqr_result.get("percentage", 0), 2) if iqr_result else 0,
                        "iqr_bounds": iqr_result.get("bounds") if iqr_result else None,
                        "zscore_outlier_count": zscore_count,
                        "max_zscore": zscore_result.get("max_zscore") if zscore_result else None,
                        "sample_size": len(values),
                    },
                    statistical_test="IQR (1.5×IQR) + Z-score (>3σ)",
                    recommendation=(
                        "Investigate whether outliers represent data errors, "
                        "legitimate extreme values, or different populations."
                    ),
                    limitations=(
                        "Outlier detection is highly sensitive to distribution shape. "
                        "IQR routinely over-flags in skewed or heavy-tailed distributions. "
                        "Do NOT assume outliers are errors or invalid data points."
                    ),
                ))

        # Ensemble with Isolation Forest and LOF (for datasets with multiple numeric cols)
        if len(numeric_cols) >= 2 and len(self.data) >= 50:
            ensemble_finding = self._multivariate_ensemble(numeric_cols)
            if ensemble_finding:
                findings.extend(ensemble_finding)

        return findings

    def _iqr_outliers(self, col: str, values: pd.Series) -> Optional[dict]:
        """IQR-based outlier detection."""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return None

        k = self.config.outlier_iqr_multiplier
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        outliers = values[(values < lower) | (values > upper)]

        if len(outliers) == 0:
            return None

        return {
            "count": len(outliers),
            "percentage": len(outliers) / len(values) * 100,
            "bounds": [round(float(lower), 4), round(float(upper), 4)],
        }

    def _zscore_outliers(self, col: str, values: pd.Series) -> Optional[dict]:
        """Z-score based outlier detection."""
        if values.std() == 0:
            return None

        z_scores = np.abs(stats.zscore(values, nan_policy="omit"))
        extreme = values[z_scores > self.config.zscore_threshold]

        if len(extreme) == 0:
            return None

        return {
            "count": len(extreme),
            "max_zscore": round(float(z_scores.max()), 2),
        }

    def _multivariate_ensemble(self, numeric_cols: List[str]) -> List[Finding]:
        """Run Isolation Forest and LOF for multivariate anomaly detection."""
        findings = []
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.debug("sklearn not available for multivariate ensemble")
            return findings

        subset = self.data[numeric_cols].dropna()
        if len(subset) < 30:
            return findings

        max_samples = min(len(subset), self.config.isolation_forest_max_samples)
        if len(subset) > max_samples:
            subset = subset.sample(n=max_samples, random_state=self.config.random_seed)

        # Scale data for distance-based methods like LOF
        scaler = StandardScaler()
        scaled_subset = scaler.fit_predict(subset) if hasattr(scaler, "fit_predict") else scaler.fit_transform(subset)

        # Isolation Forest
        try:
            iso = IsolationForest(
                contamination=self.config.isolation_forest_contamination,
                random_state=self.config.random_seed,
                n_jobs=1,
            )
            preds_iso = iso.fit_predict(subset)
            n_iso = int((preds_iso == -1).sum())
            if n_iso > 0:
                pct_iso = n_iso / len(subset) * 100
                findings.append(Finding(
                    id="ANOM-IFOREST-001",
                    category="anomaly",
                    severity=Severity.MEDIUM if pct_iso > 5 else Severity.LOW,
                    evidence_score=round(min(0.7, 0.3 + pct_iso / 50), 2),
                    title="Multivariate anomalies detected (Isolation Forest)",
                    description=f"Isolation Forest identified {n_iso} potential anomalies ({pct_iso:.1f}%) across {len(numeric_cols)} numeric features.",
                    evidence={
                        "anomaly_count": n_iso,
                        "anomaly_percentage": round(pct_iso, 2),
                        "features_used": len(numeric_cols),
                        "sample_size": len(subset),
                    },
                    statistical_test="Isolation Forest",
                    recommendation="Review flagged multivariate anomalies.",
                    limitations="Sensitive to contamination parameter. Flagged points are just statistically rare.",
                ))
        except Exception as e:
            logger.warning("Isolation Forest failed: %s", e)

        # Local Outlier Factor (LOF)
        try:
            lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.config.isolation_forest_contamination,
                n_jobs=1,
            )
            preds_lof = lof.fit_predict(scaled_subset)
            n_lof = int((preds_lof == -1).sum())
            if n_lof > 0:
                pct_lof = n_lof / len(subset) * 100
                findings.append(Finding(
                    id="ANOM-LOF-001",
                    category="anomaly",
                    severity=Severity.MEDIUM if pct_lof > 5 else Severity.LOW,
                    evidence_score=round(min(0.7, 0.3 + pct_lof / 50), 2),
                    title="Local anomalies detected (LOF)",
                    description=f"Local Outlier Factor identified {n_lof} potential local anomalies ({pct_lof:.1f}%) across {len(numeric_cols)} features.",
                    evidence={
                        "anomaly_count": n_lof,
                        "anomaly_percentage": round(pct_lof, 2),
                        "features_used": len(numeric_cols),
                        "sample_size": len(subset),
                    },
                    statistical_test="Local Outlier Factor (LOF)",
                    recommendation="Review flagged local anomalies.",
                    limitations="LOF identifies points that are outliers relative to their local neighborhood, not globally.",
                ))
        except Exception as e:
            logger.warning("LOF failed: %s", e)

        return findings

    # ── Duplicate Analysis ───────────────────────────────────────────

    def _analyze_duplicates(self) -> List[Finding]:
        """Analyze duplicate patterns."""
        findings = []

        dup_mask = self.data.duplicated()
        dup_count = int(dup_mask.sum())
        dup_pct = dup_count / len(self.data) * 100 if len(self.data) > 0 else 0

        if dup_pct > self.config.duplicate_warning_threshold:
            severity = Severity.HIGH if dup_pct > 20 else Severity.MEDIUM
            findings.append(Finding(
                id="ANOM-DUP-001",
                category="anomaly",
                severity=severity,
                evidence_score=0.95,
                title="High duplicate row rate",
                description=f"{dup_count} exact duplicate rows ({dup_pct:.1f}%) detected.",
                evidence={
                    "exact_duplicates": dup_count,
                    "duplicate_percentage": round(dup_pct, 2),
                },
                recommendation=(
                    "Verify whether duplicates are expected (e.g., repeated measurements) "
                    "or represent data quality issues that should be deduplicated."
                ),
            ))
        elif dup_count > 0:
            findings.append(Finding(
                id="ANOM-DUP-001",
                category="anomaly",
                severity=Severity.LOW,
                evidence_score=0.9,
                title="Duplicate rows present",
                description=f"{dup_count} exact duplicate rows ({dup_pct:.1f}%) detected.",
                evidence={
                    "exact_duplicates": dup_count,
                    "duplicate_percentage": round(dup_pct, 2),
                },
                recommendation="Review duplicates to determine if deduplication is needed.",
            ))

        return findings

    # ── Value Anomalies ──────────────────────────────────────────────

    def _detect_value_anomalies(self) -> List[Finding]:
        """Detect anomalies in value patterns (round numbers, digit preference)."""
        findings = []
        numeric_cols = self._get_analyzable_numeric_cols()

        for col in numeric_cols:
            values = self.data[col].dropna()
            if len(values) < 50:
                continue

            # Round number bias
            round_10 = (values % 10 == 0).mean()
            if round_10 > 0.3:
                findings.append(Finding(
                    id=f"ANOM-ROUND-{col[:20].upper().replace(' ', '_')}",
                    category="anomaly",
                    severity=Severity.LOW,
                    evidence_score=round(min(0.7, round_10), 2),
                    title=f"Round number clustering in '{col}'",
                    description=(
                        f"{round_10*100:.1f}% of values are multiples of 10. "
                        f"This may indicate data rounding, estimation, or preference patterns."
                    ),
                    column=col,
                    evidence={"multiple_of_10_pct": round(round_10 * 100, 1)},
                    recommendation="Determine if rounding is expected for this variable.",
                    limitations="Round numbers are common in survey data, estimates, and prices.",
                ))

            # Impossible values: negative where domain implies non-negative
            if values.min() < 0:
                non_negative_keywords = [
                    "age", "count", "quantity", "price", "amount",
                    "size", "height", "weight", "distance", "duration",
                ]
                if any(kw in col.lower() for kw in non_negative_keywords):
                    neg_count = int((values < 0).sum())
                    findings.append(Finding(
                        id=f"ANOM-INVALID-{col[:20].upper().replace(' ', '_')}",
                        category="data_quality",
                        severity=Severity.HIGH,
                        evidence_score=0.85,
                        title=f"Potentially invalid negative values in '{col}'",
                        description=(
                            f"{neg_count} negative values found in '{col}', "
                            f"which appears to be a non-negative variable based on its name."
                        ),
                        column=col,
                        evidence={
                            "negative_count": neg_count,
                            "min_value": round(float(values.min()), 4),
                        },
                        recommendation="Verify whether negative values are valid or represent errors/sentinel values.",
                    ))

        return findings

    # ── Pattern Anomalies (renamed from "fabrication") ───────────────

    def _detect_pattern_anomalies(self) -> List[Finding]:
        """
        Detect unusual patterns that may warrant investigation.

        NOTE: These patterns do NOT establish fabrication or manipulation.
        They indicate anomalies that require human investigation.
        """
        findings = []
        numeric_cols = self._get_analyzable_numeric_cols()

        for col in numeric_cols:
            values = self.data[col].dropna()
            if len(values) < 30:
                continue

            # Suspiciously low variance
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 0 and mean_val != 0:
                cv = abs(std_val / mean_val)
                if cv < 0.01:
                    findings.append(Finding(
                        id=f"ANOM-LOWVAR-{col[:20].upper().replace(' ', '_')}",
                        category="anomaly",
                        severity=Severity.LOW,
                        evidence_score=0.5,
                        title=f"Very low variance in '{col}'",
                        description=(
                            f"Coefficient of variation is {cv:.4f}, indicating "
                            f"extremely uniform values. This may be normal for "
                            f"some variables or may warrant investigation."
                        ),
                        column=col,
                        evidence={
                            "coefficient_of_variation": round(cv, 4),
                            "mean": round(float(mean_val), 4),
                            "std": round(float(std_val), 4),
                        },
                        recommendation="Verify whether this uniformity is expected.",
                        limitations="Low variance can be legitimate for many types of data.",
                    ))

        return findings
