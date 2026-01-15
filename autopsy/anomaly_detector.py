"""
Anomaly Detector Module

Detection of statistical anomalies, data fabrication, and inconsistencies.
Uses Benford's Law, outlier detection, and pattern analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


class AnomalyDetector:
    """
    Forensic detection of anomalies, fabrication, and data manipulation.
    """
    
    # Expected Benford's Law distribution for first digits
    BENFORD_EXPECTED = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize anomaly detector.
        
        Args:
            data: The dataset to analyze for anomalies
        """
        self.data = data
        self.findings = {}
        
    def analyze(self) -> dict:
        """
        Run complete anomaly analysis.
        
        Returns:
            dict: Anomaly detection findings
        """
        self.findings = {
            'benfords_law': self._analyze_benfords_law(),
            'statistical_outliers': self._detect_outliers(),
            'duplicates': self._analyze_duplicates(),
            'value_anomalies': self._detect_value_anomalies(),
            'fabrication_indicators': self._detect_fabrication(),
            'overall_anomaly_score': 0,
            'red_flags': []
        }
        
        self.findings['overall_anomaly_score'] = self._calculate_anomaly_score()
        
        return self.findings
    
    def _analyze_benfords_law(self) -> dict:
        """
        Apply Benford's Law to detect potential data fabrication.
        
        Benford's Law: In naturally occurring datasets, the first digit
        follows a specific logarithmic distribution. Fabricated data
        often violates this law.
        """
        results = {
            'applicable_columns': [],
            'violations': [],
            'conforming_columns': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = self.data[col].dropna()
            
            # Benford's Law works best on data spanning multiple orders of magnitude
            if len(values) < 100:
                continue
                
            # Get first digit of each value
            abs_values = values.abs()
            abs_values = abs_values[abs_values >= 1]  # Exclude values < 1
            
            if len(abs_values) < 100:
                continue
                
            first_digits = abs_values.apply(lambda x: int(str(x).replace('-', '').replace('.', '')[0]) if x != 0 else 0)
            first_digits = first_digits[first_digits > 0]  # Exclude 0
            
            if len(first_digits) < 100:
                continue
                
            results['applicable_columns'].append(col)
            
            # Calculate observed distribution
            observed_counts = first_digits.value_counts().sort_index()
            observed_freq = observed_counts / len(first_digits)
            
            # Chi-square test against Benford's distribution
            expected_counts = pd.Series({d: p * len(first_digits) 
                                        for d, p in self.BENFORD_EXPECTED.items()})
            
            # Ensure both series have same indices
            common_digits = set(observed_counts.index) & set(expected_counts.index)
            if len(common_digits) < 5:
                continue
                
            obs = [observed_counts.get(d, 0) for d in range(1, 10)]
            exp = [expected_counts.get(d, 0) for d in range(1, 10)]
            
            try:
                chi2, p_value = stats.chisquare(obs, exp)
                
                # Calculate mean absolute deviation from Benford's
                mad = sum(abs(observed_freq.get(d, 0) - self.BENFORD_EXPECTED[d]) 
                         for d in range(1, 10)) / 9
                
                if p_value < 0.01:
                    results['violations'].append({
                        'column': col,
                        'chi2_statistic': round(chi2, 2),
                        'p_value': round(p_value, 6),
                        'mean_deviation': round(mad, 4),
                        'interpretation': 'Significant deviation from Benford\'s Law',
                        'severity': 'high' if p_value < 0.001 else 'moderate'
                    })
                else:
                    results['conforming_columns'].append({
                        'column': col,
                        'p_value': round(p_value, 4),
                        'interpretation': 'Conforms to Benford\'s Law'
                    })
            except Exception as e:
                pass
                
        return results
    
    def _detect_outliers(self) -> dict:
        """Detect statistical outliers using multiple methods."""
        results = {
            'iqr_outliers': [],
            'zscore_outliers': [],
            'isolation_outliers': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = self.data[col].dropna()
            
            if len(values) < 10:
                continue
                
            # IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = values[(values < lower_bound) | (values > upper_bound)]
            outlier_pct = len(iqr_outliers) / len(values) * 100
            
            if len(iqr_outliers) > 0:
                results['iqr_outliers'].append({
                    'column': col,
                    'count': len(iqr_outliers),
                    'percentage': round(outlier_pct, 2),
                    'bounds': [round(lower_bound, 4), round(upper_bound, 4)],
                    'severity': 'high' if outlier_pct > 10 else 'moderate' if outlier_pct > 5 else 'low'
                })
                
            # Z-score method
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
                extreme_outliers = values[z_scores > 3]
                
            if len(extreme_outliers) > 0:
                results['zscore_outliers'].append({
                    'column': col,
                    'count': len(extreme_outliers),
                    'max_zscore': round(float(z_scores.max()), 2),
                    'extreme_values': [round(v, 4) for v in extreme_outliers.head(5).tolist()]
                })
                
        return results
    
    def _analyze_duplicates(self) -> dict:
        """Analyze duplicate patterns in the data."""
        results = {
            'exact_duplicates': 0,
            'duplicate_percentage': 0,
            'near_duplicates': [],
            'suspicious_patterns': []
        }
        
        # Exact row duplicates
        dup_rows = self.data.duplicated()
        results['exact_duplicates'] = int(dup_rows.sum())
        results['duplicate_percentage'] = round(dup_rows.mean() * 100, 2)
        
        if results['duplicate_percentage'] > 10:
            results['suspicious_patterns'].append({
                'type': 'high_duplicate_rate',
                'value': results['duplicate_percentage'],
                'warning': 'Unusually high duplicate rate'
            })
            
        # Check for columns with high duplicate values
        for col in self.data.columns:
            unique_ratio = self.data[col].nunique() / len(self.data)
            
            # Flag columns where single values repeat excessively
            if unique_ratio < 0.01 and len(self.data) > 100:
                top_value = self.data[col].value_counts().iloc[0]
                top_value_pct = top_value / len(self.data) * 100
                
                if top_value_pct > 50:
                    results['suspicious_patterns'].append({
                        'type': 'dominant_value',
                        'column': col,
                        'value_count': int(top_value),
                        'percentage': round(top_value_pct, 2)
                    })
                    
        return results
    
    def _detect_value_anomalies(self) -> dict:
        """Detect anomalies in value patterns."""
        results = {
            'round_number_bias': [],
            'boundary_clustering': [],
            'digit_preference': [],
            'impossible_values': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = self.data[col].dropna()
            
            if len(values) < 50:
                continue
                
            # Check for round number bias
            round_10 = (values % 10 == 0).mean()
            round_100 = (values % 100 == 0).mean()
            
            if round_10 > 0.3:  # More than 30% are multiples of 10
                results['round_number_bias'].append({
                    'column': col,
                    'multiple_of_10': round(round_10 * 100, 1),
                    'warning': 'Unusual clustering at round numbers'
                })
                
            # Check last digit distribution
            last_digits = values.abs().apply(lambda x: int(str(int(x))[-1]) if x == int(x) else -1)
            last_digits = last_digits[last_digits >= 0]
            
            if len(last_digits) > 50:
                digit_counts = last_digits.value_counts()
                # In natural data, last digits should be roughly uniform
                expected = len(last_digits) / 10
                
                for digit, count in digit_counts.items():
                    if count > expected * 2:  # More than 2x expected
                        results['digit_preference'].append({
                            'column': col,
                            'digit': int(digit),
                            'observed_pct': round(count / len(last_digits) * 100, 1),
                            'expected_pct': 10.0,
                            'warning': f'Last digit {digit} appears too frequently'
                        })
                        break
                        
            # Check for impossible values (negative where shouldn't be, etc.)
            # This is domain-specific but we can flag obvious cases
            if values.min() < 0:
                # Check common patterns suggesting non-negative values
                if any(word in col.lower() for word in ['age', 'count', 'quantity', 'price', 'amount', 'size', 'height', 'weight']):
                    negative_count = (values < 0).sum()
                    results['impossible_values'].append({
                        'column': col,
                        'issue': 'Negative values in likely non-negative field',
                        'count': int(negative_count),
                        'min_value': float(values.min())
                    })
                    
        return results
    
    def _detect_fabrication(self) -> dict:
        """Detect signs of data fabrication or manipulation."""
        results = {
            'too_perfect': [],
            'artificial_variance': [],
            'copy_paste_patterns': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = self.data[col].dropna()
            
            if len(values) < 30:
                continue
                
            # Check for "too perfect" statistics
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val > 0:
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                
                # Suspiciously low variance
                if cv < 0.01 and mean_val != 0:
                    results['too_perfect'].append({
                        'column': col,
                        'cv': round(cv, 4),
                        'warning': 'Suspiciously uniform values'
                    })
                    
            # Check for artificial variance patterns
            # Real data usually has natural clustering
            value_counts = values.value_counts()
            if len(value_counts) == len(values):
                # Every value is unique - could be fabricated to avoid detection
                if len(values) > 100:
                    results['artificial_variance'].append({
                        'column': col,
                        'unique_ratio': 1.0,
                        'warning': 'All values unique - possible artificial variance'
                    })
                    
        # Check for copy-paste patterns (repeated sequences)
        if len(self.data) > 10:
            for i in range(len(self.data) - 5):
                row = self.data.iloc[i].values
                for j in range(i + 1, min(i + 100, len(self.data))):
                    if np.array_equal(self.data.iloc[j].values, row):
                        results['copy_paste_patterns'].append({
                            'row_indices': [i, j],
                            'warning': 'Exact row duplication detected'
                        })
                        break
                if len(results['copy_paste_patterns']) >= 5:
                    break
                    
        return results
    
    def _calculate_anomaly_score(self) -> float:
        """
        Calculate overall anomaly severity score (0-100).
        Higher score = more anomalies detected.
        """
        score = 0
        red_flags = []
        
        # Benford's Law violations
        benford = self.findings.get('benfords_law', {})
        violations = benford.get('violations', [])
        if violations:
            high_severity = [v for v in violations if v.get('severity') == 'high']
            score += len(high_severity) * 15 + (len(violations) - len(high_severity)) * 8
            red_flags.append(f"Benford's Law violations in {len(violations)} column(s)")
            
        # Outliers
        outliers = self.findings.get('statistical_outliers', {})
        high_outlier_cols = [o for o in outliers.get('iqr_outliers', []) 
                            if o.get('severity') == 'high']
        if high_outlier_cols:
            score += len(high_outlier_cols) * 5
            red_flags.append(f"High outlier rates in {len(high_outlier_cols)} column(s)")
            
        # Duplicates
        dups = self.findings.get('duplicates', {})
        if dups.get('duplicate_percentage', 0) > 10:
            score += 15
            red_flags.append(f"High duplicate rate: {dups['duplicate_percentage']}%")
            
        # Value anomalies
        value_anoms = self.findings.get('value_anomalies', {})
        if value_anoms.get('round_number_bias'):
            score += 10
            red_flags.append("Round number bias detected")
        if value_anoms.get('digit_preference'):
            score += 15
            red_flags.append("Digit preference detected (possible fabrication)")
        if value_anoms.get('impossible_values'):
            score += 20
            red_flags.append("Impossible values found")
            
        # Fabrication indicators
        fab = self.findings.get('fabrication_indicators', {})
        if fab.get('too_perfect'):
            score += 15
            red_flags.append("Suspiciously perfect/uniform values")
        if fab.get('copy_paste_patterns'):
            score += 20
            red_flags.append("Copy-paste patterns detected")
            
        self.findings['red_flags'] = red_flags
        return min(100, score)
    
    def get_summary(self) -> str:
        """Generate human-readable anomaly summary."""
        if not self.findings:
            self.analyze()
            
        score = self.findings['overall_anomaly_score']
        severity = 'LOW' if score < 30 else 'MODERATE' if score < 60 else 'HIGH'
        
        summary = f"""
ANOMALY DETECTION REPORT
========================
Overall Anomaly Score: {score}/100 ({severity})

Red Flags:
"""
        for flag in self.findings.get('red_flags', []):
            summary += f"  🚨 {flag}\n"
            
        if not self.findings.get('red_flags'):
            summary += "  ✅ No major anomalies detected\n"
            
        return summary
