"""
Robustness Tester Module

Tests the robustness and fragility of conclusions drawn from the data.
Performs sensitivity analysis, bootstrap tests, and subgroup analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
import warnings


class RobustnessTester:
    """
    Forensic testing of conclusion robustness and reliability.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize robustness tester.
        
        Args:
            data: The dataset to analyze
        """
        self.data = data
        self.findings = {}
        
    def analyze(self) -> dict:
        """
        Run complete robustness analysis.
        
        Returns:
            dict: Robustness testing findings
        """
        self.findings = {
            'stability': self._test_stability(),
            'sensitivity': self._test_sensitivity(),
            'subgroup_analysis': self._analyze_subgroups(),
            'bootstrap_results': self._bootstrap_statistics(),
            'outlier_influence': self._test_outlier_influence(),
            'overall_robustness_score': 0,
            'fragility_warnings': []
        }
        
        self.findings['overall_robustness_score'] = self._calculate_robustness_score()
        
        return self.findings
    
    def _test_stability(self) -> dict:
        """Test stability of key statistics across random samples."""
        results = {
            'mean_stability': [],
            'variance_stability': [],
            'correlation_stability': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(self.data) < 100:
            results['note'] = 'Dataset too small for meaningful stability testing'
            return results
            
        n_samples = 10
        sample_size = int(len(self.data) * 0.7)
        
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            values = self.data[col].dropna()
            
            if len(values) < 50:
                continue
                
            # Sample means across bootstrap samples
            sample_means = []
            sample_stds = []
            
            for _ in range(n_samples):
                sample = values.sample(n=min(sample_size, len(values)), replace=True)
                sample_means.append(sample.mean())
                sample_stds.append(sample.std())
                
            # Calculate coefficient of variation of the means
            mean_cv = np.std(sample_means) / np.mean(sample_means) if np.mean(sample_means) != 0 else 0
            std_cv = np.std(sample_stds) / np.mean(sample_stds) if np.mean(sample_stds) != 0 else 0
            
            results['mean_stability'].append({
                'column': col,
                'mean_cv': round(abs(mean_cv), 4),
                'stable': abs(mean_cv) < 0.1
            })
            
            results['variance_stability'].append({
                'column': col,
                'std_cv': round(abs(std_cv), 4),
                'stable': abs(std_cv) < 0.2
            })
            
        # Test correlation stability
        if len(numeric_cols) >= 2:
            for _ in range(min(5, len(numeric_cols) - 1)):
                col1, col2 = np.random.choice(numeric_cols, 2, replace=False)
                
                sample_corrs = []
                for _ in range(n_samples):
                    sample = self.data[[col1, col2]].dropna().sample(
                        n=min(sample_size, len(self.data)), replace=True
                    )
                    if len(sample) > 10:
                        corr = sample[col1].corr(sample[col2])
                        if pd.notna(corr):
                            sample_corrs.append(corr)
                            
                if sample_corrs:
                    corr_std = np.std(sample_corrs)
                    results['correlation_stability'].append({
                        'columns': [col1, col2],
                        'correlation_std': round(corr_std, 4),
                        'stable': corr_std < 0.1
                    })
                    
        return results
    
    def _test_sensitivity(self) -> dict:
        """Test sensitivity of statistics to small perturbations."""
        results = {
            'mean_sensitivity': [],
            'quantile_sensitivity': [],
            'trimmed_mean_comparison': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:
            values = self.data[col].dropna()
            
            if len(values) < 30:
                continue
                
            original_mean = values.mean()
            original_median = values.median()
            
            # Test sensitivity by removing extreme values
            q01 = values.quantile(0.01)
            q99 = values.quantile(0.99)
            trimmed_values = values[(values >= q01) & (values <= q99)]
            
            trimmed_mean = trimmed_values.mean()
            
            # Compare regular mean to trimmed mean
            mean_diff_pct = abs(original_mean - trimmed_mean) / abs(original_mean) * 100 if original_mean != 0 else 0
            
            results['trimmed_mean_comparison'].append({
                'column': col,
                'original_mean': round(original_mean, 4),
                'trimmed_mean': round(trimmed_mean, 4),
                'difference_pct': round(mean_diff_pct, 2),
                'sensitive': mean_diff_pct > 10
            })
            
            # Test quantile sensitivity
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            iqr = q75 - q25
            
            if iqr > 0:
                sensitivity_ratio = (values.max() - values.min()) / iqr
                results['quantile_sensitivity'].append({
                    'column': col,
                    'range_to_iqr_ratio': round(sensitivity_ratio, 2),
                    'interpretation': 'High' if sensitivity_ratio > 10 else 'Normal'
                })
                
        return results
    
    def _analyze_subgroups(self) -> dict:
        """Analyze if conclusions hold across subgroups."""
        results = {
            'subgroup_differences': [],
            'simpson_paradox_candidates': [],
            'heterogeneous_effects': []
        }
        
        numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        cat_cols = list(self.data.select_dtypes(include=['object', 'category']).columns)
        
        if not numeric_cols or not cat_cols:
            return results
            
        # Limit analysis scope
        numeric_cols = numeric_cols[:5]
        cat_cols = [c for c in cat_cols if self.data[c].nunique() <= 10][:3]
        
        for cat_col in cat_cols:
            groups = self.data[cat_col].dropna().unique()
            
            if len(groups) < 2 or len(groups) > 10:
                continue
                
            for num_col in numeric_cols:
                group_means = {}
                group_sizes = {}
                
                for group in groups:
                    group_data = self.data[self.data[cat_col] == group][num_col].dropna()
                    if len(group_data) >= 10:
                        group_means[str(group)] = group_data.mean()
                        group_sizes[str(group)] = len(group_data)
                        
                if len(group_means) >= 2:
                    # Test for significant differences between groups
                    overall_mean = self.data[num_col].mean()
                    max_deviation = max(abs(m - overall_mean) for m in group_means.values())
                    deviation_pct = max_deviation / abs(overall_mean) * 100 if overall_mean != 0 else 0
                    
                    if deviation_pct > 20:
                        results['subgroup_differences'].append({
                            'grouping_variable': cat_col,
                            'outcome_variable': num_col,
                            'group_means': {k: round(v, 4) for k, v in group_means.items()},
                            'overall_mean': round(overall_mean, 4),
                            'max_deviation_pct': round(deviation_pct, 2),
                            'warning': 'Conclusions may differ by subgroup'
                        })
                        
        # Check for potential Simpson's paradox
        if len(numeric_cols) >= 2 and cat_cols:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    overall_corr = self.data[col1].corr(self.data[col2])
                    
                    if pd.isna(overall_corr):
                        continue
                        
                    for cat_col in cat_cols[:2]:
                        group_corrs = {}
                        
                        for group in self.data[cat_col].dropna().unique():
                            group_data = self.data[self.data[cat_col] == group]
                            if len(group_data) >= 20:
                                group_corr = group_data[col1].corr(group_data[col2])
                                if pd.notna(group_corr):
                                    group_corrs[str(group)] = group_corr
                                    
                        # Check if any group has opposite correlation
                        if group_corrs:
                            for group, corr in group_corrs.items():
                                if (overall_corr > 0.1 and corr < -0.1) or \
                                   (overall_corr < -0.1 and corr > 0.1):
                                    results['simpson_paradox_candidates'].append({
                                        'variables': [col1, col2],
                                        'grouping': cat_col,
                                        'overall_correlation': round(overall_corr, 3),
                                        'group': group,
                                        'group_correlation': round(corr, 3),
                                        'warning': 'Possible Simpson\'s Paradox!'
                                    })
                                    break
                                    
        return results
    
    def _bootstrap_statistics(self) -> dict:
        """Perform bootstrap analysis on key statistics."""
        results = {
            'confidence_intervals': [],
            'bootstrap_bias': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        n_bootstrap = 100
        
        for col in numeric_cols[:5]:
            values = self.data[col].dropna()
            
            if len(values) < 30:
                continue
                
            # Bootstrap for mean
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = values.sample(n=len(values), replace=True)
                bootstrap_means.append(sample.mean())
                
            bootstrap_means = np.array(bootstrap_means)
            
            # Calculate confidence interval
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            # Estimate bias
            original_mean = values.mean()
            bootstrap_mean_of_means = bootstrap_means.mean()
            bias = bootstrap_mean_of_means - original_mean
            
            results['confidence_intervals'].append({
                'column': col,
                'point_estimate': round(original_mean, 4),
                'ci_95_lower': round(ci_lower, 4),
                'ci_95_upper': round(ci_upper, 4),
                'ci_width': round(ci_upper - ci_lower, 4)
            })
            
            results['bootstrap_bias'].append({
                'column': col,
                'estimated_bias': round(bias, 6),
                'bias_significant': abs(bias) > (ci_upper - ci_lower) * 0.1
            })
            
        return results
    
    def _test_outlier_influence(self) -> dict:
        """Test the influence of outliers on key statistics."""
        results = {
            'influential_outliers': [],
            'robust_statistics_comparison': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:
            values = self.data[col].dropna()
            
            if len(values) < 30:
                continue
                
            original_mean = values.mean()
            original_std = values.std()
            
            # Identify outliers using IQR
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = (values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)
            outliers = values[outlier_mask]
            
            if len(outliers) == 0:
                continue
                
            # Calculate statistics without outliers
            clean_values = values[~outlier_mask]
            clean_mean = clean_values.mean()
            clean_std = clean_values.std()
            
            mean_change_pct = abs(original_mean - clean_mean) / abs(original_mean) * 100 if original_mean != 0 else 0
            std_change_pct = abs(original_std - clean_std) / abs(original_std) * 100 if original_std != 0 else 0
            
            if mean_change_pct > 5 or std_change_pct > 10:
                results['influential_outliers'].append({
                    'column': col,
                    'outlier_count': len(outliers),
                    'outlier_pct': round(len(outliers) / len(values) * 100, 2),
                    'mean_change_pct': round(mean_change_pct, 2),
                    'std_change_pct': round(std_change_pct, 2),
                    'influential': True
                })
                
            # Compare robust vs non-robust statistics
            median = values.median()
            mad = np.median(np.abs(values - median))  # Median Absolute Deviation
            
            results['robust_statistics_comparison'].append({
                'column': col,
                'mean': round(original_mean, 4),
                'median': round(median, 4),
                'mean_median_diff_pct': round(abs(original_mean - median) / abs(median) * 100, 2) if median != 0 else 0,
                'std': round(original_std, 4),
                'mad': round(mad, 4)
            })
            
        return results
    
    def _calculate_robustness_score(self) -> float:
        """
        Calculate overall robustness score (0-100).
        Higher score = MORE ROBUST (better).
        """
        score = 100  # Start with perfect score, deduct for issues
        warnings = []
        
        # Stability issues
        stability = self.findings.get('stability', {})
        unstable_means = [s for s in stability.get('mean_stability', []) 
                         if not s.get('stable')]
        if unstable_means:
            score -= len(unstable_means) * 5
            warnings.append(f"Unstable means in {len(unstable_means)} column(s)")
            
        unstable_corrs = [s for s in stability.get('correlation_stability', [])
                         if not s.get('stable')]
        if unstable_corrs:
            score -= len(unstable_corrs) * 5
            warnings.append(f"Unstable correlations detected")
            
        # Sensitivity issues
        sensitivity = self.findings.get('sensitivity', {})
        sensitive_cols = [s for s in sensitivity.get('trimmed_mean_comparison', [])
                         if s.get('sensitive')]
        if sensitive_cols:
            score -= len(sensitive_cols) * 5
            warnings.append(f"Mean sensitive to outliers in {len(sensitive_cols)} column(s)")
            
        # Subgroup issues
        subgroups = self.findings.get('subgroup_analysis', {})
        if subgroups.get('simpson_paradox_candidates'):
            score -= 25
            warnings.append("Simpson's Paradox candidates detected - conclusions may reverse!")
            
        if len(subgroups.get('subgroup_differences', [])) > 3:
            score -= 15
            warnings.append("Substantial subgroup differences - overall conclusions may be misleading")
            
        # Outlier influence
        outliers = self.findings.get('outlier_influence', {})
        influential = [o for o in outliers.get('influential_outliers', [])
                      if o.get('mean_change_pct', 0) > 10]
        if influential:
            score -= len(influential) * 5
            warnings.append(f"Highly influential outliers in {len(influential)} column(s)")
            
        self.findings['fragility_warnings'] = warnings
        return max(0, min(100, score))
    
    def get_summary(self) -> str:
        """Generate human-readable robustness summary."""
        if not self.findings:
            self.analyze()
            
        score = self.findings['overall_robustness_score']
        severity = 'HIGH' if score >= 70 else 'MODERATE' if score >= 40 else 'LOW'
        
        summary = f"""
ROBUSTNESS REPORT
=================
Robustness Score: {score}/100 ({severity} robustness)

Fragility Warnings:
"""
        for warning in self.findings.get('fragility_warnings', []):
            summary += f"  ⚠️ {warning}\n"
            
        if not self.findings.get('fragility_warnings'):
            summary += "  ✅ Conclusions appear robust\n"
            
        return summary
