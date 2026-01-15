"""
Bias Detector Module

Systematic identification of sampling, measurement, and selection biases.
Answers: What biases exist in this data? Who is over/under-represented?
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


class BiasDetector:
    """
    Forensic detection of various forms of bias in datasets.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize bias detector.
        
        Args:
            data: The dataset to analyze for bias
        """
        self.data = data
        self.findings = {}
        
    def analyze(self) -> dict:
        """
        Run complete bias analysis.
        
        Returns:
            dict: Bias detection findings
        """
        self.findings = {
            'distribution_bias': self._detect_distribution_bias(),
            'sampling_bias': self._detect_sampling_bias(),
            'representation_bias': self._detect_representation_bias(),
            'correlation_bias': self._detect_correlation_bias(),
            'temporal_bias': self._detect_temporal_bias(),
            'overall_bias_score': 0,
            'critical_warnings': []
        }
        
        self.findings['overall_bias_score'] = self._calculate_bias_score()
        
        return self.findings
    
    def _detect_distribution_bias(self) -> dict:
        """Detect skewed or abnormal distributions in numeric columns."""
        results = {
            'skewed_columns': [],
            'outlier_heavy_columns': [],
            'uniform_columns': [],
            'bimodal_columns': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = self.data[col].dropna()
            if len(values) < 10:
                continue
                
            # Calculate skewness
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)
            
            if abs(skewness) > 1.5:
                results['skewed_columns'].append({
                    'column': col,
                    'skewness': round(skewness, 3),
                    'direction': 'right' if skewness > 0 else 'left',
                    'severity': 'high' if abs(skewness) > 3 else 'moderate'
                })
                
            # Check for outlier-heavy distributions (high kurtosis)
            if kurtosis > 5:
                results['outlier_heavy_columns'].append({
                    'column': col,
                    'kurtosis': round(kurtosis, 3),
                    'interpretation': 'Many extreme values present'
                })
                
            # Check for uniform distribution (suspicious if unexpected)
            _, p_value = stats.kstest(values, 'uniform', 
                                       args=(values.min(), values.max() - values.min()))
            if p_value > 0.05:
                results['uniform_columns'].append({
                    'column': col,
                    'p_value': round(p_value, 4),
                    'warning': 'Suspiciously uniform distribution'
                })
                
            # Detect bimodal distributions using dip test approximation
            if self._is_bimodal(values):
                results['bimodal_columns'].append({
                    'column': col,
                    'interpretation': 'Possible hidden subgroups'
                })
                
        return results
    
    def _is_bimodal(self, values: pd.Series) -> bool:
        """Check if distribution appears bimodal."""
        try:
            hist, bin_edges = np.histogram(values, bins='auto')
            # Look for two distinct peaks
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append(i)
            return len(peaks) >= 2
        except:
            return False
    
    def _detect_sampling_bias(self) -> dict:
        """Detect potential sampling biases."""
        results = {
            'class_imbalance': [],
            'rare_categories': [],
            'dominant_categories': [],
            'sample_size_concerns': []
        }
        
        # Check overall sample size
        n = len(self.data)
        if n < 100:
            results['sample_size_concerns'].append({
                'issue': 'Very small sample size',
                'count': n,
                'warning': 'Results may not be generalizable'
            })
        elif n < 1000:
            results['sample_size_concerns'].append({
                'issue': 'Small sample size',
                'count': n,
                'warning': 'Limited statistical power'
            })
            
        # Analyze categorical columns for imbalance
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            value_counts = self.data[col].value_counts()
            total = len(self.data[col].dropna())
            
            if total == 0:
                continue
                
            # Check for class imbalance
            proportions = value_counts / total
            max_prop = proportions.max()
            min_prop = proportions.min()
            
            if max_prop > 0.8:
                results['dominant_categories'].append({
                    'column': col,
                    'dominant_value': str(value_counts.index[0]),
                    'proportion': round(max_prop, 3),
                    'warning': 'Severely imbalanced - one category dominates'
                })
                
            if min_prop < 0.01 and len(value_counts) > 2:
                rare_cats = proportions[proportions < 0.01].index.tolist()
                results['rare_categories'].append({
                    'column': col,
                    'rare_values': [str(v) for v in rare_cats[:5]],
                    'warning': 'Under-represented categories detected'
                })
                
            # Calculate imbalance ratio
            imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
            if imbalance_ratio > 10:
                results['class_imbalance'].append({
                    'column': col,
                    'imbalance_ratio': round(imbalance_ratio, 2),
                    'severity': 'high' if imbalance_ratio > 50 else 'moderate'
                })
                
        return results
    
    def _detect_representation_bias(self) -> dict:
        """Detect demographic and representation biases."""
        results = {
            'potential_demographic_columns': [],
            'missing_groups': [],
            'representation_issues': []
        }
        
        # Common demographic column indicators
        demographic_keywords = ['gender', 'sex', 'race', 'ethnicity', 'age', 
                               'income', 'education', 'country', 'region', 
                               'nationality', 'religion', 'occupation']
        
        for col in self.data.columns:
            col_lower = col.lower()
            for keyword in demographic_keywords:
                if keyword in col_lower:
                    results['potential_demographic_columns'].append({
                        'column': col,
                        'type': keyword,
                        'unique_values': self.data[col].nunique(),
                        'missing_pct': round(self.data[col].isnull().mean() * 100, 2)
                    })
                    
                    # Check for common representation issues
                    if self.data[col].dtype == 'object':
                        unique_vals = self.data[col].dropna().unique()
                        
                        # Check gender representation
                        if keyword in ['gender', 'sex']:
                            unique_lower = [str(v).lower() for v in unique_vals]
                            if len(unique_vals) == 2:
                                results['representation_issues'].append({
                                    'column': col,
                                    'issue': 'Binary gender only',
                                    'values': list(unique_vals),
                                    'recommendation': 'Consider non-binary representation'
                                })
                    break
                    
        return results
    
    def _detect_correlation_bias(self) -> dict:
        """Detect suspicious correlations that may indicate bias."""
        results = {
            'strong_correlations': [],
            'suspicious_patterns': [],
            'potential_confounders': []
        }
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return results
            
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        # Find strong correlations
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                
                if pd.isna(corr):
                    continue
                    
                if abs(corr) > 0.9:
                    results['strong_correlations'].append({
                        'columns': [col1, col2],
                        'correlation': round(corr, 3),
                        'warning': 'Near-perfect correlation - possible data duplication or derived feature'
                    })
                elif abs(corr) > 0.7:
                    results['potential_confounders'].append({
                        'columns': [col1, col2],
                        'correlation': round(corr, 3),
                        'interpretation': 'Strong relationship may confound analysis'
                    })
                    
        # Check for suspicious patterns (e.g., all correlations being similar)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).stack()
        
        if len(upper_triangle) > 5:
            corr_std = upper_triangle.std()
            if corr_std < 0.1:
                results['suspicious_patterns'].append({
                    'pattern': 'Uniform correlations',
                    'std': round(corr_std, 4),
                    'warning': 'All correlations suspiciously similar - possible artificial data'
                })
                
        return results
    
    def _detect_temporal_bias(self) -> dict:
        """Detect time-related biases."""
        results = {
            'temporal_columns': [],
            'time_clustering': [],
            'period_bias': []
        }
        
        # Try to find datetime columns
        for col in self.data.columns:
            try:
                if self.data[col].dtype == 'datetime64[ns]':
                    dates = self.data[col]
                elif self.data[col].dtype == 'object':
                    dates = pd.to_datetime(self.data[col], errors='coerce')
                else:
                    continue
                    
                valid_dates = dates.dropna()
                if len(valid_dates) < 10:
                    continue
                    
                results['temporal_columns'].append(col)
                
                # Check for time clustering
                if hasattr(valid_dates.dt, 'dayofweek'):
                    dow_counts = valid_dates.dt.dayofweek.value_counts()
                    if dow_counts.max() / dow_counts.sum() > 0.3:
                        results['time_clustering'].append({
                            'column': col,
                            'type': 'day_of_week',
                            'dominant_day': int(dow_counts.idxmax()),
                            'proportion': round(dow_counts.max() / dow_counts.sum(), 3)
                        })
                        
                # Check for month bias
                if hasattr(valid_dates.dt, 'month'):
                    month_counts = valid_dates.dt.month.value_counts()
                    missing_months = set(range(1, 13)) - set(month_counts.index)
                    if missing_months:
                        results['period_bias'].append({
                            'column': col,
                            'type': 'missing_months',
                            'missing': list(missing_months),
                            'warning': 'Some months not represented'
                        })
                        
            except Exception:
                continue
                
        return results
    
    def _calculate_bias_score(self) -> float:
        """
        Calculate overall bias severity score (0-100).
        Higher score = more bias detected.
        """
        score = 0
        warnings = []
        
        # Distribution bias
        dist = self.findings.get('distribution_bias', {})
        if len(dist.get('skewed_columns', [])) > 2:
            score += 15
            warnings.append("Multiple highly skewed distributions detected")
            
        if dist.get('uniform_columns'):
            score += 10
            warnings.append("Suspiciously uniform distributions found")
            
        # Sampling bias
        sampling = self.findings.get('sampling_bias', {})
        if sampling.get('dominant_categories'):
            score += 20
            warnings.append("Severe class imbalance in categorical variables")
            
        if sampling.get('sample_size_concerns'):
            score += 10
            warnings.append("Sample size may affect reliability")
            
        # Representation bias
        rep = self.findings.get('representation_bias', {})
        if rep.get('representation_issues'):
            score += 15
            warnings.append("Demographic representation issues detected")
            
        # Correlation bias
        corr = self.findings.get('correlation_bias', {})
        if corr.get('suspicious_patterns'):
            score += 25
            warnings.append("Suspicious correlation patterns - possible data manipulation")
        if len(corr.get('strong_correlations', [])) > 3:
            score += 10
            warnings.append("Multiple near-perfect correlations detected")
            
        # Temporal bias
        temporal = self.findings.get('temporal_bias', {})
        if temporal.get('period_bias'):
            score += 10
            warnings.append("Time period bias detected")
            
        self.findings['critical_warnings'] = warnings
        return min(100, score)
    
    def get_summary(self) -> str:
        """Generate human-readable bias summary."""
        if not self.findings:
            self.analyze()
            
        score = self.findings['overall_bias_score']
        severity = 'LOW' if score < 30 else 'MODERATE' if score < 60 else 'HIGH'
        
        summary = f"""
BIAS DETECTION REPORT
=====================
Overall Bias Score: {score}/100 ({severity})

Critical Warnings:
"""
        for warning in self.findings.get('critical_warnings', []):
            summary += f"  ⚠️ {warning}\n"
            
        if not self.findings.get('critical_warnings'):
            summary += "  ✅ No critical bias warnings\n"
            
        return summary
