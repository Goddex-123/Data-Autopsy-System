"""
Missing Data Analyzer Module

Forensic analysis of missing data patterns.
Detects intentional omissions, silent assumptions, and coverage gaps.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


class MissingDataAnalyzer:
    """
    Forensic analysis of missing data to detect intentional omissions
    and understand data gaps.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize missing data analyzer.
        
        Args:
            data: The dataset to analyze
        """
        self.data = data
        self.findings = {}
        
    def analyze(self) -> dict:
        """
        Run complete missing data analysis.
        
        Returns:
            dict: Missing data findings
        """
        self.findings = {
            'overview': self._get_missing_overview(),
            'patterns': self._analyze_missing_patterns(),
            'mechanisms': self._detect_missing_mechanisms(),
            'correlations': self._analyze_missing_correlations(),
            'silent_assumptions': self._detect_silent_assumptions(),
            'coverage_gaps': self._detect_coverage_gaps(),
            'overall_missing_score': 0,
            'concerns': []
        }
        
        self.findings['overall_missing_score'] = self._calculate_missing_score()
        
        return self.findings
    
    def _get_missing_overview(self) -> dict:
        """Get overview of missing data in the dataset."""
        total_cells = self.data.size
        missing_cells = self.data.isnull().sum().sum()
        
        column_missing = {}
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                column_missing[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(self.data) * 100, 2)
                }
                
        # Rows with any missing
        rows_with_missing = self.data.isnull().any(axis=1).sum()
        
        # Complete rows
        complete_rows = len(self.data) - rows_with_missing
        
        return {
            'total_cells': total_cells,
            'missing_cells': int(missing_cells),
            'missing_percentage': round(missing_cells / total_cells * 100, 2),
            'columns_with_missing': len(column_missing),
            'column_details': column_missing,
            'rows_with_missing': int(rows_with_missing),
            'complete_rows': int(complete_rows),
            'complete_percentage': round(complete_rows / len(self.data) * 100, 2)
        }
    
    def _analyze_missing_patterns(self) -> dict:
        """Analyze patterns in missing data."""
        results = {
            'missing_together': [],
            'missing_sequences': [],
            'column_patterns': []
        }
        
        # Find columns that are often missing together
        missing_matrix = self.data.isnull()
        cols_with_missing = [col for col in self.data.columns 
                            if missing_matrix[col].any()]
        
        if len(cols_with_missing) >= 2:
            for i, col1 in enumerate(cols_with_missing):
                for col2 in cols_with_missing[i+1:]:
                    # Check if they're missing together
                    both_missing = (missing_matrix[col1] & missing_matrix[col2]).sum()
                    either_missing = (missing_matrix[col1] | missing_matrix[col2]).sum()
                    
                    if either_missing > 0:
                        together_rate = both_missing / either_missing
                        if together_rate > 0.7:
                            results['missing_together'].append({
                                'columns': [col1, col2],
                                'together_rate': round(together_rate, 3),
                                'interpretation': 'These columns tend to be missing together'
                            })
                            
        # Check for sequential missing patterns (runs of missing)
        for col in cols_with_missing:
            missing_mask = missing_matrix[col].values
            runs = self._find_runs(missing_mask)
            long_runs = [r for r in runs if r['length'] >= 5]
            
            if long_runs:
                results['missing_sequences'].append({
                    'column': col,
                    'long_runs': len(long_runs),
                    'max_run_length': max(r['length'] for r in long_runs),
                    'interpretation': 'Consecutive missing values detected'
                })
                
        # Analyze per-column missing patterns
        for col in cols_with_missing:
            col_data = self.data[col]
            missing_pct = col_data.isnull().mean() * 100
            
            pattern_type = None
            if missing_pct > 90:
                pattern_type = 'almost_empty'
            elif missing_pct > 50:
                pattern_type = 'majority_missing'
            elif missing_pct > 20:
                pattern_type = 'significant_missing'
            else:
                pattern_type = 'sparse_missing'
                
            results['column_patterns'].append({
                'column': col,
                'pattern_type': pattern_type,
                'missing_pct': round(missing_pct, 2)
            })
            
        return results
    
    def _find_runs(self, mask: np.ndarray) -> List[dict]:
        """Find runs of True values in a boolean array."""
        runs = []
        in_run = False
        run_start = 0
        
        for i, val in enumerate(mask):
            if val and not in_run:
                in_run = True
                run_start = i
            elif not val and in_run:
                in_run = False
                runs.append({'start': run_start, 'length': i - run_start})
                
        if in_run:
            runs.append({'start': run_start, 'length': len(mask) - run_start})
            
        return runs
    
    def _detect_missing_mechanisms(self) -> dict:
        """
        Detect the mechanism of missing data:
        - MCAR: Missing Completely At Random
        - MAR: Missing At Random (depends on observed data)
        - MNAR: Missing Not At Random (depends on unobserved data)
        """
        results = {
            'likely_mcar': [],
            'likely_mar': [],
            'likely_mnar': [],
            'tests_performed': []
        }
        
        missing_cols = [col for col in self.data.columns 
                       if self.data[col].isnull().any()]
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in missing_cols:
            missing_mask = self.data[col].isnull()
            
            # Skip if too few missing or too few present
            if missing_mask.sum() < 10 or (~missing_mask).sum() < 10:
                continue
                
            # Test for MAR: Check if missingness is related to other variables
            mar_indicators = []
            
            for other_col in numeric_cols:
                if other_col == col:
                    continue
                    
                other_data = self.data[other_col]
                
                # Compare values when col is missing vs not missing
                present_values = other_data[~missing_mask].dropna()
                missing_values = other_data[missing_mask].dropna()
                
                if len(present_values) < 10 or len(missing_values) < 10:
                    continue
                    
                # Mann-Whitney U test
                try:
                    stat, p_value = stats.mannwhitneyu(
                        present_values, missing_values, alternative='two-sided'
                    )
                    
                    if p_value < 0.05:
                        mar_indicators.append({
                            'related_to': other_col,
                            'p_value': round(p_value, 4)
                        })
                except:
                    pass
                    
            if mar_indicators:
                results['likely_mar'].append({
                    'column': col,
                    'related_variables': mar_indicators[:3],  # Top 3
                    'interpretation': 'Missingness appears related to other variables'
                })
            else:
                # If no MAR relationship found, could be MCAR or MNAR
                # We can't definitively test for MNAR, but we can note
                results['likely_mcar'].append({
                    'column': col,
                    'interpretation': 'No clear pattern - possibly random'
                })
                
            results['tests_performed'].append(col)
            
        return results
    
    def _analyze_missing_correlations(self) -> dict:
        """Analyze correlations between missing values across columns."""
        results = {
            'correlation_matrix': {},
            'high_correlations': []
        }
        
        missing_matrix = self.data.isnull()
        cols_with_missing = [col for col in self.data.columns 
                            if missing_matrix[col].any()]
        
        if len(cols_with_missing) < 2:
            return results
            
        # Calculate correlation between missing indicators
        missing_corr = missing_matrix[cols_with_missing].corr()
        
        # Find high correlations
        for i, col1 in enumerate(cols_with_missing):
            for col2 in cols_with_missing[i+1:]:
                corr = missing_corr.loc[col1, col2]
                
                if pd.notna(corr) and abs(corr) > 0.5:
                    results['high_correlations'].append({
                        'columns': [col1, col2],
                        'correlation': round(corr, 3),
                        'interpretation': 'Missing values are correlated'
                    })
                    
        return results
    
    def _detect_silent_assumptions(self) -> dict:
        """Detect silent assumptions in the data."""
        results = {
            'default_values': [],
            'implicit_zeros': [],
            'sentinel_values': []
        }
        
        # Check for common sentinel/default values
        sentinel_candidates = [0, -1, -999, 999, 9999, -9999, 99, -99, 
                              'NA', 'N/A', 'NULL', 'None', '', ' ', 
                              'Unknown', 'UNKNOWN', 'missing', 'MISSING']
        
        for col in self.data.columns:
            values = self.data[col]
            
            # Check numeric columns for suspicious default values
            if values.dtype in ['int64', 'float64']:
                value_counts = values.value_counts()
                
                for sentinel in [0, -1, -999, 999, 9999, -9999, 99, -99]:
                    if sentinel in value_counts.index:
                        count = value_counts[sentinel]
                        pct = count / len(values) * 100
                        
                        # Flag if this value appears suspiciously often
                        if pct > 10 and count > 10:
                            results['sentinel_values'].append({
                                'column': col,
                                'value': sentinel,
                                'count': int(count),
                                'percentage': round(pct, 2),
                                'warning': 'Possible sentinel value disguising missing data'
                            })
                            
                # Check for suspicious zero inflation
                zero_count = (values == 0).sum()
                zero_pct = zero_count / len(values) * 100
                
                if zero_pct > 30 and zero_count > 10:
                    # Check if zeros might be defaults
                    non_zero = values[values != 0]
                    if len(non_zero) > 0 and non_zero.min() > 0:
                        results['implicit_zeros'].append({
                            'column': col,
                            'zero_percentage': round(zero_pct, 2),
                            'warning': 'High zero rate - possibly defaulted missing values'
                        })
                        
            # Check string columns for default text
            elif values.dtype == 'object':
                value_counts = values.value_counts()
                
                for default_val in ['Unknown', 'UNKNOWN', 'N/A', 'NA', 'None', 
                                    'missing', 'MISSING', '-', '.']:
                    if default_val in value_counts.index:
                        count = value_counts[default_val]
                        if count > 5:
                            results['default_values'].append({
                                'column': col,
                                'value': default_val,
                                'count': int(count),
                                'warning': 'Default value representing missing data'
                            })
                            
        return results
    
    def _detect_coverage_gaps(self) -> dict:
        """Detect gaps in data coverage."""
        results = {
            'temporal_gaps': [],
            'categorical_gaps': [],
            'numeric_gaps': []
        }
        
        # Check temporal columns for gaps
        for col in self.data.columns:
            try:
                if self.data[col].dtype == 'datetime64[ns]':
                    dates = self.data[col]
                else:
                    dates = pd.to_datetime(self.data[col], errors='coerce')
                    if dates.isna().all():
                        continue
                        
                valid_dates = dates.dropna().sort_values()
                
                if len(valid_dates) > 10:
                    # Check for month coverage
                    months_present = valid_dates.dt.month.unique()
                    missing_months = set(range(1, 13)) - set(months_present)
                    
                    if missing_months:
                        results['temporal_gaps'].append({
                            'column': col,
                            'type': 'missing_months',
                            'missing': list(missing_months),
                            'interpretation': 'Some months not represented'
                        })
                        
                    # Check for year coverage
                    years = valid_dates.dt.year.unique()
                    if len(years) >= 2:
                        full_range = set(range(years.min(), years.max() + 1))
                        missing_years = full_range - set(years)
                        
                        if missing_years:
                            results['temporal_gaps'].append({
                                'column': col,
                                'type': 'missing_years',
                                'missing': list(missing_years),
                                'interpretation': 'Some years not represented'
                            })
            except:
                pass
                
        # Check categorical columns for expected values that might be missing
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            unique_vals = self.data[col].dropna().unique()
            
            # Check for common expected categories that might be missing
            # This is heuristic-based
            if len(unique_vals) < 10:
                # Check if there might be missing categories
                col_lower = col.lower()
                
                expected_sets = {
                    'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday'],
                    'month': ['January', 'February', 'March', 'April', 'May', 
                             'June', 'July', 'August', 'September', 'October',
                             'November', 'December'],
                    'gender': ['Male', 'Female'],
                    'status': ['Active', 'Inactive'],
                    'type': []  # Domain-specific
                }
                
                for keyword, expected in expected_sets.items():
                    if keyword in col_lower and expected:
                        unique_lower = [str(v).lower() for v in unique_vals]
                        expected_lower = [v.lower() for v in expected]
                        
                        missing = [e for e in expected_lower 
                                  if e not in unique_lower]
                        
                        if missing and len(missing) < len(expected):
                            results['categorical_gaps'].append({
                                'column': col,
                                'potentially_missing': missing[:5],
                                'present_values': list(unique_vals)[:10],
                                'interpretation': 'Expected categories may be missing'
                            })
                        break
                        
        return results
    
    def _calculate_missing_score(self) -> float:
        """
        Calculate overall missing data concern score (0-100).
        Higher score = more concerns about missing data.
        """
        score = 0
        concerns = []
        
        # Overview concerns
        overview = self.findings.get('overview', {})
        missing_pct = overview.get('missing_percentage', 0)
        
        if missing_pct > 20:
            score += 20
            concerns.append(f"High overall missing rate: {missing_pct}%")
        elif missing_pct > 10:
            score += 10
            concerns.append(f"Moderate missing rate: {missing_pct}%")
            
        # Pattern concerns
        patterns = self.findings.get('patterns', {})
        if patterns.get('missing_together'):
            score += 10
            concerns.append("Variables missing together (possible systematic issue)")
        if patterns.get('missing_sequences'):
            score += 15
            concerns.append("Sequential missing patterns detected")
            
        # Mechanism concerns
        mechanisms = self.findings.get('mechanisms', {})
        if mechanisms.get('likely_mar'):
            score += 15
            concerns.append("Missing At Random pattern detected")
            
        # Silent assumptions
        silent = self.findings.get('silent_assumptions', {})
        if silent.get('sentinel_values'):
            score += 20
            concerns.append("Sentinel values masking missing data")
        if silent.get('implicit_zeros'):
            score += 10
            concerns.append("Implicit zeros may represent missing values")
            
        # Coverage gaps
        gaps = self.findings.get('coverage_gaps', {})
        if gaps.get('temporal_gaps'):
            score += 15
            concerns.append("Temporal coverage gaps detected")
        if gaps.get('categorical_gaps'):
            score += 10
            concerns.append("Expected categories may be missing")
            
        self.findings['concerns'] = concerns
        return min(100, score)
    
    def get_summary(self) -> str:
        """Generate human-readable missing data summary."""
        if not self.findings:
            self.analyze()
            
        overview = self.findings['overview']
        score = self.findings['overall_missing_score']
        severity = 'LOW' if score < 30 else 'MODERATE' if score < 60 else 'HIGH'
        
        summary = f"""
MISSING DATA REPORT
===================
Overall Missing: {overview['missing_percentage']}% ({overview['missing_cells']:,} cells)
Complete Rows: {overview['complete_percentage']}%
Missing Concern Score: {score}/100 ({severity})

Key Concerns:
"""
        for concern in self.findings.get('concerns', []):
            summary += f"  🕳️ {concern}\n"
            
        if not self.findings.get('concerns'):
            summary += "  ✅ No major missing data concerns\n"
            
        return summary
