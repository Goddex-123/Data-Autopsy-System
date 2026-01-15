"""
Data Provenance Analyzer

Investigates the origin, collection methods, and metadata of datasets.
Answers: Who generated this data? How was it collected? When?
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import hashlib


class DataProvenance:
    """
    Forensic analysis of data provenance - tracking data origins and lineage.
    """
    
    def __init__(self, data: pd.DataFrame, source_path: str = None):
        """
        Initialize provenance analyzer.
        
        Args:
            data: The dataset to analyze
            source_path: Original file path or source identifier
        """
        self.data = data
        self.source_path = source_path
        self.findings = {}
        
    def analyze(self) -> dict:
        """
        Run complete provenance analysis.
        
        Returns:
            dict: Provenance findings
        """
        self.findings = {
            'metadata': self._analyze_metadata(),
            'temporal': self._analyze_temporal_patterns(),
            'structure': self._analyze_structure(),
            'fingerprint': self._generate_fingerprint(),
            'credibility_score': 0,
            'concerns': []
        }
        
        # Calculate overall credibility score
        self.findings['credibility_score'] = self._calculate_credibility()
        
        return self.findings
    
    def _analyze_metadata(self) -> dict:
        """Analyze file and data metadata."""
        metadata = {
            'source': self.source_path,
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # File metadata if available
        if self.source_path and os.path.exists(str(self.source_path)):
            stat = os.stat(self.source_path)
            metadata['file_size_mb'] = stat.st_size / 1024 / 1024
            metadata['file_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            metadata['file_created'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            
        return metadata
    
    def _analyze_temporal_patterns(self) -> dict:
        """Detect temporal columns and analyze time-related patterns."""
        temporal = {
            'datetime_columns': [],
            'date_range': None,
            'temporal_gaps': [],
            'suspicious_patterns': []
        }
        
        for col in self.data.columns:
            # Try to detect datetime columns
            if self.data[col].dtype == 'datetime64[ns]':
                temporal['datetime_columns'].append(col)
            elif self.data[col].dtype == 'object':
                # Try parsing as datetime
                try:
                    sample = self.data[col].dropna().head(100)
                    pd.to_datetime(sample, infer_datetime_format=True)
                    temporal['datetime_columns'].append(col)
                except:
                    pass
        
        # Analyze first datetime column found
        if temporal['datetime_columns']:
            col = temporal['datetime_columns'][0]
            try:
                dates = pd.to_datetime(self.data[col], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) > 0:
                    temporal['date_range'] = {
                        'start': str(valid_dates.min()),
                        'end': str(valid_dates.max()),
                        'span_days': (valid_dates.max() - valid_dates.min()).days
                    }
                    
                    # Detect gaps
                    sorted_dates = valid_dates.sort_values()
                    diffs = sorted_dates.diff()
                    median_diff = diffs.median()
                    
                    # Find significant gaps (>3x median)
                    if pd.notna(median_diff) and median_diff.days > 0:
                        large_gaps = diffs[diffs > median_diff * 3]
                        temporal['temporal_gaps'] = [
                            {'gap_days': g.days, 'after': str(sorted_dates.iloc[i-1])}
                            for i, g in large_gaps.items() if i > 0
                        ][:5]  # Top 5 gaps
                        
            except Exception as e:
                temporal['analysis_error'] = str(e)
                
        return temporal
    
    def _analyze_structure(self) -> dict:
        """Analyze data structure for signs of manipulation or issues."""
        structure = {
            'uniformity_score': 0,
            'naming_convention': 'unknown',
            'encoding_issues': [],
            'structural_anomalies': []
        }
        
        # Check column naming conventions
        col_names = list(self.data.columns)
        if all(c.islower() for c in ''.join(col_names) if c.isalpha()):
            structure['naming_convention'] = 'lowercase'
        elif all(c.isupper() for c in ''.join(col_names) if c.isalpha()):
            structure['naming_convention'] = 'uppercase'
        elif all('_' in str(c) for c in col_names):
            structure['naming_convention'] = 'snake_case'
        else:
            structure['naming_convention'] = 'mixed'
            
        # Check for encoding issues in string columns
        for col in self.data.select_dtypes(include=['object']).columns:
            sample = self.data[col].dropna().head(1000).astype(str)
            # Look for encoding artifacts
            encoding_artifacts = sample[sample.str.contains(r'[\x00-\x1f\x7f-\x9f]', regex=True, na=False)]
            if len(encoding_artifacts) > 0:
                structure['encoding_issues'].append({
                    'column': col,
                    'count': len(encoding_artifacts)
                })
                
        # Check for structural anomalies
        # Look for columns with single value (potential constants)
        for col in self.data.columns:
            unique_count = self.data[col].nunique()
            if unique_count == 1:
                structure['structural_anomalies'].append({
                    'type': 'constant_column',
                    'column': col,
                    'value': str(self.data[col].iloc[0]) if len(self.data) > 0 else None
                })
            elif unique_count == len(self.data):
                structure['structural_anomalies'].append({
                    'type': 'unique_identifier',
                    'column': col
                })
                
        # Calculate uniformity score (how consistent is the data structure)
        issues = len(structure['encoding_issues']) + len(structure['structural_anomalies'])
        structure['uniformity_score'] = max(0, 100 - (issues * 10))
        
        return structure
    
    def _generate_fingerprint(self) -> dict:
        """Generate a unique fingerprint for the dataset."""
        # Create hash of data content
        data_str = self.data.to_csv(index=False)
        content_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        
        # Create schema hash
        schema_str = str(list(self.data.columns)) + str(list(self.data.dtypes))
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
        
        return {
            'content_hash': content_hash,
            'schema_hash': schema_hash,
            'fingerprint': f"{schema_hash}-{content_hash}",
            'row_signature': len(self.data),
            'column_signature': len(self.data.columns)
        }
    
    def _calculate_credibility(self) -> float:
        """
        Calculate overall data credibility score (0-100).
        
        Based on:
        - Structural consistency
        - Temporal completeness
        - Metadata availability
        """
        score = 100.0
        concerns = []
        
        # Penalize for structural issues
        struct = self.findings.get('structure', {})
        if struct.get('naming_convention') == 'mixed':
            score -= 5
            concerns.append("Inconsistent column naming suggests multiple data sources")
            
        if struct.get('encoding_issues'):
            score -= 10
            concerns.append("Encoding issues detected - possible data corruption")
            
        anomalies = struct.get('structural_anomalies', [])
        if len(anomalies) > 3:
            score -= 10
            concerns.append(f"Multiple structural anomalies detected ({len(anomalies)})")
            
        # Penalize for temporal gaps
        temporal = self.findings.get('temporal', {})
        if temporal.get('temporal_gaps'):
            gap_count = len(temporal['temporal_gaps'])
            score -= min(20, gap_count * 4)
            concerns.append(f"Temporal gaps detected ({gap_count} significant gaps)")
            
        # Penalize for missing metadata
        metadata = self.findings.get('metadata', {})
        if not metadata.get('file_modified'):
            score -= 5
            concerns.append("File metadata not available")
            
        self.findings['concerns'] = concerns
        return max(0, min(100, score))
    
    def get_summary(self) -> str:
        """Generate a human-readable summary of provenance findings."""
        if not self.findings:
            self.analyze()
            
        meta = self.findings['metadata']
        cred = self.findings['credibility_score']
        
        summary = f"""
DATA PROVENANCE REPORT
======================
Source: {meta['source']}
Size: {meta['rows']:,} rows × {meta['columns']} columns
Memory: {meta['memory_usage_mb']:.2f} MB

Credibility Score: {cred:.1f}/100 {'✅' if cred >= 70 else '⚠️' if cred >= 50 else '❌'}

Concerns:
"""
        for concern in self.findings.get('concerns', []):
            summary += f"  • {concern}\n"
            
        if not self.findings.get('concerns'):
            summary += "  • No major concerns detected\n"
            
        return summary
