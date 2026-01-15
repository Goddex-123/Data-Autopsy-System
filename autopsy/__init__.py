"""
Data Autopsy System - Forensic Dataset Analysis

A master-level forensic analysis toolkit for investigating datasets.
"""

from .provenance import DataProvenance
from .bias_detector import BiasDetector
from .anomaly_detector import AnomalyDetector
from .missing_analyzer import MissingDataAnalyzer
from .robustness import RobustnessTester
from .visualizer import ForensicVisualizer

__version__ = "1.0.0"
__author__ = "Data Autopsy Team"


class DataAutopsy:
    """
    Main forensic investigation class that orchestrates all analysis modules.
    
    Treats datasets like crime scenes - investigating provenance, bias,
    anomalies, missing data, and conclusion robustness.
    """
    
    def __init__(self, data_source):
        """
        Initialize the Data Autopsy investigation.
        
        Args:
            data_source: Path to CSV file, pandas DataFrame, or URL
        """
        import pandas as pd
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
            self.source_path = "DataFrame"
        elif isinstance(data_source, str):
            self.source_path = data_source
            self.data = pd.read_csv(data_source)
        else:
            raise ValueError("data_source must be a file path or pandas DataFrame")
        
        # Initialize all forensic modules
        self.provenance = DataProvenance(self.data, self.source_path)
        self.bias_detector = BiasDetector(self.data)
        self.anomaly_detector = AnomalyDetector(self.data)
        self.missing_analyzer = MissingDataAnalyzer(self.data)
        self.robustness_tester = RobustnessTester(self.data)
        self.visualizer = ForensicVisualizer(self.data)
        
        # Investigation results
        self.findings = {}
        
    def investigate(self, output_dir="output"):
        """
        Run complete forensic investigation on the dataset.
        
        Args:
            output_dir: Directory to save visual evidence
            
        Returns:
            ForensicReport: Complete investigation report
        """
        from reports.generator import ForensicReport
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔬 Starting forensic investigation...")
        
        # 1. Provenance Analysis
        print("  📋 Analyzing data provenance...")
        self.findings['provenance'] = self.provenance.analyze()
        
        # 2. Bias Detection
        print("  ⚖️ Detecting biases...")
        self.findings['bias'] = self.bias_detector.analyze()
        
        # 3. Anomaly Detection
        print("  🚨 Scanning for anomalies...")
        self.findings['anomalies'] = self.anomaly_detector.analyze()
        
        # 4. Missing Data Analysis
        print("  🕳️ Investigating missing data...")
        self.findings['missing'] = self.missing_analyzer.analyze()
        
        # 5. Robustness Testing
        print("  🧪 Testing conclusion robustness...")
        self.findings['robustness'] = self.robustness_tester.analyze()
        
        # 6. Generate Visual Evidence
        print("  📊 Generating visual evidence...")
        self.findings['visualizations'] = self.visualizer.generate_all(
            self.findings, output_dir
        )
        
        print("✅ Investigation complete!")
        
        # Generate report
        report = ForensicReport(self.findings, self.source_path, output_dir)
        return report
    
    def quick_scan(self):
        """
        Perform a quick preliminary scan without full analysis.
        
        Returns:
            dict: Quick scan results with key concerns
        """
        return {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'missing_percentage': (self.data.isnull().sum().sum() / self.data.size) * 100,
            'duplicate_rows': self.data.duplicated().sum(),
            'numeric_columns': len(self.data.select_dtypes(include=['number']).columns),
            'categorical_columns': len(self.data.select_dtypes(include=['object']).columns),
        }


__all__ = [
    'DataAutopsy',
    'DataProvenance',
    'BiasDetector', 
    'AnomalyDetector',
    'MissingDataAnalyzer',
    'RobustnessTester',
    'ForensicVisualizer'
]
