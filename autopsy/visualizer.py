"""
Forensic Visualizer Module

Generates visual evidence for forensic analysis reports.
Creates professional-quality visualizations for all findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import warnings

# Set style for all visualizations
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore', category=UserWarning)


class ForensicVisualizer:
    """
    Professional forensic data visualization generator.
    """
    
    # Color palette for forensic theme
    COLORS = {
        'primary': '#2C3E50',
        'warning': '#E74C3C',
        'success': '#27AE60',
        'info': '#3498DB',
        'secondary': '#95A5A6',
        'accent': '#9B59B6'
    }
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize forensic visualizer.
        
        Args:
            data: The dataset to visualize
        """
        self.data = data
        self.figures = []
        
    def generate_all(self, findings: dict, output_dir: str = "output") -> dict:
        """
        Generate all visualizations based on findings.
        
        Args:
            findings: All forensic findings from analyzers
            output_dir: Directory to save visualizations
            
        Returns:
            dict: Paths to generated visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated = {
            'data_overview': None,
            'bias_analysis': None,
            'anomaly_detection': None,
            'missing_data': None,
            'robustness': None
        }
        
        # 1. Data Overview Dashboard
        try:
            path = os.path.join(output_dir, 'data_overview.png')
            self._create_overview_dashboard(path)
            generated['data_overview'] = path
        except Exception as e:
            print(f"  Warning: Could not create overview: {e}")
            
        # 2. Bias Analysis Visualizations
        try:
            path = os.path.join(output_dir, 'bias_analysis.png')
            self._create_bias_visualization(findings.get('bias', {}), path)
            generated['bias_analysis'] = path
        except Exception as e:
            print(f"  Warning: Could not create bias viz: {e}")
            
        # 3. Anomaly Detection Visualizations
        try:
            path = os.path.join(output_dir, 'anomaly_detection.png')
            self._create_anomaly_visualization(findings.get('anomalies', {}), path)
            generated['anomaly_detection'] = path
        except Exception as e:
            print(f"  Warning: Could not create anomaly viz: {e}")
            
        # 4. Missing Data Visualizations
        try:
            path = os.path.join(output_dir, 'missing_data.png')
            self._create_missing_visualization(findings.get('missing', {}), path)
            generated['missing_data'] = path
        except Exception as e:
            print(f"  Warning: Could not create missing viz: {e}")
            
        # 5. Executive Summary Dashboard
        try:
            path = os.path.join(output_dir, 'executive_summary.png')
            self._create_executive_summary(findings, path)
            generated['executive_summary'] = path
        except Exception as e:
            print(f"  Warning: Could not create summary: {e}")
            
        return generated
    
    def _create_overview_dashboard(self, path: str):
        """Create data overview dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('DATA OVERVIEW DASHBOARD', fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Data Types Distribution
        ax = axes[0, 0]
        dtype_counts = self.data.dtypes.value_counts()
        colors = [self.COLORS['primary'], self.COLORS['info'], 
                 self.COLORS['accent'], self.COLORS['secondary']][:len(dtype_counts)]
        ax.pie(dtype_counts.values, labels=[str(d) for d in dtype_counts.index], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Data Types Distribution', fontweight='bold')
        
        # 2. Missing Data Overview
        ax = axes[0, 1]
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100).sort_values(ascending=False)
        cols_to_show = missing_pct[missing_pct > 0].head(10)
        if len(cols_to_show) > 0:
            bars = ax.barh(range(len(cols_to_show)), cols_to_show.values, color=self.COLORS['warning'])
            ax.set_yticks(range(len(cols_to_show)))
            ax.set_yticklabels([str(c)[:15] for c in cols_to_show.index])
            ax.set_xlabel('Missing %')
            ax.set_title('Top Columns with Missing Data', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', fontsize=14)
            ax.set_title('Missing Data', fontweight='bold')
            ax.axis('off')
            
        # 3. Numeric Distribution Summary
        ax = axes[0, 2]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sample_col = numeric_cols[0]
            values = self.data[sample_col].dropna()
            ax.hist(values, bins=30, color=self.COLORS['primary'], edgecolor='white', alpha=0.7)
            ax.axvline(values.mean(), color=self.COLORS['warning'], linestyle='--', 
                      linewidth=2, label=f'Mean: {values.mean():.2f}')
            ax.axvline(values.median(), color=self.COLORS['success'], linestyle='--', 
                      linewidth=2, label=f'Median: {values.median():.2f}')
            ax.legend()
            ax.set_title(f'Distribution: {sample_col[:20]}', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Numeric Data', ha='center', va='center', fontsize=14)
            ax.axis('off')
            
        # 4. Correlation Heatmap
        ax = axes[1, 0]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:8]
        if len(numeric_cols) >= 2:
            corr_matrix = self.data[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                       annot=True, fmt='.2f', ax=ax, cbar_kws={'shrink': 0.5})
            ax.set_title('Correlation Matrix', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient numeric columns', ha='center', va='center')
            ax.axis('off')
            
        # 5. Value Counts for Top Categorical Column
        ax = axes[1, 1]
        cat_cols = self.data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            col = cat_cols[0]
            top_values = self.data[col].value_counts().head(8)
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_values)))
            ax.barh(range(len(top_values)), top_values.values[::-1], color=colors)
            ax.set_yticks(range(len(top_values)))
            ax.set_yticklabels([str(v)[:15] for v in top_values.index[::-1]])
            ax.set_title(f'Top Values: {col[:20]}', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Categorical Data', ha='center', va='center', fontsize=14)
            ax.axis('off')
            
        # 6. Data Quality Quick Metrics
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate metrics
        total_cells = self.data.size
        missing_pct = self.data.isnull().sum().sum() / total_cells * 100
        duplicate_pct = self.data.duplicated().mean() * 100
        complete_rows_pct = (len(self.data) - self.data.isnull().any(axis=1).sum()) / len(self.data) * 100
        
        metrics_text = f"""
DATA QUALITY METRICS
━━━━━━━━━━━━━━━━━━━━

📊 Total Rows:          {len(self.data):,}
📋 Total Columns:       {len(self.data.columns)}
📦 Total Cells:         {total_cells:,}

🕳️ Missing Data:        {missing_pct:.1f}%
📑 Duplicate Rows:      {duplicate_pct:.1f}%
✅ Complete Rows:       {complete_rows_pct:.1f}%

🔢 Numeric Columns:     {len(self.data.select_dtypes(include=[np.number]).columns)}
📝 Text Columns:        {len(self.data.select_dtypes(include=['object']).columns)}
"""
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def _create_bias_visualization(self, bias_findings: dict, path: str):
        """Create bias analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('BIAS DETECTION ANALYSIS', fontsize=16, fontweight='bold', 
                    color=self.COLORS['primary'])
        
        # 1. Distribution Skewness
        ax = axes[0, 0]
        dist_bias = bias_findings.get('distribution_bias', {})
        skewed = dist_bias.get('skewed_columns', [])
        
        if skewed:
            cols = [s['column'][:15] for s in skewed[:8]]
            skews = [s['skewness'] for s in skewed[:8]]
            colors = [self.COLORS['warning'] if s['severity'] == 'high' 
                     else self.COLORS['info'] for s in skewed[:8]]
            bars = ax.barh(cols, skews, color=colors)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_title('Distribution Skewness', fontweight='bold')
            ax.set_xlabel('Skewness Value')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=self.COLORS['warning'], label='High Severity'),
                             Patch(facecolor=self.COLORS['info'], label='Moderate')]
            ax.legend(handles=legend_elements, loc='lower right')
        else:
            ax.text(0.5, 0.5, 'No significant skewness detected', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Distribution Skewness', fontweight='bold')
            
        # 2. Class Imbalance
        ax = axes[0, 1]
        sampling = bias_findings.get('sampling_bias', {})
        imbalance = sampling.get('class_imbalance', [])
        
        if imbalance:
            cols = [i['column'][:15] for i in imbalance[:8]]
            ratios = [min(i['imbalance_ratio'], 100) for i in imbalance[:8]]
            colors = [self.COLORS['warning'] if i['severity'] == 'high' 
                     else self.COLORS['secondary'] for i in imbalance[:8]]
            bars = ax.barh(cols, ratios, color=colors)
            ax.set_title('Class Imbalance Ratio', fontweight='bold')
            ax.set_xlabel('Imbalance Ratio')
            ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='Threshold')
        else:
            ax.text(0.5, 0.5, 'No significant class imbalance', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Class Imbalance', fontweight='bold')
            
        # 3. Overall Bias Score Gauge
        ax = axes[1, 0]
        score = bias_findings.get('overall_bias_score', 0)
        self._draw_gauge(ax, score, 'Bias Score', 
                        'Higher = More Bias Detected')
        
        # 4. Bias Summary Text
        ax = axes[1, 1]
        ax.axis('off')
        
        warnings = bias_findings.get('critical_warnings', [])
        
        summary = "BIAS ASSESSMENT SUMMARY\n" + "=" * 30 + "\n\n"
        
        if warnings:
            summary += "⚠️ CRITICAL WARNINGS:\n\n"
            for w in warnings[:5]:
                summary += f"  • {w}\n\n"
        else:
            summary += "✅ No critical bias warnings detected.\n\n"
            summary += "The data appears to have reasonable\n"
            summary += "balance and representation.\n"
            
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def _create_anomaly_visualization(self, anomaly_findings: dict, path: str):
        """Create anomaly detection visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ANOMALY DETECTION ANALYSIS', fontsize=16, fontweight='bold',
                    color=self.COLORS['warning'])
        
        # 1. Benford's Law Analysis
        ax = axes[0, 0]
        benford = anomaly_findings.get('benfords_law', {})
        violations = benford.get('violations', [])
        conforming = benford.get('conforming_columns', [])
        
        if violations or conforming:
            # Show Benford's Law expected distribution
            digits = list(range(1, 10))
            expected = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
            
            ax.bar(digits, expected, alpha=0.7, color=self.COLORS['success'], 
                  label='Expected (Benford)')
            ax.set_xlabel('First Digit')
            ax.set_ylabel('Frequency')
            ax.set_title("Benford's Law Analysis", fontweight='bold')
            ax.legend()
            
            # Add annotation
            if violations:
                ax.annotate(f'{len(violations)} column(s) violate\nBenford\'s Law',
                           xy=(7, 0.1), fontsize=10, color=self.COLORS['warning'],
                           fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Benford\'s Law not applicable\n(insufficient data)', 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Benford's Law", fontweight='bold')
            
        # 2. Outlier Summary
        ax = axes[0, 1]
        outliers = anomaly_findings.get('statistical_outliers', {})
        iqr_outliers = outliers.get('iqr_outliers', [])
        
        if iqr_outliers:
            cols = [o['column'][:15] for o in iqr_outliers[:8]]
            pcts = [o['percentage'] for o in iqr_outliers[:8]]
            colors = [self.COLORS['warning'] if o['severity'] == 'high' 
                     else self.COLORS['info'] if o['severity'] == 'moderate'
                     else self.COLORS['secondary'] for o in iqr_outliers[:8]]
            
            ax.barh(cols, pcts, color=colors)
            ax.set_xlabel('Outlier Percentage')
            ax.set_title('Outlier Distribution by Column', fontweight='bold')
            ax.axvline(5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No significant outliers detected', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Outlier Analysis', fontweight='bold')
            
        # 3. Anomaly Score Gauge
        ax = axes[1, 0]
        score = anomaly_findings.get('overall_anomaly_score', 0)
        self._draw_gauge(ax, score, 'Anomaly Score', 
                        'Higher = More Anomalies')
        
        # 4. Red Flags Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        red_flags = anomaly_findings.get('red_flags', [])
        
        summary = "🚨 ANOMALY RED FLAGS\n" + "=" * 30 + "\n\n"
        
        if red_flags:
            for flag in red_flags[:6]:
                summary += f"  🔴 {flag}\n\n"
        else:
            summary += "✅ No major anomalies detected.\n\n"
            summary += "Data appears to be naturally\n"
            summary += "generated without obvious\n"
            summary += "fabrication or manipulation.\n"
            
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#fff0f0', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def _create_missing_visualization(self, missing_findings: dict, path: str):
        """Create missing data visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MISSING DATA FORENSICS', fontsize=16, fontweight='bold',
                    color=self.COLORS['info'])
        
        # 1. Missing Data Heatmap
        ax = axes[0, 0]
        missing_matrix = self.data.isnull()
        cols_with_missing = [c for c in self.data.columns if missing_matrix[c].any()]
        
        if cols_with_missing and len(cols_with_missing) <= 15:
            sample_size = min(100, len(self.data))
            sample_indices = np.linspace(0, len(self.data)-1, sample_size, dtype=int)
            
            sns.heatmap(missing_matrix.iloc[sample_indices][cols_with_missing[:15]], 
                       cbar=True, cmap='YlOrRd', ax=ax, yticklabels=False)
            ax.set_title('Missing Data Pattern (Sample)', fontweight='bold')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Rows (sampled)')
        else:
            # Show bar chart instead
            missing_pct = (self.data.isnull().sum() / len(self.data) * 100)
            missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
            
            if len(missing_pct) > 0:
                ax.barh(range(len(missing_pct)), missing_pct.values[::-1], 
                       color=self.COLORS['warning'])
                ax.set_yticks(range(len(missing_pct)))
                ax.set_yticklabels([str(c)[:15] for c in missing_pct.index[::-1]])
                ax.set_xlabel('Missing %')
                ax.set_title('Missing Data by Column', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No missing data!', 
                       ha='center', va='center', fontsize=14, color=self.COLORS['success'])
                ax.set_title('Missing Data', fontweight='bold')
                ax.axis('off')
                
        # 2. Missing Data Overview Pie
        ax = axes[0, 1]
        overview = missing_findings.get('overview', {})
        total_cells = overview.get('total_cells', self.data.size)
        missing_cells = overview.get('missing_cells', 0)
        complete_cells = total_cells - missing_cells
        
        if missing_cells > 0:
            sizes = [complete_cells, missing_cells]
            labels = [f'Complete\n({complete_cells:,})', f'Missing\n({missing_cells:,})']
            colors = [self.COLORS['success'], self.COLORS['warning']]
            explode = (0, 0.05)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('Data Completeness', fontweight='bold')
        else:
            ax.text(0.5, 0.5, '100% Complete!', 
                   ha='center', va='center', fontsize=16, color=self.COLORS['success'])
            ax.set_title('Data Completeness', fontweight='bold')
            ax.axis('off')
            
        # 3. Missing Data Score
        ax = axes[1, 0]
        score = missing_findings.get('overall_missing_score', 0)
        self._draw_gauge(ax, score, 'Missing Data\nConcern Score', 
                        'Higher = More Concern')
        
        # 4. Summary and Recommendations
        ax = axes[1, 1]
        ax.axis('off')
        
        concerns = missing_findings.get('concerns', [])
        silent = missing_findings.get('silent_assumptions', {})
        
        summary = "MISSING DATA SUMMARY\n" + "=" * 30 + "\n\n"
        summary += f"Total Missing: {overview.get('missing_percentage', 0):.1f}%\n"
        summary += f"Columns Affected: {overview.get('columns_with_missing', 0)}\n\n"
        
        if concerns:
            summary += "⚠️ CONCERNS:\n"
            for c in concerns[:4]:
                summary += f"  • {c[:40]}\n"
        
        if silent.get('sentinel_values'):
            summary += "\n🔍 HIDDEN MISSING:\n"
            for s in silent['sentinel_values'][:2]:
                summary += f"  • {s['column']}: value {s['value']}\n"
                
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def _create_executive_summary(self, findings: dict, path: str):
        """Create executive summary dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('FORENSIC ANALYSIS EXECUTIVE SUMMARY', 
                    fontsize=18, fontweight='bold', color=self.COLORS['primary'])
        
        # Get all scores
        provenance_score = findings.get('provenance', {}).get('credibility_score', 0)
        bias_score = findings.get('bias', {}).get('overall_bias_score', 0)
        anomaly_score = findings.get('anomalies', {}).get('overall_anomaly_score', 0)
        missing_score = findings.get('missing', {}).get('overall_missing_score', 0)
        robustness_score = findings.get('robustness', {}).get('overall_robustness_score', 100)
        
        # 1. Provenance Gauge
        self._draw_gauge(axes[0, 0], provenance_score, 'Credibility', '(Higher = Better)', 
                        invert=True)
        
        # 2. Bias Gauge
        self._draw_gauge(axes[0, 1], bias_score, 'Bias Score', '(Lower = Better)')
        
        # 3. Anomaly Gauge
        self._draw_gauge(axes[0, 2], anomaly_score, 'Anomaly Score', '(Lower = Better)')
        
        # 4. Missing Gauge
        self._draw_gauge(axes[1, 0], missing_score, 'Missing Score', '(Lower = Better)')
        
        # 5. Robustness Gauge
        self._draw_gauge(axes[1, 1], robustness_score, 'Robustness', '(Higher = Better)', 
                        invert=True)
        
        # 6. Overall Assessment
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate overall health score
        # High credibility + high robustness = good
        # Low bias + low anomaly + low missing = good
        overall = (provenance_score + robustness_score + 
                  (100 - bias_score) + (100 - anomaly_score) + (100 - missing_score)) / 5
        
        if overall >= 70:
            emoji = "✅"
            verdict = "HEALTHY"
            color = self.COLORS['success']
        elif overall >= 50:
            emoji = "⚠️"
            verdict = "CONCERNS"
            color = self.COLORS['info']
        else:
            emoji = "🚨"
            verdict = "PROBLEMATIC"
            color = self.COLORS['warning']
            
        summary = f"""
{emoji} OVERALL DATA HEALTH

━━━━━━━━━━━━━━━━━━━━━━━

Overall Score: {overall:.0f}/100

Verdict: {verdict}

━━━━━━━━━━━━━━━━━━━━━━━

Review the detailed reports
for specific findings and
recommendations.
"""
        ax.text(0.1, 0.85, summary, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor=color, 
                        linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def _draw_gauge(self, ax, value: float, title: str, subtitle: str = "", 
                   invert: bool = False):
        """Draw a gauge chart for a score value."""
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Determine color based on value
        if invert:
            # High is good (green), low is bad (red)
            if value >= 70:
                color = self.COLORS['success']
            elif value >= 40:
                color = self.COLORS['info']
            else:
                color = self.COLORS['warning']
        else:
            # Low is good (green), high is bad (red)
            if value <= 30:
                color = self.COLORS['success']
            elif value <= 60:
                color = self.COLORS['info']
            else:
                color = self.COLORS['warning']
                
        # Draw background arc
        theta = np.linspace(0, np.pi, 100)
        x_bg = np.cos(theta)
        y_bg = np.sin(theta)
        ax.plot(x_bg, y_bg, color='#e0e0e0', linewidth=20, solid_capstyle='round')
        
        # Draw value arc
        value_angle = np.pi * (value / 100)
        theta_val = np.linspace(0, value_angle, 50)
        x_val = np.cos(np.pi - theta_val)
        y_val = np.sin(np.pi - theta_val)
        ax.plot(x_val, y_val, color=color, linewidth=20, solid_capstyle='round')
        
        # Add value text
        ax.text(0, 0.4, f'{value:.0f}', fontsize=28, fontweight='bold',
               ha='center', va='center', color=color)
        
        # Add title
        ax.text(0, -0.1, title, fontsize=12, fontweight='bold',
               ha='center', va='center', color=self.COLORS['primary'])
        
        if subtitle:
            ax.text(0, -0.25, subtitle, fontsize=9, ha='center', va='center',
                   color=self.COLORS['secondary'])
