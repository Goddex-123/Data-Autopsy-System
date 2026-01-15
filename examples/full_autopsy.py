#!/usr/bin/env python3
"""
Full Autopsy Example

Demonstrates the complete forensic analysis capabilities of the Data Autopsy System
using a synthetic dataset that contains various data quality issues.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autopsy import DataAutopsy


def generate_problematic_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic dataset with intentional issues for demonstration.
    
    This dataset simulates a "crime scene" with:
    - Sampling bias (over-representation of certain groups)
    - Benford's Law violations (fabricated numbers)
    - Missing data patterns (systematic omissions)
    - Outliers and anomalies
    - Data quality issues
    """
    np.random.seed(42)
    
    # Age: biased towards working age, with some impossible values
    age_main = np.random.normal(35, 10, n_rows - 5)
    age_outliers = np.array([-5, -10, -3, 150, 200])
    age = np.concatenate([age_main, age_outliers])
    np.random.shuffle(age)
    
    # Income: Fabricated data with round numbers (doesn't follow Benford's Law well)
    income_round = np.round(np.random.uniform(30000, 100000, n_rows // 3), -3)
    income_repeated = np.array([50000] * (n_rows // 3))
    income_natural = np.random.lognormal(10.5, 0.5, n_rows - 2 * (n_rows // 3))
    income = np.concatenate([income_round, income_repeated, income_natural])
    np.random.shuffle(income)
    
    # Create base dataset
    data = {
        'record_id': range(1, n_rows + 1),
        'age': age,
        'income': income,
        'test_score': np.random.uniform(60, 100, n_rows),
        'gender': np.random.choice(['Male', 'Female'], n_rows, p=[0.8, 0.2]),
        'region': np.random.choice(['North', 'South', 'East', 'West', None], n_rows,
                                   p=[0.25, 0.25, 0.2, 0.2, 0.1]),
        'status': np.random.choice(['Active', 'Inactive', 'Pending'], n_rows,
                                   p=[0.9, 0.05, 0.05]),
        'transaction_date': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
        'amount': np.where(
            np.random.random(n_rows) < 0.15,
            -999,
            np.random.exponential(500, n_rows)
        ),
        'metric_a': np.random.normal(100, 15, n_rows),
    }
    
    # metric_b is almost perfectly correlated with metric_a (suspicious)
    data['metric_b'] = data['metric_a'] * 1.05 + np.random.normal(0, 0.5, n_rows)
    
    # Add some digit preference in quantity field
    quantities = []
    for _ in range(n_rows):
        if np.random.random() < 0.4:
            quantities.append(np.random.choice([10, 20, 30, 50, 100]))
        else:
            quantities.append(np.random.randint(1, 150))
    data['quantity'] = quantities
    
    df = pd.DataFrame(data)
    
    # Add systematic missing data pattern
    # Missing region is correlated with low income
    low_income_mask = df['income'] < df['income'].quantile(0.3)
    missing_region_indices = df[low_income_mask].sample(frac=0.5).index
    df.loc[missing_region_indices, 'region'] = None
    
    # Add duplicate rows (copy-paste pattern)
    duplicate_rows = df.sample(n=20)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df


def run_full_autopsy():
    """Run complete forensic analysis on the problematic dataset."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    🔬 DATA AUTOPSY DEMONSTRATION                      ║
║                                                                       ║
║   This example generates a synthetic dataset with intentional         ║
║   data quality issues and runs a full forensic investigation.         ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate problematic dataset
    print("📊 Generating synthetic dataset with intentional issues...")
    df = generate_problematic_dataset(1000)
    
    # Save the dataset for reference
    dataset_path = os.path.join(output_dir, 'sample_dataset.csv')
    df.to_csv(dataset_path, index=False)
    print(f"   Dataset saved to: {dataset_path}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Initialize autopsy
    print("🔬 Initializing Data Autopsy System...")
    autopsy = DataAutopsy(df)
    
    # Quick scan first
    print("\n📋 Quick Scan Results:")
    print("-" * 50)
    quick_results = autopsy.quick_scan()
    for key, value in quick_results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Full investigation
    print("\n" + "=" * 50)
    print("🔬 STARTING FULL FORENSIC INVESTIGATION")
    print("=" * 50 + "\n")
    
    report = autopsy.investigate(output_dir=output_dir)
    
    # Print summary
    report.print_summary()
    
    # Save all report formats
    print("\n📁 Saving Reports...")
    report.save('forensic_report.html')
    report.save('forensic_report.md')
    report.save('forensic_report.json')
    
    # Print individual module summaries
    print("\n" + "=" * 50)
    print("DETAILED MODULE REPORTS")
    print("=" * 50)
    
    print(autopsy.provenance.get_summary())
    print(autopsy.bias_detector.get_summary())
    print(autopsy.anomaly_detector.get_summary())
    print(autopsy.missing_analyzer.get_summary())
    print(autopsy.robustness_tester.get_summary())
    
    print("\n" + "=" * 50)
    print("✅ INVESTIGATION COMPLETE")
    print("=" * 50)
    print(f"\n📂 All outputs saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated files:")
    for f in os.listdir(output_dir):
        filepath = os.path.join(output_dir, f)
        size = os.path.getsize(filepath)
        print(f"   📄 {f} ({size:,} bytes)")
        
    print("\n🌐 Open forensic_report.html in your browser for the full interactive report!")
    
    return report


if __name__ == '__main__':
    run_full_autopsy()
