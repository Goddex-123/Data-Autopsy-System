import pandas as pd
import numpy as np

def generate_massive_sample(n_rows: int = 25000) -> pd.DataFrame:
    """
    Generates a massive synthetic dataset containing various data quality issues,
    anomalies, biases, privacy risks, and target leakage to demonstrate all features.
    """
    np.random.seed(42)
    
    # Base target variable (for leakage and bias)
    target = np.random.choice([0, 1], size=n_rows, p=[0.85, 0.15])  # Class imbalance
    
    df = pd.DataFrame()
    
    # 1. Normal numeric data (clean)
    df['age'] = np.random.normal(35, 10, n_rows).clip(18, 90).astype(int)
    
    # 2. Missing data patterns (MCAR, MAR)
    income = np.random.lognormal(11, 0.8, n_rows)
    # MAR: Older people missing more income data
    income_missing_prob = np.where(df['age'] > 60, 0.4, 0.05)
    df['annual_income'] = np.where(np.random.random(n_rows) < income_missing_prob, np.nan, income)
    
    # Sentinel values
    df['credit_score'] = np.where(np.random.random(n_rows) < 0.1, -999, np.random.normal(650, 50, n_rows))
    
    # 3. Anomalies & Benford's Law
    df['transaction_amount'] = np.random.lognormal(5, 2, n_rows)  # Benford applicable
    # Inject outliers
    outlier_indices = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df.loc[outlier_indices, 'transaction_amount'] = df.loc[outlier_indices, 'transaction_amount'] * 50
    
    # 4. Target Leakage (ML Audit)
    df['target'] = target
    # Direct leakage
    df['is_churned_flag'] = np.where(np.random.random(n_rows) < 0.95, target, 1 - target)
    
    # 5. Bias & Representation
    # Region A is 90% of the data, Region B, C, D are minority
    df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_rows, p=[0.85, 0.05, 0.05, 0.05])
    
    # 6. Privacy & PII
    df['customer_email'] = [f"user_{i}@example.com" for i in range(n_rows)]
    df['phone_number'] = [f"+1-555-{np.random.randint(1000, 9999)}" for _ in range(n_rows)]
    df['ssn_last_4'] = np.random.randint(1000, 9999, n_rows).astype(str)
    
    # 7. Data Quality & Consistency
    # Mixed types
    mixed_col = ["Valid" if np.random.random() > 0.05 else 42 for _ in range(n_rows)]
    df['status_code'] = mixed_col
    
    # Case inconsistencies
    cities = ['New York', 'new york', 'NEW YORK', 'Boston', 'boston', 'Chicago']
    df['city'] = np.random.choice(cities, size=n_rows)
    
    # Duplicates
    # Duplicate the first 100 rows
    df = pd.concat([df, df.iloc[:100]], ignore_index=True)
    
    return df
