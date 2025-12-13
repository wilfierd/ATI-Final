import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("Reading entire dataset 'sensor.csv'...")
try:
    df = pd.read_csv('sensor.csv')
    print(f"Successfully read {len(df)} rows and {len(df.columns)} columns.")
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 1. Class Imbalance Analysis
print("\n=== Class Distribution ===")
if 'machine_status' in df.columns:
    counts = df['machine_status'].value_counts()
    percents = df['machine_status'].value_counts(normalize=True) * 100
    for cls, count in counts.items():
        print(f"{cls}: {count} ({percents[cls]:.4f}%)")
else:
    print("Column 'machine_status' not found!")

# 2. Time Series Continuity Check
print("\n=== Time Series Analysis ===")
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Check if sorted
    is_sorted = df['timestamp'].is_monotonic_increasing
    print(f"Is data chronologically sorted? {is_sorted}")
    
    # Check time deltas
    deltas = df['timestamp'].diff().dropna()
    print(f"Most common time interval: {deltas.mode()[0]}")
    print(f"Min interval: {deltas.min()}")
    print(f"Max interval: {deltas.max()}")
else:
    print("No 'timestamp' column found.")

# 3. Missing Data Patterns
print("\n=== Missing Data Patterns ===")
missing_counts = df.isnull().sum().sort_values(ascending=False)
print(f"Top 5 columns with missing values:\n{missing_counts.head(5)}")
print(f"Total rows with at least one missing value: {df.isnull().any(axis=1).sum()}")

# 4. Sensor Value Patterns (Sample)
print("\n=== Sensor Statistics (First 5 sensors) ===")
sensor_cols = [c for c in df.columns if 'sensor' in c][:5]
print(df[sensor_cols].describe())

# 5. Check for "Perfect" Correlations or Leaks
print("\n=== Correlation with Target (Simple Label Encoding) ===")
if 'machine_status' in df.columns:
    df['target_enc'] = df['machine_status'].astype('category').cat.codes
    # Correlation of sensors with target
    corrs = df[sensor_cols].corrwith(df['target_enc'])
    print(corrs)
