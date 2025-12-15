import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('sensor.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get first break index
idx = 17155  # First break

# Select 10 sensors (some high variance, some low)
sensors = ['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 
           'sensor_05', 'sensor_06', 'sensor_10', 'sensor_50', 'sensor_51']

start = idx - 10
end = idx + 5
subset = df.iloc[start:end]

plt.figure(figsize=(20, 15))
for i, sensor in enumerate(sensors):
    plt.subplot(5, 2, i+1)
    plt.plot(subset['timestamp'], subset[sensor], marker='o')
    plt.axvline(subset.iloc[10]['timestamp'], color='red', linestyle='--', label='BROKEN Label')
    plt.title(sensor)
    plt.grid(True)

plt.tight_layout()
plt.savefig('all_sensors_break.png')
print("Plot saved to all_sensors_break.png")

# Check which sensors change significantly > 1 min before break
print("\n=== SENSOR ANALYSIS (t-5 mins vs t-1 min) ===")
t_minus_5 = subset.iloc[5]
t_minus_1 = subset.iloc[9] # 1 min before label
diffs = []
for col in df.columns:
    if 'sensor' in col and col != 'sensor_15':
        val5 = t_minus_5[col]
        val1 = t_minus_1[col]
        # Calculate % change
        if val5 != 0:
            pct_change = (val1 - val5) / val5 * 100
        else:
            pct_change = 0
            
        if abs(pct_change) > 10: # Significant change
            diffs.append((col, val5, val1, pct_change))

diffs.sort(key=lambda x: abs(x[3]), reverse=True)
for d in diffs:
    print(f"{d[0]}: {d[1]:.2f} -> {d[2]:.2f} ({d[3]:+.1f}%)")
