import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data from the parent directory
# Ensure you specify the semicolon separator
df = pd.read_csv('../market_data.csv', sep=';')

# 2. Convert 'Date' to datetime and set it as index 
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 3. Calculate weekly logarithmic returns for all assets
# Formula: ln(P_t / P_{t-1})
log_returns = np.log(df / df.shift(1)).dropna()

print(log_returns.head())


# Simulation setup

n_scenarios = 5000
n_steps = 480
last_prices = df.iloc[-1].values

# Random sample of historical log returns
historical_indices = np.random.choice(
    a=len(log_returns), 
    size=(n_scenarios, n_steps), 
    replace=True
)

simulated_log_returns = log_returns.values[historical_indices]

cumulative_log_returns = np.cumsum(simulated_log_returns, axis=1)
simulated_paths = last_prices * np.exp(cumulative_log_returns)

# Calculate percentiles
percentiles = [5, 10, 25, 50, 75, 90, 95]
simulated_percentiles = np.percentile(simulated_paths, percentiles, axis=0)

# Get column (asset) names
asset_names = df.columns.tolist() 

t240_data = simulated_percentiles[:, 239, :].T  # Transpose to get assets as rows
t480_data = simulated_percentiles[:, 479, :].T

# Convert to Pandas DataFrames for nice formatting
table_240 = pd.DataFrame(t240_data, index=asset_names, columns=[f"{p}%" for p in percentiles])
table_480 = pd.DataFrame(t480_data, index=asset_names, columns=[f"{p}%" for p in percentiles])

print("--- Percentiles at Time Step 240 ---")
print(table_240)

print("\n--- Percentiles at Time Step 480 ---")
print(table_480)

# Find the column indices for the four required assets
assets_to_plot = ['CAT', 'WMT', 'Y05', 'Y30']
asset_indices = [asset_names.index(asset) for asset in assets_to_plot]

time_steps = np.arange(1, 481) # X-axis from 1 to 480

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (asset_name, asset_idx) in enumerate(zip(assets_to_plot, asset_indices)):
    ax = axes[i]
    
    # Plot each percentile as a dashed line
    for p_idx, p_val in enumerate(percentiles):
        # Extract the line for this specific percentile and asset over all 480 steps
        line_data = simulated_percentiles[p_idx, :, asset_idx]
        
        # Plot with dashed lines ('--') for Historical Simulation
        ax.plot(time_steps, line_data, linestyle='--', label=f'{p_val}th pctl')
        
    ax.set_title(f'Historical Simulation Percentiles: {asset_name}')
    ax.set_xlabel('Time Step (Weeks)')
    ax.set_ylabel('Price / Rate')
    # Place legend outside the plot to avoid covering the lines
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()