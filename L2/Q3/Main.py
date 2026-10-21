import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Data preparation
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, '..', 'market_data.csv'), sep=';')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate log returns for ALL assets
log_returns = np.log(df / df.shift(1)).dropna()
asset_names = df.columns.tolist()

## 2. Historical simulation
n_scenarios = 5000
n_steps = 480
np.random.seed(42) 

# Random sample of historical log returns
historical_indices = np.random.randint(0, len(log_returns), size=(n_scenarios, n_steps))
simulated_log_returns = log_returns.values[historical_indices]

# Simulate paths for all assets using exponentiated log-returns
last_prices = df.iloc[-1].values
cumulative_log_returns = np.cumsum(simulated_log_returns, axis=1)
simulated_paths = last_prices * np.exp(cumulative_log_returns)

# 3. Percentiles & tables
percentiles = [5, 10, 25, 50, 75, 90, 95]
simulated_percentiles = np.percentile(simulated_paths, percentiles, axis=0)

# Filter to the 4 required assets and format row names to match Task 2 perfectly
plot_assets = {
    "CAT": "Caterpillar (CAT)", 
    "WMT": "Walmart (WMT)", 
    "Y05": "5-year rate (Y05) [%]", 
    "Y30": "30-year rate (Y30) [%]"
}

def generate_table(week_idx):
    rows = {}
    for asset, row_name in plot_assets.items():
        asset_idx = asset_names.index(asset)
        values = simulated_percentiles[:, week_idx, asset_idx]
        
        # Multiply interest rates by 100 to match the [%] unit in the table
        if asset in ['Y05', 'Y30']:
            values = values * 100
            
        rows[row_name] = values
    return pd.DataFrame(rows, index=[f"{p}%" for p in percentiles]).T

# Generate tables (index 239 = week 240, index 479 = week 480)
table_240 = generate_table(239)
table_480 = generate_table(479)

# Export directly to LaTeX format
table_240.round(2).to_latex(os.path.join(script_dir, 'table_240.tex'))
table_480.round(2).to_latex(os.path.join(script_dir, 'table_480.tex'))

# 4. Plotting
try:
    var1_percentiles = np.load(os.path.join(script_dir, '..', 'Q2', 'percentiles_var1.npy'))
    has_var1 = True
except FileNotFoundError:
    print("\nNote: Q2/percentiles_var1.npy not found.")
    has_var1 = False

assets_to_plot = ['CAT', 'WMT', 'Y05', 'Y30']
asset_indices = [asset_names.index(asset) for asset in assets_to_plot]
time_steps = np.arange(1, n_steps + 1)

for asset_name, asset_idx in zip(assets_to_plot, asset_indices):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for p_idx, p_val in enumerate(percentiles):
        # Plot VAR(1) as solid black lines
        if has_var1:
            var1_line = var1_percentiles[p_idx, 1:, asset_idx]
            ax.plot(time_steps, var1_line, color='black', linestyle='-', alpha=0.6, 
                    label='VAR(1)' if p_val == 50 else None)
        
        # Plot Historical Simulation as dashed blue lines
        hs_line = simulated_percentiles[p_idx, :, asset_idx]
        ax.plot(time_steps, hs_line, color='blue', linestyle='--', alpha=0.8,
                label='Historical Simulation' if p_val == 50 else None)
        
    ax.set_title(f'Historical Simulation vs VAR(1): {asset_name}')
    ax.set_xlabel('Time Step (Weeks)')
    ax.set_ylabel('Price / Yield')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'Q3_plot_{asset_name}.png'), dpi=150)
    plt.close()

# Export to LaTeX
table_240.round(2).to_latex(os.path.join(script_dir, 'table_240.tex'))
table_480.round(2).to_latex(os.path.join(script_dir, 'table_480.tex'))