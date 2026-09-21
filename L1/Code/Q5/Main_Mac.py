import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Import S&P500 data from spx_data.csv
# Fetch data from further back in time to accommodate rolling windows
df = pd.read_csv('spx_data.csv', 
                 skiprows=3, 
                 names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])

# Data setup
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
prices = df['Close']

# Calculate daily log-returns
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

# VaR at 98% confidence level with windows 500 and 1000 days
var_500 = -returns.rolling(window=500).quantile(0.02)
var_1000 = -returns.rolling(window=1000).quantile(0.02)

# Q2 parameters were in percentages, so we scale returns by 100 for the math
returns_pct = returns * 100

params_path = pathlib.Path(__file__).parent.parent / "Q2" / "garch11_params.json"
garch_params = json.loads(params_path.read_text())
omega = garch_params["omega"]
alpha = garch_params["alpha"]
beta = garch_params["beta"]
sigma2_init = returns_pct.var()

sigma2 = np.zeros(len(returns_pct))
sigma2[0] = sigma2_init

# Recursive loop for conditional variance
for i in range(1, len(returns_pct)):
    sigma2[i] = omega + alpha * returns_pct.iloc[i-1]**2 + beta * sigma2[i-1]

# Convert GARCH standard deviation back to decimal scale for the plot (/ 100)
sigma_series = pd.Series(np.sqrt(sigma2) / 100, index=returns.index)

# FHS Setup (Standardize using 2012-2021 data)
hist_returns = returns.loc[:'2021-12-31']
hist_sigma = sigma_series.loc[:'2021-12-31']
z_t = hist_returns / hist_sigma
q_fhs = np.percentile(z_t.dropna(), 2)
q_norm = norm.ppf(0.02)

# --- Define Forecasting Period ---
# Ensure sufficient data for rolling windows by fetching data from earlier
forecast_dates = slice('2022-01-01', '2025-12-31')
actual_losses = -returns.loc[forecast_dates]

# Slice all models to the forecast period
var_500_forecast = var_500.loc[forecast_dates]
var_1000_forecast = var_1000.loc[forecast_dates]
forecast_sigma = sigma_series.loc[forecast_dates]

var_GARCH_normal = -(forecast_sigma * q_norm)
var_GARCH_fhs = -(forecast_sigma * q_fhs)

# Count violations (actual loss > VaR forecast)
violations_500 = (actual_losses > var_500_forecast).sum()
violations_1000 = (actual_losses > var_1000_forecast).sum()
violations_GARCH_normal = (actual_losses > var_GARCH_normal).sum()
violations_GARCH_fhs = (actual_losses > var_GARCH_fhs).sum()

# Calculate ratios
total_days = len(actual_losses)
expected_violations = total_days * 0.02
ratio_500 = violations_500 / expected_violations
ratio_1000 = violations_1000 / expected_violations
ratio_GARCH_normal = violations_GARCH_normal / expected_violations
ratio_GARCH_fhs = violations_GARCH_fhs / expected_violations

# Print violation results
print(f"500-Day Window: {violations_500} violations, VR: {ratio_500:.4f}")
print(f"1000-Day Window: {violations_1000} violations, VR: {ratio_1000:.4f}")
print(f"GARCH Normal: {violations_GARCH_normal} violations, VR: {ratio_GARCH_normal:.4f}")
print(f"GARCH FHS: {violations_GARCH_fhs} violations, VRo: {ratio_GARCH_fhs:.4f}")

# --- Create Figure (Using your Q4 formatting) ---
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual daily losses
ax.plot(actual_losses.index, actual_losses, 
        label='Actual Losses (Negative Returns)', color='gray', alpha=0.6, linewidth=1)

# Plot Q4 Historical VaR
ax.plot(var_500_forecast.index, var_500_forecast, 
        label='98% VaR (500-day window)', color='blue', linewidth=1.5)
ax.plot(var_1000_forecast.index, var_1000_forecast, 
        label='98% VaR (1000-day window)', color='orange', linewidth=1.5)

# Plot Q5 GARCH VaR
ax.plot(var_GARCH_normal.index, var_GARCH_normal, 
        label='98% VaR (GARCH Normal)', color='red', linewidth=1.5)
ax.plot(var_GARCH_fhs.index, var_GARCH_fhs, 
        label='98% VaR (GARCH FHS)', color='purple', linewidth=1.5)

# Formatting the chart
ax.set_title('98% VaR Forecasts vs Actual Losses (All Models: 2022-2025)')
ax.set_ylabel('Loss / VaR')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)

# Display the plot
plt.tight_layout()
plt.show()