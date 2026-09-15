import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import S&P500 data from spx_data.csv
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

# Choose forecasting period
var_500_forecast = var_500.loc['2022-01-01':'2025-12-31']
var_1000_forecast = var_1000.loc['2022-01-01':'2025-12-31']
negative_returns_forecast = -returns.loc['2022-01-01':'2025-12-31']

# Count violations (actual loss > VaR forecast)
violations_500 = (negative_returns_forecast > var_500_forecast).sum()
violations_1000 = (negative_returns_forecast > var_1000_forecast).sum()

# Calculate ratios
total_days = len(negative_returns_forecast)
ratio_500 = violations_500 / total_days
ratio_1000 = violations_1000 / total_days

# Print results
print(f"500-Day Window: {violations_500} violations, Ratio: {ratio_500:.4f}")
print(f"1000-Day Window: {violations_1000} violations, Ratio: {ratio_1000:.4f}")

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual daily losses (negative returns)
ax.plot(negative_returns_forecast.index, negative_returns_forecast, 
        label='Actual Losses (Negative Returns)', color='gray', alpha=0.6, linewidth=1)

# Plot the 500-day VaR forecast
ax.plot(var_500_forecast.index, var_500_forecast, 
        label='98% VaR (500-day window)', color='blue', linewidth=1.5)

# Plot the 1000-day VaR forecast
ax.plot(var_1000_forecast.index, var_1000_forecast, 
        label='98% VaR (1000-day window)', color='orange', linewidth=1.5)

# Formatting the chart
ax.set_title('98% VaR Forecasts vs Actual Losses (2022-2025)')
ax.set_ylabel('Loss / VaR')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)

# Display the plot
plt.tight_layout()
plt.show()