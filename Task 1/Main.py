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

# Calculate daily log-returns and squared returns
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()
y2 = returns ** 2

# Moving Average (MA) Volatility Forecast
W = 60
ma_var = y2.rolling(window=W).mean()
ma_vol = np.sqrt(ma_var)
ma_vol = ma_vol.dropna()
ma_vol_forecast = ma_vol.loc['2022-01-01':'2025-12-31']

# Exponentially Weighted Moving Average (EWMA) Volatility Forecast
lmbda = 0.94  # Decay factor for EWMA
W = 60 # Window size for EWMA

powers = np.arange(W, 0, -1)
multiplier = (1 - lmbda) / (lmbda * (1 - lmbda**W)) # Calculate the multiplier expression
weights = multiplier * (lmbda ** powers) # Calculate the weights for the EWMA

# Define a function to calculate the EWMA variance
def calc_ewma(y2):
    return np.sum(weights * y2)

# Calculate the EWMA volatility using the rolling apply method
ewma_var = y2.rolling(window=W).apply(calc_ewma, raw=True)
ewma_vol = np.sqrt(ewma_var)
ewma_vol = ewma_vol.dropna()
ewma_vol_forecast = ewma_vol.loc['2022-01-01':'2025-12-31']

# Plot setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot the volatility forecasts in the first subplot
ax1.plot(ma_vol_forecast.index, ma_vol_forecast, label='60-Day MA', color='blue', linewidth=1.5)
ax1.plot(ewma_vol_forecast.index, ewma_vol_forecast, label='EWMA (λ=0.94)', color='orange', alpha=0.8, linewidth=1.5)
ax1.set_title('S&P 500 Daily Volatility Forecasts (2022-2025)')
ax1.set_ylabel('Daily Volatility')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot the daily log-returns in the second subplot
returns_forecast = returns.loc['2022-01-01':'2025-12-31']
ax2.plot(returns_forecast.index, returns_forecast, label='Daily Log-Returns', color='gray', linewidth=0.8)
ax2.set_title('S&P 500 Daily Log-Returns')
ax2.set_ylabel('Log-Return')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Display the plot
plt.tight_layout()
plt.show()