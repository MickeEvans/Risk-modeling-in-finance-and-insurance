import yfinance as yf

# Download the data
data = yf.download("^SPX", start="2018-01-01", end="2025-12-31")

# Save it to a CSV file
data.to_csv("spx_data.csv")