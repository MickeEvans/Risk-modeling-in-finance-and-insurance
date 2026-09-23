import yfinance as yf

# Settings
TICKER = "^GSPC"          # S&P 500 index ticker on Yahoo Finance
START_DATE = "2012-01-01"
END_DATE = "2021-12-31"
OUTPUT_FILE = "Q2/sp500_2012_2021.csv"

# Download the data
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Save it to a CSV file
data.to_csv(OUTPUT_FILE)
