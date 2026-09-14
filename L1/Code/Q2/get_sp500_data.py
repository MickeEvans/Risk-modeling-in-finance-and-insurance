"""
Download historical S&P 500 (^GSPC) price data and save it as a CSV file.

Requires: yfinance (pip install yfinance)
Usage:    python get_sp500_data.py
"""

import numpy as np
import yfinance as yf

#  Settings 
TICKER = "^GSPC"          # S&P 500 index ticker on Yahoo Finance
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"   # note: yfinance's 'end' is exclusive, handled below
OUTPUT_FILE = "Q2/sp500_2022_2025.csv"

def main():
    # yfinance's `end` param is exclusive, so add one day to include END_DATE
    import datetime
    end_inclusive = (
        datetime.datetime.strptime(END_DATE, "%Y-%m-%d") + datetime.timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # Download a buffer of extra days before START_DATE so the log return for
    # the first trading day in range has a previous close to compute against
    # (otherwise it comes out NaN and gets dropped, shifting the series).
    download_start = (
        datetime.datetime.strptime(START_DATE, "%Y-%m-%d") - datetime.timedelta(days=10)
    ).strftime("%Y-%m-%d")

    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")

    data = yf.download(
        TICKER,
        start=download_start,
        end=end_inclusive,
        auto_adjust=False,   # keep both 'Close' and 'Adj Close' columns
        progress=False,
    )

    if data is None or data.empty:
        print("No data returned. Check your internet connection or the ticker symbol.")
        return

    # yfinance can return a MultiIndex on columns when downloading a single
    # ticker in newer versions — flatten it just in case.
    if isinstance(data.columns, __import__("pandas").MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Daily log return: ln(Close_t / Close_{t-1}); computed on the buffered
    # series, then trimmed back to the requested window.
    data["Log_returns"] = np.log(data["Close"] / data["Close"].shift(1))
    data = data[data.index >= START_DATE]

    data.index.name = "Date"
    data.to_csv(OUTPUT_FILE)

    print(f"Saved {len(data)} rows to {OUTPUT_FILE}")
    print(data.head())


if __name__ == "__main__":
    main()