"""
Download historical NASDAQ (^IXIC) price data and save it as a CSV file.

Requires: yfinance (pip install yfinance)
Usage:    python get_nasdaq_data.py
"""

import numpy as np 
import yfinance as yf

# --- Settings ---------------------------------------------------------
TICKER = "^IXIC"          # NASDAQ index ticker on Yahoo Finance
START_DATE = "2012-01-01"
END_DATE = "2021-12-31"   # note: yfinance's 'end' is exclusive, handled below
OUTPUT_FILE = "Q2/nasdaq_2012_2021.csv"
# -----------------------------------------------------------------------

def main():
    # yfinance's `end` param is exclusive, so add one day to include END_DATE
    import datetime
    end_inclusive = (
        datetime.datetime.strptime(END_DATE, "%Y-%m-%d") + datetime.timedelta(days=1)
    ).strftime("%Y-%m-%d")

    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")

    data = yf.download(
        TICKER,
        start=START_DATE,
        end=end_inclusive,
        auto_adjust=False,   # keep both 'Close' and 'Adj Close' columns
        progress=False,
    )

    if  data is None or data.empty:
        print("No data returned. Check your internet connection or the ticker symbol.")
        return

    # yfinance can return a MultiIndex on columns when downloading a single
    # ticker in newer versions — flatten it just in case.
    if isinstance(data.columns, __import__("pandas").MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Daily log return: ln(Close_t / Close_{t-1}); first row is NaN.
    data["Log_returns"] = np.log(data["Close"] / data["Close"].shift(1))

    data.index.name = "Date"
    data.to_csv(OUTPUT_FILE)

    print(f"Saved {len(data)} rows to {OUTPUT_FILE}")
    print(data.head())


if __name__ == "__main__":
    main()