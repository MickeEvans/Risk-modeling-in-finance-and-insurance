import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RATE_COLUMNS, STOCK_COLUMNS, load_data, style_axes  # noqa: E402

# Tab10 qualitative palette: 10 distinguishable hues, one per stock.
STOCK_COLORS = plt.get_cmap("tab10").colors

# Fixed-order categorical palette (blue..red), one per interest rate tenor.
RATE_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e34948",  # red
]


def plot_stocks(df: pd.DataFrame) -> None:
    normalized = df[STOCK_COLUMNS] / df[STOCK_COLUMNS].iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for column, color in zip(STOCK_COLUMNS, STOCK_COLORS):
        ax.plot(normalized.index, normalized[column], label=column, color=color, linewidth=1.5)

    ax.set_title("Stock price evolution, 1996-2025 (indexed to 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized price (start = 100)")
    style_axes(ax)
    ax.legend(title="Stock", ncol=2, frameon=False)
    fig.tight_layout()


def plot_rates(df: pd.DataFrame) -> None:
    rates_pct = df[RATE_COLUMNS] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for column, color in zip(RATE_COLUMNS, RATE_COLORS):
        ax.plot(rates_pct.index, rates_pct[column], label=column, color=color, linewidth=1.5)

    ax.set_title("Interest rate evolution, 1996-2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (%)")
    style_axes(ax)
    ax.legend(title="Tenor", frameon=False)
    fig.tight_layout()


def main() -> None:
    df = load_data()
    plot_stocks(df)
    plot_rates(df)
    plt.show()


if __name__ == "__main__":
    main()
