import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ASSET_COLUMNS, PERIODS_PER_YEAR, STOCK_COLUMNS, load_data  # noqa: E402

CORR_MATRIX_PATH = "L2/Q1/CorrelationMatrix.png"

# Sequential blue ramp, light -> dark (no negative correlations in this dataset, so 0-1 magnitude).
BLUE_SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_sequential", BLUE_SEQUENTIAL)
TEXT_MUTED = "#898781"
TEXT_PRIMARY = "#0b0b0b"


def log_return_stats(df: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(df[STOCK_COLUMNS] / df[STOCK_COLUMNS].shift(1)).dropna()

    stats = pd.DataFrame({
        "Annualized mean log-return": log_returns.mean() * PERIODS_PER_YEAR,
        "Annualized volatility": log_returns.std() * np.sqrt(PERIODS_PER_YEAR),
        "Min log-return": log_returns.min(),
        "Max log-return": log_returns.max(),
    })
    return stats


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(df[ASSET_COLUMNS] / df[ASSET_COLUMNS].shift(1)).dropna()
    return log_returns.corr()


def plot_correlation_matrix(corr: pd.DataFrame, path: str) -> None:
    n = len(corr.columns)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2, 0.9 * n + 1))
    im = ax.imshow(corr.values, cmap=BLUE_CMAP, vmin=0, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    ax.tick_params(length=0, labelsize=10, colors=TEXT_MUTED)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.values[i, j]
            color = "white" if abs(value) > 0.6 else TEXT_PRIMARY
            ax.text(j, i, f"{value * 100:.0f}%", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Correlation matrix of log-returns (stocks & rates)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    df = load_data()
    stats = log_return_stats(df)
    print((stats * 100).round(2).to_string())

    corr = correlation_matrix(df)
    plot_correlation_matrix(corr, CORR_MATRIX_PATH)


if __name__ == "__main__":
    main()
