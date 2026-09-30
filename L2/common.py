"""Shared data loading and plotting for the L2 questions (run scripts from the Laboration root)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "L2/market_data.csv"
START_DATE = "1996-01-01"
END_DATE = "2025-12-31"

STOCK_COLUMNS = ["BA", "CAT", "KO", "DIS", "JPM", "MMM", "MSFT", "PFE", "WMT", "XOM"]
RATE_COLUMNS = ["Y05", "Y10", "Y15", "Y20", "Y30"]
ASSET_COLUMNS = STOCK_COLUMNS + RATE_COLUMNS

# Data is sampled weekly.
PERIODS_PER_YEAR = 52

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
PERCENTILE_COLORS = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(PERCENTILES)))

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df.loc[START_DATE:END_DATE, ASSET_COLUMNS]


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly log-returns for all assets (stocks first, rates last)."""
    return np.log(df / df.shift(1)).dropna()


def style_axes(ax) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_COLOR)


def plot_fan(ax, q: np.ndarray, asset: str, model: str = "VAR(1)", linestyle: str = "-") -> None:
    """Add one line per percentile for `asset` to `ax`.

    q has shape (len(PERCENTILES), weeks + 1, n_assets) with index 0 = today. Rates are shown in %.
    Later questions call this again on the same ax with linestyle "--" or ":" to overlay their model.
    """
    i = ASSET_COLUMNS.index(asset)
    scale = 100 if asset in RATE_COLUMNS else 1
    weeks = np.arange(q.shape[1])
    for k in reversed(range(len(PERCENTILES))):  # highest first so the legend reads top-down
        ax.plot(weeks, q[k, :, i] * scale, color=PERCENTILE_COLORS[k], linestyle=linestyle,
                linewidth=2 if PERCENTILES[k] == 50 else 1.3, label=f"{model} {PERCENTILES[k]}%")
    ax.set_xlim(0, weeks[-1])
    ax.set_xlabel("Weeks ahead (0 = 2025-12-30)")
    ax.set_ylabel("Rate (%)" if scale == 100 else "Price (USD)")
    style_axes(ax)
