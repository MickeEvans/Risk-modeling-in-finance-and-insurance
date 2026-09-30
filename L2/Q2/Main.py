import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ASSET_COLUMNS, PERCENTILES, RATE_COLUMNS, load_data, log_returns, plot_fan  # noqa: E402
from var1 import fit_var1, returns_to_levels, simulate_var1  # noqa: E402

OUT_DIR = Path("L2/Q2")
SEED = 20260929
N_SCENARIOS = 5000
N_STEPS = 480
TABLE_WEEKS = [240, 480]
PLOT_ASSETS = {"CAT": "Caterpillar", "WMT": "Walmart", "Y05": "5-year rate", "Y30": "30-year rate"}


def percentile_table(q: np.ndarray, week: int) -> pd.DataFrame:
    """Rows = assets, columns = percentiles; q index 0 is today, so week w is index w."""
    rows = {}
    for asset, name in PLOT_ASSETS.items():
        values = q[:, week, ASSET_COLUMNS.index(asset)]
        is_rate = asset in RATE_COLUMNS
        rows[f"{name} ({asset}){' [%]' if is_rate else ''}"] = values * 100 if is_rate else values
    return pd.DataFrame(rows, index=[f"{p}%" for p in PERCENTILES]).T


def main() -> None:
    prices = load_data()
    R = log_returns(prices).to_numpy()

    # 2a: estimate
    c, A, sigma = fit_var1(R)
    c_df = pd.DataFrame({"c": c}, index=ASSET_COLUMNS)
    A_df = pd.DataFrame(A, index=ASSET_COLUMNS, columns=ASSET_COLUMNS)
    sigma_df = pd.DataFrame(sigma, index=ASSET_COLUMNS, columns=ASSET_COLUMNS)
    for name, df in [("c", c_df), ("A", A_df), ("Sigma", sigma_df)]:
        df.to_csv(OUT_DIR / f"var1_{name}.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("2a) c:\n", c_df.round(5).to_string())
        print("\n2a) A (row i = equation for asset i):\n", A_df.round(3).to_string())
        print("\n2a) Sigma x 1e4:\n", (sigma_df * 1e4).round(2).to_string())

    # 2b: stability
    abs_eig = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    print("\n2b) |eigenvalues of A|:", abs_eig.round(4))
    print(f"    max = {abs_eig[0]:.4f} < 1 -> stable" if abs_eig[0] < 1 else f"    max = {abs_eig[0]:.4f} >= 1 -> NOT stable")

    # 2c: simulate, percentiles, plots, tables
    rng = np.random.default_rng(SEED)
    sim_returns = simulate_var1(c, A, sigma, R[-1], N_SCENARIOS, N_STEPS, rng)
    start_levels = prices.iloc[-1].to_numpy()
    levels = returns_to_levels(sim_returns, start_levels)

    q = np.concatenate([np.broadcast_to(start_levels, (len(PERCENTILES), 1, len(start_levels))),
                        np.percentile(levels, PERCENTILES, axis=0)], axis=1)
    np.save(OUT_DIR / "scenarios_returns.npy", sim_returns.astype(np.float32))
    np.save(OUT_DIR / "start_levels.npy", start_levels)
    np.save(OUT_DIR / "percentiles_var1.npy", q)

    for asset, name in PLOT_ASSETS.items():
        fig, ax = plt.subplots(figsize=(10, 5.5))
        plot_fan(ax, q, asset)
        ax.set_title(f"{name} ({asset}): simulated percentiles, {N_SCENARIOS} scenarios")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"Fan_{asset}.png", dpi=300)
        plt.close(fig)

    for week in TABLE_WEEKS:
        table = percentile_table(q, week)
        table.to_csv(OUT_DIR / f"Percentiles_week{week}.csv")
        print(f"\n2c) Percentiles at week {week} (stocks in USD, rates in %):\n", table.round(2).to_string())


if __name__ == "__main__":
    main()
