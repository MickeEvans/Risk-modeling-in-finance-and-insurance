"""
Check: the observed yearly total losses (11 years, 1980-1990) against the 10 000 simulated years from Q3.
Both are summarised by mean, std and VaR_p = smallest s with empirical F(s) >= p.
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q3", OUT_DIR / "Main.py")
q3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q3)
q2 = q3.q2


def measures(x):
    """Mean, std and empirical VaR_p of the yearly losses x."""
    return pd.Series({"Mean": x.mean(), "Std": x.std(ddof=1),
                      **{f"VaR {100 * p:g}%": np.quantile(x, p, method="inverted_cdf") for p in q2.LEVELS}})


if __name__ == "__main__":
    fire = q2.load_fire()
    mon = q2.monthly(fire)
    observed = mon.Total.resample("YS").sum().to_numpy()
    S = q3.simulate_years(*q3.monthly_history(fire, mon), q3.N_SIM, np.random.default_rng(1))
    table = pd.DataFrame({"Observed": measures(observed), "Simulated": measures(S)}).T
    print(f"Yearly loss (MDKK): {len(observed)} observed years vs {len(S)} simulated years")
    print("Observed years:", np.round(observed, 1).tolist())
    print(table.round(1).to_string())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(S, bins=100, density=True, color="lightgray", edgecolor="gray", label="Simulated years")
    ax.plot(observed, np.zeros_like(observed), "|", ms=30, mew=2.5, color="C3", zorder=5, clip_on=False,
            label="Observed years")
    for (name, row), color in zip(table.iterrows(), ["C3", "C0"]):
        ax.axvspan(row.Mean - row.Std, row.Mean + row.Std, color=color, alpha=0.08,
                   label=f"{name}: mean ± std ({row.Std:.0f})")
        for key, ls in zip(["Mean", *table.columns[2:]], ["-", "--", "-.", ":"]):
            ax.axvline(row[key], color=color, ls=ls, label=f"{name}: {key} = {row[key]:.0f}")
    ax.set_xlabel("Yearly loss s (MDKK)")
    ax.set_ylabel("Density (per MDKK)")
    ax.set_title("Observed vs simulated yearly total loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q3_checkplot.png", dpi=150)
    plt.close(fig)
