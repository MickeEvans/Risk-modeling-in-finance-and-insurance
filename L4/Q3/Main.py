
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
N_SIM = 10_000
spec = importlib.util.spec_from_file_location("q2", OUT_DIR.parent / "Q2" / "Main.py")
q2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q2)


def monthly_history(fire, mon):
    """Per calendar month m = 1..12: counts[m] = number of fires in month m of each year (0 included),
    losses[m] = all single losses (Total) in month m, pooled over all years."""
    counts = {m: mon.N[mon.index.month == m].to_numpy() for m in range(1, 13)}
    losses = {m: fire.Total[fire.Date.dt.month == m].to_numpy() for m in range(1, 13)}
    return counts, losses


def simulate_years(counts, losses, n_sim, rng):
    """n_sim synthetic yearly losses. Per calendar month: draw a count from that month's historical counts,
    then draw that many losses with replacement from that month's pooled losses, and add them to the year."""
    S = np.zeros(n_sim)
    for m in range(1, 13):
        n = rng.choice(counts[m], size=n_sim, replace=True)         # one claim count per simulated year
        x = rng.choice(losses[m], size=n.sum(), replace=True)       # all claims for this month
        year = np.repeat(np.arange(n_sim), n)                       # simulated year of each claim
        S += np.bincount(year, weights=x, minlength=n_sim)          # monthly total per year, 0 if n = 0
    return S


def collective_model(fire, mon):
    """Panjer pmf of the yearly loss for the best frequency and severity by AIC, as in Q2c (no Q2 plots)."""
    counts, x = mon.N.to_numpy(), fire.Total.to_numpy()
    freqs, sevs = q2.fit_frequency(counts), q2.fit_severity(x)
    f_name = min(freqs, key=lambda k: q2.log_lik_aic(k, freqs[k], counts)[1])
    s_name = min(sevs, key=lambda k: q2.log_lik_aic(k, sevs[k], x)[1])
    print(f"\nCollective model (Q2c): {f_name} frequency, {s_name} severity")
    s, fx = q2.discretize(sevs[s_name])
    return s, q2.panjer(q2.yearly_frequency(freqs[f_name]), fx)


def q3a(S, s, fs):
    """Histogram of the simulated yearly losses with the collective density fs / H on top."""
    upper = max(np.quantile(S, 0.999), s[np.searchsorted(np.cumsum(fs), 0.999)])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(S, bins=100, density=True, color="lightgray", edgecolor="gray", label="Historical simulation")
    ax.plot(s, fs / q2.H, "k", label="Collective model (Q2c)")
    ax.set_xlim(0, upper)
    ax.set_xlabel("Yearly loss s (MDKK)")
    ax.set_ylabel("Density (per MDKK)")
    ax.set_title("Yearly total loss: historical simulation vs collective model")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q3a_comparison.png", dpi=150)
    plt.close(fig)


def q3b(S, s, fs):
    """Mean, std and VaR_p = smallest s with empirical F(s) >= p of the simulated yearly losses,
    next to the same measures of the collective model."""
    hist = pd.Series({"Mean": S.mean(), "Std": S.std(ddof=1),
                      **{f"VaR {100 * p:g}%": np.quantile(S, p, method="inverted_cdf") for p in q2.LEVELS}})
    table = pd.DataFrame({"Historical simulation": hist, "Collective model (Q2c)": q2.pmf_measures(s, fs)}).T
    table.to_csv(OUT_DIR / "Q3b_table.csv")
    print(f"\n3b) yearly loss S (MDKK), {len(S)} simulated years\n", table.round(1).to_string())
    return table


if __name__ == "__main__":
    fire = q2.load_fire()
    mon = q2.monthly(fire)
    counts, losses = monthly_history(fire, mon)
    print("Month  years  counts per year                              fires  mean loss")
    for m in range(1, 13):
        print(f"{m:5d}  {len(counts[m]):5d}  {str(counts[m].tolist()):43s}  {len(losses[m]):5d}  {losses[m].mean():9.2f}")

    S = simulate_years(counts, losses, N_SIM, np.random.default_rng(1))
    print(f"\n{len(S)} simulated years, first five: {np.round(S[:5], 1).tolist()}")
    s, fs = collective_model(fire, mon)
    q3a(S, s, fs)
    q3b(S, s, fs)
