from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent


def load_river():
    """Daily river levels as a Series indexed by date."""
    df = pd.read_csv(DATA_DIR / "river_level.csv", parse_dates=["date"])
    return df.set_index("date")["value"]


def q5a(level):
    print(f"5a) {len(level)} daily levels, {level.index.min():%Y-%m-%d} to {level.index.max():%Y-%m-%d}: "
          f"mean {level.mean():.1f}, std {level.std():.1f}, min {level.min():.0f}, max {level.max():.0f}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(level.index, level.to_numpy(), lw=0.5, color="C0")
    ax.set_xlabel("Date")
    ax.set_ylabel("River level")
    ax.set_title(f"Daily river levels {level.index.year.min()}-{level.index.year.max()}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q5a_daily.png", dpi=150)
    plt.close(fig)


def yearly_maxima(level):
    """M_y = max of the daily levels in calendar year y."""
    return level.groupby(level.index.year).max().rename("max")


def q5b(level):
    m = yearly_maxima(level)
    m.to_frame().to_csv(OUT_DIR / "Q5b_yearly_maxima.csv", index_label="year")
    print(f"\n5b) {len(m)} yearly maxima: mean {m.mean():.1f}, std {m.std():.1f}, "
          f"min {m.min():.0f} ({m.idxmin()}), max {m.max():.0f} ({m.idxmax()})")
    print(m.to_frame().T.to_string())
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(m.index, m.to_numpy(), color="C0")
    ax.set_xticks(m.index)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Yearly maximum river level")
    ax.set_title(f"Yearly maxima {m.index.min()}-{m.index.max()}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q5b_yearly_maxima.png", dpi=150)
    plt.close(fig)
    return m


def q5c(m):
    """Block maxima method with calendar years as blocks.
    Method: GEV H(x) = exp(-(1 + xi (x - mu) / sigma)^(-1/xi)) fitted by ML to the yearly maxima.
    scipy's shape c has the opposite sign of the lecture's xi, so xi = -c.
    T-year return level: z_T = H^{-1}(1 - 1/T) = mu + sigma / xi * ((-ln(1 - 1/T))^(-xi) - 1).
    Assumption: the yearly maxima are iid."""
    x = np.sort(m.to_numpy())
    n = len(x)
    c, mu, sigma = stats.genextreme.fit(x)
    gev = stats.genextreme(c, loc=mu, scale=sigma)
    xi = -c
    print(f"\n5c) GEV fitted by ML to the {n} yearly maxima: mu {mu:.2f}, sigma {sigma:.2f}, xi {xi:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    grid = np.linspace(m.min() * 0.9, m.max() * 1.1, 500)
    ax.step(x, np.arange(1, n + 1) / n, "k", where="post", label="Empirical")
    ax.plot(grid, gev.cdf(grid), label="Fitted GEV")
    ax.set_xlabel("Yearly maximum river level")
    ax.set_ylabel("F(x)")
    ax.set_title("cdf of the yearly maxima")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q5c_cdf.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    q = gev.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.plot(q, x, "o")
    lims = [min(q.min(), x.min()), max(q.max(), x.max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("Fitted GEV quantiles")
    ax.set_ylabel("Observed yearly maxima")
    ax.set_title("QQ-plot against the fitted GEV")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q5c_qq.png", dpi=150)
    plt.close(fig)

    T = np.array([100, 1_000, 10_000])
    table = pd.DataFrame({"T": T, "Return level": gev.ppf(1 - 1 / T)})
    table.to_csv(OUT_DIR / "Q5c_return_levels.csv", index=False)
    print(table.round(1).to_string(index=False))


def q5d(level):
    """Threshold exceedance method.
    Method: u = 95th percentile of the daily levels, excesses y = x - u for the days with x > u.
    GPD G(y) = 1 - (1 + xi y / beta)^(-1/xi) fitted by ML to the excesses with loc = 0 (scipy's c = xi).
    lam = N_u / n_years exceedances per year, and the T-year return level solves lam (1 - G(x_T - u)) = 1/T:
        x_T = u + beta / xi * ((lam T)^xi - 1).
    Assumption: the excesses are iid. In practice they are not independent, since consecutive
    high days belong to the same flood."""
    u = level.quantile(0.95)
    y = (level[level > u] - u).to_numpy()
    n = len(y)
    n_years = level.index.year.nunique()
    lam = n / n_years
    print(f"\n5d) threshold u = {u:.1f} (95th percentile), N_u = {n} exceedances, lam = {lam:.2f} per year")

    xi, _, beta = stats.genpareto.fit(y, floc=0)
    gpd = stats.genpareto(xi, loc=0, scale=beta)
    print(f"    GPD fitted by ML to the excesses: xi {xi:.4f}, beta {beta:.2f}")

    x = np.sort(y)
    fig, ax = plt.subplots(figsize=(7, 5))
    grid = np.linspace(0, y.max() * 1.1, 500)
    ax.step(x, np.arange(1, n + 1) / n, "k", where="post", label="Empirical")
    ax.plot(grid, gpd.cdf(grid), label="Fitted GPD")
    ax.set_xlabel("Excess over threshold y = x - u")
    ax.set_ylabel("G(y)")
    ax.set_title(f"cdf of the excesses, u = {u:.0f} (95th percentile)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q5d_cdf.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    q = gpd.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.plot(q, x, "o", ms=4)
    lims = [0, max(q.max(), x.max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("Fitted GPD quantiles")
    ax.set_ylabel("Observed excesses")
    ax.set_title(f"QQ-plot against the fitted GPD, u = {u:.0f} (95th percentile)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q5d_qq.png", dpi=150)
    plt.close(fig)

    T = np.array([100, 1_000, 10_000])
    table = pd.DataFrame({"T": T, "Return level": u + gpd.ppf(1 - 1 / (lam * T))})
    table.to_csv(OUT_DIR / "Q5d_return_levels.csv", index=False)
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    river = load_river()
    q5a(river)
    maxima = q5b(river)
    q5c(maxima)
    q5d(river)
