"""
Q5 - Backtesting VaR forecasts from the GARCH(1,1) model fitted in Q2.

Steps completed so far:
  1) Load and align the S&P 500 backtest data (same source as Q4).
  2) Reuse garch11_variance() to get sigma2 over the full series, then slice
     to the 2022-2025 backtest window.
  3) Normal-quantile helper -> VaR_t^{GARCH-N} = -z_0.02 * sigma_t.
  4) Standardized residuals z from the Nasdaq estimation sample = the fixed
     FHS pool (not recalibrated during the backtest).
  5) Empirical 2% quantile of that pool -> VaR_t^{FHS} = -q_0.02(z) * sigma_t.

Unit convention
---------------
The GARCH(1,1) parameters in Q2/garch11_params.json were estimated on Nasdaq
log-returns expressed in PERCENT (returns * 100). The recursion must therefore
be run in percent units. Everything is converted back to decimal log-return
units at the end so the VaR forecasts live on the same scale as the Q4 plot.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

# Reuse the GARCH recursion from Q2 rather than re-implementing it.
CODE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR / "Q2"))
from garch11 import garch11_variance  # noqa: E402

BACKTEST_START = "2022-01-01"
BACKTEST_END = "2025-12-31"
VAR_LEVEL = 0.02  # 98% VaR -> 2% lower tail


# Step 1: Load and align the S&P 500 backtest data

def load_sp500_returns():
    """Load spx_data.csv exactly as Q4/Main.py does and return log-returns.

    Returns a pd.Series of daily log-returns (decimal units) indexed by date,
    covering the full 2018-2025 sample. The pre-2022 part is not backtested but
    is needed to warm up the GARCH recursion.
    """
    csv_path = CODE_DIR / "Q4" / "spx_data.csv"
    df = pd.read_csv(
        csv_path,
        skiprows=3,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    prices = df["Close"]

    returns = np.log(prices / prices.shift(1))
    return returns.dropna()



# Step 2: Conditional variance from the Q2 GARCH(1,1) model

def load_garch_params():
    """Load the Nasdaq-fitted GARCH(1,1) parameters from Q2."""
    params = json.loads((CODE_DIR / "Q2" / "garch11_params.json").read_text())
    return params["omega"], params["alpha"], params["beta"]


def garch_sigma(returns_decimal, omega, alpha, beta):
    """Run the GARCH(1,1) recursion on the FULL series, return sigma in decimals.

    The recursion is run in percent units (to match the units the parameters
    were estimated in) over the whole sample, so that by the time we reach
    2022 the variance no longer depends on the arbitrary starting value.
    """
    returns_pct = (returns_decimal * 100).values
    sigma2_init = returns_pct.var()

    sigma2_pct = garch11_variance(returns_pct, omega, alpha, beta, sigma2_init)
    sigma_pct = np.sqrt(sigma2_pct)

    # Back to decimal log-return units, keeping the date index aligned.
    sigma_decimal = pd.Series(sigma_pct / 100, index=returns_decimal.index)
    return sigma_decimal


# Step 3: Normal quantile -> GARCH-N VaR

def normal_quantile(p):
    """Standard-normal inverse CDF, with a hard-coded fallback for p = 0.02."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:
        if abs(p - 0.02) < 1e-12:
            return -2.053748910631823
        raise


def garch_normal_var(sigma, level=VAR_LEVEL):
    """VaR_t = -z_level * sigma_t  (a positive number = loss quantile)."""
    z = normal_quantile(level)
    return -z * sigma



# Step 4: Standardized residuals over the Nasdaq estimation sample (FHS pool)
def nasdaq_standardized_residuals(omega, alpha, beta):
    """z_t = r_t / sigma_t over the Nasdaq 2012-2021 estimation sample.

    This reproduces the calculation in Q2/garch11.py:

        sigma2_fitted = garch11_variance(returns, omega, alpha, beta, sample_var)
        z = returns / np.sqrt(sigma2_fitted)

    z is unitless, so the percent-vs-decimal convention does not matter here:
    the same pool applies whichever units sigma_t is expressed in.

    The pool is fixed once, from the ESTIMATION sample only. It is never
    refreshed with 2022-2025 data, which keeps the backtest out-of-sample.
    """
    csv_path = CODE_DIR / "Q2" / "nasdaq_2012_2021.csv"
    data = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    returns_pct = (data["Log_returns"].dropna() * 100).values

    sigma2_fitted = garch11_variance(
        returns_pct, omega, alpha, beta, returns_pct.var()
    )
    return returns_pct / np.sqrt(sigma2_fitted)


# Step 5: Empirical quantile of the z-pool -> FHS VaR
def empirical_quantile(sample, p):
    """Empirical p-quantile, written out by hand (sort + order statistic).

    Uses the same linear-interpolation convention as numpy's default:
    with n observations, the quantile sits at position h = (n - 1) * p in the
    sorted array; if h falls between two order statistics we interpolate
    linearly between them.
    """
    x = np.sort(np.asarray(sample))
    n = len(x)

    h = (n - 1) * p
    lo = int(np.floor(h))
    hi = int(np.ceil(h))

    if lo == hi:
        return float(x[lo])

    weight = h - lo
    return float(x[lo] + weight * (x[hi] - x[lo]))


def fhs_var(sigma, z_pool, level=VAR_LEVEL):
    """One-period FHS VaR: VaR_t = -q_level(z) * sigma_t.

    Same structure as the normal case, but the quantile is read off the
    empirical distribution of the standardized residuals instead of N(0,1),
    so it inherits the fat tails and skew of the historical shocks.
    """
    q = empirical_quantile(z_pool, level)
    return -q * sigma


# ---------------------------------------------------------------------------
# Step 6: Historical simulation VaR + one shared violation counter
# ---------------------------------------------------------------------------
def hs_var(returns, window, level=VAR_LEVEL, shift=False):
    """Rolling historical-simulation VaR, as in Q4/Main.py.

    shift=False reproduces Main.py exactly: the quantile at date t is taken over
    the window ENDING at t, so the forecast for day t already contains day t's
    own return. shift=True lags the window by one day, making it a genuine
    out-of-sample forecast (see the note printed in __main__).
    """
    var = -returns.rolling(window=window).quantile(level)
    return var.shift(1) if shift else var


def count_violations(losses, var_forecasts, level=VAR_LEVEL):
    """Count VaR violations and return (count, rate, violation_ratio, n).

    A violation is a day where the realised loss exceeds the VaR forecast,
    which generalises the inline comparison in Q4/Main.py.

      rate            = x / N            (what Main.py called "ratio")
      violation_ratio = (x / N) / level  (textbook VR: observed / expected;
                                          1.0 = perfectly calibrated)
    """
    losses, var_forecasts = losses.align(var_forecasts, join="inner")
    valid = losses.notna() & var_forecasts.notna()
    losses, var_forecasts = losses[valid], var_forecasts[valid]

    n = len(losses)
    count = int((losses > var_forecasts).sum())
    rate = count / n
    return count, rate, rate / level, n


# ---------------------------------------------------------------------------
# Step 8: Binomial acceptance interval for the number of violations
# ---------------------------------------------------------------------------
def binom_log_pmf(k, n, p):
    """log P(X = k) for X ~ Bin(n, p), via log-gamma (no overflow at n ~ 1000)."""
    from math import lgamma, log, log1p
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
            + k * log(p) + (n - k) * log1p(-p))


def binom_cdf(k, n, p):
    """P(X <= k), summed from the log-pmf."""
    from math import exp
    return sum(exp(binom_log_pmf(i, n, p)) for i in range(0, k + 1))


def binomial_acceptance(n, p=VAR_LEVEL, conf=0.95):
    """Exact equal-tailed 95% acceptance region for the violation count.

    Keeps every k whose two one-sided tail probabilities both exceed
    alpha/2, i.e. the set of counts a correct model would not be rejected on.
    """
    from math import exp

    tail = (1 - conf) / 2
    pmf = [exp(binom_log_pmf(k, n, p)) for k in range(n + 1)]
    cdf, running = [], 0.0
    for val in pmf:
        running += val
        cdf.append(running)

    keep = [k for k in range(n + 1)
            if cdf[k] >= tail and (1.0 if k == 0 else 1 - cdf[k - 1]) >= tail]
    return min(keep), max(keep)


def normal_acceptance(n, p=VAR_LEVEL, z=1.959963984540054):
    """Normal-approximation acceptance bounds for the violation count."""
    half = z * np.sqrt(n * p * (1 - p))
    return n * p - half, n * p + half


if __name__ == "__main__":
    #  Step 1 
    returns = load_sp500_returns()
    returns_bt = returns.loc[BACKTEST_START:BACKTEST_END]
    dates_bt = returns_bt.index

    print("=== Step 1: data ===")
    print(f"full sample:     {returns.index[0].date()} to {returns.index[-1].date()} "
          f"({len(returns)} obs)")
    print(f"backtest window: {dates_bt[0].date()} to {dates_bt[-1].date()} "
          f"({len(returns_bt)} obs)")

    #  Step 2 
    omega, alpha, beta = load_garch_params()
    print("\n=== Step 2: GARCH(1,1) from Q2 ===")
    print(f"omega={omega:.6f}  alpha={alpha:.6f}  beta={beta:.6f}  "
          f"alpha+beta={alpha + beta:.6f}")

    sigma = garch_sigma(returns, omega, alpha, beta)
    sigma_bt = sigma.loc[BACKTEST_START:BACKTEST_END]

    # Index alignment sanity check: return t must line up with forecast t.
    assert sigma_bt.index.equals(returns_bt.index), "date index misaligned"

    print(f"sigma over backtest window (daily, %): "
          f"min={sigma_bt.min() * 100:.3f}  mean={sigma_bt.mean() * 100:.3f}  "
          f"max={sigma_bt.max() * 100:.3f}")
    print(f"annualised mean vol: {sigma_bt.mean() * np.sqrt(252) * 100:.2f}%")
    print(f"realised sd of returns in window: {returns_bt.std() * 100:.3f}%")

    # Step 3 
    z = normal_quantile(VAR_LEVEL)
    var_garch_n = garch_normal_var(sigma_bt)

    print("\n=== Step 3: GARCH-N VaR ===")
    print(f"z_{VAR_LEVEL} = {z:.6f}")
    print(f"VaR (daily, %): min={var_garch_n.min() * 100:.3f}  "
          f"mean={var_garch_n.mean() * 100:.3f}  max={var_garch_n.max() * 100:.3f}")
    print("\nfirst 5 forecasts:")
    print((var_garch_n.head() * 100).round(3).to_string())

    # --- Step 4 ---
    z_pool = nasdaq_standardized_residuals(omega, alpha, beta)

    print("\n=== Step 4: FHS pool (standardized residuals, Nasdaq 2012-2021) ===")
    print(f"pool size: {len(z_pool)}")
    print(f"mean={z_pool.mean():.4f}  var={z_pool.var():.4f}  "
          f"min={z_pool.min():.3f}  max={z_pool.max():.3f}")

    # If the GARCH fit is sound the pool should have variance ~1. Skew/kurtosis
    # are what FHS exploits and the normal assumption throws away.
    skew = ((z_pool - z_pool.mean()) ** 3).mean() / z_pool.std() ** 3
    kurt = ((z_pool - z_pool.mean()) ** 4).mean() / z_pool.std() ** 4
    print(f"skewness={skew:.4f}  kurtosis={kurt:.4f}  (normal: 0.0 and 3.0)")

    # --- Step 5 ---
    q_fhs = empirical_quantile(z_pool, VAR_LEVEL)
    var_fhs = fhs_var(sigma_bt, z_pool)

    print("\n=== Step 5: FHS VaR ===")
    print(f"empirical q_{VAR_LEVEL}(z) = {q_fhs:.6f}   (hand-rolled)")
    print(f"np.quantile cross-check   = {np.quantile(z_pool, VAR_LEVEL):.6f}")
    assert np.isclose(q_fhs, np.quantile(z_pool, VAR_LEVEL)), "quantile mismatch"

    print(f"normal z_{VAR_LEVEL}           = {z:.6f}")
    print(f"ratio FHS/normal          = {q_fhs / z:.4f}")
    print(f"VaR (daily, %): min={var_fhs.min() * 100:.3f}  "
          f"mean={var_fhs.mean() * 100:.3f}  max={var_fhs.max() * 100:.3f}")

    # --- Step 6: all four models through one violation counter ---
    losses = -returns_bt  # realised loss = negative log-return

    var_hs500 = hs_var(returns, 500).loc[BACKTEST_START:BACKTEST_END]
    var_hs1000 = hs_var(returns, 1000).loc[BACKTEST_START:BACKTEST_END]

    models = {
        "HS-500": var_hs500,
        "HS-1000": var_hs1000,
        "GARCH-Normal": var_garch_n,
        "FHS": var_fhs,
    }

    rows = []
    for name, var_series in models.items():
        cnt, rate, vr, n_used = count_violations(losses, var_series)
        rows.append({
            "model": name, "N": n_used, "violations": cnt,
            "rate": rate, "VR": vr, "mean_VaR_%": var_series.mean() * 100,
        })
    results = pd.DataFrame(rows).set_index("model")

    print("\n=== Step 6: violations ===")
    print(f"expected violations at p={VAR_LEVEL}: {VAR_LEVEL * len(losses):.1f} "
          f"of {len(losses)} days")
    print(results.round({"rate": 4, "VR": 3, "mean_VaR_%": 3}).to_string())

    # --- Step 8: acceptance interval ---
    N = len(losses)
    k_lo, k_hi = binomial_acceptance(N)
    n_lo, n_hi = normal_acceptance(N)

    print("\n=== Step 8: 95% acceptance interval ===")
    print(f"N={N}, p={VAR_LEVEL}, expected={N * VAR_LEVEL:.1f} violations")
    print(f"exact binomial : counts [{k_lo}, {k_hi}]  "
          f"rate [{k_lo / N:.4f}, {k_hi / N:.4f}]  "
          f"VR [{k_lo / N / VAR_LEVEL:.3f}, {k_hi / N / VAR_LEVEL:.3f}]")
    print(f"normal approx  : counts [{n_lo:.2f}, {n_hi:.2f}]  "
          f"rate [{n_lo / N:.4f}, {n_hi / N:.4f}]  "
          f"VR [{n_lo / N / VAR_LEVEL:.3f}, {n_hi / N / VAR_LEVEL:.3f}]")

    results["verdict"] = [
        "ACCEPT" if k_lo <= c <= k_hi else "REJECT" for c in results["violations"]
    ]
    print("\n--- verdicts ---")
    print(results[["violations", "rate", "VR", "verdict"]]
          .round({"rate": 4, "VR": 3}).to_string())

    # Sensitivity: the Q4 rolling window at date t includes date t's own return,
    # so the HS forecasts peek one day ahead. GARCH sigma_t does not (it uses
    # only r_{t-1}). Re-run the HS models lagged by one day to size that effect.
    print("\n--- sensitivity: HS with a genuinely lagged window ---")
    for name, w in (("HS-500", 500), ("HS-1000", 1000)):
        v = hs_var(returns, w, shift=True).loc[BACKTEST_START:BACKTEST_END]
        cnt, rate, vr, _ = count_violations(losses, v)
        tag = "ACCEPT" if k_lo <= cnt <= k_hi else "REJECT"
        print(f"{name}(lagged): {cnt} violations, rate={rate:.4f}, VR={vr:.3f} -> {tag}")

    # --- Step 7: the Q4 plot, extended with the two GARCH-based models ---
    import matplotlib.pyplot as plt

    COLORS = {  # validated categorical palette (see palette validation)
        "HS-500": "#2a78d6", "HS-1000": "#eb6834",
        "GARCH-Normal": "#1baf7a", "FHS": "#4a3aa7",
    }
    STYLES = {"HS-500": "-", "HS-1000": "-", "GARCH-Normal": "-", "FHS": "--"}

    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.plot(losses.index, losses, color="#9a9a93", alpha=0.55, linewidth=0.8,
            label="Actual losses (negative returns)", zorder=1)

    for name, var_series in models.items():
        ax.plot(var_series.index, var_series, color=COLORS[name],
                linestyle=STYLES[name], linewidth=1.8, label=f"98% VaR - {name}",
                zorder=3)

    ax.axhline(0, color="#c9c9c2", linewidth=0.8, zorder=0)
    ax.set_title("98% VaR forecasts vs actual losses, S&P 500 (2022-2025)",
                 fontsize=13, pad=12)
    ax.set_ylabel("Loss / VaR (log-return)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.18, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    out_png = pathlib.Path(__file__).parent / "q5_var_backtest.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nplot saved to {out_png.name}")
    plt.show()
