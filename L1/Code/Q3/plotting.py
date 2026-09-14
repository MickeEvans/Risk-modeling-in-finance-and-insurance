import json
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t

Q2_DIR = pathlib.Path(__file__).parent.parent / "Q2"
sys.path.insert(0, str(Q2_DIR))
from garch11 import garch11_variance  # noqa: E402

PARAMS_FILE = Q2_DIR / "garch11_params.json"
CSV_FILE = Q2_DIR / "nasdaq_2012_2021.csv"
MAX_LAG = 200


def acf(x, max_lag):
    x = np.asarray(x)
    x_demeaned = x - x.mean()
    denom = np.sum(x_demeaned ** 2)

    rho = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        rho[k - 1] = np.dot(x_demeaned[k:], x_demeaned[:-k]) / denom

    return rho


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plotting_positions(n):
    return (np.arange(1, n + 1) - 0.5) / n


def t_quantiles_std(p, df):
    """Student-t inverse CDF, rescaled to unit variance."""
    return student_t.ppf(p, df) / np.sqrt(df / (df - 2))


def plot_qq(ax, theoretical, empirical, title):
    ax.scatter(theoretical, empirical, s=8)
    lo = min(theoretical.min(), empirical.min())
    hi = max(theoretical.max(), empirical.max())
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")


def plot_acf(ax, rho, band, title):
    lags = np.arange(1, len(rho) + 1)
    markerline, stemlines, baseline = ax.stem(lags, rho, markerfmt=" ", basefmt="black")
    stemlines.set_linewidth(1)
    ax.axhline(band, color="red", linestyle="--", linewidth=1)
    ax.axhline(-band, color="red", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")


if __name__ == "__main__":
    params = json.loads(PARAMS_FILE.read_text())
    omega, alpha, beta = params["omega"], params["alpha"], params["beta"]

    data = pd.read_csv(CSV_FILE, index_col="Date", parse_dates=True)
    returns = (data["Log_returns"].dropna() * 100).values
    T = len(returns)

    sigma2_fitted = garch11_variance(returns, omega, alpha, beta, returns.var())
    z = returns / np.sqrt(sigma2_fitted)

    band = 1.96 / np.sqrt(T)
    print(f"T = {T}, confidence band = +/-{band:.4f}")

    returns_series = {
        "Returns": returns,
        "Squared returns": returns ** 2,
    }
    residual_series = {
        "Standardized residuals": z,
        "Squared standardized residuals": z ** 2,
    }

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, x) in zip(axes1, returns_series.items()):
        rho = acf(x, MAX_LAG)
        plot_acf(ax, rho, band, f"ACF of {name}")
    fig1.tight_layout()
    fig1.savefig(pathlib.Path(__file__).parent / "acf_returns.png", dpi=150)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, x) in zip(axes2, residual_series.items()):
        rho = acf(x, MAX_LAG)
        plot_acf(ax, rho, band, f"ACF of {name}")
    fig2.tight_layout()
    fig2.savefig(pathlib.Path(__file__).parent / "acf_standardized_residuals.png", dpi=150)

    z_mean, z_std = z.mean(), z.std()
    x_grid = np.linspace(z.min(), z.max(), 500)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.hist(z, bins=100, density=True, label="Standardized residuals")
    ax3.plot(x_grid, normal_pdf(x_grid, z_mean, z_std), color="red",
             label=f"Normal pdf (mean={z_mean:.3f}, std={z_std:.3f})")
    ax3.set_title("Histogram of standardized residuals vs. normal pdf")
    ax3.set_xlabel("z")
    ax3.set_ylabel("Density")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(pathlib.Path(__file__).parent / "residuals_histogram.png", dpi=150)

    # --- QQ-plots: normal vs. Student-t, and best-fit df for the left tail ---
    z_sorted = np.sort(z)
    n = len(z_sorted)
    p = plotting_positions(n)

    normal_q = norm.ppf(p)

    left_mask = p <= 0.10
    df_grid = np.arange(3, 31)
    sse = np.array([
        np.sum((z_sorted[left_mask] - t_quantiles_std(p[left_mask], df)) ** 2)
        for df in df_grid
    ])
    best_df = df_grid[np.argmin(sse)]
    print(f"Best-fit Student-t df for left tail (bottom 10%): {best_df} (SSE={sse.min():.4f})")

    best_t_q = t_quantiles_std(p, best_df)

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    plot_qq(axes4[0], normal_q, z_sorted, "QQ-plot vs Normal")
    plot_qq(axes4[1], best_t_q, z_sorted, f"QQ-plot vs Student-t (df={best_df})")
    fig4.tight_layout()
    fig4.savefig(pathlib.Path(__file__).parent / "qq_plots.png", dpi=150)

    plt.show()
