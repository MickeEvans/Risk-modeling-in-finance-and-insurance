import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAMS_FILE = pathlib.Path(__file__).parent / "garch11_params.json"


def garch11_variance(returns, omega, alpha, beta, sigma2_init):
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = sigma2_init

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    return sigma2


def garch11_loglik(returns, omega, alpha, beta, sigma2_init):
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return -np.inf

    sigma2 = garch11_variance(returns, omega, alpha, beta, sigma2_init)

    loglik = -0.5 * np.sum(np.log(sigma2) + (returns ** 2) / sigma2)

    if not np.isfinite(loglik):
        return -np.inf

    return loglik


def garch11_grid_search(returns, omega_bounds, alpha_bounds, beta_bounds, n=15):
    sigma2_init = returns.var()

    omega_grid = np.linspace(*omega_bounds, n)
    alpha_grid = np.linspace(*alpha_bounds, n)
    beta_grid = np.linspace(*beta_bounds, n)

    best_loglik = -np.inf
    best_omega = best_alpha = best_beta = None

    for omega in omega_grid:
        for alpha in alpha_grid:
            for beta in beta_grid:
                if alpha + beta >= 1:
                    continue
                loglik = garch11_loglik(returns, omega, alpha, beta, sigma2_init)
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_omega, best_alpha, best_beta = omega, alpha, beta

    if best_omega in (omega_grid[0], omega_grid[-1]):
        print(f"WARNING: omega hit grid boundary at {best_omega}")
    if best_alpha in (alpha_grid[0], alpha_grid[-1]):
        print(f"WARNING: alpha hit grid boundary at {best_alpha}")
    if best_beta in (beta_grid[0], beta_grid[-1]):
        print(f"WARNING: beta hit grid boundary at {best_beta}")

    return best_omega, best_alpha, best_beta, best_loglik


def garch11_refine(returns, omega_bounds, alpha_bounds, beta_bounds, n=15, tol=1e-6):
    prev_loglik = -np.inf
    round_num = 0

    while True:
        best_omega, best_alpha, best_beta, best_loglik = garch11_grid_search(
            returns, omega_bounds, alpha_bounds, beta_bounds, n=n
        )

        omega_step = (omega_bounds[1] - omega_bounds[0]) / (n - 1)
        alpha_step = (alpha_bounds[1] - alpha_bounds[0]) / (n - 1)
        beta_step = (beta_bounds[1] - beta_bounds[0]) / (n - 1)

        print(
            f"round {round_num}: omega={best_omega:.6f} alpha={best_alpha:.6f} "
            f"beta={best_beta:.6f} alpha+beta={best_alpha + best_beta:.6f} "
            f"loglik={best_loglik:.6f}"
        )

        if best_loglik < prev_loglik:
            print("WARNING: likelihood decreased from previous round — check bracket clamping")

        prev_loglik = best_loglik
        round_num += 1

        step = max(omega_step, alpha_step, beta_step)
        if step < tol:
            break

        omega_bounds = (max(best_omega - omega_step, 1e-8), best_omega + omega_step)
        alpha_bounds = (max(best_alpha - alpha_step, 0.0), best_alpha + alpha_step)
        beta_bounds = (max(best_beta - beta_step, 0.0), min(best_beta + beta_step, 1 - 1e-8))

    return best_omega, best_alpha, best_beta, best_loglik


def rolling_ma_ewma_vol(prices, window=60, lmbda=0.94):
    returns = np.log(prices / prices.shift(1)).dropna()
    y2 = returns ** 2

    ma_vol = np.sqrt(y2.rolling(window=window).mean()).dropna()

    powers = np.arange(window, 0, -1)
    multiplier = (1 - lmbda) / (lmbda * (1 - lmbda ** window))
    weights = multiplier * (lmbda ** powers)
    ewma_var = y2.rolling(window=window).apply(lambda w: np.sum(weights * w), raw=True)
    ewma_vol = np.sqrt(ewma_var).dropna()

    return ma_vol, ewma_vol


if __name__ == "__main__":

    csv_path = pathlib.Path(__file__).parent / "sp500_2012_2021.csv"
    data = pd.read_csv(
        csv_path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"]
    )
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    returns = (log_returns * 100).values
    sample_var = returns.var()

    refit_answer = input("Refit the GARCH(1,1) model on S&P 500 data? [y/n]: ").strip().lower()
    refit = refit_answer.startswith("y") or not PARAMS_FILE.exists()

    if refit:
        best_omega, best_alpha, best_beta, best_loglik = garch11_refine(
            returns,
            omega_bounds=(0.001, 0.20),
            alpha_bounds=(0.0, 0.3),
            beta_bounds=(0.60, 0.999),
            n=15,
        )
        PARAMS_FILE.write_text(json.dumps({
            "omega": best_omega,
            "alpha": best_alpha,
            "beta": best_beta,
            "loglik": best_loglik,
        }))
    else:
        params = json.loads(PARAMS_FILE.read_text())
        best_omega = params["omega"]
        best_alpha = params["alpha"]
        best_beta = params["beta"]
        best_loglik = params["loglik"]

    print(
        f"best: omega={best_omega:.6f} alpha={best_alpha:.6f} "
        f"beta={best_beta:.6f} loglik={best_loglik:.6f}"
    )

    #  Apply the fitted GARCH(1,1) out-of-sample to S&P 500 2022-2025 returns
    sp500_csv_path = pathlib.Path(__file__).parent / "sp500_2022_2025.csv"
    sp500_data = pd.read_csv(
        sp500_csv_path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"]
    )
    sp500_data["Date"] = pd.to_datetime(sp500_data["Date"])
    sp500_data.set_index("Date", inplace=True)
    sp500_log_returns = np.log(sp500_data["Close"] / sp500_data["Close"].shift(1)).dropna()
    sp500_returns = (sp500_log_returns * 100).values
    sp500_dates = sp500_log_returns.index

    sigma2_init_sp500 = sp500_returns.var()
    sp500_sigma2 = garch11_variance(
        sp500_returns, best_omega, best_alpha, best_beta, sigma2_init_sp500
    )
    sp500_vol = np.sqrt(sp500_sigma2)

    #  Compare against the MA / EWMA volatility forecasts from Q1
    q1_csv_path = pathlib.Path(__file__).parent.parent / "Q1" / "spx_data.csv"
    q1_data = pd.read_csv(
        q1_csv_path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"]
    )
    q1_data["Date"] = pd.to_datetime(q1_data["Date"])
    q1_data.set_index("Date", inplace=True)

    ma_vol, ewma_vol = rolling_ma_ewma_vol(q1_data["Close"])
    ma_vol_pct = (ma_vol * 100).loc["2022-01-01":"2025-12-31"]
    ewma_vol_pct = (ewma_vol * 100).loc["2022-01-01":"2025-12-31"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(sp500_dates, sp500_vol, label="GARCH(1,1)", color="tab:blue")
    ax1.plot(ma_vol_pct.index, ma_vol_pct, label="60-day MA", color="tab:orange")
    ax1.plot(ewma_vol_pct.index, ewma_vol_pct, label="EWMA (λ=0.94)", color="tab:green")
    ax1.set_title("S&P 500 daily volatility: GARCH(1,1) vs MA vs EWMA")
    ax1.set_ylabel("Volatility (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(sp500_dates, sp500_log_returns, label="Daily Log-Returns", color="gray", linewidth=0.8)
    ax2.set_title("S&P 500 Daily Log-Returns")
    ax2.set_ylabel("Log-Return")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
