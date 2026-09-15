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


if __name__ == "__main__":

    csv_path = pathlib.Path(__file__).parent / "nasdaq_2012_2021.csv"
    data = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    returns = (data["Log_returns"].dropna() * 100).values
    sample_var = returns.var()

    refit_answer = input("Refit the GARCH(1,1) model on Nasdaq data? [y/n]: ").strip().lower()
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
        print("Loaded cached parameters from", PARAMS_FILE.name)

    print("best omega:", best_omega)
    print("best alpha:", best_alpha)
    print("best beta:", best_beta)
    print("best loglik:", best_loglik)

    omega_from_unconditional_var = sample_var * (1 - best_alpha - best_beta)
    print("omega implied by unconditional variance:", omega_from_unconditional_var)

    sigma2_fitted = garch11_variance(returns, best_omega, best_alpha, best_beta, sample_var)
    z = returns / np.sqrt(sigma2_fitted)
    print("standardized residual variance:", z.var())

    #  Apply the Nasdaq-fitted GARCH(1,1) to S&P 500 returns 
    sp500_csv_path = pathlib.Path(__file__).parent / "sp500_2022_2025.csv"
    sp500_data = pd.read_csv(sp500_csv_path, index_col="Date", parse_dates=True)
    sp500_returns = (sp500_data["Log_returns"].dropna() * 100).values
    sp500_dates = sp500_data["Log_returns"].dropna().index

    sigma2_init_sp500 = sp500_returns.var()
    print("sp500 sigma2_init (sample variance):", sigma2_init_sp500)

    sp500_sigma2 = garch11_variance(
        sp500_returns, best_omega, best_alpha, best_beta, sigma2_init_sp500
    )
    sp500_vol = np.sqrt(sp500_sigma2)

    print("sp500 any NaN:", np.isnan(sp500_sigma2).any())
    print("sp500 all positive:", (sp500_sigma2 > 0).all())
    print("sp500 vol range (%):", sp500_vol.min(), "to", sp500_vol.max())

    hand_calc_sigma2_1 = (
        best_omega + best_alpha * sp500_returns[0] ** 2 + best_beta * sigma2_init_sp500
    )
    print(
        "sp500 sigma2[1] matches hand calc:",
        np.isclose(sp500_sigma2[1], hand_calc_sigma2_1),
    )

    plt.figure(figsize=(10, 4))
    plt.plot(sp500_dates, sp500_vol)
    plt.title("S&P 500 conditional volatility (Nasdaq-fitted GARCH(1,1))")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.tight_layout()
    plt.show()
