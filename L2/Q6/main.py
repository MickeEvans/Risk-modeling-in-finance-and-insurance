import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
from scb import load_mortality  # noqa: E402

SEXES = {"män": "Men", "kvinnor": "Women"}


def estimate_mu(deaths, pop, ages):
    """Pooled estimate mu(x) = sum_t D(t,x) / sum_t N(t,x) over 2021-2025, per sex."""
    D = deaths.groupby(["sex", "age"]).sum()
    N = pop.groupby(["sex", "age"]).sum()
    mu = (D / N).unstack("sex").loc[ages]
    return mu[list(SEXES)].rename(columns=SEXES)


def fit_makeham(ages, mu_hat):
    """Least squares fit of mu(x) = alpha + beta*e^(gamma*x), method from the lecture."""
    x, mu_hat = np.asarray(ages, float), np.asarray(mu_hat)
    m, N = mu_hat.sum(), len(x)

    def alpha_beta(gamma):
        e = np.exp(gamma * x)
        g, f1, f2 = (mu_hat * e).sum(), e.sum(), (e**2).sum()
        alpha = (m * f2 - g * f1) / (N * f2 - f1**2)
        beta = (m * f1 - N * g) / (f1**2 - N * f2)
        return alpha, beta

    def third_eq(gamma):
        alpha, beta = alpha_beta(gamma)
        e = np.exp(gamma * x)
        return np.sum((alpha + beta * e - mu_hat) * x * beta * e)

    lo, hi = 0.05, 0.2
    assert third_eq(lo) * third_eq(hi) < 0, "no sign change over bracket"
    gamma = brentq(third_eq, lo, hi)
    return (*alpha_beta(gamma), gamma)


def makeham_mu(x, alpha, beta, gamma):
    return alpha + beta * np.exp(gamma * np.asarray(x))


def survival(x, alpha, beta, gamma):
    """l(x) = exp(-alpha*x - (beta/gamma)*(e^(gamma*x) - 1))."""
    x = np.asarray(x)
    return np.exp(-alpha * x - beta / gamma * (np.exp(gamma * x) - 1))


def life_expectancy(alpha, beta, gamma, start_age):
    """e(x0) = int_{x0}^inf l(x) dx / l(x0), upper limit 150 (l is negligible beyond)."""
    integral, _ = quad(survival, start_age, 150, args=(alpha, beta, gamma))
    return integral / survival(start_age, alpha, beta, gamma)


if __name__ == "__main__":
    deaths, pop = load_mortality()

    # 6a
    ages = range(20, 100)
    mu = estimate_mu(deaths, pop, ages)

    plt.figure()
    for sex in mu:
        plt.plot(mu.index, mu[sex], ".", label=sex)
    plt.yscale("log")
    plt.xlabel("Age")
    plt.ylabel(r"$\hat\mu(x)$")
    plt.title("Estimated force of mortality, 2021-2025")
    plt.legend()
    plt.savefig(OUT_DIR / "6a_mu.png", dpi=150)

    print(mu.loc[[60, 70, 80, 90]].round(5))

    # 6b
    fit_ages = range(20, 91)
    params = {sex: fit_makeham(fit_ages, mu.loc[fit_ages, sex]) for sex in mu}
    print(pd.DataFrame(params, index=["alpha", "beta", "gamma"]).T)

    plt.figure()
    for sex, color in zip(mu, ["C0", "C1"]):
        plt.plot(mu.index, mu[sex], ".", color=color, label=f"{sex}, estimate")
        plt.plot(mu.index, makeham_mu(mu.index, *params[sex]), "-", color=color, label=f"{sex}, Makeham")
    plt.yscale("log")
    plt.xlabel("Age")
    plt.ylabel(r"$\mu(x)$")
    plt.title("Force of mortality with fitted Makeham functions")
    plt.legend()
    plt.savefig(OUT_DIR / "6b_makeham.png", dpi=150)

    # 6c
    x = np.arange(0, 106)
    plt.figure()
    for sex in mu:
        plt.plot(x, survival(x, *params[sex]), label=sex)
    plt.xlabel("Age")
    plt.ylabel(r"$l(x)$")
    plt.title("Survival functions (Makeham)")
    plt.legend()
    plt.savefig(OUT_DIR / "6c_survival.png", dpi=150)

    # 6d
    e = {sex: {f"e({a})": life_expectancy(*params[sex], a) for a in (0, 65)} for sex in mu}
    print(pd.DataFrame(e).T.round(2))

    plt.show()
