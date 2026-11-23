"""
Q2. Collective model for the yearly total loss of the Danish fire data

Data: the monthly dataframe from Q1 (132 months, 1980-1990) and the 2167 single losses (Total, MDKK).

a) Frequency: number of fires per month
   Method: Poisson fitted by ML (lambda = mean). Negative binomial fitted by ML, using the profile
   likelihood in r with p = r / (r + mean). Compared by log-likelihood and AIC, and by their pmfs
   plotted over the histogram of the monthly counts.
   Assumption: the monthly counts are iid, so the seasonality and the upward trend found in Q1 are ignored.

b) Severity: size of each fire (Total)
   Method: gamma, log-normal and Pareto (lecture form, F(x) = 1 - (alpha / (alpha + x))^gamma, x > 0,
   i.e. scipy's lomax) all fitted by ML with location 0, so all three have support starting at 0.
   Compared by log-likelihood, AIC and the Kolmogorov-Smirnov distance, and by their cdfs plotted
   over the empirical cdf.
   Assumption: the losses are iid and independent of the number of fires.

c) Yearly loss S = X_1 + ... + X_N, collective model with the best frequency and severity (lowest AIC)
   Method: N is the sum of 12 iid monthly counts, so N ~ Poisson(12 lambda) or NB(12 r, p) (same p).
   X is discretised on the grid 0, H, 2H, ..., S_MAX (rounding method) and the pmf of S is computed
   on the same grid with the Panjer recursion (Lecture 22), valid since Poisson and NB are in the (a, b, 0) class:
       s_0 = P_N(g_0),  s_k = 1 / (1 - a g_0) sum_{i=1}^{k} (a + b i / k) g_i s_{k-i}.
   VaR_p(S) = smallest grid value s with F_S(s) >= p.
   Mean and std in theory (Lecture 21): E[S] = E[N] E[X],  Var(S) = E[N] Var(X) + Var(N) E[X]^2,
   and also computed from the Panjer pmf on [0, S_MAX].
   Assumption: if Var(X) is infinite (Pareto, alpha <= 2), the theoretical std of S is infinite.

d) Normal and translated gamma approximations of S (Lecture 21)
   Method: match the moments of S, computed from the moments of N and X:
       E[S] = E[N] mu,  Var(S) = E[N] sigma^2 + Var(N) mu^2,
       kappa3(S) = E[N] kappa3(X) + 3 Var(N) mu sigma^2 + kappa3(N) mu^3,  skew(S) = kappa3(S) / Var(S)^(3/2).
   Normal: S ~ N(E[S], Var(S)), VaR_p = E[S] + z_p Std(S).
   Translated gamma: S ~ k + Gamma(alpha, rate beta) with alpha = 4 / skew^2, beta = 2 / (skew Std(S)),
   k = E[S] - 2 Std(S) / skew.
   Assumption: the moments of X are exact if finite, otherwise (Pareto, alpha <= 3) they are taken from
   the discretised X in c), i.e. the same severity that Panjer uses.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent
H = 1.0          # grid step (MDKK)
S_MAX = 20_000   # largest yearly loss on the grid (MDKK)
LEVELS = (0.95, 0.99, 0.995)
N_PARAMS = {"Poisson": 1, "Neg. binomial": 2, "Gamma": 2, "Log-normal": 2, "Pareto": 2}


def load_fire():
    """One row per fire, losses in MDKK. Other = Contents + Profits."""
    df = pd.read_csv(DATA_DIR / "danish_fire_data.csv", parse_dates=["Date"])
    df["Other"] = df.Contents + df.Profits
    return df[["Date", "Building", "Other", "Total"]]


def monthly(df):
    """Summed losses and number of fires N per month. Months without fires get 0."""
    m = df.set_index("Date").resample("MS")
    out = m[["Building", "Other", "Total"]].sum()
    out["N"] = m.size()
    return out


def log_lik_aic(name, dist, data):
    """Log-likelihood and AIC = 2k - 2 logL of a fitted (frozen) distribution."""
    logf = dist.logpmf if hasattr(dist, "logpmf") else dist.logpdf
    ll = logf(data).sum()
    return ll, 2 * N_PARAMS[name] - 2 * ll


def fit_frequency(counts):
    """ML fits of the monthly counts. For the NB, p = r / (r + mean) maximises the likelihood for
    a given r, so only r is optimised (on log scale to keep it positive)."""
    m = counts.mean()

    def nb_nll(log_r):
        r = np.exp(log_r)
        return -stats.nbinom.logpmf(counts, r, r / (r + m)).sum()

    r = np.exp(optimize.minimize_scalar(nb_nll, bounds=(-5, 10), method="bounded").x)
    return {"Poisson": stats.poisson(m), "Neg. binomial": stats.nbinom(r, r / (r + m))}


def q2a(mon):
    counts = mon.N.to_numpy()
    fits = fit_frequency(counts)
    print(f"2a) {counts.sum()} fires in {len(counts)} months: "
          f"mean {counts.mean():.2f}, variance {counts.var(ddof=1):.2f}")
    aics = {}
    for name, dist in fits.items():
        ll, aics[name] = log_lik_aic(name, dist, counts)
        print(f"    {name:14s} params {np.round(dist.args, 4)}, mean {dist.mean():.2f}, "
              f"var {dist.var():.2f}, logL {ll:.1f}, AIC {aics[name]:.1f}")

    k = np.arange(counts.min() - 3, counts.max() + 4)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5, density=True,
            color="lightgray", edgecolor="gray", label="Observed")
    for name, dist in fits.items():
        ax.plot(k, dist.pmf(k), "o-", ms=4, label=name)
    ax.set_xlabel("Number of fires per month")
    ax.set_ylabel("Probability")
    ax.set_title("Monthly claim counts 1980-1990")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q2a_frequency.png", dpi=150)
    plt.close(fig)
    return fits, min(aics, key=aics.get)


def fit_severity(x):
    """ML fits of the single losses, all with location 0. Pareto = lomax: c = gamma, scale = alpha (lecture notation)."""
    return {"Gamma": stats.gamma(*stats.gamma.fit(x, floc=0)),
            "Log-normal": stats.lognorm(*stats.lognorm.fit(x, floc=0)),
            "Pareto": stats.lomax(*stats.lomax.fit(x, floc=0))}


def q2b(df):
    x = df.Total.to_numpy()
    fits = fit_severity(x)
    print(f"\n2b) {len(x)} losses: mean {x.mean():.2f}, std {x.std(ddof=1):.2f} MDKK")
    aics = {}
    for name, dist in fits.items():
        ll, aics[name] = log_lik_aic(name, dist, x)
        ks = stats.kstest(x, dist.cdf).statistic
        print(f"    {name:10s} params {np.round(dist.args, 4)}, mean {dist.mean():.2f}, "
              f"logL {ll:.1f}, AIC {aics[name]:.1f}, KS {ks:.3f}")

    xs = np.sort(x)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    grid = np.logspace(0, np.log10(xs.max()), 500)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(xs, ecdf, "k", where="post", label="Empirical")
    for name, dist in fits.items():
        ax.plot(grid, dist.cdf(grid), label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Loss x (MDKK, log scale)")
    ax.set_ylabel("F(x)")
    ax.set_title("Fitted cdfs of the losses")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q2b_severity.png", dpi=150)
    plt.close(fig)
    return fits, min(aics, key=aics.get)


def yearly_frequency(dist):
    """Sum of 12 iid monthly counts: Poisson(lambda) -> Poisson(12 lambda), NB(r, p) -> NB(12 r, p)."""
    if dist.dist.name == "poisson":
        return stats.poisson(12 * dist.args[0])
    r, p = dist.args
    return stats.nbinom(12 * r, p)


def discretize(sev):
    """Rounding method: f_X(kH) = F(kH + H/2) - F(kH - H/2), k = 0, ..., S_MAX / H."""
    s = np.arange(0, S_MAX + H, H)
    return s, np.diff(sev.cdf(np.concatenate([[0], s + H / 2])))


def panjer(freq, fx):
    """pmf of S on the grid of fx, for N in the (a, b, 0) class:
        f_S(0) = P_N(f_X(0)),  f_S(s) = sum_{j=1}^{s} (a + b j / s) f_X(j) f_S(s - j) / (1 - a f_X(0)).
    Poisson(lambda): a = 0, b = lambda.  NB(r, p): a = 1 - p, b = (r - 1)(1 - p)."""
    if freq.dist.name == "poisson":
        lam = freq.args[0]
        a, b = 0.0, lam
        f0 = np.exp(lam * (fx[0] - 1))
    else:
        r, p = freq.args
        a, b = 1 - p, (r - 1) * (1 - p)
        f0 = (p / (1 - a * fx[0])) ** r
    jfx = np.arange(len(fx)) * fx
    fs = np.zeros(len(fx))
    fs[0] = f0
    for s in range(1, len(fx)):
        prev = fs[s - 1::-1]  # f_S(s - 1), ..., f_S(0)
        fs[s] = (a * fx[1:s + 1] @ prev + b / s * jfx[1:s + 1] @ prev) / (1 - a * fx[0])
    return fs


def model_moments(freq, sev):
    """Theoretical E[S] and Std(S) of the collective model (Std = inf if Var(X) is infinite)."""
    EN, VN, EX, VX = freq.mean(), freq.var(), sev.mean(), sev.var()
    return EN * EX, np.sqrt(EN * VX + VN * EX ** 2)


def pmf_measures(s, f):
    """Mean, std and VaR_p = min{s : F(s) >= p} of a pmf f on the grid s."""
    mean = s @ f
    cdf = np.cumsum(f)
    return pd.Series({"Mean": mean, "Std": np.sqrt((s - mean) ** 2 @ f),
                      **{f"VaR {100 * p:g}%": s[np.searchsorted(cdf, p)] for p in LEVELS}})


def aggregate_moments(freq, sev, s, fx):
    """E[S], Var(S) and skew(S) from the moments of N and X (Lecture 21):
        kappa3(S) = E[N] kappa3(X) + 3 Var(N) E[X] Var(X) + kappa3(N) E[X]^3.
    Moments of X are exact if finite, otherwise from the discretised X (s, fx)."""
    EN, VN, gN = (float(v) for v in freq.stats(moments="mvs"))
    mu, var, gX = (float(v) for v in sev.stats(moments="mvs"))
    if np.isfinite([mu, var, gX]).all():
        print("    moments of X: exact, from the fitted distribution")
        k3 = gX * var ** 1.5
    else:
        print("    moments of X: infinite for the fitted distribution, taken from the discretised X on [0, S_MAX]")
        mu = s @ fx
        var, k3 = (s - mu) ** 2 @ fx, (s - mu) ** 3 @ fx
    ES = EN * mu
    VS = EN * var + VN * mu ** 2
    k3S = EN * k3 + 3 * VN * mu * var + gN * VN ** 1.5 * mu ** 3
    return ES, VS, k3S / VS ** 1.5


def q2c(freqs, sevs, best):
    """Collective model for the best frequency/severity pair: Panjer pmf of the yearly loss S."""
    freq, sev = yearly_frequency(freqs[best[0]]), sevs[best[1]]
    s, fx = discretize(sev)
    fs = panjer(freq, fx)
    mean, std = model_moments(freq, sev)
    m = pmf_measures(s, fs)
    table = pd.Series({"Mean (theory)": mean, "Std (theory)": std,
                       "Mean (pmf)": m.Mean, "Std (pmf)": m.Std, **m.filter(like="VaR")}, name=" + ".join(best))
    table.to_frame().T.to_csv(OUT_DIR / "Q2c_table.csv")
    print(f"\n2c) yearly loss S (MDKK), collective model {best}, Panjer on [0, {S_MAX}] with step {H}")
    if np.isinf(std):
        print("    Var(X) is infinite for the fitted Pareto (alpha <= 2), so the theoretical std of S is inf")
    print(table.to_frame().T.round(1).to_string())

    upper = s[np.searchsorted(np.cumsum(fs), 0.999)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s[s <= upper], fs[s <= upper] / H, "k")
    for p, ls in zip(LEVELS, ["--", "-.", ":"]):
        key = f"VaR {100 * p:g}%"
        ax.axvline(m[key], color="r", ls=ls, label=f"{key} = {m[key]:.0f}")
    ax.set_xlabel("Yearly loss s (MDKK), cut at the 99.9% quantile")
    ax.set_ylabel("Density f_S(s)")
    ax.set_title(f"Collective model: {best[0]} frequency, {best[1]} severity")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q2c_collective.png", dpi=150)
    plt.close(fig)
    return freq, sev, s, fx, upper


def q2d(freq, sev, s, fx, upper):
    """Normal N(E[S], Var(S)) and translated gamma k + Gamma(alpha, rate beta) with alpha = 4 / skew^2,
    beta = 2 / (skew Std(S)), k = E[S] - 2 Std(S) / skew, where the moments of S come from N and X.
    Plotted as densities on the same range as the Panjer pmf in c) (up to its 99.9% quantile `upper`)."""
    print("\n2d) moments of S from the moments of N and X")
    ES, VS, skew = aggregate_moments(freq, sev, s, fx)
    sd = np.sqrt(VS)
    print(f"    E[S] {ES:.1f}, Std(S) {sd:.1f}, skew(S) {skew:.2f}")
    alpha, beta, k = 4 / skew ** 2, 2 / (skew * sd), ES - 2 * sd / skew
    approx = {"Normal": stats.norm(ES, sd),
              "Translated gamma": stats.gamma(a=alpha, loc=k, scale=1 / beta)}

    table = {}
    for name, dist in approx.items():
        table[name] = pd.Series({"Mean": dist.mean(), "Std": dist.std(),
                                 **{f"VaR {100 * p:g}%": dist.ppf(p) for p in LEVELS}})
    table = pd.DataFrame(table).T
    table.to_csv(OUT_DIR / "Q2d_table.csv")
    print(table.round(1).to_string())

    grid = s[s <= upper]
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, dist), color in zip(approx.items(), ["C0", "C1"]):
        ax.plot(grid, dist.pdf(grid), color=color, label=name)
        for p, ls in zip(LEVELS, ["--", "-.", ":"]):
            key = f"VaR {100 * p:g}%"
            ax.axvline(table.loc[name, key], color=color, ls=ls, label=f"  {key} = {table.loc[name, key]:.0f}")
    ax.set_xlabel("Yearly loss s (MDKK), same range as in 2c")
    ax.set_ylabel("Density f_S(s)")
    ax.set_title("Normal and translated gamma approximations")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q2d_approximations.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fire = load_fire()
    freqs, best_freq = q2a(monthly(fire))
    sevs, best_sev = q2b(fire)
    q2d(*q2c(freqs, sevs, (best_freq, best_sev)))