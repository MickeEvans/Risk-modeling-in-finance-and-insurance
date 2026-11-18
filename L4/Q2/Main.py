"""
Q2. Collective model for the yearly total loss of the Danish fire data

Data: the monthly dataframe from Q1 (132 months, 1980-1990) and the 2167 single losses (Total, MDKK).

a) Frequency: number of fires per month
   Method: Poisson fitted by ML (lambda = mean). Negative binomial fitted by ML, using the profile
   likelihood in r with p = r / (r + mean). Compared by log-likelihood and AIC, and by their pmfs
   plotted over the histogram of the monthly counts.
   Assumption: the monthly counts are iid, so the seasonality and the upward trend found in Q1 are ignored.

b) Severity: size of each fire (Total)
   Method: gamma and log-normal fitted by ML with location 0. Pareto with x_m = 1 MDKK (the reporting
   threshold), where the ML estimate is alpha = n / sum(log x). Compared by log-likelihood, AIC and
   the Kolmogorov-Smirnov distance, and by their cdfs plotted over the empirical cdf.
   Assumption: the losses are iid and independent of the number of fires.

c) Yearly loss S = X_1 + ... + X_N
   Method: N is the sum of 12 iid monthly counts, so N ~ Poisson(12 lambda) or NB(12 r, p).
   X is discretised on the grid 0, H, 2H, ..., S_MAX (rounding method) and the pmf of S is computed
   exactly on the same grid with the Panjer recursion, valid since Poisson and NB are in the (a, b, 0) class.
   VaR_p(S) = smallest s with F_S(s) >= p. Mean and std are the theoretical values
       E[S] = E[N] E[X],  Var(S) = E[N] Var(X) + Var(N) E[X]^2.
   The collective model is the best frequency and the best severity, both by lowest AIC.
   All six pairs are computed to show how the choice of distributions affects VaR.

d) Normal and translated gamma approximations of the collective model in c)
   Method: match the mean, variance and (translated gamma) skewness of the Panjer pmf of S.
   Assumption: the fitted Pareto has alpha < 2, so the true Var(S) and skewness are infinite. The moments
   used are those of the pmf on [0, S_MAX], so they depend on the cut-off S_MAX.
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
N_PARAMS = {"Poisson": 1, "Neg. binomial": 2, "Gamma": 2, "Log-normal": 2, "Pareto": 1}


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
    """ML fits of the single losses. Pareto: x_m = 1 MDKK (threshold), alpha = n / sum(log x)."""
    alpha = len(x) / np.log(x).sum()
    return {"Gamma": stats.gamma(*stats.gamma.fit(x, floc=0)),
            "Log-normal": stats.lognorm(*stats.lognorm.fit(x, floc=0)),
            "Pareto": stats.pareto(alpha, scale=1)}


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.step(xs, ecdf, "k", where="post", label="Empirical")
    ax2.loglog(xs, 1 - ecdf + 1 / len(xs), "k.", ms=3, label="Empirical")
    for name, dist in fits.items():
        ax1.plot(grid, dist.cdf(grid), label=name)
        ax2.loglog(grid, dist.sf(grid), label=name)
    ax1.set_xscale("log")
    ax1.set_ylabel("F(x)")
    ax1.set_title("Fitted cdfs of the losses")
    ax2.set_ylim(1 / (2 * len(xs)), 1.1)
    ax2.set_ylabel("1 - F(x)")
    ax2.set_title("Tail, log-log scale")
    for ax in (ax1, ax2):
        ax.set_xlabel("Loss x (MDKK, log scale)")
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
    """Mean, std, skewness and VaR_p = min{s : F(s) >= p} of a pmf f on the grid s."""
    mean = s @ f
    var = (s - mean) ** 2 @ f
    skew = (s - mean) ** 3 @ f / var ** 1.5
    cdf = np.cumsum(f)
    return pd.Series({"Mean": mean, "Std": np.sqrt(var), "Skew": skew,
                      **{f"VaR {100 * p:g}%": s[np.searchsorted(cdf, p)] for p in LEVELS}})


def q2c(freqs, sevs, best):
    """Panjer pmf of the yearly loss for every frequency/severity pair; the best pair is plotted and returned."""
    table = {}
    for f_name, freq in freqs.items():
        for s_name, sev in sevs.items():
            s, fx = discretize(sev)
            fs = panjer(yearly_frequency(freq), fx)
            mean, std = model_moments(yearly_frequency(freq), sev)
            var_p = pmf_measures(s, fs).filter(like="VaR")
            table[(f_name, s_name)] = pd.Series({"Mean": mean, "Std": std, **var_p, "Mass on grid": fs.sum()})
            if (f_name, s_name) == best:
                fs_best = fs
    table = pd.DataFrame(table).T
    table.to_csv(OUT_DIR / "Q2c_table.csv")
    print(f"\n2c) yearly loss S (MDKK), Panjer on [0, {S_MAX}] with step {H}, collective model = {best}\n",
          table.round(1).assign(**{"Mass on grid": table["Mass on grid"].round(4)}).to_string())

    row = table.loc[best]
    upper = s[np.searchsorted(np.cumsum(fs_best), 0.999)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s[s <= upper], fs_best[s <= upper] / H, "k")
    for p, ls in zip(LEVELS, ["--", "-.", ":"]):
        key = f"VaR {100 * p:g}%"
        ax.axvline(row[key], color="r", ls=ls, label=f"{key} = {row[key]:.0f}")
    ax.set_xlabel("Yearly loss s (MDKK), cut at the 99.9% quantile")
    ax.set_ylabel("Density f_S(s)")
    ax.set_title(f"Collective model: {best[0]} frequency, {best[1]} severity (mean {row['Mean']:.0f})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q2c_collective.png", dpi=150)
    plt.close(fig)
    return s, fs_best


def q2d(s, fs):
    """Normal N(mu, sd^2) and translated gamma x0 + Gamma(a, scale) with a = 4/skew^2,
    scale = sd skew / 2, x0 = mu - 2 sd / skew, where mu, sd, skew are the moments of the Panjer pmf fs."""
    print("\n2d) moments of the collective pmf for different cut-offs (true std and skewness are infinite)")
    for cut in (5_000, 10_000, S_MAX):
        m = pmf_measures(s[s <= cut], fs[s <= cut])
        print(f"    S_MAX = {cut:6d}: mean {m.Mean:.1f}, std {m.Std:.1f}, skewness {m.Skew:.2f}")
    m = pmf_measures(s, fs)
    mu, sd, skew = m.Mean, m.Std, m.Skew
    approx = {"Normal": stats.norm(mu, sd),
              "Translated gamma": stats.gamma(4 / skew ** 2, loc=mu - 2 * sd / skew, scale=sd * skew / 2)}

    table = {"Collective (Panjer)": m.drop("Skew")}
    for name, dist in approx.items():
        table[name] = pd.Series({"Mean": dist.mean(), "Std": dist.std(),
                                 **{f"VaR {100 * p:g}%": dist.ppf(p) for p in LEVELS}})
    table = pd.DataFrame(table).T
    table.to_csv(OUT_DIR / "Q2d_table.csv")
    print(table.round(1).to_string())

    grid = np.linspace(0, 1.2 * table[f"VaR {100 * LEVELS[-1]:g}%"].max(), 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for (name, dist), color in zip(approx.items(), ["C0", "C1"]):
        ax1.plot(grid, dist.pdf(grid), color=color, label=name)
        ax2.plot(grid, dist.cdf(grid), color=color, label=name)
        ax2.plot(table.loc[name, [f"VaR {100 * p:g}%" for p in LEVELS]], LEVELS, "o", color=color)
    for p in LEVELS:
        ax2.axhline(p, color="gray", ls=":", lw=0.8)
    ax1.set_ylabel("Density")
    ax1.set_title("Approximations of the yearly loss")
    ax2.set_ylabel("F(s)")
    ax2.set_title("cdf, dots = VaR 95%, 99%, 99.5%")
    for ax in (ax1, ax2):
        ax.set_xlabel("Yearly loss s (MDKK)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q2d_approximations.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fire = load_fire()
    freqs, best_freq = q2a(monthly(fire))
    sevs, best_sev = q2b(fire)
    s, fs = q2c(freqs, sevs, (best_freq, best_sev))
    q2d(s, fs)