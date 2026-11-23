from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

OUT_DIR = Path(__file__).resolve().parent
AMOUNT = np.array([1, 2, 3])
P_DEATH = np.array([0.005, 0.010, 0.015])
N_POL = np.array([1000, 500, 500])
BUFFER = 5.0       # capital buffer U (MSEK)
LOADING = 0.10     # insurer's premium loading
SL_LOADING = 0.20  # reinsurer's loading on the stop-loss premium


def de_pril_weights(amount, p, n):
    """Weights w_ik, k = 0, ..., K_MAX - 1, per group, with f_ik = 1 for k = amount_i and 0 otherwise:
        w_ik = p_i / q_i ((k + 1) f_{i,k+1} - sum_{j=1}^{min(k, m_i)} f_ij w_{i,k-j})."""
    k_max = int(n @ amount)
    F = np.zeros((len(amount), k_max + 2))
    F[np.arange(len(amount)), amount] = 1.0
    ratio = p / (1 - p)
    w = np.zeros((len(amount), k_max))
    for k in range(k_max):
        j = np.arange(1, min(k, amount.max()) + 1)
        w[:, k] = ratio * ((k + 1) * F[:, k + 1] - (F[:, j] * w[:, k - j]).sum(axis=1))
    return w


def de_pril(amount, p, n):
    """pmf s_k = P(S = k), k = 0, ..., K_MAX, of the individual model:
        s_0 = prod_i q_i^{n_i},  s_{k+1} = 1 / (k + 1) sum_{j=0}^{k} W_j s_{k-j},  W_j = sum_i n_i w_ij."""
    W = n @ de_pril_weights(amount, p, n)
    k = np.arange(len(W) + 1)
    s = np.zeros(len(k))
    s[0] = np.prod((1 - p) ** n)
    for i in range(len(W)):
        s[i + 1] = (W[:i + 1] @ s[i::-1]) / (i + 1)
    return k, s


def q4a(k, s, ES, sd):
    """Proportional reinsurance: the insurer keeps D = retention S and A = BUFFER + retention P.
    Insolvency P(D > A), exact from the pmf and approximated by D ~ N(retention E[S], (retention sd)^2)."""
    P = (1 + LOADING) * ES
    retention = np.linspace(0, 1, 1001)
    A = BUFFER + retention * P
    exact = np.array([s[r * k > a].sum() for r, a in zip(retention, A)])
    with np.errstate(divide="ignore"):
        normal = np.where(retention > 0, stats.norm.sf((A - retention * ES) / (retention * sd)), 0.0)
    df = pd.DataFrame({"Retention": retention, "Exact": exact, "Normal": normal,
                       "Expected surplus": A - retention * ES})
    df.to_csv(OUT_DIR / "Q4a_proportional.csv", index=False)
    print(f"\n4a) proportional reinsurance, premium P = {P:.2f}, buffer {BUFFER} MSEK")
    print(df.iloc[::250].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"    check at retention 1: P(S >= 47) = {s[k >= 47].sum():.6f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(100 * retention, exact, drawstyle="steps-post", label="Exact (de Pril)")
    ax.plot(100 * retention, normal, label="Normal approximation")
    ax.set_xlabel("Retention (%)")
    ax.set_ylabel("Insolvency probability")
    ax.set_title("Proportional reinsurance")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q4a_proportional.png", dpi=150)
    plt.close(fig)


def q4b(k, s, ES):
    """Stop-loss reinsurance with retention level M: the insurer keeps D = min(S, M) and pays the premium
    (1 + SL_LOADING) Pi(M), Pi(M) = E[(S - M)^+]. Insolvency P(D > A) with A = BUFFER + P - (1 + SL_LOADING) Pi(M),
    exact from the pmf and approximated by D ~ N(E[D], Var(D))."""
    P = (1 + LOADING) * ES
    M = np.arange(1, 51)
    Pi = np.array([np.maximum(k - m, 0) @ s for m in M])
    ED = ES - Pi
    sdD = np.sqrt(np.array([np.minimum(k, m) ** 2 @ s for m in M]) - ED ** 2)
    A = BUFFER + P - (1 + SL_LOADING) * Pi
    exact = np.array([s[np.minimum(k, m) > a].sum() for m, a in zip(M, A)])
    normal = stats.norm.sf((A - ED) / sdD)
    df = pd.DataFrame({"Retention level": M, "Pi(M)": Pi, "SL premium": (1 + SL_LOADING) * Pi, "E[D]": ED,
                       "Std(D)": sdD, "Assets": A, "Exact": exact, "Normal": normal, "Expected surplus": A - ED})
    df.to_csv(OUT_DIR / "Q4b_stoploss.csv", index=False)
    print(f"\n4b) stop-loss reinsurance, premium P = {P:.2f}, buffer {BUFFER} MSEK, stop-loss loading {SL_LOADING}")
    print(df[M % 10 == 0].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"    check: Pi(M) decreasing in M: {bool(np.all(np.diff(Pi) < 0))}, "
          f"stop-loss premium at M = 50: {(1 + SL_LOADING) * Pi[-1]:.2e}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(M, exact, drawstyle="steps-post", label="Exact (de Pril)")
    ax.plot(M, normal, label="Normal approximation")
    ax.set_xlabel("Retention level M (MSEK)")
    ax.set_ylabel("Insolvency probability")
    ax.set_title("Stop-loss reinsurance")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q4b_stoploss.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    k, s = de_pril(AMOUNT, P_DEATH, N_POL)
    ES, VS = k @ s, (k - k @ s) ** 2 @ s
    print(f"Exact distribution of S (de Pril), k = 0, ..., {k[-1]} MSEK")
    print(f"    s_0 = {s[0]:.4e}   (prod q_i^n_i = {np.prod((1 - P_DEATH) ** N_POL):.4e})")

    w = de_pril_weights(AMOUNT, P_DEATH, N_POL)
    for i, (m, p) in enumerate(zip(AMOUNT, P_DEATH)):
        closed = np.zeros(w.shape[1])
        l = np.arange((w.shape[1] - m) // m + 1)
        closed[m - 1 + l * m] = m * p / (1 - p) * (-p / (1 - p)) ** l
        print(f"    group {m}: max |w - closed form| = {np.abs(w[i] - closed).max():.2e}")

    print(f"    sum(s) = {s.sum():.10f}")
    print(f"    E[S]   = {ES:.4f}   (sum n_i i p_i = {N_POL @ (AMOUNT * P_DEATH):.4f})")
    print(f"    Var(S) = {VS:.4f}   (sum n_i i^2 p_i q_i = {N_POL @ (AMOUNT ** 2 * P_DEATH * (1 - P_DEATH)):.4f})")
    print(f"    Premium P = (1 + {LOADING}) E[S] = {(1 + LOADING) * ES:.2f} MSEK")

    q4a(k, s, ES, np.sqrt(VS))
    q4b(k, s, ES)
