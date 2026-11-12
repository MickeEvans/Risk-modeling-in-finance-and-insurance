'''
Q3. SCR by simulation: 1-year VaR 95% of the loss in basic own funds, for five investment strategies.

1. Inputs
2. Liability model  (same as Q2): populations, b, G, cf1, a_due
3. Asset helpers: load_scenarios, best_estimate BE(t), liability_duration D_L(t)
4. Strategies: strategy(t, A, units, prices_t, BE, D) -> units held of each of the 15 assets
5. Company model: run_scenario -> BOF(t) for t = 0..12
       (i) returns, (ii) cf2 and V/B update, (iii) pay cf2, (iv) rebalance, (v) record
6. q3(): SCR = VaR 95% of -(BOF1 - BOF0) and histograms of BOF1
'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent  # L3/, where the scenario files are

# 1. Inputs
AGE0 = 60                                 # age at calculation date 2025-12-31
RETIREMENT_AGE = 65
MAX_AGE = 120
N_MONTHS = 12 * (MAX_AGE - AGE0)          # 720, liability horizon
RET_MONTH = 12 * (RETIREMENT_AGE - AGE0)  # 60 = last premium month, pension paid from month 61

PREMIUM = 120_000          # yearly, paid monthly
G0 = 2_000_000             # 1st order reserve at t = 0
V0 = 2_100_000             # 2nd order reserve at t = 0, also initial asset value A(0)
R_GUAR = 0.02              # guaranteed interest rate (1st order basis)
R_PROG = 0.02              # prognosis rate (2nd order basis)
TRANSFER_RATE = 0.02       # yearly
S_M = 1 - (1 - TRANSFER_RATE) ** (1 / 12)  # monthly transfer probability

ALPHA, BETA, GAMMA = 0.000739, 5.691932e-07, 0.13631  # Makeham, women (Q1/MakehamResults.txt)

STOCKS = ["BA", "CAT", "KO", "DIS", "JPM", "MMM", "MSFT", "PFE", "WMT", "XOM"]
BONDS = ["Z5", "Z10", "Z15", "Z20", "Z30"]
RATES = ["Y5", "Y10", "Y15", "Y20", "Y30"]
MATURITIES = np.array([5, 10, 15, 20, 30])  # years, matching RATES

MONTHS = np.arange(N_MONTHS + 1)
PAYS_PREMIUM = (MONTHS >= 1) & (MONTHS <= RET_MONTH)  # months 1..60
PAYS_PENSION = MONTHS > RET_MONTH                     # months 61..720

SCENARIO_FILE = "ScenarioFile.csv"   # 2000 scenarios, monthly periods 0..120
HORIZON = 12                         # 1 year
CONFIDENCE = 0.95
STOCK_SHARE = 0.5                    # 50/50 strategies
M_CPPI = 2                           # CPPI multiplier, floor = BE(t)
HIST_BINS = np.linspace(-500_000, 3_000_000, 51)  # 50 bins
HIST_XLIM = (-500_000, 3_000_000)

N_STOCKS = len(STOCKS)
N_ASSETS = len(STOCKS) + len(BONDS)  # asset order is STOCKS + BONDS
IDX_Z5, IDX_Z15, IDX_Z30 = (N_STOCKS + BONDS.index(z) for z in ["Z5", "Z15", "Z30"])


#  2. Liability model
def survival(x):
    """Makeham survival function l(x) = P(alive at age x)."""
    return np.exp(-ALPHA * x - BETA / GAMMA * (np.exp(GAMMA * x) - 1))


def state_model():
    """Expected populations PP, PO at the end of each month; to_TR[k] transfers during month k."""
    l = survival(AGE0 + MONTHS / 12)
    q = np.zeros(N_MONTHS + 1)
    q[1:] = 1 - l[1:] / l[:-1]

    pp, po, to_tr = (np.zeros(N_MONTHS + 1) for _ in range(3))
    pp[0] = 1.0
    for k in range(1, N_MONTHS + 1):
        pp_de = pp[k - 1] * q[k]
        po_de = po[k - 1] * q[k]
        to_tr[k] = (pp[k - 1] - pp_de) * S_M
        pp_po = pp[k - 1] - pp_de - to_tr[k] if k == RET_MONTH + 1 else 0.0
        pp[k] = pp[k - 1] - pp_de - to_tr[k] - pp_po
        po[k] = po[k - 1] - po_de + pp_po
    return pd.DataFrame({"PP": pp, "PO": po, "to_TR": to_tr})


def annuity_factors(rate):
    """Value at t = 0, per living policyholder, of 1/year paid monthly as pension and as premium."""
    l = survival(AGE0 + MONTHS / 12) / survival(AGE0)
    v = (1 + rate) ** (-MONTHS / 12)
    return np.sum(v * l * PAYS_PENSION) / 12, np.sum(v * l * PAYS_PREMIUM) / 12


def guaranteed_payout(reserve0, rate):
    """Yearly pension from reserve0 + PV(premiums) = PV(pension). (G0, R_GUAR) gives the guaranteed b."""
    a_pension, a_premium = annuity_factors(rate)
    return (reserve0 + PREMIUM * a_premium) / a_pension


def annuity_due(rate):
    """a_due(t) = sum_{tau=0}^{720-t} l(60 + (t+tau)/12) / l(60 + t/12) (1+rate)^(-tau/12). Turns V into B."""
    l = survival(AGE0 + MONTHS / 12)
    a = np.empty(N_MONTHS + 1)
    for t in MONTHS:
        tau = MONTHS[:N_MONTHS + 1 - t]
        a[t] = np.sum(l[t:] / l[t] * (1 + rate) ** (-tau / 12))
    return a


def reserve(reserve0, rate, payout):
    """Per-policy reserve while alive: R(k+1) = R(k) (1+rate)^(1/12) l_k / l_{k+1} - cash_{k+1}."""
    l = survival(AGE0 + MONTHS / 12)
    cash = payout / 12 * PAYS_PENSION - PREMIUM / 12 * PAYS_PREMIUM
    R = np.empty(N_MONTHS + 1)
    R[0] = reserve0
    for k in range(N_MONTHS):
        R[k + 1] = R[k] * (1 + rate) ** (1 / 12) * l[k] / l[k + 1] - cash[k + 1]
    return R


def liabilities():
    """Populations, G, the 1st order cash flow cf1 and a_due per month.

    cf1(t) = PO(t) b/12 (t > 60) + to_TR(t) G(t-1) - PP(t) PREMIUM/12 (t <= 60), positive when the company pays.
    """
    pop = state_model()
    b = guaranteed_payout(G0, R_GUAR)
    G = reserve(G0, R_GUAR, b)
    G_prev = np.concatenate([[G0], G[:-1]])  # G(t-1): transfers take last month's reserve

    cf1 = b / 12 * pop.PO * PAYS_PENSION + pop.to_TR * G_prev - PREMIUM / 12 * pop.PP * PAYS_PREMIUM
    liab = pd.DataFrame({"PP": pop.PP, "PO": pop.PO, "to_TR": pop.to_TR, "G": G,
                         "cf1": cf1, "a_due": annuity_due(R_PROG)})
    return liab, b


# 3. Asset helpers
def load_scenarios(file):
    """Prices (n_scen, T+1, 15: stocks then bonds) and zero rates (n_scen, T+1, 5)."""
    df = pd.read_csv(DATA_DIR / file)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])  # index column in ScenarioFile
    df = df.sort_values(["Scenario", "Period"])
    n_scen, n_periods = df.Scenario.nunique(), df.Period.nunique()
    prices = df[STOCKS + BONDS].to_numpy().reshape(n_scen, n_periods, -1)
    rates = df[RATES].to_numpy().reshape(n_scen, n_periods, -1)
    return prices, rates


def discounted_cf(t, cf, zero_rates):
    """Time to payment tau (years) and discounted future cash flows, continuous rates flat outside 5-30 years."""
    tau = (MONTHS[t + 1:] - t) / 12
    y = np.interp(tau, MATURITIES, zero_rates)
    return tau, cf[t + 1:] * np.exp(-y * tau)


def best_estimate(t, cf, zero_rates):
    """BE(t): future expected cash flows discounted with the zero curve at t."""
    return np.sum(discounted_cf(t, cf, zero_rates)[1])


def liability_duration(t, cf, zero_rates):
    """D_L(t): Macaulay duration of the remaining cash flows, sum(tau PV) / sum(PV)."""
    tau, pv = discounted_cf(t, cf, zero_rates)
    return np.sum(tau * pv) / np.sum(pv)


# 4. Strategies
def portfolio(A, S, D, prices_t):
    """Units: S equally over the ten stocks, A - S in Z5/Z30 with duration D (rollover Zn has duration n).

    w5 * 5 + (1 - w5) * 30 = D, clipped so that no bond is shorted. D = 5 puts all bonds in Z5.
    """
    w5 = np.clip((30 - D) / 25, 0, 1)
    amounts = np.zeros(N_ASSETS)
    amounts[:N_STOCKS] = S / N_STOCKS
    amounts[IDX_Z5] = w5 * (A - S)
    amounts[IDX_Z30] = (1 - w5) * (A - S)
    return amounts / prices_t


def bh_z15(t, A, units, prices_t, BE, D):
    """a) BH(0/100): everything in Z15 at t = 0, then hold."""
    if t > 0:
        return units
    amounts = np.zeros(N_ASSETS)
    amounts[IDX_Z15] = A
    return amounts / prices_t


def bh_50_50(t, A, units, prices_t, BE, D):
    """b) BH(50/50): 50% equally in the stocks, 50% in Z5 at t = 0, then hold."""
    return portfolio(A, STOCK_SHARE * A, 5, prices_t) if t == 0 else units


def cm_50_50(t, A, units, prices_t, BE, D):
    """c) CM(50/50): 50% equally in the stocks, 50% in Z5, every month."""
    return portfolio(A, STOCK_SHARE * A, 5, prices_t)


def cm_50_50_duration(t, A, units, prices_t, BE, D):
    """d) CM(50/50): 50% equally in the stocks, 50% in Z5/Z30 with duration D_L(t), every month."""
    return portfolio(A, STOCK_SHARE * A, D, prices_t)


def cppi(t, A, units, prices_t, BE, D):
    """e) CPPI(2): stocks S = M (A - BE(t)) with floor BE(t), no shorting; bonds Z5/Z30 with duration D_L(t)."""
    S = np.clip(M_CPPI * (A - BE), 0, A)
    return portfolio(A, S, D, prices_t)


# 5. Company model
def run_scenario(prices, rates, liab, strategy, T=HORIZON):
    """Run one scenario up to month T. Returns BOF(t) = A(t) - BE(t) for t = 0..T."""
    PP, PO, to_TR, G = (liab[c].to_numpy() for c in ["PP", "PO", "to_TR", "G"])
    cf1, a_due = liab.cf1.to_numpy(), liab.a_due.to_numpy()
    l = survival(AGE0 + MONTHS / 12)
    b = guaranteed_payout(G0, R_GUAR)

    A, V, BE = np.empty(T + 1), np.empty(T + 1), np.empty(T + 1)
    A[0], V[0], B = V0, V0, b
    BE[0] = best_estimate(0, cf1, rates[0])
    units = strategy(0, A[0], None, prices[0], BE[0], liability_duration(0, cf1, rates[0]))
    for t in range(1, T + 1):
        # (i) returns on the units carried from last month
        A1 = units @ prices[t]
        R = A1 / A[t - 1] - 1

        # (ii) actual cash flow with last month's B and V, then update V and B
        cf2 = (PO[t] * B / 12 * (t > RET_MONTH) + to_TR[t] * max(G[t - 1], V[t - 1])
               - PP[t] * PREMIUM / 12 * (t <= RET_MONTH))
        V[t] = (V[t - 1] * (1 + R) * l[t - 1] / l[t]
                + PREMIUM / 12 * (t <= RET_MONTH) - B / 12 * (t > RET_MONTH))
        if t >= RET_MONTH:  # B/12 is paid next month, never below the guarantee
            B = max(b, 12 * V[t] / a_due[t])

        # (iii) pay/receive cf2 proportionally over the holdings
        units = units * (A1 - cf2) / A1
        A[t] = A1 - cf2

        # (iv) rebalance with BE(t) and D_L(t)
        BE[t] = best_estimate(t, cf1, rates[t])
        units = strategy(t, A[t], units, prices[t], BE[t], liability_duration(t, cf1, rates[t]))
    return A - BE  # (v) BOF


def q3():
    liab, _ = liabilities()
    prices, rates = load_scenarios(SCENARIO_FILE)
    strategies = {"a": ("BH(0/100)", bh_z15), "b": ("BH(50/50)", bh_50_50), "c": ("CM(50/50)", cm_50_50),
                  "d": ("CM(50/50) duration", cm_50_50_duration), "e": ("CPPI(2)", cppi)}

    rows = []
    for part, (name, strategy) in strategies.items():
        BOF = np.array([run_scenario(p, r, liab, strategy) for p, r in zip(prices, rates)])
        bof0, bof1 = BOF[0, 0], BOF[:, HORIZON]  # BOF0 is the same in every scenario
        scr = np.quantile(-(bof1 - bof0), CONFIDENCE)  # 1-year VaR 95% of the loss in BOF
        rows.append({"Strategy": f"{part}) {name}", "BOF0": bof0, "Mean BOF1": bof1.mean(), "SCR": scr})

        # one histogram per part question
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(bof1, bins=HIST_BINS)
        ax.axvline(bof0, color="k", linestyle="--", label="BOF0")
        ax.set_xlim(HIST_XLIM)
        ax.set_title(f"3{part}) {name}: BOF after 1 year, SCR = {scr:,.0f}")
        ax.set_xlabel("BOF after 1 year (SEK)")
        ax.set_ylabel("Scenarios")
        ax.grid(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"Q3{part}_BOF1_histogram.png", dpi=150)
        plt.close(fig)

    table = pd.DataFrame(rows).set_index("Strategy")
    table.to_csv(OUT_DIR / "Q3_SCR_table.csv")
    print("3) SCR = VaR 95% of the 1-year loss in BOF\n", table.round(0).to_string())


if __name__ == "__main__":
    q3()
