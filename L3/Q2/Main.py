
'''
1. Inputs
   constants: ages, premium, G0, V0, rates, Makeham parameters, column names

   Makeham: Women  0.000739  5.691932e-07  0.13631

   Age 60

   Status, PP
   Premium : 120 000
   G(0) = 2 000 000
   V(0) = 2 100 000

   Assets 210 000 in every stock, see OneScenario.csv

2. Liability model  (runs once)
   survival(x)                 -> probability of being alive at age x
   state_model()               -> expected populations PP, PO, TR, DE per month
   guaranteed_payout()         -> b
   first_order_reserve(b)      -> G(t) for every month
   first_order_cashflows()     -> cf1(t) for every month
   annuity_factors()           -> used to turn V into a pension B

3. Asset helpers
   load_scenarios(file)        -> prices and rates per scenario
   best_estimate(t, rates)     -> BE(t)

4. Strategies
   one function per strategy: given the asset value, return how much to hold of each asset

5. Company model  (the loop)
   run_scenario(prices, rates, liabilities, strategy)
       for each month t:
           (i)   apply returns
           (ii)  update V, B and the actual cash flow cf2
           (iii) pay or receive the cash flow
           (iv)  rebalance
           (v)   record A, BE, BOF, E[G], E[V]

6. Questions
   q2(), q3(), ...  call the pieces above and make plots and tables
'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent  # L3/, where the scenario files are

# ---------------------------------------------------------------- 1. Inputs
AGE0 = 60                                 # age at calculation date 2025-12-31 (same as Q1)
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


# ---------------------------------------------------------------- 2. Liability model
def survival(x, mortality_factor=1.0):
    """Makeham survival function l(x) = P(alive at age x).

    mortality_factor scales ALPHA and BETA, i.e. the whole mortality intensity mu(x).
    """
    alpha, beta = mortality_factor * ALPHA, mortality_factor * BETA
    return np.exp(-alpha * x - beta / GAMMA * (np.exp(GAMMA * x) - 1))


def state_model(mortality_factor=1.0):
    """Expected populations PP, PO, TR, DE at the end of each month (same transitions as Q1).

    to_TR[k] is the part that transfers during month k. mortality_factor = 0.9 gives the
    longevity shock from the lecture (mu -10%). Only the populations use the shocked mortality.
    """
    ages = AGE0 + MONTHS / 12
    l = survival(ages, mortality_factor)
    q = np.zeros(N_MONTHS + 1)
    q[1:] = 1 - l[1:] / l[:-1]

    pp, po, tr, de, to_tr = (np.zeros(N_MONTHS + 1) for _ in range(5))
    pp[0] = 1.0
    for k in range(1, N_MONTHS + 1):
        pp_de = pp[k - 1] * q[k]
        po_de = po[k - 1] * q[k]
        to_tr[k] = (pp[k - 1] - pp_de) * S_M
        pp_po = pp[k - 1] - pp_de - to_tr[k] if k == RET_MONTH + 1 else 0.0
        pp[k] = pp[k - 1] - pp_de - to_tr[k] - pp_po
        po[k] = po[k - 1] - po_de + pp_po
        tr[k] = tr[k - 1] + to_tr[k]
        de[k] = de[k - 1] + pp_de + po_de
    return pd.DataFrame({"PP": pp, "PO": po, "TR": tr, "DE": de, "to_TR": to_tr})


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
    """a_due(t) = sum_{tau=0}^{720-t} l(60 + (t+tau)/12) / l(60 + t/12) (1+rate)^(-tau/12), for all t.

    Value of 1 per month, paid from now on, at the current age. Turns V into the pension B (Lecture 11).
    """
    l = survival(AGE0 + MONTHS / 12)
    a = np.empty(N_MONTHS + 1)
    for t in MONTHS:
        tau = MONTHS[:N_MONTHS + 1 - t]
        a[t] = np.sum(l[t:] / l[t] * (1 + rate) ** (-tau / 12))
    return a


def reserve(reserve0, rate, payout):
    """Per-policy reserve while alive (as L2/Q7): R(k+1) = R(k) (1+rate)^(1/12) l_k / l_{k+1} - cash_{k+1}.

    Gives G with (G0, R_GUAR, b). V is scenario dependent and is updated in run_scenario().
    """
    l = survival(AGE0 + MONTHS / 12)
    cash = payout / 12 * PAYS_PENSION - PREMIUM / 12 * PAYS_PREMIUM
    R = np.empty(N_MONTHS + 1)
    R[0] = reserve0
    for k in range(N_MONTHS):
        R[k + 1] = R[k] * (1 + rate) ** (1 / 12) * l[k] / l[k + 1] - cash[k + 1]
    return R


def liabilities(mortality_factor=1.0):
    """Scenario independent part: populations, G, E[G], the 1st order cash flow cf1 and a_due per month.

    Cash flow is positive when the company pays:
        cf1(t) = PO(t) b/12 (t > 60) + to_TR(t) G(t-1) - PP(t) PREMIUM/12 (t <= 60)
    mortality_factor only changes the populations; b, G and a_due always use the unshocked survival.
    """
    pop = state_model(mortality_factor)
    b = guaranteed_payout(G0, R_GUAR)
    G = reserve(G0, R_GUAR, b)
    G_prev = np.concatenate([[G0], G[:-1]])  # G(t-1): transfers take last month's reserve

    cf1 = b / 12 * pop.PO * PAYS_PENSION + pop.to_TR * G_prev - PREMIUM / 12 * pop.PP * PAYS_PREMIUM
    liab = pd.DataFrame({"PP": pop.PP, "PO": pop.PO, "to_TR": pop.to_TR, "G": G,
                         "EG": G * (pop.PP + pop.PO), "cf1": cf1, "a_due": annuity_due(R_PROG)})
    return liab, b


# ---------------------------------------------------------------- 3. Asset helpers
def load_scenarios(file):
    """Prices (n_scen, T+1, 15: stocks then bonds) and zero rates (n_scen, T+1, 5)."""
    df = pd.read_csv(DATA_DIR / file)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])  # index column in ScenarioFile
    df = df.sort_values(["Scenario", "Period"])
    n_scen, n_periods = df.Scenario.nunique(), df.Period.nunique()
    prices = df[STOCKS + BONDS].to_numpy().reshape(n_scen, n_periods, -1)
    rates = df[RATES].to_numpy().reshape(n_scen, n_periods, -1)
    return prices, rates


def best_estimate(t, cf, zero_rates):
    """BE(t): future expected cash flows discounted with the zero curve at t.

    Rates are continuously compounded (exp(Y) = 1.03 in OneScenario); flat outside 5-30 years.
    """
    tau = (MONTHS[t + 1:] - t) / 12
    y = np.interp(tau, MATURITIES, zero_rates)
    return np.sum(cf[t + 1:] * np.exp(-y * tau))


# ---------------------------------------------------------------- 4. Strategies
# strategy(A, prices_t, rates_t, t) -> units held of each of the 15 assets (STOCKS + BONDS)
def equal_stocks(A, prices_t, rates_t, t):
    """All assets split equally over the ten stocks, no bonds (rebalanced every month)."""
    units = np.zeros(len(STOCKS) + len(BONDS))
    units[:len(STOCKS)] = A / len(STOCKS) / prices_t[:len(STOCKS)]
    return units


# ---------------------------------------------------------------- 5. Company model
def run_scenario(prices, rates, liab, strategy):
    """Run one scenario. prices (T+1, 15), rates (T+1, 5). Returns E[G], E[V], A, BE, BOF per month.

    V, B and the actual cash flow cf2 depend on the asset return, so they are updated month by month.
    """
    T = len(prices) - 1
    PP, PO, to_TR, G = (liab[c].to_numpy() for c in ["PP", "PO", "to_TR", "G"])
    cf1, a_due = liab.cf1.to_numpy(), liab.a_due.to_numpy()
    l = survival(AGE0 + MONTHS / 12)  # unshocked, per-policy mortality gains in V
    b = guaranteed_payout(G0, R_GUAR)

    A, V, BE = np.empty(T + 1), np.empty(T + 1), np.empty(T + 1)
    A[0], V[0], B = V0, V0, b
    BE[0] = best_estimate(0, cf1, rates[0])
    units = strategy(A[0], prices[0], rates[0], 0)
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

        BE[t] = best_estimate(t, cf1, rates[t])
        units = strategy(A[t], prices[t], rates[t], t)  # (iv) rebalance
    EV = V * (PP[:T + 1] + PO[:T + 1])
    return pd.DataFrame({"EG": liab.EG.to_numpy()[:T + 1], "EV": EV,
                         "A": A, "BE": BE, "BOF": A - BE})  # (v) record


# ---------------------------------------------------------------- 6. Questions
def q2():
    liab, b = liabilities()
    prices, rates = load_scenarios("OneScenario.csv")
    res = run_scenario(prices[0], rates[0], liab, equal_stocks)
    print(f"Guaranteed pension b = {b:,.0f}")

    # a) reserves, assets and BE over 30 years
    first30 = res.loc[:360]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # E[V] is drawn last and dashed: with 100% stocks it lies almost on top of the assets
    for col, label, style in [("EG", "E[G] (1st order reserve)", "-"), ("A", "Assets", "-"),
                              ("BE", "Best estimate", "-"), ("EV", "E[V] (2nd order reserve)", "--")]:
        ax.plot(first30.index / 12, first30[col], style, label=label)
    ax.set_xlabel("Years from 2025-12-31")
    ax.set_ylabel("SEK")
    ax.set_title("ALM model, OneScenario, equally weighted stocks")
    ax.set_xlim(0, 30)
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q2a_reserves.png", dpi=150)
    plt.close(fig)

    # b) table every fifth year
    table = res.loc[0:360:60].copy()
    table.index = table.index // 12
    table.index.name = "Year"
    table.to_csv(OUT_DIR / "Q2b_table.csv")
    print("\n2b)\n", table.round(0).to_string())

    # c) SCR longevity: mu -10% (populations only), assets unchanged, so SCR = max(BE_shock(0) - BE(0), 0)
    liab_shock, _ = liabilities(mortality_factor=0.9)
    be_shock = best_estimate(0, liab_shock.cf1.to_numpy(), rates[0, 0])
    scr_longevity = max(be_shock - res.BE[0], 0)
    print(f"\n2c) BE(0) = {res.BE[0]:,.0f}, BE shocked = {be_shock:,.0f}, SCR_longevity = {scr_longevity:,.0f}")


if __name__ == "__main__":
    q2()