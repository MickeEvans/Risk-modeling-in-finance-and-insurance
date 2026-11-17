'''
Q5. Stress test: re-run 2008 for strategies a) and b) and compute dBOF = BOF(12) - BOF(0).

1. Monthly 2008 log-returns of stocks and rates from market_data.csv
2. Apply them to the end-2025 levels
3. Rollover zcb prices, Zn(m) = Zn(m-1) exp(n y_n(m-1) - (n - 1/12) y_n(m))
4. Run the ALM model on the stress scenario
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parent))  # L3/, so that Q3 can be imported
from Q3.Main import BONDS, HORIZON, MATURITIES, RATES, STOCKS, STRATEGIES, liabilities, run_scenario

MARKET_FILE = OUT_DIR.parents[1] / "L2" / "market_data.csv"
Z0 = 100.0  # zcb start value, as period 0 in ScenarioFile.csv


def stress_scenario():
    """Scenario file for 2008 applied to end-2025. Rates in market_data are already the ScenarioFile rates."""
    df = pd.read_csv(MARKET_FILE, sep=";", index_col="Date", parse_dates=True)
    df = df.rename(columns={"Y05": "Y5"})

    # 1. Monthly log-returns Jan..Dec 2008
    monthly = df.resample("ME").last().loc["2007-12":"2008-12"]
    returns = np.log(monthly / monthly.shift(1)).iloc[1:].to_numpy()  # (12, 15)

    # 2. X(m) = X_2025 exp(sum_{k<=m} r_k), m = 0..12
    start = df.iloc[-1].to_numpy()
    levels = np.vstack([start, start * np.exp(np.cumsum(returns, axis=0))])
    scen = pd.DataFrame(levels, columns=df.columns)

    # 3. Rollover zcb prices
    y = scen[RATES].to_numpy()
    log_growth = MATURITIES * y[:-1] - (MATURITIES - 1 / 12) * y[1:]
    Z = Z0 * np.exp(np.vstack([np.zeros(len(BONDS)), np.cumsum(log_growth, axis=0)]))
    scen[BONDS] = Z

    scen.insert(0, "Period", np.arange(HORIZON + 1))
    scen.insert(0, "Scenario", 1)
    return scen[["Scenario", "Period"] + STOCKS + BONDS + RATES]


def q5():
    scen = stress_scenario()
    scen.to_csv(OUT_DIR / "Q5_stress_scenario.csv", index=False)
    print("5) Stress scenario 2008 applied to end-2025\n", scen.round(4).to_string(index=False))

    prices, rates = scen[STOCKS + BONDS].to_numpy(), scen[RATES].to_numpy()
    liab, _ = liabilities()
    rows = []
    for part in ["a", "b"]:
        name, strategy = STRATEGIES[part]
        A, BE, _, _ = run_scenario(prices, rates, liab, strategy, HORIZON)
        BOF = A - BE
        rows.append({"Strategy": f"{part}) {name}", "BOF(0)": BOF[0], "BOF(12)": BOF[HORIZON],
                     "dBOF": BOF[HORIZON] - BOF[0]})
    table = pd.DataFrame(rows).set_index("Strategy")
    table.to_csv(OUT_DIR / "Q5_dBOF_table.csv")
    print("\n5) Change in BOF during the 2008 stress year\n", table.round(0).to_string())


if __name__ == "__main__":
    q5()
    
