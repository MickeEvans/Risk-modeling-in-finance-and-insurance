'''
Q4. The five strategies from Q3 over 10 years (120 months).

1. Insolvency probability: P(BOF(t) < 0 for some t = 0..120)
2. Median BOF(120) over the scenarios that never became insolvent
3. Table and scatter plot of median BOF against insolvency probability
4. A, E[G], E[V], BE and BOF in scenarios 76 and 82
5. CPPI(2) allocation between stocks and bonds in scenarios 76 and 82
'''
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parent))  # L3/, so that Q3 can be imported
from Q3.Main import SCENARIO_FILE, SIM_MONTHS, STRATEGIES, liabilities, load_scenarios, run_all

T = SIM_MONTHS
SCENARIOS = [76, 82]
YEARS = np.arange(T + 1) / 12


def q4():
    liab, _ = liabilities()
    _, _, scen_ids = load_scenarios(SCENARIO_FILE)
    paths = run_all(T)
    scen_row = {s: int(np.flatnonzero(scen_ids == s)[0]) for s in SCENARIOS}

    # 1-2. Insolvency probability and median BOF(120) of the solvent scenarios
    rows = []
    for part, (name, _) in STRATEGIES.items():
        BOF = paths[part]["BOF"]
        insolvent = (BOF < 0).any(axis=1)
        rows.append({"Strategy": name, "P(insolvency)": insolvent.mean(),
                     "Median BOF": np.median(BOF[~insolvent, T])})
    table = pd.DataFrame(rows).set_index("Strategy")
    table.to_csv(OUT_DIR / "Q4_table.csv")
    print("4) Insolvency probability and median BOF after 10 years\n", table.round(4).to_string())

    # 3. Scatter plot: up and to the left is best
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(table["P(insolvency)"], table["Median BOF"])
    for name, p, m in zip(table.index, table["P(insolvency)"], table["Median BOF"]):
        ax.annotate(name, (p, m), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Insolvency probability")
    ax.set_ylabel("Median BOF after 10 years (SEK)")
    ax.set_title("Median BOF against insolvency probability")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q4_scatter.png", dpi=150)
    plt.close(fig)

    # 4. Scenario 76 and 82: one png per scenario, one plot per strategy
    EG = liab.EG.to_numpy()[:T + 1]  # same in every scenario
    for s in SCENARIOS:
        r = scen_row[s]
        fig, axes = plt.subplots(len(STRATEGIES), 1, figsize=(9, 16), sharex=True, sharey=True)
        for ax, (part, (name, _)) in zip(axes, STRATEGIES.items()):
            p = paths[part]
            ax.plot(YEARS, p["A"][r], label="A")
            ax.plot(YEARS, EG, label="E[G]")
            ax.plot(YEARS, p["EV"][r], label="E[V]")
            ax.plot(YEARS, p["BE"][r], label="BE")
            ax.plot(YEARS, p["BOF"][r], label="BOF")
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_title(f"{name}, scenario {s}")
            ax.set_ylabel("SEK")
            ax.grid(True)
        axes[-1].set_xlabel("Years")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"Q4_scenario_{s}.png", dpi=150)
        plt.close(fig)

    # 5. CPPI(2) allocation: stocks and bonds = A - stocks, top edge is A
    name, cppi = STRATEGIES["e"][0], paths["e"]
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(12, 5), sharey=True)
    for ax, s in zip(axes, SCENARIOS):
        r = scen_row[s]
        stocks = cppi["stocks"][r]
        ax.stackplot(YEARS, stocks, cppi["A"][r] - stocks, labels=["Stocks", "Bonds"])
        ax.plot(YEARS, cppi["BE"][r], "k--", label="BE (floor)")
        ax.set_title(f"{name} allocation, scenario {s}")
        ax.set_xlabel("Years")
        ax.grid(True)
    axes[0].set_ylabel("SEK")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q4_CPPI_allocation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    q4()
