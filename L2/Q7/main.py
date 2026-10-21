import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Q6"))
from main import estimate_mu, fit_makeham, survival  # Q6/main.py
from scb import load_mortality

# Women's Makeham parameters, fitted as in 6b
deaths, pop = load_mortality()
fit_ages = range(20, 91)
params = fit_makeham(fit_ages, estimate_mu(deaths, pop, fit_ages)["Women"])

# Policy
x = 60.0  # age at calculation date 2025-12-31
u = 65 - x  # years to retirement
r = 1.02  # guaranteed interest rate
P = 120_000  # yearly premium
G0 = 2_000_000

# Month t = end of month t after 2025-12-31, age x + t/12
t = np.arange(0, int(12 * (120 - x)) + 1)
l = survival(x + t / 12, *params) / survival(x, *params)  # P(alive at t | alive at 0)
v = r ** (-t / 12)
premium = np.where((t >= 1) & (t / 12 <= u), P / 12, 0.0)  # t = 1..60
pays = (t / 12 > u)  # t = 61..720

# 7a: equivalence G(0) = sum v*l*(b/12*pays - premium)
b = 12 * (G0 + np.sum(v * l * premium)) / np.sum(v * l * pays)
cash = b / 12 * pays - premium
print(f"Yearly payout b = {b:,.0f}")

# Reserve by recursion: G(k+1) = G(k) * r^(1/12) * l_k / l_{k+1} - cash_{k+1}
G = np.empty(len(t))
G[0] = G0
for k in range(len(t) - 1):
    G[k + 1] = G[k] * r ** (1 / 12) * l[k] / l[k + 1] - cash[k + 1]

# Check against the prospective reserve G(k) = sum_{j>k} v_j l_j cash_j / (v_k l_k)
disc = v * l * cash
G_prosp = (disc[::-1].cumsum()[::-1] - disc) / (v * l)
assert np.isclose(G_prosp[0], G0)
assert np.allclose(G[:361], G_prosp[:361], rtol=1e-6)

# 7b
T = 360
years = 2025 + t[: T + 1] / 12
plt.figure()
plt.plot(years, G[: T + 1])
plt.axvline(2025 + u, color="gray", linestyle="--", label="Retirement (age 65)")
plt.xlabel("Year")
plt.ylabel("Reserve G(t) [SEK]")
plt.title("1st order reserve")
plt.legend()
plt.savefig(Path(__file__).resolve().parent / "7b_reserve.png", dpi=150)

# 7c
idx = np.arange(0, T + 1, 60)
print(pd.DataFrame({"year": 2025 + idx // 12, "age": (x + idx / 12).round(1), "reserve": G[idx].round(0)},
                   index=pd.Index(idx, name="t (months)")))

plt.show()
