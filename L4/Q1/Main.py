'''
Q1. Explore the Danish fire data (losses > 1 MDKK, 1980-01-03 to 1990-12-31, 2167 fires)
'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR.parent
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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


def q1a(df):
    mon = monthly(df)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, col in zip(axes, ["Building", "Other"]):
        ax.plot(mon.index, mon[col])
        ax.set_ylabel("MDKK")
        ax.set_title(f"Monthly {col} losses")
        ax.grid(True)
    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q1a_monthly_losses.png", dpi=150)
    plt.close(fig)


def q1b(df):
    counts = df.Date.dt.month.value_counts().sort_index().reindex(range(1, 13), fill_value=0)
    n = counts.sum()

    chi2, p = stats.chisquare(counts)
    # expected counts proportional to the number of days in each month
    days = pd.date_range("1980-01-01", "1990-12-31").month.value_counts().sort_index()
    expected_days = n * days / days.sum()
    chi2_d, p_d = stats.chisquare(counts, expected_days)
    print("1b) fires per calendar month:", counts.tolist())
    print(f"    uniform over months: chi2 = {chi2:.2f}, df = 11, p = {p:.4f}")
    print(f"    uniform over days:   chi2 = {chi2_d:.2f}, df = 11, p = {p_d:.4f}")

    mon = monthly(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(MONTH_NAMES, counts, label="Observed")
    ax1.axhline(n / 12, color="k", ls="--", label="Expected, uniform")
    ax1.plot(MONTH_NAMES, expected_days, "r.", ms=10, label="Expected, by days in month")
    ax1.set_ylabel("Number of fires 1980-1990")
    ax1.set_title("Fires per calendar month")
    ax1.set_ylim(0, 270)
    ax1.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
    ax2.boxplot([mon.N[mon.index.month == k] for k in range(1, 13)], tick_labels=MONTH_NAMES)
    ax2.set_ylabel("Fires per month")
    ax2.set_title("Monthly counts across years")
    for ax in (ax1, ax2):
        ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Q1b_seasonality.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fire = load_fire()
    q1a(fire)
    q1b(fire)


