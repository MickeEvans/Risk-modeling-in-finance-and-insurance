from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent
YEARS = range(2021, 2026)


def _load(name):
    df = pd.concat([pd.read_csv(DATA / f"{name}_{y}.csv", encoding="latin-1") for y in (2024, 2025)])
    df.columns = ["region", "age", "sex", "year", "value"]
    df = df[(df.region == "00 Riket") & df.sex.isin(["män", "kvinnor"])]
    df = df[df.age.str.fullmatch(r"\d+ år")]  # single ages only (drops totals, 0-4 år, 100+ år)
    df = df.assign(age=df.age.str.extract(r"(\d+)", expand=False).astype(int),
                   year=df.year.astype(int))
    df = df[df.year.isin(YEARS)]
    return df.set_index(["year", "sex", "age"])["value"].sort_index()


def load_mortality():
    deaths, pop = _load("doda"), _load("medelfolklangd")
    assert len(deaths) == len(pop) == 5 * 2 * 100  # years x sexes x single ages 0-99
    assert deaths.index.equals(pop.index)  # keys line up 1:1
    print(pd.DataFrame({"deaths": deaths.groupby("year").sum(), "pop": pop.groupby("year").sum()}))
    return deaths, pop
