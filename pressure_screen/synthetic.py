"""Invented data with a known answer. Used by the tests and for a dry run. Not market data.

Each issuer gets a hidden `pressure` number. With signal=True, pressure drives both the
fundamentals at formation and the later downgrades, defaults and spread moves. With
signal=False, later outcomes are random, so a correct backtest must NOT pass.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SECTORS = ["Healthcare", "Consumer cyclical", "Industrials", "Technology", "Energy", "Utilities", "Communications", "Real estate"]
RATINGS = ["A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+"]
BASE_SPREAD = {"A": 80, "BBB": 130, "BB": 230, "B": 400, "CCC": 900}
FORMATION = pd.Timestamp("2021-12-31")
END = pd.Timestamp("2023-12-31")
PEAK = pd.Timestamp("2022-10-14")


def make(signal=True, n=400, seed=11):
    rng = np.random.default_rng(seed)
    ids = [f"I{i:04d}" for i in range(n)]
    sector = rng.choice(SECTORS, n)
    rating = rng.choice(RATINGS, n, p=np.array([4, 5, 8, 10, 9, 8, 10, 10, 10, 10, 9, 7]) / 100)
    tier = np.array([r.rstrip("+-") for r in rating])
    pressure = rng.normal(0, 1, n)                    # hidden truth
    mkt_view = 0.4 * pressure + rng.normal(0, 1, n)   # the market sees part of it
    issuers = pd.DataFrame({"issuer_id": ids, "name": [f"Issuer {i}" for i in range(n)], "sector": sector,
                            "rating": rating, "is_financial": False})

    quarters = pd.date_range("2017-03-31", END, freq="QE")
    ism_m = pd.date_range("2016-01-31", END, freq="ME")
    ism = pd.DataFrame({"date": ism_m, "ism": 52 + 5 * np.sin(np.arange(len(ism_m)) / 7) + rng.normal(0, 1, len(ism_m))})
    q_ism = ism.assign(q=ism["date"].dt.to_period("Q")).groupby("q")["ism"].mean()
    rows = []
    for i in range(n):
        p = pressure[i]
        rev0 = rng.uniform(200, 2000)
        beta = 0.004 + 0.003 * max(p, -1) + rng.normal(0, 0.001)
        debt = rev0 * rng.uniform(1.0, 2.5) * (1 + 0.25 * p).clip(0.4)
        margin0 = rng.uniform(0.25, 0.5)
        for k, q in enumerate(quarters):
            post = q > FORMATION
            cyc = beta * (q_ism[q.to_period("Q")] - 50)
            rev = rev0 / 4 * (1 + 0.01 * k) * (1 + cyc) * (1 + rng.normal(0, 0.01))
            gm = margin0 - 0.004 * p * max(k - 12, 0) / 4 - (0.02 * max(p, 0) if post and signal else 0)
            ebitda = rev * (gm - 0.12)
            rate = 0.045 + 0.01 * p + (0.03 * (0.3 + 0.2 * p) if post and signal else 0)
            rows.append((ids[i], q, q + pd.Timedelta(days=45), rev, rev * (1 - gm), ebitda, debt * rate / 4, debt,
                         rev0 * rng.uniform(0.05, 0.2) * (1 - 0.15 * p).clip(0.2), rev0 * 0.1, ebitda * 0.4,
                         debt * np.clip(0.3 + 0.2 * p + rng.normal(0, 0.1), 0, 1), rev * (0.18 + 0.04 * p), rev * 0.03,
                         np.clip(0.15 + 0.08 * p + rng.normal(0, 0.05), 0.02, 0.9), bool(rng.random() < 0.3)))
    fundamentals = pd.DataFrame(rows, columns=["issuer_id", "period_end", "filed_date", "revenue", "cogs", "ebitda",
                                               "cash_interest", "total_debt", "cash", "undrawn_revolver", "fcf",
                                               "floating_debt", "staff_cost", "energy_cost", "top_customer_share", "ig_backed"])

    debt_now = fundamentals.groupby("issuer_id")["total_debt"].last()
    mats = []
    for i in range(n):
        years = np.clip(rng.normal(4 - 1.2 * pressure[i], 1.5, 3), 0.3, 12)
        for j, y in enumerate(years):
            mats.append((ids[i], FORMATION + pd.Timedelta(days=int(365 * y)), debt_now[ids[i]] / 3,
                         3.5 + 0.6 * RATINGS.index(rating[i]) / 3 + rng.normal(0, 0.3), bool(j == 0 and pressure[i] > 0),
                         pd.Timestamp("2019-01-01")))
    debt_maturities = pd.DataFrame(mats, columns=["issuer_id", "maturity_date", "amount", "coupon", "is_floating", "known_date"])

    days = pd.bdate_range("2021-06-01", END)
    t = np.arange(len(days))
    after = (days > FORMATION).astype(float)
    hump = after * np.exp(-((days - PEAK).days.values / 120.0) ** 2)
    tsy = 1.5 + 3.0 * after * np.minimum((days - FORMATION).days.values.clip(0) / 500, 1)
    treasury_curve = pd.concat([pd.DataFrame({"date": days, "tenor_years": ten, "yield_pct": tsy + 0.1 * np.log(ten)}) for ten in (2.0, 5.0, 10.0)])
    bonds, eq, defaults, ratings_hist = [], [], [], []
    drive = pressure if signal else rng.normal(0, 1, n)
    for i in range(n):
        base = BASE_SPREAD[tier[i]] * np.exp(0.25 * mkt_view[i])
        spread = base * (1 + hump * (0.5 + 0.35 * drive[i] - (0.25 * mkt_view[i] if signal else 0)).clip(0)) + rng.normal(0, 4, len(days))
        yld = tsy + spread / 100
        price = 100 - (yld - (2.0 + base / 100)) * 4.5
        prob_default = 1 / (1 + np.exp(-(-4.2 + 1.3 * drive[i] + 0.25 * RATINGS.index(rating[i]) / 3)))
        dead = rng.random() < prob_default
        d_day = FORMATION + pd.Timedelta(days=int(rng.uniform(200, 700)))
        keep = days <= d_day if dead else np.ones(len(days), bool)
        if dead:
            defaults.append((ids[i], d_day, "default"))
            price = np.where(days > d_day - pd.Timedelta(days=90), price - 30, price)
        bonds.append(pd.DataFrame({"issuer_id": ids[i], "cusip": f"C{i:05d}", "date": days[keep], "price": price[keep],
                                   "yield_pct": yld[keep], "volume": 1e6, "years_to_maturity": 5.0}))
        px = 50 * np.exp(np.cumsum(rng.normal(0.0003 - 0.0004 * mkt_view[i], 0.015, len(days))))
        eq.append(pd.DataFrame({"issuer_id": ids[i], "date": days[keep], "close": px[keep]}))
        ratings_hist.append((ids[i], pd.Timestamp("2020-01-15"), rating[i]))
        if dead or rng.random() < 1 / (1 + np.exp(-(-2.2 + 1.2 * drive[i]))):
            new = RATINGS[min(RATINGS.index(rating[i]) + int(rng.integers(2, 4)), len(RATINGS) - 1)]
            ratings_hist.append((ids[i], FORMATION + pd.Timedelta(days=int(rng.uniform(60, 600))), new))
    tiers = pd.concat([pd.DataFrame({"tier": k, "date": days[::21], "spread_bp": v, "yield_pct": tsy[::21] + v / 100})
                       for k, v in list(BASE_SPREAD.items()) + [("AA", 60), ("AAA", 45)]])
    ppi = pd.concat([pd.DataFrame({"sector": s, "date": ism_m, "index": 100 * (1 + 0.004) ** np.arange(len(ism_m))}) for s in SECTORS])
    return {"issuers": issuers, "fundamentals": fundamentals, "debt_maturities": debt_maturities,
            "bond_prices": pd.concat(bonds), "equity_prices": pd.concat(eq),
            "ratings_history": pd.DataFrame(ratings_hist, columns=["issuer_id", "date", "rating"]),
            "defaults": pd.DataFrame(defaults, columns=["issuer_id", "date", "event"]),
            "macro_ism": ism, "macro_input_prices": ppi, "tier_yields": tiers, "treasury_curve": treasury_curve}, pressure


def write(out_dir, signal=True, n=400, seed=11):
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    tables, _ = make(signal=signal, n=n, seed=seed)
    spec = {"tables": {}}
    for name, df in tables.items():
        df.to_parquet(out_dir / "data" / f"{name}.parquet", index=False)
        spec["tables"][name] = {"kind": "parquet", "path": f"data/{name}.parquet"}
    (out_dir / "sources.yaml").write_text("# INVENTED DATA for a dry run. Not market data.\n" + yaml.safe_dump(spec))
    return out_dir / "sources.yaml"
