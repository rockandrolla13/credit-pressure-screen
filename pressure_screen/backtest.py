"""Backtest: form the screen on one date, then ask whether it predicted what followed.

The screen FAILS if any of the five tests fails. A test that the data cannot run is reported
as NOT TESTABLE with the reason. It is never counted as a pass.
"""
import numpy as np
import pandas as pd

from .schema import to_notch
from .score import score
from .variables import _spread, compute


def auc(scores, outcome):
    """Chance that a random bad issuer has a higher score than a random good one. 0.5 is no skill."""
    d = pd.DataFrame({"s": scores, "y": outcome}).dropna()
    pos, neg = d[d["y"] == 1], d[d["y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = d["s"].rank()
    return (ranks[pos.index].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def outcomes(tables, formation, end, peak, cfg):
    """What happened to each issuer between formation and end."""
    formation, end, peak = map(pd.Timestamp, (formation, end, peak))
    bt = cfg["backtest"]
    ids = tables["issuers"]["issuer_id"].drop_duplicates()
    out = pd.DataFrame(index=pd.Index(ids, name="issuer_id"))
    notes = []

    rh = tables.get("ratings_history")
    if rh is not None and len(rh):
        rh = rh.assign(notch=rh["rating"].map(to_notch)).dropna(subset=["notch"]).sort_values("date")
        start = rh[rh["date"] <= formation].groupby("issuer_id")["notch"].last()
        worst = rh[(rh["date"] > formation) & (rh["date"] <= end)].groupby("issuer_id")["notch"].max()
        worst = worst.reindex(start.index).fillna(start)
        out["downgrade"] = (((worst - start) >= bt["downgrade_notches"]) | ((worst >= 17) & (start < 17))).reindex(out.index)
    else:
        out["downgrade"] = np.nan
        notes.append("No ratings history: downgrades cannot be measured.")

    df = tables.get("defaults")
    if df is not None and len(df):
        hit = df[(df["date"] > formation) & (df["date"] <= end)]["issuer_id"].unique()
        out["default"] = out.index.isin(hit)
        if len(hit) == 0:
            notes.append("The defaults table has no event in the window. Check that the table covers it.")
    else:
        out["default"] = np.nan
        notes.append("No defaults table: defaults cannot be measured.")
    known = out[["downgrade", "default"]].notna().any(axis=1)
    out["bad"] = ((out["downgrade"] == True) | (out["default"] == True)).where(known).astype(float)

    s0 = _spread(tables, formation, cfg["price_window_days"])["spread_bp"]
    s1 = _spread(tables, peak, cfg["price_window_days"])["spread_bp"]
    out["spread_start"] = s0.reindex(out.index)
    out["widening"] = (s1 - s0).reindex(out.index)
    out["lost_price"] = out["spread_start"].notna() & s1.reindex(out.index).isna()

    cov0 = compute(tables, formation, cfg)["coverage_now"]
    cov1 = compute(tables, end, cfg)["coverage_now"]
    out["coverage_fall"] = ((cov0 - cov1) >= bt["coverage_fall"]).where(cov0.notna() & cov1.notna()).reindex(out.index)
    out.attrs["notes"] = notes
    return out


def _rates(d, col, q):
    d = d.dropna(subset=["F", col])
    if len(d) < 20:
        return np.nan, np.nan, len(d)
    hi, lo = d["F"].quantile(1 - q), d["F"].quantile(q)
    return d.loc[d["F"] >= hi, col].mean(), d.loc[d["F"] <= lo, col].mean(), len(d)


def _ratio(top, bottom):
    if np.isnan(top) or np.isnan(bottom):
        return np.nan
    return np.inf if bottom == 0 and top > 0 else (np.nan if bottom == 0 else top / bottom)


def evaluate(scored, out, cfg):
    """Run the five failure tests. Returns (table of tests, dict of numbers)."""
    bt = cfg["backtest"]
    q = bt["quantile"]
    d = scored.join(out, how="left")
    d["excess_widening"] = d["widening"] - d.groupby(["tier", "sector"])["widening"].transform("median")
    s = d[d["gate0"]]
    tests, nums = [], {"scored": int(len(s)), "universe": int(len(d)), "notes": list(out.attrs.get("notes", []))}

    def add(n, name, verdict, detail):
        tests.append({"test": n, "name": name, "verdict": verdict, "detail": detail})

    # 1. Top fifth against bottom fifth on downgrades and defaults.
    top, bottom, n = _rates(s, "bad", q)
    ratio = _ratio(top, bottom)
    nums.update(bad_rate_top=top, bad_rate_bottom=bottom, rate_ratio=ratio, base_bad_rate=s["bad"].mean())
    rng = np.random.default_rng(bt["seed"])
    boot = []
    sb = s.dropna(subset=["F", "bad"])
    for _ in range(bt["bootstrap"] if len(sb) >= 20 else 0):
        r = sb.iloc[rng.integers(0, len(sb), len(sb))]
        t, b, _n = _rates(r, "bad", q)
        boot.append(_ratio(t, b))
    boot = [x for x in boot if np.isfinite(x)]
    lo, hi = (np.percentile(boot, [5, 95]) if len(boot) > 50 else (np.nan, np.nan))
    nums.update(ratio_ci_low=lo, ratio_ci_high=hi)
    if np.isnan(ratio):
        add(1, "Top fifth has 1.5 times the downgrade and default rate of the bottom fifth", "NOT TESTABLE",
            f"{n} issuers have both a score and an outcome, or the bottom fifth has no event to compare with.")
    else:
        add(1, "Top fifth has 1.5 times the downgrade and default rate of the bottom fifth",
            "PASS" if ratio >= bt["min_rate_ratio"] else "FAIL",
            f"Top {top:.1%}, bottom {bottom:.1%}, ratio {ratio:.2f}. 90% bootstrap range {lo:.2f} to {hi:.2f}.")

    # 2. Does F add to the starting spread?
    nums["auc_F"] = auc(s["F"], s["bad"])
    nums["auc_spread"] = auc(s["spread_start"], s["bad"])
    nums["auc_rating"] = auc(s["notch"], s["bad"])
    nums["auc_leverage"] = auc(s["net_leverage"], s["bad"])
    w = s.dropna(subset=["F", "bad", "spread_start"])
    inside = []
    if len(w) >= 60:
        for _, part in w.groupby(pd.qcut(w["spread_start"], 3, duplicates="drop"), observed=True):
            inside.append(auc(part["F"], part["bad"]))
    inside = [x for x in inside if not np.isnan(x)]
    nums["auc_F_within_spread_terciles"] = float(np.mean(inside)) if inside else np.nan
    if not inside:
        add(2, "F adds information to the starting spread", "NOT TESTABLE", "Too few issuers with a spread, a score and an outcome.")
    else:
        m = nums["auc_F_within_spread_terciles"]
        add(2, "F adds information to the starting spread", "PASS" if m >= bt["min_auc_within_spread_tercile"] else "FAIL",
            f"AUC of F inside spread thirds {m:.2f}. Alone: F {nums['auc_F']:.2f}, spread {nums['auc_spread']:.2f}, "
            f"rating {nums['auc_rating']:.2f}, leverage {nums['auc_leverage']:.2f}.")

    # 3. Drop one sector at a time.
    worst = (None, np.inf)
    for sec in s["sector"].dropna().unique():
        t, b, _n = _rates(s[s["sector"] != sec], "bad", q)
        r = _ratio(t, b)
        if not np.isnan(r) and r < worst[1]:
            worst = (sec, r)
    if worst[0] is None:
        add(3, "The result survives dropping any one sector", "NOT TESTABLE", "Test 1 could not be computed on the subsets.")
    else:
        nums["worst_drop_sector"], nums["worst_drop_ratio"] = worst
        add(3, "The result survives dropping any one sector", "PASS" if worst[1] >= bt["min_rate_ratio"] else "FAIL",
            f"Lowest ratio {worst[1]:.2f}, when '{worst[0]}' is dropped.")

    # 4. The gates must discriminate.
    rate = (s["gate1"] & s["gate2"]).mean() if len(s) else np.nan
    nums["gate_pass_rate"] = rate
    add(4, "The two gates pass at most 40% of scored issuers", "NOT TESTABLE" if np.isnan(rate) else ("PASS" if rate <= bt["max_gate_pass_rate"] else "FAIL"),
        "No issuer was scored." if np.isnan(rate) else f"The gates pass {rate:.0%}.")

    # 5. High F with low M should do worse afterwards than high F with high M.
    h = s[s["F"] >= s["F"].quantile(0.6)].dropna(subset=["M", "excess_widening"])
    if len(h) < 30:
        add(5, "High F with low M widens more afterwards than high F with high M", "NOT TESTABLE", f"Only {len(h)} high-F issuers have M and a later spread.")
    else:
        med = h["M"].median()
        low_m, high_m = h.loc[h["M"] <= med, "excess_widening"].mean(), h.loc[h["M"] > med, "excess_widening"].mean()
        nums.update(widening_highF_lowM=low_m, widening_highF_highM=high_m)
        add(5, "High F with low M widens more afterwards than high F with high M", "PASS" if low_m > high_m else "FAIL",
            f"Excess widening: low M {low_m:+.0f}bp, high M {high_m:+.0f}bp.")

    nums["lost_price_share"] = float(d["lost_price"].mean()) if "lost_price" in d else np.nan
    nums["unscored_share"] = float(1 - len(s) / len(d)) if len(d) else np.nan
    verdicts = [t["verdict"] for t in tests]
    nums["overall"] = "FAIL" if "FAIL" in verdicts else ("INCOMPLETE" if "NOT TESTABLE" in verdicts else "PASS")
    return pd.DataFrame(tests), nums


def run(tables, formation, end, peak, cfg):
    scored = score(compute(tables, formation, cfg), cfg)
    out = outcomes(tables, formation, end, peak, cfg)
    tests, nums = evaluate(scored, out, cfg)
    return scored, out, tests, nums
