"""Compute the screen variables for one as-of date, using only data known on that date.

Rule: a missing input gives a missing variable. It never gives a safe value.
"""
import numpy as np
import pandas as pd

from .schema import TIERS, to_notch, to_tier

CAP = 99.0
TIER_MID = {"AAA": 1, "AA": 3, "A": 6, "BBB": 9, "BB": 12, "B": 15, "CCC": 18}
FLOWS = ["revenue", "ebitda", "cash_interest", "cogs", "fcf", "staff_cost", "energy_cost"]


def _bool(s):
    """Missing counts as False. Written out so pandas does not guess the type."""
    return s.map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _ratings(tables, as_of):
    base = tables["issuers"].set_index("issuer_id")["rating"]
    hist = tables.get("ratings_history")
    if hist is not None and len(hist):
        h = hist[hist["date"] <= as_of].sort_values("date").groupby("issuer_id")["rating"].last()
        base = h.reindex(base.index).fillna(base) if len(h) else base
    return base


def _fundamentals(tables, as_of):
    """Trailing-four-quarter flows and latest stocks, from filings dated on or before as_of."""
    f = tables["fundamentals"]
    f = f[(f["filed_date"] <= as_of) & (f["period_end"] <= as_of)].sort_values(["issuer_id", "period_end"])
    f = f.drop_duplicates(["issuer_id", "period_end"], keep="last")
    f = f.assign(rank_back=f.groupby("issuer_id").cumcount(ascending=False))
    out = pd.DataFrame(index=pd.Index(f["issuer_id"].unique(), name="issuer_id"))
    last = f[f["rank_back"] == 0].set_index("issuer_id")
    out["last_period"] = last["period_end"]
    out["stale"] = (as_of - last["filed_date"]).dt.days > 200
    for name, lo in (("ttm", 0), ("prev", 4)):
        w = f[(f["rank_back"] >= lo) & (f["rank_back"] < lo + 4)]
        g = w.groupby("issuer_id")
        full = g.size() == 4
        span_ok = (g["period_end"].max() - g["period_end"].min()).dt.days <= 300
        for c in FLOWS:
            s = g[c].sum(min_count=4) if c in w.columns else pd.Series(np.nan, index=full.index)
            out[f"{c}_{name}"] = s.where(full & span_ok)
    for c in ["total_debt", "cash", "undrawn_revolver", "floating_debt", "top_customer_share", "ig_backed"]:
        out[c] = _col(last, c)
    return out, f


def _window(prices, as_of, days):
    p = prices[prices["date"] <= as_of]
    dates = np.sort(p["date"].unique())[-days:]
    return p[p["date"].isin(dates)]


def _vw(df, value, weight="volume"):
    w = df[weight].fillna(0) if weight in df.columns else pd.Series(1.0, index=df.index)
    w = w.where(w > 0, 1.0)
    return (df[value] * w).groupby(df["issuer_id"]).sum() / w.groupby(df["issuer_id"]).sum()


def _treasury_at(tables, as_of, tenor):
    tc = tables.get("treasury_curve")
    if tc is None or not len(tc):
        return pd.Series(np.nan, index=tenor.index)
    tc = tc[tc["date"] <= as_of]
    if not len(tc):
        return pd.Series(np.nan, index=tenor.index)
    curve = tc[tc["date"] == tc["date"].max()].sort_values("tenor_years")
    return pd.Series(np.interp(tenor.fillna(5.0), curve["tenor_years"], curve["yield_pct"]), index=tenor.index)


def _spread(tables, as_of, days):
    """Issuer spread in basis points from its own traded bonds. Missing when no bond traded lately."""
    bp = tables["bond_prices"]
    w = _window(bp, as_of, days)
    if not len(w):
        return pd.DataFrame(columns=["own_yield", "spread_bp", "min_price", "last_trade"])
    out = pd.DataFrame({"own_yield": _vw(w, "yield_pct")})
    tenor = _vw(w, "years_to_maturity") if "years_to_maturity" in w.columns else pd.Series(np.nan, index=out.index)
    out["spread_bp"] = (out["own_yield"] - _treasury_at(tables, as_of, tenor.reindex(out.index))) * 100
    out["min_price"] = w.groupby("issuer_id")["price"].min()
    out["last_trade"] = w.groupby("issuer_id")["date"].max()
    fresh = (as_of - out["last_trade"]).dt.days <= 30
    return out.where(fresh, np.nan)


def _tier_table(tables, as_of, col):
    ty = tables.get("tier_yields")
    if ty is None or not len(ty) or col not in ty.columns:
        return pd.Series(dtype=float)
    ty = ty[ty["date"] <= as_of]
    return ty.sort_values("date").groupby("tier")[col].last()


def _cyclicality(f, tables, min_q):
    ism = tables.get("macro_ism")
    if ism is None or not len(ism):
        return pd.Series(dtype=float)
    q_ism = (ism.assign(q=ism["date"].dt.to_period("Q")).groupby("q")["ism"].mean() - 50.0)
    g = f[["issuer_id", "period_end", "revenue"]].copy()
    g["growth"] = g.groupby("issuer_id")["revenue"].pct_change(4)
    g["x"] = g["period_end"].dt.to_period("Q").map(q_ism)
    g = g.dropna(subset=["growth", "x"])

    def beta(d):
        if len(d) < min_q or d["x"].var() == 0:
            return np.nan
        return np.cov(d["x"], d["growth"])[0, 1] / d["x"].var()
    return g.groupby("issuer_id")[["x", "growth"]].apply(beta)


def _equity(tables, as_of):
    eq = tables.get("equity_prices")
    if eq is None or not len(eq):
        return pd.DataFrame(columns=["eq_drawdown", "eq_below_200d"])
    e = eq[(eq["date"] <= as_of) & (eq["date"] > as_of - pd.Timedelta(days=400))].sort_values(["issuer_id", "date"])
    g = e.groupby("issuer_id")["close"]
    last, n = g.last(), g.size()
    ma200 = g.apply(lambda s: s.tail(200).mean())
    fresh = (as_of - e.groupby("issuer_id")["date"].max()).dt.days <= 30
    out = pd.DataFrame({"eq_drawdown": 1 - last / g.max(), "eq_below_200d": (ma200 - last) / ma200})
    return out.where((n >= 200) & fresh, np.nan)


def compute(tables, as_of, cfg):
    """Return one row per issuer: identifiers, diagnostics and the variable components."""
    as_of = pd.Timestamp(as_of)
    iss = tables["issuers"].drop_duplicates("issuer_id").set_index("issuer_id")
    out = pd.DataFrame(index=iss.index)
    out["name"] = _col(iss, "name")
    out["sector"] = iss["sector"]
    out["rating"] = _ratings(tables, as_of)
    out["notch"] = out["rating"].map(to_notch)
    out["tier"] = out["rating"].map(to_tier)
    out["is_financial"] = _bool(_col(iss, "is_financial"))

    fund, f_all = _fundamentals(tables, as_of)
    fund = fund.reindex(out.index)
    ok = ~fund["stale"].map(lambda v: True if pd.isna(v) else bool(v)).astype(bool)          # stale or absent filings give missing variables
    keep = [c for c in fund.columns if c not in ("last_period", "stale")]
    fund[keep] = fund[keep].astype(object).where(ok, np.nan).infer_objects()
    for c in keep:
        if c != "ig_backed":
            fund[c] = pd.to_numeric(fund[c], errors="coerce")
    out["stale_fundamentals"] = ~ok

    mkt = _spread(tables, as_of, cfg["price_window_days"]).reindex(out.index)
    out["no_traded_bond"] = mkt["own_yield"].isna()
    tier_yield = out["tier"].map(_tier_table(tables, as_of, "yield_pct"))
    market_yield = mkt["own_yield"].fillna(tier_yield)
    out["market_yield"] = market_yield
    out["yield_source"] = np.where(mkt["own_yield"].notna(), "own bonds", np.where(tier_yield.notna(), "rating tier", "none"))

    dm = tables["debt_maturities"]
    if "known_date" in dm.columns:
        dm = dm[dm["known_date"].isna() | (dm["known_date"] <= as_of)]
    dm = dm[dm["maturity_date"] > as_of]
    horizon = as_of + pd.DateOffset(months=cfg["refi_horizon_months"])
    has_terms = pd.Series(out.index.isin(dm["issuer_id"].unique()), index=out.index)
    due = dm[dm["maturity_date"] <= horizon].groupby("issuer_id")["amount"].sum().reindex(out.index)
    debt, cash = fund["total_debt"], fund["cash"]
    due = due.where(has_terms, np.nan).fillna(0).where(has_terms | (debt == 0), np.nan)
    if "coupon" in dm.columns:
        d = dm.dropna(subset=["coupon"])
        avg_coupon = ((d["coupon"] * d["amount"]).groupby(d["issuer_id"]).sum() / d["amount"].groupby(d["issuer_id"]).sum()).reindex(out.index)
    else:
        avg_coupon = pd.Series(np.nan, index=out.index)
    avg_coupon = avg_coupon.fillna(fund["cash_interest_ttm"] / debt.where(debt > 0) * 100)

    interest = fund["cash_interest_ttm"]
    if cfg.get("refi_scope", "full") == "full":
        repriced = debt * market_yield / 100
    else:
        repriced = interest + due * (market_yield - avg_coupon).clip(lower=0) / 100
    repriced = np.maximum(repriced, interest).where(repriced.notna() & interest.notna())
    out["coverage_now"] = (fund["ebitda_ttm"] / interest.where(interest > 0)).clip(-CAP, CAP)
    out["coverage_refi"] = (fund["ebitda_ttm"] / repriced.where(repriced > 0)).clip(-CAP, CAP)
    out.loc[(debt == 0) & fund["ebitda_ttm"].notna(), ["coverage_now", "coverage_refi"]] = CAP

    liquidity = cash + fund["undrawn_revolver"].fillna(0) + (2 * fund["fcf_ttm"].fillna(0)).clip(lower=0)
    out["refi_need"] = (due / liquidity.where(liquidity > 0)).where(liquidity > 0, CAP).clip(upper=CAP).where(due.notna() & cash.notna())
    out.loc[due == 0, "refi_need"] = 0.0

    float_terms = pd.Series(np.nan, index=out.index)
    if "is_floating" in dm.columns:
        fl = dm.assign(fa=dm["amount"] * _bool(dm["is_floating"]).astype(float))
        float_terms = (fl.groupby("issuer_id")["fa"].sum() / fl.groupby("issuer_id")["amount"].sum()).reindex(out.index)
    out["floating_share"] = (fund["floating_debt"] / debt.where(debt > 0)).fillna(float_terms).clip(0, 1)
    out["coupon_stepup"] = market_yield - avg_coupon

    net_debt = debt - cash
    lev = net_debt / fund["ebitda_ttm"].where(fund["ebitda_ttm"] > 0)
    lev = lev.where(fund["ebitda_ttm"] > 0, CAP).where(net_debt > 0, 0.0).where(net_debt.notna() & fund["ebitda_ttm"].notna())
    out["net_leverage"] = lev.clip(upper=CAP)

    gm = lambda s: (fund[f"revenue_{s}"] - fund[f"cogs_{s}"]) / fund[f"revenue_{s}"].where(fund[f"revenue_{s}"] > 0)
    out["margin_change"] = (gm("ttm") - gm("prev")) * 100
    costs = fund[["staff_cost_ttm", "energy_cost_ttm"]]
    out["cost_intensity"] = (costs.sum(axis=1, min_count=1) / fund["revenue_ttm"].where(fund["revenue_ttm"] > 0))

    out["cyclicality"] = _cyclicality(f_all, tables, cfg["cyclicality_min_quarters"]).reindex(out.index).where(ok)
    backed = _bool(fund["ig_backed"])
    out["concentration"] = fund["top_customer_share"] * np.where(backed, cfg["backed_discount"], 1.0)

    out["spread_bp"] = mkt["spread_bp"]
    out["min_price"] = mkt["min_price"]
    out["spread_per_turn"] = (mkt["spread_bp"] / lev.where(lev > 0).clip(lower=0.5)).where(lev < CAP)
    back = _spread(tables, as_of - pd.tseries.offsets.BDay(cfg["spread_change_days"]), cfg["price_window_days"]).reindex(out.index)
    out["spread_change_3m"] = mkt["spread_bp"] - back["spread_bp"]

    out = out.join(_equity(tables, as_of).reindex(out.index))
    tier_spread = _tier_table(tables, as_of, "spread_bp")
    if not len(tier_spread):
        tier_spread = out.groupby("tier")["spread_bp"].median()
    tier_spread = tier_spread.reindex(TIERS).dropna()
    tier_spread = tier_spread[tier_spread > 0]
    if len(tier_spread) >= 3:
        logs = np.log(tier_spread.values)
        s = out["spread_bp"].where(out["spread_bp"] > 0)
        idx = pd.Series([np.nan if np.isnan(v) else int(np.abs(logs - np.log(v)).argmin()) for v in s.values], index=out.index)
        out["implied_tier"] = idx.map(lambda i: tier_spread.index[int(i)] if pd.notna(i) else None)
        out["implied_gap"] = out["implied_tier"].map(TIER_MID) - out["notch"]
    else:
        out["implied_tier"], out["implied_gap"] = None, np.nan
    out["ponzi"] = out["coverage_refi"] < cfg["ponzi_coverage"]
    out.attrs["as_of"] = as_of
    return out
