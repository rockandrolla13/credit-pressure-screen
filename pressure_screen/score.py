"""Gates first, rank second. Equal weights on purpose: nothing is fitted."""
import numpy as np
import pandas as pd

from .schema import FUNDAMENTAL_GROUPS, VARIABLES

# A variable is the mean percentile of its components. Most have one component.
COMPONENTS = {v: [(v, d)] for v, (g, d, _) in VARIABLES.items()}
COMPONENTS["spread_per_turn"] = [("spread_per_turn", +1), ("spread_change_3m", +1)]
COMPONENTS["market_stress"] = [("eq_drawdown", +1), ("eq_below_200d", +1), ("implied_gap", +1)]


def _pct(values, direction):
    """Percentile from 0 to 100 where 100 is worst. Missing stays missing."""
    return (values * direction).rank(pct=True, method="average") * 100


def peer_percentile(df, col, direction, min_group):
    """Percentile inside tier and sector. Falls back to tier, then to everyone, when the group is small."""
    out = pd.Series(np.nan, index=df.index)
    level = pd.Series("", index=df.index)
    valid = df[col].notna()
    for keys, label in ((["tier", "sector"], "tier+sector"), (["tier"], "tier"), (None, "all")):
        todo = valid & out.isna()
        if not todo.any():
            break
        if keys is None:
            out[todo] = _pct(df.loc[valid, col], direction)[todo]
            level[todo] = label
            continue
        grp = df.loc[valid].groupby(keys, dropna=False)[col]
        size = grp.transform("size")
        p = grp.transform(lambda s: _pct(s, direction))
        take = todo & (size.reindex(df.index).fillna(0) >= min_group)
        out[take] = p.reindex(df.index)[take]
        level[take] = label
    return out, level


def score(variables, cfg):
    """Add percentiles, group scores, F, M, gates and a status to the variable table."""
    df = variables.copy()
    if cfg.get("exclude_financials", True):
        df = df[~df["is_financial"]]
    min_group = cfg["min_peer_group"]
    var_pct = pd.DataFrame(index=df.index)
    levels = []
    for var, comps in COMPONENTS.items():
        parts = []
        for col, direction in comps:
            if col not in df.columns:
                continue
            p, lvl = peer_percentile(df, col, direction, min_group)
            parts.append(p)
            levels.append(lvl)
        var_pct[var] = pd.concat(parts, axis=1).mean(axis=1) if parts else np.nan
    df["n_variables"] = var_pct.notna().sum(axis=1)
    lv = pd.concat(levels, axis=1)
    df["peer_group"] = np.where((lv == "all").any(axis=1), "all", np.where((lv == "tier").any(axis=1), "tier", "tier+sector"))

    for g in list(FUNDAMENTAL_GROUPS) + ["market"]:
        members = [v for v, (grp, _, _) in VARIABLES.items() if grp == g]
        df[f"g_{g}"] = var_pct[members].mean(axis=1)
    df["F"] = df[[f"g_{g}" for g in FUNDAMENTAL_GROUPS]].mean(axis=1)
    df["F_groups"] = df[[f"g_{g}" for g in FUNDAMENTAL_GROUPS]].notna().sum(axis=1)
    df["M"] = df["g_market"]
    df["F_minus_M"] = df["F"] - df["M"]
    for v in var_pct:
        df[f"pct_{v}"] = var_pct[v]

    g1, g2 = cfg["gate1"], cfg["gate2"]
    df["gate0"] = df["n_variables"] >= cfg["gate0_min_variables"]
    df["gate1"] = ((df["refi_need"] > g1["refi_need_above"]) | (df["floating_share"] > g1["floating_share_above"])
                   | (df["coverage_refi"] < g1["coverage_refi_below"]))
    cut = 1 - g2["worst_fraction"]
    in_sector = lambda c: df.groupby("sector")[c].rank(pct=True)
    df["gate2"] = (in_sector("g_margin") > cut) | (in_sector("g_demand") > cut)
    ap = cfg["already_priced"]
    df["already_priced"] = (df["min_price"] < ap["price_below"]) | (df["implied_tier"] == ap["implied_tier"])

    df["status"] = "screened_out"
    df.loc[df["gate1"] & df["gate2"], "status"] = "flagged"
    df.loc[(df["status"] == "flagged") & df["already_priced"], "status"] = "already_priced"
    df.loc[~df["gate0"], "status"] = "unscored"
    df["rank"] = df.loc[df["status"] == "flagged", "F"].rank(ascending=False, method="first")
    df.attrs.update(variables.attrs)
    return df.sort_values(["rank", "F"], ascending=[True, False], na_position="last")
