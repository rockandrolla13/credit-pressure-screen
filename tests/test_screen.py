"""Each test names one way the screen could give a silent wrong answer."""
import numpy as np
import pandas as pd
import pytest

from pressure_screen import backtest, sources, synthetic
from pressure_screen.schema import to_notch, to_tier
from pressure_screen.score import peer_percentile, score
from pressure_screen.variables import compute

F, E, PK = synthetic.FORMATION, synthetic.END, synthetic.PEAK


def test_ratings_map_and_unknown_is_none():
    assert to_notch("AAA") == 1 and to_notch("Baa3") == to_notch("BBB-") == 10
    assert to_tier("B+") == "B" and to_tier("Caa2") == "CCC"
    assert to_notch("NR") is None and to_tier(None) is None


def test_no_look_ahead(signal_tables, cfg):
    """Changing any data dated after the as-of date must not change one number."""
    tables, _ = signal_tables
    base = compute(tables, F, cfg)
    t2 = dict(tables)
    f = tables["fundamentals"].copy()
    f.loc[f["filed_date"] > F, ["revenue", "ebitda", "cash_interest", "total_debt"]] *= 7.0
    bp = tables["bond_prices"].copy()
    bp.loc[bp["date"] > F, ["price", "yield_pct"]] = 1.0
    rh = tables["ratings_history"].copy()
    rh.loc[rh["date"] > F, "rating"] = "D"
    t2.update(fundamentals=f, bond_prices=bp, ratings_history=rh)
    num = base.select_dtypes("number").columns
    pd.testing.assert_frame_equal(base[num], compute(t2, F, cfg)[num])
    assert (base["rating"] == compute(t2, F, cfg)["rating"]).all()


def test_filing_lag_is_respected(signal_tables, cfg):
    """A quarter that ended before the as-of date but was filed after it must not be used."""
    tables, _ = signal_tables
    v = compute(tables, F, cfg)
    assert (v["coverage_now"].notna()).all()
    f = tables["fundamentals"]
    assert f[(f["period_end"] <= F) & (f["filed_date"] > F)].shape[0] > 0     # the trap exists in the data
    used = f[(f["filed_date"] <= F)].groupby("issuer_id")["period_end"].max()
    assert (used < F).all()


def test_missing_data_is_unscored_not_safe(signal_tables, cfg):
    tables, _ = signal_tables
    victim = tables["issuers"]["issuer_id"].iloc[0]
    t2 = dict(tables)
    t2["fundamentals"] = tables["fundamentals"][tables["fundamentals"]["issuer_id"] != victim]
    s = score(compute(t2, F, cfg), cfg)
    assert s.loc[victim, "status"] == "unscored"
    assert np.isnan(s.loc[victim, "coverage_refi"]) and not s.loc[victim, "gate0"]


def test_stale_filings_give_missing_values(signal_tables, cfg):
    tables, _ = signal_tables
    victim = tables["issuers"]["issuer_id"].iloc[1]
    f = tables["fundamentals"]
    t2 = dict(tables)
    t2["fundamentals"] = f[~((f["issuer_id"] == victim) & (f["filed_date"] > F - pd.Timedelta(days=400)))]
    v = compute(t2, F, cfg)
    assert v.loc[victim, "stale_fundamentals"] and np.isnan(v.loc[victim, "net_leverage"])


def test_no_bond_means_market_variables_missing(signal_tables, cfg):
    tables, _ = signal_tables
    victim = tables["issuers"]["issuer_id"].iloc[2]
    t2 = dict(tables)
    t2["bond_prices"] = tables["bond_prices"][tables["bond_prices"]["issuer_id"] != victim]
    v = compute(t2, F, cfg)
    assert v.loc[victim, "no_traded_bond"] and np.isnan(v.loc[victim, "spread_bp"])
    assert v.loc[victim, "yield_source"] == "rating tier" and not np.isnan(v.loc[victim, "coverage_refi"])


def test_percentile_direction_and_fallback():
    df = pd.DataFrame({"tier": ["B"] * 10 + ["BB"] * 2, "sector": ["x"] * 12, "v": list(range(10)) + [5.0, np.nan]})
    worse_high, lvl = peer_percentile(df, "v", +1, min_group=8)
    worse_low, _ = peer_percentile(df, "v", -1, min_group=8)
    assert worse_high.iloc[9] == 100 and worse_low.iloc[0] == 100
    assert lvl.iloc[0] == "tier+sector" and lvl.iloc[10] == "all"
    assert np.isnan(worse_high.iloc[11])


def test_known_values_on_a_hand_built_issuer(cfg):
    """One issuer whose numbers can be checked by hand."""
    q = pd.date_range("2019-03-31", "2021-09-30", freq="QE")
    tables = {
        "issuers": pd.DataFrame({"issuer_id": ["X"], "sector": ["Tech"], "rating": ["B"]}),
        "fundamentals": pd.DataFrame({"issuer_id": "X", "period_end": q, "filed_date": q + pd.Timedelta(days=40),
                                      "revenue": 100.0, "cogs": 60.0, "ebitda": 25.0, "cash_interest": 5.0,
                                      "total_debt": 400.0, "cash": 50.0, "undrawn_revolver": 50.0, "fcf": 5.0,
                                      "floating_debt": 300.0}),
        "debt_maturities": pd.DataFrame({"issuer_id": ["X", "X"], "amount": [200.0, 200.0], "coupon": [5.0, 5.0],
                                         "maturity_date": pd.to_datetime(["2023-06-30", "2028-06-30"])}),
        "bond_prices": pd.DataFrame({"issuer_id": "X", "date": pd.bdate_range("2021-06-01", "2021-12-31"),
                                     "price": 95.0, "yield_pct": 10.0, "years_to_maturity": 5.0}),
        "treasury_curve": pd.DataFrame({"date": pd.Timestamp("2021-12-30"), "tenor_years": [2.0, 5.0, 10.0], "yield_pct": [1.0, 2.0, 3.0]}),
        "equity_prices": None, "ratings_history": None, "defaults": None, "macro_ism": None,
        "macro_input_prices": None, "tier_yields": None,
    }
    v = compute(tables, "2021-12-31", cfg).loc["X"]
    assert v["coverage_now"] == pytest.approx(100 / 20)            # EBITDA 100 over interest 20
    assert v["coverage_refi"] == pytest.approx(100 / 40)           # 400 of debt at a 10% market yield
    assert v["net_leverage"] == pytest.approx(350 / 100)
    assert v["floating_share"] == pytest.approx(0.75)
    assert v["coupon_stepup"] == pytest.approx(5.0)
    assert v["refi_need"] == pytest.approx(200 / (50 + 50 + 40))   # 200 due, cash 50, revolver 50, two years of FCF 40
    assert v["spread_bp"] == pytest.approx(800.0)
    assert v["margin_change"] == pytest.approx(0.0)
    assert np.isnan(v["cyclicality"]) and np.isnan(v["concentration"])


def test_gate1_any_condition(signal_tables, cfg):
    tables, _ = signal_tables
    s = score(compute(tables, F, cfg), cfg)
    g = cfg["gate1"]
    expect = (s["refi_need"] > g["refi_need_above"]) | (s["floating_share"] > g["floating_share_above"]) | (s["coverage_refi"] < g["coverage_refi_below"])
    assert (s["gate1"] == expect).all()
    flagged = s[s["status"] == "flagged"]
    assert (flagged["gate1"] & flagged["gate2"] & flagged["gate0"] & ~flagged["already_priced"]).all()
    assert flagged["rank"].is_unique and flagged["rank"].min() == 1


def test_screen_finds_the_planted_pressure(signal_tables, cfg):
    tables, pressure = signal_tables
    s = score(compute(tables, F, cfg), cfg)
    truth = pd.Series(pressure, index=tables["issuers"]["issuer_id"])
    assert s["F"].corr(truth.reindex(s.index), method="spearman") > 0.5


def test_backtest_passes_when_the_signal_is_real(signal_tables, cfg):
    tables, _ = signal_tables
    _, _, tests, nums = backtest.run(tables, F, E, PK, cfg)
    assert nums["overall"] == "PASS", tests.to_string()
    assert nums["auc_F"] > 0.65


def test_backtest_does_not_pass_on_noise(noise_tables, cfg):
    """Outcomes are random here. A backtest that passes is broken."""
    tables, _ = noise_tables
    _, _, tests, nums = backtest.run(tables, F, E, PK, cfg)
    assert nums["overall"] != "PASS", tests.to_string()
    assert abs(nums["auc_F"] - 0.5) < 0.12


def test_backtest_reports_untestable_not_pass(signal_tables, cfg):
    tables, _ = signal_tables
    t2 = dict(tables, ratings_history=None, defaults=None)
    _, _, tests, nums = backtest.run(t2, F, E, PK, cfg)
    assert nums["overall"] == "INCOMPLETE"
    assert tests.set_index("test").loc[1, "verdict"] == "NOT TESTABLE"
    assert any("cannot be measured" in n for n in nums["notes"])


def test_auc_basics():
    assert backtest.auc(pd.Series([1, 2, 3, 4]), pd.Series([0, 0, 1, 1])) == 1.0
    assert backtest.auc(pd.Series([4, 3, 2, 1]), pd.Series([0, 0, 1, 1])) == 0.0
    assert np.isnan(backtest.auc(pd.Series([1, 2]), pd.Series([0, 0])))


def test_doctor_flags_empty_missing_and_unmatched(tmp_path):
    src = synthetic.write(tmp_path, n=40)
    rep, _ = sources.doctor(src)
    assert sources.usable(rep)
    data = tmp_path / "data"
    pd.read_parquet(data / "bond_prices.parquet").head(0).to_parquet(data / "bond_prices.parquet")
    f = pd.read_parquet(data / "fundamentals.parquet")
    f["issuer_id"] = "Z" + f["issuer_id"]
    f.to_parquet(data / "fundamentals.parquet")
    (data / "debt_maturities.parquet").unlink()
    rep, _ = sources.doctor(src)
    r = rep.set_index("table")
    assert r.loc["bond_prices", "status"] == "EMPTY"
    assert r.loc["fundamentals", "status"] == "ERROR" and "id" in r.loc["fundamentals", "note"]
    assert r.loc["debt_maturities", "status"] == "ERROR"
    assert not sources.usable(rep)


def test_doctor_reports_absent_required_column(tmp_path):
    src = synthetic.write(tmp_path, n=40)
    p = tmp_path / "data" / "issuers.parquet"
    pd.read_parquet(p).drop(columns=["rating"]).to_parquet(p)
    rep, tables = sources.doctor(src)
    assert tables["issuers"] is None and "rating" in rep.set_index("table").loc["issuers", "note"]
