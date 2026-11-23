"""Canonical input tables. Every data source is mapped onto these names and columns.

`required` columns must exist. `optional` columns may be absent: the variable that needs
them is then missing for every issuer, and Gate 0 decides whether an issuer is still scored.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Table:
    name: str
    required: tuple
    optional: tuple = ()
    dates: tuple = ()
    must_have_rows: bool = True
    purpose: str = ""


TABLES = {t.name: t for t in [
    Table("issuers", ("issuer_id", "sector", "rating"), ("name", "is_financial"),
          purpose="Universe. One row per issuer. `rating` is the agency rating at the as-of date "
                  "unless ratings_history is supplied."),
    Table("fundamentals", ("issuer_id", "period_end", "filed_date", "revenue", "ebitda", "cash_interest",
                           "total_debt", "cash"),
          ("cogs", "undrawn_revolver", "fcf", "floating_debt", "staff_cost", "energy_cost",
           "top_customer_share", "ig_backed"),
          dates=("period_end", "filed_date"),
          purpose="Quarterly figures, one row per issuer and quarter. Flows are single-quarter values. "
                  "`filed_date` is the day the market could first read them."),
    Table("debt_maturities", ("issuer_id", "maturity_date", "amount"), ("coupon", "is_floating", "known_date"),
          dates=("maturity_date", "known_date"),
          purpose="One row per bond or loan. `known_date` is the first day the instrument was public."),
    Table("bond_prices", ("issuer_id", "date", "price", "yield_pct"), ("cusip", "volume", "years_to_maturity"),
          dates=("date",), purpose="Daily bond prices. Yield in percent."),
    Table("equity_prices", ("issuer_id", "date", "close"), dates=("date",), must_have_rows=False,
          purpose="Daily adjusted close."),
    Table("ratings_history", ("issuer_id", "date", "rating"), dates=("date",), must_have_rows=False,
          purpose="Agency rating actions. Needed for the backtest."),
    Table("defaults", ("issuer_id", "date"), ("event",), dates=("date",), must_have_rows=False,
          purpose="Defaults and distressed exchanges. Needed for the backtest."),
    Table("macro_ism", ("date", "ism"), dates=("date",), must_have_rows=False,
          purpose="Monthly manufacturing ISM."),
    Table("macro_input_prices", ("sector", "date", "index"), dates=("date",), must_have_rows=False,
          purpose="Producer price index by sector."),
    Table("tier_yields", ("tier", "date", "yield_pct"), ("spread_bp",), dates=("date",), must_have_rows=False,
          purpose="Market yield by rating tier. Used when an issuer has no traded bond."),
    Table("treasury_curve", ("date", "tenor_years", "yield_pct"), dates=("date",), must_have_rows=False,
          purpose="Treasury yields by tenor. Used to turn a bond yield into a spread."),
]}

# The eleven screen variables. direction: +1 means a higher value is worse.
VARIABLES = {
    "coverage_refi":      ("balance", -1, "EBITDA over interest, with debt repriced at today's market yield"),
    "refi_need":          ("balance", +1, "Debt due in 24 months over cash, undrawn lines and 24 months of free cash flow"),
    "floating_share":     ("balance", +1, "Floating-rate debt over total debt"),
    "coupon_stepup":      ("balance", +1, "Market yield minus average coupon, in points"),
    "net_leverage":       ("balance", +1, "Net debt over EBITDA"),
    "margin_change":      ("margin",  -1, "Change in gross margin over four quarters, in points"),
    "cost_intensity":     ("margin",  +1, "Staff and energy cost over revenue"),
    "cyclicality":        ("demand",  +1, "Sensitivity of revenue growth to the manufacturing ISM"),
    "concentration":      ("demand",  +1, "Revenue share of the top customer, discounted when an IG party backs it"),
    "spread_per_turn":    ("market",  +1, "Bond spread per turn of net leverage"),
    "market_stress":      ("market",  +1, "Mean of equity drawdown, distance under the 200-day average and implied rating gap"),
}
FUNDAMENTAL_GROUPS = ("balance", "margin", "demand")

NOTCHES = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
           "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]
NOTCH = {r: i + 1 for i, r in enumerate(NOTCHES)}
MOODYS = dict(zip(["Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3", "Ba1", "Ba2", "Ba3",
                   "B1", "B2", "B3", "Caa1", "Caa2", "Caa3", "Ca", "C"], NOTCHES))
TIERS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]


def to_notch(rating):
    """Rating string to a number. 1 is AAA, 22 is D. Unknown ratings give None, never a guess."""
    if rating is None:
        return None
    r = str(rating).strip()
    r = MOODYS.get(r, r).upper()
    return NOTCH.get(r)


def to_tier(rating):
    n = to_notch(rating)
    if n is None:
        return None
    if n >= 17:
        return "CCC"
    return NOTCHES[n - 1].rstrip("+-")
