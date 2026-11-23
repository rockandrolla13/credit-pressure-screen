"""Write the results as CSV and as a short markdown note."""
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import VARIABLES

HEADER = "Research screen. It ranks issuers by exposure. It gives no trade, size, hedge or stop.\n"
SHOW = ["rank", "name", "sector", "rating", "F", "M", "F_minus_M", "coverage_refi", "refi_need", "floating_share",
        "net_leverage", "n_variables", "peer_group"]


def _md(df, digits=2):
    if not len(df):
        return "_none_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    d = d.fillna("").astype(str)
    lines = ["| " + " | ".join(["issuer_id"] + list(d.columns)) + " |", "|" + "---|" * (len(d.columns) + 1)]
    lines += ["| " + " | ".join([str(i)] + list(r)) + " |" for i, r in zip(d.index, d.values)]
    return "\n".join(lines) + "\n"


def write_run(scored, doctor_report, out_dir, top=40):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = str(scored.attrs.get("as_of", ""))[:10]
    scored.to_csv(out_dir / "screen.csv")
    doctor_report.to_csv(out_dir / "doctor.csv", index=False)
    counts = scored["status"].value_counts()
    flagged = scored[scored["status"] == "flagged"]
    cols = [c for c in SHOW if c in scored.columns]
    n = len(scored)
    md = [f"# Pressure screen, as of {as_of}\n", HEADER,
          "## Counts\n",
          f"- Issuers in the universe: {n}",
          f"- Flagged (both gates passed, not yet priced): {counts.get('flagged', 0)}",
          f"- Already priced (both gates passed, bond under 80 or CCC-implied): {counts.get('already_priced', 0)}",
          f"- Screened out: {counts.get('screened_out', 0)}",
          f"- Unscored for lack of data: {counts.get('unscored', 0)}. An unscored issuer is unknown, not safe.",
          f"- No traded bond in the last month: {int(scored['no_traded_bond'].sum())}. The market cannot show their weakness.",
          f"- Interest cover below 1.0 after refinancing: {int((scored['ponzi'] == True).sum())}\n",
          "## Variable coverage\n",
          "| variable | group | issuers with a value |", "|---|---|---|"]
    for v, (g, _, desc) in VARIABLES.items():
        md.append(f"| {v} | {g} | {int(scored[f'pct_{v}'].notna().sum())} of {n} |")
    md += ["\n## Flagged, ranked by F\n",
           "F is the fundamental score, 0 to 100, where 100 is worst. M is the market score. "
           "A high F with a low M means pressure that prices do not yet show.\n",
           _md(flagged[cols].head(top)),
           "\n## Largest gap between F and M among flagged issuers\n",
           _md(flagged.sort_values("F_minus_M", ascending=False)[cols].head(15)),
           "\n## Already priced\n", _md(scored[scored["status"] == "already_priced"][cols].head(top)),
           "\n## Data health\n", _md(doctor_report.set_index("table"))]
    (out_dir / "screen.md").write_text("\n".join(md))
    return out_dir / "screen.md"


def write_backtest(scored, out, tests, nums, out_dir, formation, end, peak):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.join(out, how="left").to_csv(out_dir / "backtest_issuers.csv")
    tests.to_csv(out_dir / "backtest_tests.csv", index=False)
    f = lambda k, p="{:.2f}": "n/a" if pd.isna(nums.get(k, np.nan)) else p.format(nums[k])
    md = [f"# Backtest: formed {formation}, outcomes to {end}, spread peak {peak}\n", HEADER,
          f"## Overall: {nums['overall']}\n",
          "The screen fails if any test fails. INCOMPLETE means a test could not run.\n",
          "| # | test | verdict | detail |", "|---|---|---|---|"]
    md += [f"| {t.test} | {t.name} | {t.verdict} | {t.detail} |" for t in tests.itertuples()]
    md += ["\n## Numbers\n",
           f"- Issuers scored: {nums['scored']} of {nums['universe']}. Unscored share {f('unscored_share', '{:.0%}')}.",
           f"- Downgrade or default rate among scored issuers: {f('base_bad_rate', '{:.1%}')}",
           f"- Top fifth {f('bad_rate_top', '{:.1%}')}, bottom fifth {f('bad_rate_bottom', '{:.1%}')}",
           f"- Skill alone (0.5 is none): F {f('auc_F')}, starting spread {f('auc_spread')}, rating {f('auc_rating')}, leverage {f('auc_leverage')}",
           f"- Issuers with a price at formation and none at the peak: {f('lost_price_share', '{:.0%}')}. Check that these are not dropped failures.\n",
           "## Notes\n"] + [f"- {x}" for x in nums["notes"]] + [
           "- A window with no recession tests the rate and margin legs well and the demand leg weakly.",
           "- The universe must include issuers that later defaulted or were delisted."]
    (out_dir / "backtest.md").write_text("\n".join(md))
    return out_dir / "backtest.md"
