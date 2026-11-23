"""Command line.

    python -m pressure_screen doctor   --sources config/sources.yaml
    python -m pressure_screen run      --sources config/sources.yaml --as-of 2026-11-20 --out out/run
    python -m pressure_screen backtest --sources config/sources.yaml --formation 2021-12-31 --end 2023-12-31 --peak 2022-10-14 --out out/bt
    python -m pressure_screen synth    --out demo [--no-signal]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import backtest, report, sources
from .score import score
from .variables import compute

ROOT = Path(__file__).resolve().parent.parent


def _cfg(path):
    return yaml.safe_load(Path(path).read_text())


def main(argv=None):
    p = argparse.ArgumentParser(prog="pressure_screen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("doctor", "run", "backtest"):
        s = sub.add_parser(name)
        s.add_argument("--sources", required=True)
        s.add_argument("--config", default=str(ROOT / "config/screen.yaml"))
        if name != "doctor":
            s.add_argument("--out", required=True)
    sub.choices["run"].add_argument("--as-of", required=True)
    b = sub.choices["backtest"]
    b.add_argument("--formation", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--peak", required=True, help="date at which spread widening is measured")
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--no-signal", action="store_true")
    s.add_argument("--issuers", type=int, default=400)
    a = p.parse_args(argv)

    if a.cmd == "synth":
        from .synthetic import write
        print(f"wrote demo data and {write(a.out, signal=not a.no_signal, n=a.issuers)}")
        return 0

    rep, tables = sources.doctor(a.sources)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    if a.cmd == "doctor":
        print(rep.to_string(index=False))
        print("\nUSABLE" if sources.usable(rep) else "\nNOT USABLE: fix every ERROR row first.")
        return 0 if sources.usable(rep) else 1
    if not sources.usable(rep):
        print(rep[rep["status"] == "ERROR"][["table", "status", "note"]].to_string(index=False))
        print("\nNOT USABLE: fix every ERROR row first. Nothing was computed.")
        return 1
    cfg = _cfg(a.config)
    if a.cmd == "run":
        scored = score(compute(tables, a.as_of, cfg), cfg)
        path = report.write_run(scored, rep, a.out)
        print(scored["status"].value_counts().to_string())
        print(f"\nreport: {path}")
        return 0
    scored, out, tests, nums = backtest.run(tables, a.formation, a.end, a.peak, cfg)
    path = report.write_backtest(scored, out, tests, nums, a.out, a.formation, a.end, a.peak)
    print(tests[["test", "verdict", "detail"]].to_string(index=False))
    print(f"\nOVERALL: {nums['overall']}\nreport: {path}")
    return 0 if nums["overall"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
