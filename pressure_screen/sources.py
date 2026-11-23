"""Load the canonical tables from whatever the machine has, and say plainly what is missing.

sources.yaml maps each canonical table to a file or a query, with a column rename map:

    tables:
      issuers:      {kind: parquet, path: data/issuers.parquet, columns: {gvkey: issuer_id}}
      bond_prices:  {kind: marketdb_sql, sql: "SELECT ..."}
      fundamentals: {kind: duckdb_sql, database: /data/wh.duckdb, sql: "SELECT ..."}
      defaults:     {kind: missing}

The warehouse answers "no rows" and "no such data" the same way. So `doctor` never reports
an empty table as fine: it reports it as EMPTY, and it checks that issuer ids overlap.
"""
from pathlib import Path

import pandas as pd
import yaml

from .schema import TABLES


class SourceError(Exception):
    pass


def _read(spec, base):
    kind = spec.get("kind", "missing")
    if kind == "missing":
        return None
    if kind in ("parquet", "csv"):
        path = Path(spec["path"])
        path = path if path.is_absolute() else base / path
        if not path.exists():
            raise SourceError(f"file not found: {path}")
        return pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path)
    if kind == "duckdb_sql":
        import duckdb
        con = duckdb.connect(spec.get("database", ":memory:"), read_only=spec.get("database", ":memory:") != ":memory:")
        try:
            return con.sql(spec["sql"]).df()
        finally:
            con.close()
    if kind == "marketdb_sql":
        import marketdb
        con = marketdb.connect()
        try:
            return con.sql(spec["sql"]).df()
        finally:
            con.close()
    raise SourceError(f"unknown kind: {kind}")


def load(sources_path):
    """Return (tables, problems). `tables` maps name to DataFrame or None. Never raises on bad data."""
    sources_path = Path(sources_path)
    spec_all = (yaml.safe_load(sources_path.read_text()) or {}).get("tables", {})
    tables, problems = {}, []
    for name, t in TABLES.items():
        spec = spec_all.get(name, {"kind": "missing"})
        try:
            df = _read(spec, sources_path.parent)
        except Exception as e:  # report it, do not hide it
            problems.append((name, "ERROR", f"{type(e).__name__}: {e}"))
            tables[name] = None
            continue
        if df is None:
            tables[name] = None
            problems.append((name, "MISSING", "no source configured"))
            continue
        df = df.rename(columns=spec.get("columns", {}))
        lack = [c for c in t.required if c not in df.columns]
        if lack:
            problems.append((name, "ERROR", f"required columns absent: {', '.join(lack)}"))
            tables[name] = None
            continue
        for c in t.dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None) if getattr(df[c].dtype, "tz", None) else pd.to_datetime(df[c], errors="coerce")
        if "issuer_id" in df.columns:
            df["issuer_id"] = df["issuer_id"].astype(str)
        if len(df) == 0:
            problems.append((name, "EMPTY", "the source returned zero rows. Check that the data exists before you trust this."))
        tables[name] = df
    return tables, problems


def doctor(sources_path):
    """One row per canonical table: status, rows, date range, issuer overlap, absent optional columns."""
    tables, problems = load(sources_path)
    prob = {n: (s, m) for n, s, m in problems}
    universe = set(tables["issuers"]["issuer_id"]) if tables.get("issuers") is not None else set()
    rows = []
    for name, t in TABLES.items():
        df = tables.get(name)
        status, note = prob.get(name, ("OK", ""))
        row = {"table": name, "status": status, "rows": 0 if df is None else len(df), "issuers": "", "overlap_with_universe": "",
               "first": "", "last": "", "optional_absent": "", "note": note}
        if df is not None and len(df):
            if "issuer_id" in df.columns:
                ids = set(df["issuer_id"])
                row["issuers"] = len(ids)
                if universe and name != "issuers":
                    share = len(ids & universe) / len(universe)
                    row["overlap_with_universe"] = f"{share:.0%}"
                    if share == 0:
                        row["status"], row["note"] = "ERROR", "no issuer id matches the issuers table. Check the id mapping."
                    elif share < 0.5 and t.must_have_rows:
                        row["status"] = "WARN"
                        row["note"] = "fewer than half of the issuers have rows here"
            d = t.dates[0] if t.dates and t.dates[0] in df.columns else None
            if d:
                row["first"], row["last"] = str(df[d].min())[:10], str(df[d].max())[:10]
            row["optional_absent"] = ", ".join(c for c in t.optional if c not in df.columns or df[c].isna().all())
        if status == "MISSING" and t.must_have_rows:
            row["status"] = "ERROR"
            row["note"] = "this table is required"
        rows.append(row)
    return pd.DataFrame(rows), tables


def usable(report):
    return not (report["status"] == "ERROR").any()
