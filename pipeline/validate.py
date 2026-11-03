"""Failing quality gate for the NIRF master dataset."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "nirf_master.parquet"
EXPECTED = ["year", "category", "original_category", "institution_id", "institution", "city", "state",
            "score", "rank", "rank_band", "missing_fields"]


def validate(path=MASTER):
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found: {path}; run 04_build_master.py")
    df = pd.read_parquet(path)
    errors = []
    if list(df.columns) != EXPECTED:
        errors.append(f"schema/order mismatch: expected {EXPECTED}, got {list(df.columns)}")
    keys = ["year", "category", "institution_id"]
    dupes = df[df.duplicated(keys, keep=False)] if set(keys).issubset(df.columns) else pd.DataFrame()
    if not dupes.empty:
        errors.append(f"{len(dupes)} rows participate in duplicate (year, category, institution_id) keys")
    if "rank" in df:
        out = df[df["rank"].notna() & ((df["rank"] < 1) | (df["rank"] > 10000))]
        if not out.empty:
            errors.append(f"{len(out)} ranks outside supported range 1..10000")
    if "year" in df:
        out = df[df.year.notna() & ~df.year.between(2000, 2100)]
        if not out.empty:
            errors.append(f"{len(out)} years outside supported range 2000..2100")
    missing_report = df[EXPECTED].isna().sum().to_dict() if set(EXPECTED).issubset(df.columns) else {}
    if missing_report:
        print("Missing-value counts:")
        for col, count in missing_report.items():
            print(f"  {col}: {count}")
    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))
    print(f"Validation passed: {len(df)} rows; no duplicate keys or out-of-range ranks.")


if __name__ == "__main__":
    try:
        validate()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
