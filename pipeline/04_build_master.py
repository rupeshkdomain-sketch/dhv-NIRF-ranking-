"""Assemble standardized yearly files into the canonical Parquet dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "interim" / "standardized"
OUTPUT = ROOT / "data" / "processed" / "nirf_master.parquet"
SCHEMA = ["year", "category", "original_category", "institution_id", "institution", "city", "state",
          "score", "rank", "rank_band", "missing_fields"]


def main():
    files = sorted(INPUT.glob("*_standardized.csv"))
    if not files:
        raise FileNotFoundError(f"No standardized CSV files found in {INPUT}; run 03_standardize.py first")
    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    for col in SCHEMA:
        if col not in df:
            df[col] = pd.NA
    df = df[SCHEMA]
    df["year"] = pd.to_numeric(df.year, errors="coerce").astype("Int64")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df["score"] = pd.to_numeric(df.score, errors="coerce")
    df["category"] = df.category.fillna("Uncategorized").astype("string")
    df["original_category"] = df.original_category.fillna(df.category).astype("string")
    df["institution"] = df.institution.astype("string").str.strip()
    df["rank_band"] = df.rank_band.astype("string")
    df["missing_fields"] = df.missing_fields.astype("string").fillna("")
    # Repeated rows for a year/category/institution are resolved by keeping the record
    # with the most populated fields, then the smaller (better) rank.
    df["_filled"] = df[SCHEMA].notna().sum(axis=1)
    df = df.sort_values(["_filled", "rank"], ascending=[False, True], na_position="last")
    df = df.drop_duplicates(["year", "category", "institution_id"], keep="first")
    df = df.drop(columns="_filled").sort_values(["year", "category", "rank", "institution"], na_position="last")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
