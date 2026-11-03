"""First-pass cleaning for the yearly NIRF ranking CSVs."""
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "cleaned"


def parse_rank(value):
    """Return numeric rank and original rank band (if any)."""
    if pd.isna(value):
        return pd.NA, pd.NA
    s = str(value).strip().replace("–", "-").replace("—", "-")
    m = re.search(r"\d+", s)
    if not m:
        return pd.NA, s or pd.NA
    return int(m.group()), s if re.search(r"\d+\s*[-]\s*\d+", s) else pd.NA


def clean_file(path: Path) -> pd.DataFrame:
    year_match = re.search(r"20\d{2}", path.name)
    if not year_match:
        raise ValueError(f"Cannot determine year from {path.name}")
    year = int(year_match.group())
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
    aliases = {"Institute ID": "institution_id", "Name": "raw_name", "State": "raw_state",
               "City": "city", "Score": "score", "Rank": "raw_rank"}
    missing = set(aliases) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {sorted(missing)}")
    df = raw.rename(columns=aliases)[list(aliases.values())].copy()
    for col in ["institution_id", "raw_name", "raw_state", "city"]:
        df[col] = (df[col].fillna("").astype(str)
                   .str.replace(r"\s*More Details\s*\|(?:\s*\|)*", "", regex=True)
                   .str.replace(r"\s+", " ", regex=True).str.strip(" ,|"))
        df[col] = df[col].replace("", pd.NA)
    df["score"] = pd.to_numeric(df.score.str.replace(",", "", regex=False).str.strip(), errors="coerce")
    ranks = df.raw_rank.map(parse_rank)
    df["rank"] = pd.array([x[0] for x in ranks], dtype="Int64")
    df["rank_band"] = [x[1] for x in ranks]
    df["year"] = year
    df["category"] = "Overall"
    df["original_category"] = "Overall"
    # Entirely empty rows are junk; partially missing records remain visible for QA.
    df = df.dropna(subset=["institution_id", "raw_name", "rank"], how="all")
    df["missing_fields"] = df[["institution_id", "raw_name", "raw_state", "city", "score", "rank"]].isna().apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1)
    return df[["year", "category", "original_category", "institution_id", "raw_name", "city", "raw_state",
               "score", "rank", "rank_band", "missing_fields"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW}")
    for path in files:
        df = clean_file(path)
        year = re.search(r"20\d{2}", path.name).group()
        target = OUT / f"nirf_{year}_cleaned.csv"
        df.to_csv(target, index=False)
        print(f"{path.name}: {len(df)} records -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
