"""Apply reviewed institution, state, and category mapping tables."""
from pathlib import Path
import re
import unicodedata

import pandas as pd
from rapidfuzz import process, fuzz

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim" / "cleaned"
OUT = ROOT / "data" / "interim" / "standardized"
REF = ROOT / "data" / "reference"


def key(value):
    s = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bmore details\b", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def mapping(path, required):
    if not path.exists():
        return pd.DataFrame(columns=required)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} must have columns {required}; missing {sorted(missing)}")
    return df


def apply_map(values, df, raw_col, canonical_col):
    exact = {key(r[raw_col]): r[canonical_col] for _, r in df.iterrows() if r[raw_col] and r[canonical_col]}
    canonical = sorted(set(exact.values()))
    result = []
    for value in values.fillna(""):
        k = key(value)
        match = exact.get(k)
        # Only accept a high-confidence suggestion; ambiguous/low-confidence names stay as supplied.
        if match is None and k and canonical:
            choices = {key(x): x for x in canonical}
            suggestion = process.extractOne(k, list(choices), scorer=fuzz.WRatio)
            if suggestion and suggestion[1] >= 96:
                match = choices[suggestion[0]]
        result.append(match or value)
    return pd.Series(result, index=values.index, dtype="string")


def main():
    REF.mkdir(parents=True, exist_ok=True)
    inst = mapping(REF / "institution_mapping.csv", ["raw_name", "canonical_name", "institution_id"])
    states = mapping(REF / "state_mapping.csv", ["raw_state", "standard_state"])
    cats = mapping(REF / "category_mapping.csv", ["year", "original_category", "category"])
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(INTERIM.glob("*_cleaned.csv"))
    if not files:
        raise FileNotFoundError(f"No cleaned files found in {INTERIM}; run 02_clean.py first")
    for path in files:
        df = pd.read_csv(path, dtype={"institution_id": "string", "rank": "Int64"})
        df["original_category"] = df["original_category"].fillna(df["category"])
        if not inst.empty:
            df["institution"] = apply_map(df.raw_name, inst, "raw_name", "canonical_name")
            id_map = {key(r.raw_name): r.institution_id for r in inst.itertuples() if r.raw_name and r.institution_id}
            df["institution_id"] = [id_map.get(key(name), current) for name, current in zip(df.raw_name, df.institution_id)]
        else:
            df["institution"] = df.raw_name
        df["state"] = apply_map(df.raw_state, states, "raw_state", "standard_state") if not states.empty else df.raw_state
        if not cats.empty:
            cats["year"] = cats.year.astype(str)
            cm = {(str(r.year), key(r.original_category)): r.category for r in cats.itertuples()}
            df["category"] = [cm.get((str(y), key(c)), c) for y, c in zip(df.year, df.original_category)]
        df.to_csv(OUT / path.name.replace("_cleaned", "_standardized"), index=False)
        print(f"{path.name}: standardized {len(df)} records")


if __name__ == "__main__":
    main()
