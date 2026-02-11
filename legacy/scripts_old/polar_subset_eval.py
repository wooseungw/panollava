#!/usr/bin/env python3
"""Filter panorama CSVs for polar-region queries and run eval.

Usage:
  python scripts/polar_subset_eval.py \
    --input data/quic360/test.csv \
    --output data/quic360/test_polar.csv \
    --lat-column lat --threshold 60 \
    --keyword-filter

Then run evaluation (example):
  python scripts/eval.py --csv-input data/quic360/test_polar.csv --config configs/anyres_e2p_bimamba.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DEFAULT_LAT_CANDIDATES = ["lat", "latitude", "pitch"]
POLAR_KEYWORDS = [
    "north pole",
    "south pole",
    "polar",
    "zenith",
    "nadir",
    "ceiling",
    "sky",
    "overhead",
    "above",
    "below",
    "top",
    "bottom",
]


def _find_lat_column(df: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    for cand in DEFAULT_LAT_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def filter_by_lat(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    series = pd.to_numeric(df[col], errors="coerce")
    mask = series.abs() >= threshold
    return df[mask]


def filter_by_keywords(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    patt = "|".join(POLAR_KEYWORDS)
    mask = False
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(patt)
    return df[mask]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Filtered CSV path")
    parser.add_argument("--lat-column", default=None, help="Latitude column name (optional)")
    parser.add_argument("--threshold", type=float, default=60.0, help="Latitude threshold in degrees")
    parser.add_argument("--keyword-filter", action="store_true", help="Enable keyword-based fallback when lat is missing")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.is_file():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    df = pd.read_csv(inp)

    lat_col = _find_lat_column(df, args.lat_column)
    filtered: List[pd.DataFrame] = []

    if lat_col:
        df_lat = filter_by_lat(df, lat_col, args.threshold)
        filtered.append(df_lat)
        print(f"✓ Filtered by latitude column '{lat_col}': {len(df_lat)} / {len(df)} rows")
    else:
        print("! No latitude column found; skipping lat-based filter", file=sys.stderr)

    if args.keyword_filter:
        df_kw = filter_by_keywords(df, ["query", "instruction", "annotation", "response"])
        filtered.append(df_kw)
        print(f"✓ Keyword-based polar filter: {len(df_kw)} / {len(df)} rows")

    if not filtered:
        raise RuntimeError("No filters applied; please provide a latitude column or enable --keyword-filter")

    result = pd.concat(filtered).drop_duplicates()
    result.to_csv(out, index=False)
    print(f"✓ Saved polar subset to {out} ({len(result)} rows)")


if __name__ == "__main__":
    main()
