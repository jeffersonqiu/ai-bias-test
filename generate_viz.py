#!/usr/bin/env python3
"""
Generate all charts from a completed results CSV.

Usage
-----
python generate_viz.py results/raw/results_20260501T120000.csv
python generate_viz.py results/raw/results_20260501T120000.csv --out results/charts/phase2/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Generate bias visualisations from results CSV")
    p.add_argument("csv", metavar="PATH", help="Path to results CSV")
    p.add_argument("--out", metavar="DIR", default="results/charts",
                   help="Output directory for PNG charts (default: results/charts)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    outdir = Path(args.out)

    df = pd.read_csv(csv_path)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Coerce dimension columns and compute composite score
    dim_cols = ["score_technical", "score_business", "score_experience",
                "score_communication", "score_education"]
    for col in dim_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    present_dims = [c for c in dim_cols if c in df.columns and df[c].notna().any()]
    if present_dims:
        df["score_composite"] = df[present_dims].mean(axis=1).round(3)

    n_total = len(df)
    score_col = "score_composite" if "score_composite" in df.columns else "score"
    n_valid = df[score_col].notna().sum()
    n_error = n_total - n_valid
    print(f"Loaded {n_total} rows  ({n_valid} valid {score_col} scores, {n_error} errors/refusals)")

    if n_valid == 0:
        print(f"ERROR: No valid {score_col} scores in CSV. Cannot generate charts.")
        sys.exit(1)

    models_found = sorted(df["model_id"].unique())
    conditions_found = sorted(df["condition_id"].unique())
    print(f"Models:     {', '.join(models_found)}")
    print(f"Conditions: {', '.join(conditions_found)}")

    from bias_framework.visualization import generate_all
    generate_all(df, outdir)


if __name__ == "__main__":
    main()
