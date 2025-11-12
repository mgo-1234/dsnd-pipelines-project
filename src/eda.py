from __future__ import annotations
import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from src.utils import infer_column_types, summarize_missingness, safe_read_csv


def main():
    p = argparse.ArgumentParser(description="Quick EDA for StyleSense dataset")
    p.add_argument("--csv", required=True, help="Path to CSV file")
    p.add_argument("--target", required=True, help="Binary target column (e.g., Recommended)")
    p.add_argument("--text", nargs="*", default=None, help="Text column hint(s), e.g. --text 'Review Text'")
    p.add_argument("--outdir", default="reports", help="Where to write reports/figures")
    args = p.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = safe_read_csv(args.csv)

    # Basic shape & peek
    head_path = outdir / "head_sample.csv"
    df.head(20).to_csv(head_path, index=False)

    # Target sanity
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    # Infer column types
    num_cols, cat_cols, text_cols = infer_column_types(df, target=args.target, text_hint=args.text)

    # Missingness
    miss_df = summarize_missingness(df)
    miss_path = outdir / "missingness.csv"
    miss_df.to_csv(miss_path, index=False)

    # Target distribution
    tgt_counts = df[args.target].value_counts(dropna=False)
    tgt_norm = df[args.target].value_counts(normalize=True, dropna=False)

    # Save summary
    summary = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "target": args.target,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "text_cols": text_cols,
        "target_counts": {str(k): int(v) for k, v in tgt_counts.to_dict().items()},
        "target_proportions": {str(k): float(v) for k, v in tgt_norm.round(4).to_dict().items()},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- Figures ----------
    # 1) Target distribution bar
    plt.figure()
    tgt_counts.plot(kind="bar")
    plt.title(f"Target distribution: {args.target}")
    plt.xlabel(args.target)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figdir / "target_distribution.png", dpi=150)
    plt.close()

    # 2) Missingness top 20
    plt.figure()
    top_miss = miss_df.head(20)
    plt.barh(top_miss["column"], top_miss["missing_rate"])
    plt.gca().invert_yaxis()
    plt.title("Top missingness (rate)")
    plt.xlabel("missing rate")
    plt.tight_layout()
    plt.savefig(figdir / "missingness_top20.png", dpi=150)
    plt.close()

    # 3) Numeric distributions (first 6 to keep it light)
    for c in num_cols[:6]:
        plt.figure()
        df[c].dropna().plot(kind="hist", bins=30)
        plt.title(f"Numeric distribution: {c}")
        plt.xlabel(c)
        plt.tight_layout()
        plt.savefig(figdir / f"num_{c}.png", dpi=150)
        plt.close()

    # 4) Categorical top frequencies (first 6)
    for c in cat_cols[:6]:
        vc = df[c].astype(str).value_counts().head(15)
        plt.figure()
        vc.plot(kind="bar")
        plt.title(f"Top categories: {c}")
        plt.xlabel(c)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figdir / f"cat_{c}.png", dpi=150)
        plt.close()

    # 5) Text length histograms (first 2 text cols)
    for c in text_cols[:2]:
        lens = df[c].dropna().astype(str).str.len()
        plt.figure()
        lens.plot(kind="hist", bins=40)
        plt.title(f"Text length: {c}")
        plt.xlabel("characters")
        plt.tight_layout()
        plt.savefig(figdir / f"textlen_{c}.png", dpi=150)
        plt.close()

    print("✅ EDA complete.")
    print(f"- Summary JSON: {outdir / 'summary.json'}")
    print(f"- Head sample:  {head_path}")
    print(f"- Missingness:  {miss_path}")
    print(f"- Figures in:   {figdir}")
    print("\nColumn inference:")
    print(json.dumps({ "num_cols": num_cols, "cat_cols": cat_cols, "text_cols": text_cols }, indent=2))


if __name__ == "__main__":
    main()
