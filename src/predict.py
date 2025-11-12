
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import joblib
import src.transformers
try:
    import src.tune_pipeline  
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description="Predict with saved StyleSense pipeline")
    ap.add_argument("--model", default="models/best_pipeline.joblib")
    ap.add_argument("--csv", help="CSV with feature columns (no target)")
    ap.add_argument("--jsonl", help="JSONL with one record per line")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    print(f"[predict] loading model: {args.model}", file=sys.stderr)
    pipe = joblib.load(args.model)
    print("[predict] model loaded", file=sys.stderr)

    if args.csv:
        print(f"[predict] reading CSV: {args.csv}", file=sys.stderr)
        X = pd.read_csv(args.csv)
    elif args.jsonl:
        print(f"[predict] reading JSONL: {args.jsonl}", file=sys.stderr)
        rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
        X = pd.DataFrame(rows)
    else:
        raise SystemExit("Provide --csv or --jsonl")

    print(f"[predict] input shape: {X.shape}", file=sys.stderr)

    # Predict
    preds = pipe.predict(X)
    proba = None
    if "clf" in pipe.named_steps and hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]

    out_df = X.copy()
    out_df["predicted_recommend"] = preds
    if proba is not None:
        out_df["recommend_proba"] = proba

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[predict] ✅ wrote {args.out} with {len(out_df)} rows", file=sys.stderr)

if __name__ == "__main__":
    main()
