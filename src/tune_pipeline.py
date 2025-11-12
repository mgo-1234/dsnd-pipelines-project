from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
import joblib
from src.transformers import to_1d_str


def build_pipeline(num_cols, cat_cols, text_col: str) -> Pipeline:
    # numeric
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=False)),   
    ])

    # categorical (note: sklearn >=1.4 uses sparse_output)
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    txt = Pipeline([
        ("sel", FunctionTransformer(to_1d_str, validate=False)),
        ("tfidf", TfidfVectorizer(
            max_features=20000,        
            ngram_range=(1, 2),        
            stop_words="english",
            strip_accents="unicode"
        ))
    ])

    pre = ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols),
        ("txt", txt, [text_col]),
    ])

    clf = LogisticRegression(max_iter=500, n_jobs=None)

    return Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], outpath: Path, title="Confusion matrix"):
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # labels
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Grid search + evaluation for StyleSense pipeline")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--text", default="Review Text")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    # LOAD
    df = pd.read_csv(args.csv)
    print("✅ Data:", df.shape)

    # Columns for this dataset
    num_cols = ["Age", "Positive Feedback Count"]
    cat_cols = ["Division Name", "Department Name", "Class Name"]
    text_col = args.text

    # Split
    X = df[num_cols + cat_cols + [text_col]]
    y = df[args.target]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"Split → Train: {Xtr.shape}, Test: {Xte.shape}")

    pipe = build_pipeline(num_cols, cat_cols, text_col)

    # PARAM GRID (tunes TF-IDF + LogisticRegression)
    param_grid = {
        "pre__txt__tfidf__max_features": [10000, 20000],
        "pre__txt__tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
        "clf__class_weight": [None, "balanced"],
        "clf__solver": ["liblinear", "lbfgs"],   # both support l2
        "clf__penalty": ["l2"]
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    print("🔎 Grid search starting…")
    grid.fit(Xtr, ytr)
    print("✅ Grid done. Best F1 (CV):", grid.best_score_)
    print("✅ Best params:", grid.best_params_)

    # EVALUATE on TEST
    best = grid.best_estimator_
    yp = best.predict(Xte)

    # Probabilities for ROC (if available)
    yproba = best.predict_proba(Xte)[:, 1] if hasattr(best.named_steps["clf"], "predict_proba") else None

    acc = accuracy_score(yte, yp)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)
    roc = float(roc_auc_score(yte, yproba)) if yproba is not None else None
    cm = confusion_matrix(yte, yp)

    report = classification_report(yte, yp, output_dict=True)

    results = {
        "best_params": grid.best_params_,
        "cv_best_f1": float(grid.best_score_),
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": roc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    # SAVE ARTIFACTS
    joblib.dump(best, outdir / "best_pipeline.joblib")
    Path(outdir / "best_params.json").write_text(json.dumps(grid.best_params_, indent=2))
    # CV results as CSV
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv(outdir / "cv_results.csv", index=False)
    # Test report JSON
    Path(outdir / "test_report.json").write_text(json.dumps(results, indent=2))

    # FIGURES
    plot_confusion_matrix(cm, classes=["0", "1"], outpath=outdir / "figures" / "confusion_matrix.png")

    if yproba is not None:
        RocCurveDisplay.from_predictions(yte, yproba)
        plt.title("ROC Curve (test)")
        plt.tight_layout()
        plt.savefig(outdir / "figures" / "roc_curve.png", dpi=150)
        plt.close()

    print("✅ Saved:")
    print(" - models/best_pipeline.joblib")
    print(" - models/best_params.json")
    print(" - models/cv_results.csv")
    print(" - models/test_report.json")
    print(" - models/figures/confusion_matrix.png")
    if yproba is not None:
        print(" - models/figures/roc_curve.png")


if __name__ == "__main__":
    main()
