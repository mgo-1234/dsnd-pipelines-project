from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd

def infer_column_types(
    df: pd.DataFrame,
    target: Optional[str] = None,
    text_hint: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    if text_hint is None:
        text_hint = []

    cols = [c for c in df.columns if c != target]
    num_cols, cat_cols, text_cols = [], [], []

    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s) or s.dtype == "object":
            if c in text_hint:
                text_cols.append(c)
            else:
                avg_len = s.dropna().astype(str).str.len().mean() if s.dropna().size else 0
                nunique = s.nunique(dropna=True)
                if avg_len >= 30 and nunique > 50:
                    text_cols.append(c)
                else:
                    cat_cols.append(c)
        else:
            cat_cols.append(c)

    cat_cols = [c for c in cat_cols if c not in text_cols]
    num_cols = [c for c in num_cols if c not in text_cols]
    return num_cols, cat_cols, text_cols


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return miss.to_frame("missing_rate").reset_index(names="column")


def safe_read_csv(path: str) -> pd.DataFrame:
    try_encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err = None
    for enc in try_encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err
