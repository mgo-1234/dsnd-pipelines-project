def to_1d_str(x):
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    x = pd.Series(x).fillna("").astype(str)
    return x.values
