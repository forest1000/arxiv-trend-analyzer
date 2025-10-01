import pandas as pd

def weekly_counts(df: pd.DataFrame, date_col: str = "published") -> pd.Series:
    if df.empty:
        return pd.Series(dtype="int64")
    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    out = s.dt.to_period("W").dt.to_timestamp().value_counts().sort_index()
    out.name = "count"
    return out