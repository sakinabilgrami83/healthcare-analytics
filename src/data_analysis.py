"""
Exploratory Data Analysis (EDA) Module.
Calculates summary statistics, categorical distributions, and multi-granularity time-series trends.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def analyze_numerical(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Compute comprehensive descriptive statistics for a numerical column:
    Mean, Median, Min, Max, Standard Deviation, Variance, Skewness, and Percentiles.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return {}
    
    series = df[column].dropna()
    if len(series) == 0:
        return {}
    
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "variance": float(series.var()) if len(series) > 1 else 0.0,
        "skew": float(series.skew()) if len(series) > 2 else 0.0,
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95))
    }


def analyze_all_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tabular statistical summary for all numerical columns.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    summary_rows = []
    
    for col in numeric_cols:
        stats = analyze_numerical(df, col)
        if stats:
            summary_rows.append({
                "Column": col,
                "Count": stats["count"],
                "Mean": round(stats["mean"], 2),
                "Median": round(stats["median"], 2),
                "Std Dev": round(stats["std"], 2),
                "Min": round(stats["min"], 2),
                "P25": round(stats["p25"], 2),
                "P75": round(stats["p75"], 2),
                "Max": round(stats["max"], 2),
                "Skewness": round(stats["skew"], 2)
            })
    return pd.DataFrame(summary_rows)


def analyze_categorical(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """
    Calculate frequency, percentage distribution, unique count, and cumulative share.
    """
    if column not in df.columns:
        return pd.DataFrame()
    
    counts = df[column].value_counts(dropna=False)
    total = len(df)
    
    results = []
    for cat, cnt in counts.head(top_n).items():
        pct = (cnt / total) * 100.0 if total > 0 else 0.0
        results.append({
            "Category": str(cat),
            "Count": int(cnt),
            "Share %": round(pct, 2)
        })
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["Cumulative Share %"] = res_df["Share %"].cumsum().round(2)
    return res_df


def analyze_time_series(
    df: pd.DataFrame, 
    date_col: str, 
    metric_col: str, 
    granularity: str = "M", 
    agg_func: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate metric over time based on chosen granularity:
    - 'D' for Daily
    - 'W' for Weekly
    - 'M' for Monthly
    - 'Q' for Quarterly
    - 'Y' for Yearly
    Returns DataFrame with Period, Metric Value, Previous Period Value, and Growth %.
    """
    if date_col not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()
    
    sub = df[[date_col, metric_col]].dropna().copy()
    if not pd.api.types.is_datetime64_any_dtype(sub[date_col]):
        sub[date_col] = pd.to_datetime(sub[date_col], errors='coerce')
        sub = sub.dropna(subset=[date_col])

    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(by=date_col)
    
    # Map friendly granularity names
    granularity_map = {
        "Daily": "D",
        "Weekly": "W-MON",
        "Monthly": "ME",
        "Quarterly": "QE",
        "Yearly": "YE",
        "D": "D",
        "W": "W-MON",
        "M": "ME",
        "Q": "QE",
        "Y": "YE"
    }
    freq = granularity_map.get(granularity, "ME")
    
    # Resample
    ts = sub.set_index(date_col).resample(freq)[metric_col].agg(agg_func).reset_index()
    
    # Period label formatting
    if freq == "D":
        ts["Period_Label"] = ts[date_col].dt.strftime("%Y-%m-%d")
    elif freq.startswith("W"):
        ts["Period_Label"] = ts[date_col].dt.strftime("W%W %Y")
    elif freq.startswith("M"):
        ts["Period_Label"] = ts[date_col].dt.strftime("%b %Y")
    elif freq.startswith("Q"):
        ts["Period_Label"] = ts[date_col].dt.to_period("Q").astype(str)
    elif freq.startswith("Y"):
        ts["Period_Label"] = ts[date_col].dt.strftime("%Y")
    else:
        ts["Period_Label"] = ts[date_col].astype(str)

    # Compute period-over-period growth
    ts["Previous_Period"] = ts[metric_col].shift(1)
    ts["Growth_Abs"] = ts[metric_col] - ts["Previous_Period"]
    ts["Growth_Pct"] = np.where(
        (ts["Previous_Period"].isna()) | (ts["Previous_Period"] == 0),
        0.0,
        ((ts[metric_col] - ts["Previous_Period"]) / ts["Previous_Period"].abs()) * 100.0
    ).round(2)
    
    # 3-period rolling average
    ts["Rolling_3P_Avg"] = ts[metric_col].rolling(window=3, min_periods=1).mean().round(2)

    return ts
