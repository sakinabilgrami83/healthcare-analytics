"""
Data Quality Assessment and Cleaning Pipeline Module.
Implements non-destructive quality reporting and transformation.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform deep data quality analysis:
    - Missing values count and percentage per column
    - Duplicate row count and percentage
    - Data type distribution
    - Cardinality (unique values) per column
    - Outlier counts per numerical column (using IQR 1.5x)
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Missing values
    missing_series = df.isnull().sum()
    total_missing_cells = int(missing_series.sum())
    missing_by_col = {
        col: {"count": int(count), "pct": round((count / total_rows) * 100, 2) if total_rows > 0 else 0}
        for col, count in missing_series.items() if count > 0
    }
    
    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0
    
    # Column classifications
    num_cols = list(df.select_dtypes(include=['number']).columns)
    date_cols = list(df.select_dtypes(include=['datetime', 'datetimetz']).columns)
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'string']).columns)
    
    # Unique values per column
    cardinality = {col: int(df[col].nunique(dropna=True)) for col in df.columns}
    
    # Outlier detection (IQR method for numeric columns)
    outliers_by_col = {}
    total_outliers = 0
    for col in num_cols:
        series = df[col].dropna()
        if len(series) >= 10:
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                cnt = int(outlier_mask.sum())
                if cnt > 0:
                    outliers_by_col[col] = {
                        "count": cnt,
                        "pct": round((cnt / len(series)) * 100, 2),
                        "lower_bound": round(float(lower_bound), 2),
                        "upper_bound": round(float(upper_bound), 2)
                    }
                    total_outliers += cnt

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_missing_cells": total_missing_cells,
        "missing_by_col": missing_by_col,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "date_cols": date_cols,
        "cardinality": cardinality,
        "outliers_by_col": outliers_by_col,
        "total_outliers": total_outliers
    }


def generate_quality_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the standardized Data Quality Summary Table:
    | Check | Result |
    """
    quality = assess_data_quality(df)
    
    records = [
        {"Check": "Total Rows", "Result": f"{quality['total_rows']:,}"},
        {"Check": "Total Columns", "Result": f"{quality['total_cols']:,}"},
        {"Check": "Missing Values", "Result": f"{quality['total_missing_cells']:,} cells ({len(quality['missing_by_col'])} affected columns)"},
        {"Check": "Duplicate Rows", "Result": f"{quality['duplicate_rows']:,} ({quality['duplicate_pct']}%)"},
        {"Check": "Outliers Detected", "Result": f"{quality['total_outliers']:,} potential records"},
        {"Check": "Numeric Columns", "Result": f"{len(quality['numeric_cols'])} ({', '.join(quality['numeric_cols'][:3]) + ('...' if len(quality['numeric_cols']) > 3 else '')})"},
        {"Check": "Categorical Columns", "Result": f"{len(quality['categorical_cols'])} ({', '.join(quality['categorical_cols'][:3]) + ('...' if len(quality['categorical_cols']) > 3 else '')})"},
        {"Check": "Date Columns", "Result": f"{len(quality['date_cols'])} ({', '.join(quality['date_cols']) if quality['date_cols'] else 'Pending Type Conversion'})"}
    ]
    return pd.DataFrame(records)


def clean_data(
    raw_df: pd.DataFrame, 
    date_col: str = None, 
    remove_duplicates: bool = True,
    handle_missing: bool = True,
    standardize_names: bool = True,
    handle_outliers: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Dedicated non-destructive cleaning pipeline.
    Leaves original raw_df untouched and returns (clean_df, audit_log).
    """
    df = raw_df.copy(deep=True)
    audit_log = []
    
    # 1. Standardize column names
    if standardize_names:
        original_cols = list(df.columns)
        new_cols = [
            str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
            for col in df.columns
        ]
        col_rename_map = dict(zip(original_cols, new_cols))
        df.rename(columns=col_rename_map, inplace=True)
        if date_col and date_col in col_rename_map:
            date_col = col_rename_map[date_col]
        audit_log.append("Standardized column names to clean lowercase snake_case.")
    
    # 2. Remove duplicate records
    if remove_duplicates:
        dups_count = int(df.duplicated().sum())
        if dups_count > 0:
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            audit_log.append(f"Removed {dups_count:,} duplicate rows.")
        else:
            audit_log.append("Checked for duplicate rows: none found.")

    # 3. Convert & Format Dates
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            invalid_dates = int(df[date_col].isna().sum() - raw_df[date_col].isna().sum() if date_col in raw_df.columns else df[date_col].isna().sum())
            if invalid_dates > 0:
                audit_log.append(f"Parsed datetime column '{date_col}' ({invalid_dates} unparseable entries set to NaT).")
            else:
                audit_log.append(f"Successfully converted '{date_col}' to datetime.")
        except Exception as e:
            audit_log.append(f"Date conversion note for '{date_col}': {str(e)}")

    # 4. Standardize Categorical Strings
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        if c == date_col:
            continue
        # Strip trailing/leading spaces and clean excessive internal spaces
        try:
            df[c] = df[c].astype(str).str.strip()
            # Replace 'nan', 'none', 'null' with actual NaN
            df[c] = df[c].replace({'nan': np.nan, 'None': np.nan, 'null': np.nan, '': np.nan})
        except Exception:
            pass
    audit_log.append("Cleaned and trimmed categorical text fields.")

    # 5. Handle Missing Values
    if handle_missing:
        # Numeric columns: impute with median
        num_cols = df.select_dtypes(include=['number']).columns
        imputed_nums = 0
        for col in num_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
                imputed_nums += n_missing
        if imputed_nums > 0:
            audit_log.append(f"Imputed {imputed_nums:,} missing values across numeric columns using column medians.")

        # Categorical columns: fill missing with 'Unknown'
        cat_cols = df.select_dtypes(include=['object']).columns
        imputed_cats = 0
        for col in cat_cols:
            if col == date_col:
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                df[col] = df[col].fillna('Unknown')
                imputed_cats += n_missing
        if imputed_cats > 0:
            audit_log.append(f"Imputed {imputed_cats:,} missing categorical values with 'Unknown'.")

    # 6. Optional Outlier Handling (capping at 99.5th percentile or IQR bounds)
    if handle_outliers:
        num_cols = df.select_dtypes(include=['number']).columns
        capped_count = 0
        for col in num_cols:
            upper_cap = df[col].quantile(0.99)
            lower_cap = df[col].quantile(0.01)
            outliers = (df[col] > upper_cap) | (df[col] < lower_cap)
            cnt = int(outliers.sum())
            if cnt > 0:
                df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                capped_count += cnt
        if capped_count > 0:
            audit_log.append(f"Capped {capped_count:,} extreme outlier records at 1st/99th percentiles.")

    return df, audit_log
