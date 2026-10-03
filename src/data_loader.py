"""
Data Ingestion Module for Data Analytics Dashboard.
Supports loading CSV, Excel, and future SQL connections.
Provides metadata extraction and schema auto-detection.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import io
import pandas as pd


def load_data(file_or_path: Union[str, io.BytesIO], file_type: Optional[str] = None, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load dataset from a file path or file-like buffer (CSV or Excel).
    Returns a raw, unmodified Pandas DataFrame.
    """
    # Detect file type if not explicitly provided
    if file_type is None:
        if isinstance(file_or_path, str):
            if file_or_path.lower().endswith(('.xlsx', '.xls')):
                file_type = 'excel'
            else:
                file_type = 'csv'
        else:
            # For uploaded file objects in Streamlit
            file_name = getattr(file_or_path, 'name', '')
            if file_name.lower().endswith(('.xlsx', '.xls')):
                file_type = 'excel'
            else:
                file_type = 'csv'

    if file_type == 'excel':
        # Load excel file
        if sheet_name:
            df = pd.read_excel(file_or_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_or_path)
    else:
        # Load CSV file with encoding fallback
        try:
            df = pd.read_csv(file_or_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_or_path, encoding='latin1')

    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect raw dataset and extract fundamental metadata:
    - Number of rows
    - Number of columns
    - Column names
    - Data types
    - Memory usage
    - Sample records
    """
    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "sample_head": df.head(5).copy()
    }


def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Intelligently infer semantic roles of columns in the dataset:
    - Date / Time dimension
    - Primary value / monetary metric (Revenue, Sales, Total, Amount, etc.)
    - Secondary numeric metrics (Profit, Quantity, Discount, Target, Score, etc.)
    - Primary categorical grouping dimensions (Category, Region, Segment, Department)
    - Entity identifiers (Customer, Order, Employee, Product ID)
    - Status / Conversion dimension
    """
    cols_lower = {c: str(c).lower().strip().replace(' ', '_') for c in df.columns}
    
    # 1. Detect Date Column
    date_candidates = []
    # Prioritize actual datetime dtypes
    for orig_col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[orig_col]):
            date_candidates.append(orig_col)

    # If no datetime dtype, check string/object columns with date-like names
    if not date_candidates:
        date_name_terms = ['date', 'timestamp', 'datetime', 'order_date', 'trans_date', 'year_month', 'period']
        for orig_col, low_col in cols_lower.items():
            if not pd.api.types.is_numeric_dtype(df[orig_col]):
                if any(term in low_col for term in date_name_terms):
                    try:
                        sample = df[orig_col].dropna().iloc[:30]
                        if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notna().mean() > 0.7:
                            date_candidates.append(orig_col)
                    except Exception:
                        pass
    
    # Also check other object columns if still not found
    if not date_candidates:
        for orig_col in df.select_dtypes(include=['object']).columns[:5]:
            try:
                sample = df[orig_col].dropna().iloc[:20]
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().mean() > 0.8:
                        date_candidates.append(orig_col)
                        break
            except Exception:
                pass

    date_col = date_candidates[0] if date_candidates else None

    # 2. Detect Numeric Columns
    numeric_cols = list(df.select_dtypes(include=['number']).columns)

    # 3. Detect Primary Metric (Revenue / Sales / Amount / Value / Billing)
    primary_metric_candidates = []
    value_keywords = ['billing_amount', 'billing', 'revenue', 'sales', 'sales_amount', 'amount', 'total_amount', 'value', 'price', 'spend', 'gmv', 'charge', 'cost']
    for num_col in numeric_cols:
        low = cols_lower[num_col]
        if any(term == low or term in low for term in value_keywords):
            primary_metric_candidates.append(num_col)

    if not primary_metric_candidates:
        # Fallback to any positive continuous numeric column with high variance
        primary_metric_candidates = numeric_cols

    primary_metric = primary_metric_candidates[0] if primary_metric_candidates else None

    # 4. Detect Target Column
    target_candidates = [
        c for c in numeric_cols 
        if any(term in cols_lower[c] for term in ['target', 'quota', 'goal', 'budget', 'benchmark'])
    ]
    target_col = target_candidates[0] if target_candidates else None

    # 5. Detect Categorical Columns
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'string']).columns)
    # Remove candidate date col from categoricals if present
    if date_col and date_col in cat_cols:
        cat_cols = [c for c in cat_cols if c != date_col]

    # Category hierarchy candidates
    category_candidates = [
        c for c in cat_cols
        if any(term in cols_lower[c] for term in ['department', 'category', 'sub_category', 'diagnosis', 'treatment', 'specialty', 'type', 'genre', 'segment', 'industry'])
    ]
    if not category_candidates and cat_cols:
        category_candidates = cat_cols

    primary_category = category_candidates[0] if category_candidates else (cat_cols[0] if cat_cols else None)

    # 6. Detect Region / Location / Facility
    region_candidates = [
        c for c in cat_cols
        if any(term in cols_lower[c] for term in ['hospital', 'facility', 'center', 'branch', 'region', 'country', 'state', 'city', 'zone', 'territory', 'market'])
    ]
    primary_region = region_candidates[0] if region_candidates else None

    # 7. Detect Entity Identifiers (Patient, Customer, Product, Order, User)
    entity_candidates = [
        c for c in df.columns
        if any(term in cols_lower[c] for term in ['patient', 'customer', 'client', 'entity', 'account', 'user_id', 'product', 'order_id', 'doctor', 'id'])
    ]
    primary_entity = entity_candidates[0] if entity_candidates else None

    # 8. Detect Status / Conversion / Outcome Column
    status_candidates = [
        c for c in cat_cols
        if any(term in cols_lower[c] for term in ['outcome', 'status', 'state', 'stage', 'result', 'converted', 'completed', 'condition'])
    ]
    status_col = status_candidates[0] if status_candidates else None

    return {
        "date_col": date_col,
        "primary_metric": primary_metric,
        "target_col": target_col,
        "primary_category": primary_category,
        "primary_region": primary_region,
        "primary_entity": primary_entity,
        "status_col": status_col,
        "all_numeric": numeric_cols,
        "all_categorical": cat_cols
    }


def load_from_sql(connection_string: str, query: str) -> pd.DataFrame:
    """
    Extensible SQL Loader interface for enterprise database connection.
    Supports PostgreSQL, MySQL, SQLite, Snowflake, and SQL Server via SQLAlchemy.
    """
    try:
        from sqlalchemy import create_engine
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except ImportError:
        raise ImportError("SQLAlchemy is required for SQL connectivity. Run: pip install sqlalchemy")
    except Exception as e:
        raise RuntimeError(f"Database connection error: {str(e)}")
