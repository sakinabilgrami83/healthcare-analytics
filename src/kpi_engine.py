"""
KPI Engine Module for Data Analytics Dashboard.
Calculates dynamic KPIs, period-over-period comparative metrics, and the standardized KPI Calculation Table.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def calculate_kpis(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    metric_col: Optional[str] = None,
    target_col: Optional[str] = None,
    entity_col: Optional[str] = None,
    status_col: Optional[str] = None,
    secondary_metric_col: Optional[str] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Compute 8 dynamic KPIs with prior-period comparison:
    1. Total Records
    2. Total Revenue / Value
    3. Average Value (AOV)
    4. Growth % (Period-over-Period)
    5. Conversion / Completion Rate
    6. Average Margin / Performance %
    7. Target Achievement %
    8. Entity Count (Unique Entities)
    """
    if df.empty:
        return {}
    
    # Work with filtered slice
    curr_df = df.copy()
    prev_df = pd.DataFrame()
    
    has_date = bool(date_col and date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]))
    
    if has_date:
        if start_date is not None and end_date is not None:
            mask_curr = (df[date_col] >= start_date) & (df[date_col] <= end_date)
            curr_df = df[mask_curr].copy()
            
            # Calculate previous period of equal length
            period_len = end_date - start_date
            prev_end = start_date - pd.Timedelta(days=1)
            prev_start = prev_end - period_len
            mask_prev = (df[date_col] >= prev_start) & (df[date_col] <= prev_end)
            prev_df = df[mask_prev].copy()
        else:
            # If no start/end specified, split chronological halves
            sorted_df = df.sort_values(by=date_col)
            mid = len(sorted_df) // 2
            prev_df = sorted_df.iloc[:mid].copy()
            curr_df = sorted_df.iloc[mid:].copy()
    else:
        # Fallback split
        mid = len(df) // 2
        prev_df = df.iloc[:mid].copy()
        curr_df = df.iloc[mid:].copy()

    def safe_pct_diff(curr_val: float, prev_val: float) -> Tuple[float, str]:
        if prev_val is None or prev_val == 0 or np.isnan(prev_val):
            return 0.0, "neutral"
        pct = ((curr_val - prev_val) / abs(prev_val)) * 100.0
        status = "positive" if pct >= 0 else "negative"
        return round(pct, 2), status

    # KPI 1: Total Records
    curr_records = len(curr_df)
    prev_records = len(prev_df)
    rec_pct, rec_status = safe_pct_diff(curr_records, prev_records)

    # KPI 2: Total Revenue / Value
    has_metric = bool(metric_col and metric_col in curr_df.columns and pd.api.types.is_numeric_dtype(curr_df[metric_col]))
    curr_total_val = float(curr_df[metric_col].sum()) if has_metric else 0.0
    prev_total_val = float(prev_df[metric_col].sum()) if (has_metric and not prev_df.empty) else 0.0
    val_pct, val_status = safe_pct_diff(curr_total_val, prev_total_val)

    # KPI 3: Average Value
    curr_avg_val = float(curr_df[metric_col].mean()) if has_metric and curr_records > 0 else 0.0
    prev_avg_val = float(prev_df[metric_col].mean()) if has_metric and prev_records > 0 else 0.0
    avg_pct, avg_status = safe_pct_diff(curr_avg_val, prev_avg_val)

    # KPI 4: Period-over-Period Growth %
    growth_pct = val_pct

    # KPI 5: Conversion / Completion Rate
    has_status = bool(status_col and status_col in curr_df.columns)
    curr_completion_rate = 0.0
    prev_completion_rate = 0.0
    if has_status:
        # Check for positive status flags
        positive_keywords = ['complete', 'completed', 'delivered', 'success', 'converted', 'paid', 'closed', 'won', 'recovered', 'improved', 'cured']
        status_series = curr_df[status_col].astype(str).str.lower()
        curr_pos = status_series.apply(lambda s: any(k in s for k in positive_keywords)).sum()
        curr_completion_rate = round((curr_pos / curr_records) * 100.0, 2) if curr_records > 0 else 0.0

        if not prev_df.empty:
            prev_pos = prev_df[status_col].astype(str).str.lower().apply(lambda s: any(k in s for k in positive_keywords)).sum()
            prev_completion_rate = round((prev_pos / prev_records) * 100.0, 2) if prev_records > 0 else 0.0
    comp_pct, comp_status = safe_pct_diff(curr_completion_rate, prev_completion_rate)

    # KPI 6: Average Performance / Margin %
    # Try profit or secondary metric / primary metric, or mean of secondary
    has_secondary = bool(secondary_metric_col and secondary_metric_col in curr_df.columns and pd.api.types.is_numeric_dtype(curr_df[secondary_metric_col]))
    curr_margin_pct = 0.0
    prev_margin_pct = 0.0
    if has_secondary and has_metric and curr_total_val != 0:
        curr_margin_pct = round((curr_df[secondary_metric_col].sum() / curr_total_val) * 100.0, 2)
        if not prev_df.empty and prev_total_val != 0:
            prev_margin_pct = round((prev_df[secondary_metric_col].sum() / prev_total_val) * 100.0, 2)
    elif has_secondary and curr_records > 0:
        curr_margin_pct = round(float(curr_df[secondary_metric_col].mean()), 2)
        prev_margin_pct = round(float(prev_df[secondary_metric_col].mean()), 2) if not prev_df.empty else 0.0
    margin_pct_change, margin_status = safe_pct_diff(curr_margin_pct, prev_margin_pct)

    # KPI 7: Target Achievement %
    has_target = bool(target_col and target_col in curr_df.columns and pd.api.types.is_numeric_dtype(curr_df[target_col]))
    curr_target_achieved = 0.0
    prev_target_achieved = 0.0
    if has_target and has_metric:
        curr_target_sum = curr_df[target_col].sum()
        curr_target_achieved = round((curr_total_val / curr_target_sum) * 100.0, 2) if curr_target_sum > 0 else 100.0
        if not prev_df.empty:
            prev_target_sum = prev_df[target_col].sum()
            prev_target_achieved = round((prev_total_val / prev_target_sum) * 100.0, 2) if prev_target_sum > 0 else 100.0
    else:
        # Default benchmark baseline (100% target)
        curr_target_achieved = 100.0
        prev_target_achieved = 100.0
    target_pct_diff, target_status = safe_pct_diff(curr_target_achieved, prev_target_achieved)

    # KPI 8: Customer / Entity Count
    has_entity = bool(entity_col and entity_col in curr_df.columns)
    curr_entities = int(curr_df[entity_col].nunique()) if has_entity else curr_records
    prev_entities = int(prev_df[entity_col].nunique()) if (has_entity and not prev_df.empty) else prev_records
    entity_pct, entity_status = safe_pct_diff(curr_entities, prev_entities)

    return {
        "curr_records": curr_records,
        "prev_records": prev_records,
        "rec_pct": rec_pct,
        "rec_status": rec_status,
        
        "curr_total_val": curr_total_val,
        "prev_total_val": prev_total_val,
        "val_pct": val_pct,
        "val_status": val_status,
        
        "curr_avg_val": curr_avg_val,
        "prev_avg_val": prev_avg_val,
        "avg_pct": avg_pct,
        "avg_status": avg_status,
        
        "growth_pct": growth_pct,
        
        "has_status": has_status,
        "curr_completion_rate": curr_completion_rate,
        "prev_completion_rate": prev_completion_rate,
        "comp_pct": comp_pct,
        "comp_status": comp_status,
        
        "has_secondary": has_secondary,
        "curr_margin_pct": curr_margin_pct,
        "prev_margin_pct": prev_margin_pct,
        "margin_pct_change": margin_pct_change,
        "margin_status": margin_status,
        
        "has_target": has_target,
        "curr_target_achieved": curr_target_achieved,
        "prev_target_achieved": prev_target_achieved,
        "target_pct_diff": target_pct_diff,
        "target_status": target_status,
        
        "has_entity": has_entity,
        "curr_entities": curr_entities,
        "prev_entities": prev_entities,
        "entity_pct": entity_pct,
        "entity_status": entity_status,
        
        "metric_name": metric_col or "Value",
        "entity_name": entity_col or "Records",
        "secondary_metric_name": secondary_metric_col or "Performance"
    }


def generate_kpi_calculation_table(kpis: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate the standardized KPI Calculation Table:
    | KPI | Formula | Actual Value | Previous Period | Change % | Status | Business Meaning |
    """
    if not kpis:
        return pd.DataFrame()

    def status_label(pct: float, benchmark: float = 0.0) -> str:
        if pct > benchmark:
            return "🟢 On Track"
        elif pct == benchmark:
            return "🟡 Neutral"
        else:
            return "🔴 Watch"

    table_data = [
        {
            "KPI": f"Total {kpis.get('metric_name', 'Revenue')}",
            "Formula": f"SUM({kpis.get('metric_name', 'Value')})",
            "Actual Value": f"${kpis.get('curr_total_val', 0):,.2f}",
            "Previous Period": f"${kpis.get('prev_total_val', 0):,.2f}",
            "Change %": f"{kpis.get('val_pct', 0):+.1f}%",
            "Status": status_label(kpis.get('val_pct', 0)),
            "Business Meaning": "Aggregate monetary volume generated in the selected timeframe."
        },
        {
            "KPI": f"Average {kpis.get('metric_name', 'Value')}",
            "Formula": f"AVG({kpis.get('metric_name', 'Value')})",
            "Actual Value": f"${kpis.get('curr_avg_val', 0):,.2f}",
            "Previous Period": f"${kpis.get('prev_avg_val', 0):,.2f}",
            "Change %": f"{kpis.get('avg_pct', 0):+.1f}%",
            "Status": status_label(kpis.get('avg_pct', 0)),
            "Business Meaning": "Average financial realization per transaction or order (AOV)."
        },
        {
            "KPI": "Period Growth %",
            "Formula": "(Current - Previous) / Previous * 100",
            "Actual Value": f"{kpis.get('growth_pct', 0):+.1f}%",
            "Previous Period": "Baseline",
            "Change %": f"{kpis.get('growth_pct', 0):+.1f}%",
            "Status": status_label(kpis.get('growth_pct', 0)),
            "Business Meaning": "Velocity of commercial expansion compared to prior period."
        },
        {
            "KPI": "Total Records",
            "Formula": "COUNT(Rows)",
            "Actual Value": f"{kpis.get('curr_records', 0):,}",
            "Previous Period": f"{kpis.get('prev_records', 0):,}",
            "Change %": f"{kpis.get('rec_pct', 0):+.1f}%",
            "Status": status_label(kpis.get('rec_pct', 0)),
            "Business Meaning": "Operational throughput and transaction velocity."
        },
        {
            "KPI": f"Distinct {kpis.get('entity_name', 'Entities')}",
            "Formula": f"DISTINCT_COUNT({kpis.get('entity_name', 'Entity_ID')})",
            "Actual Value": f"{kpis.get('curr_entities', 0):,}",
            "Previous Period": f"{kpis.get('prev_entities', 0):,}",
            "Change %": f"{kpis.get('entity_pct', 0):+.1f}%",
            "Status": status_label(kpis.get('entity_pct', 0)),
            "Business Meaning": "Active customer base or unique trading participants."
        }
    ]

    if kpis.get("has_status"):
        table_data.append({
            "KPI": "Completion / Conversion Rate",
            "Formula": "COUNT(Completed) / COUNT(Total) * 100",
            "Actual Value": f"{kpis.get('curr_completion_rate', 0):.1f}%",
            "Previous Period": f"{kpis.get('prev_completion_rate', 0):.1f}%",
            "Change %": f"{kpis.get('comp_pct', 0):+.1f}%",
            "Status": status_label(kpis.get('comp_pct', 0)),
            "Business Meaning": "Proportion of orders successfully fulfilled without cancellation."
        })

    if kpis.get("has_secondary"):
        table_data.append({
            "KPI": f"Average {kpis.get('secondary_metric_name', 'Profit')} Yield %",
            "Formula": f"SUM({kpis.get('secondary_metric_name', 'Profit')}) / SUM(Revenue) * 100",
            "Actual Value": f"{kpis.get('curr_margin_pct', 0):.1f}%",
            "Previous Period": f"{kpis.get('prev_margin_pct', 0):.1f}%",
            "Change %": f"{kpis.get('margin_pct_change', 0):+.1f}%",
            "Status": status_label(kpis.get('margin_pct_change', 0)),
            "Business Meaning": "Effective operational margin or return on sales."
        })

    if kpis.get("has_target"):
        table_data.append({
            "KPI": "Target Achievement %",
            "Formula": "SUM(Actual) / SUM(Target) * 100",
            "Actual Value": f"{kpis.get('curr_target_achieved', 0):.1f}%",
            "Previous Period": f"{kpis.get('prev_target_achieved', 0):.1f}%",
            "Change %": f"{kpis.get('target_pct_diff', 0):+.1f}%",
            "Status": status_label(kpis.get('curr_target_achieved', 0), benchmark=100.0),
            "Business Meaning": "Execution against planned corporate budget quotas."
        })

    return pd.DataFrame(table_data)
