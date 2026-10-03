"""
Automated Business Insights Engine Module.
Synthesizes descriptive statistics, KPIs, and categorical distributions into concise, data-driven executive findings.
Organized across 7 core analytical categories.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from utils.helpers import format_currency, format_number, format_percentage


def generate_business_insights(
    df: pd.DataFrame,
    kpis: Dict[str, Any],
    date_col: Optional[str] = None,
    metric_col: Optional[str] = None,
    category_col: Optional[str] = None,
    entity_col: Optional[str] = None,
    df_trend: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Generate rigorous, calculation-backed business statements across 7 analytical categories:
    1. Overall Performance
    2. Trend
    3. Category Performance
    4. Top Performers
    5. Low Performers
    6. Anomalies & Outliers
    7. Opportunity Areas
    """
    categorized_insights: Dict[str, List[str]] = {
        "Overall Performance": [],
        "Trend": [],
        "Category": [],
        "Top Performers": [],
        "Low Performers": [],
        "Anomalies": [],
        "Opportunity Areas": []
    }
    
    key_takeaways: List[str] = []
    
    if df.empty or not kpis:
        return {"key_takeaways": ["Insufficient data available to generate insights."], "categorized": categorized_insights}

    metric_name = kpis.get("metric_name", "Value")
    curr_total = kpis.get("curr_total_val", 0.0)
    curr_records = kpis.get("curr_records", 0)
    curr_avg = kpis.get("curr_avg_val", 0.0)
    growth_pct = kpis.get("growth_pct", 0.0)
    
    # 1. Overall Performance Insights
    p1 = f"Total {metric_name} reached {format_currency(curr_total)} across {format_number(curr_records)} validated transactions."
    categorized_insights["Overall Performance"].append(p1)
    
    p2 = f"Average ticket realization stood at {format_currency(curr_avg)} per transaction."
    categorized_insights["Overall Performance"].append(p2)
    
    if kpis.get("has_target"):
        target_ach = kpis.get("curr_target_achieved", 100.0)
        p3 = f"Target quota attainment achieved {target_ach:.1f}% of corporate benchmark."
        categorized_insights["Overall Performance"].append(p3)

    key_takeaways.append(p1)

    # 2. Trend Insights
    if growth_pct != 0.0:
        dir_word = "expanded by" if growth_pct > 0 else "contracted by"
        t1 = f"Period-over-period {metric_name} {dir_word} {abs(growth_pct):.1f}% relative to the baseline comparison period."
        categorized_insights["Trend"].append(t1)
        key_takeaways.append(t1)
    else:
        categorized_insights["Trend"].append("Performance remained flat against the comparison baseline.")

    if df_trend is not None and not df_trend.empty and len(df_trend) >= 3:
        # Check consecutive trend
        recent_periods = df_trend.tail(3)
        if "Growth_Pct" in recent_periods.columns:
            recent_growths = recent_periods["Growth_Pct"].values
            if all(g > 0 for g in recent_growths):
                t2 = "Consecutive upward momentum observed over the last 3 reporting intervals."
                categorized_insights["Trend"].append(t2)
            elif all(g < 0 for g in recent_growths):
                t2 = "Consecutive downward pressure observed over the last 3 reporting intervals."
                categorized_insights["Trend"].append(t2)

    # 3. Category Insights
    if category_col and category_col in df.columns and metric_col and metric_col in df.columns:
        cat_group = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False)
        if len(cat_group) > 0:
            top_cat = cat_group.index[0]
            top_cat_val = cat_group.iloc[0]
            top_cat_share = (top_cat_val / curr_total * 100.0) if curr_total > 0 else 0
            
            c1 = f"'{top_cat}' is the primary category driver, contributing {format_currency(top_cat_val)} ({top_cat_share:.1f}% of total volume)."
            categorized_insights["Category"].append(c1)
            key_takeaways.append(c1)

            if len(cat_group) > 1:
                bottom_cat = cat_group.index[-1]
                bottom_cat_val = cat_group.iloc[-1]
                bottom_cat_share = (bottom_cat_val / curr_total * 100.0) if curr_total > 0 else 0
                c2 = f"'{bottom_cat}' contributed the lowest category share at {format_currency(bottom_cat_val)} ({bottom_cat_share:.1f}%)."
                categorized_insights["Category"].append(c2)

    # 4. Top Performers Insights
    ent_col = entity_col if (entity_col and entity_col in df.columns) else category_col
    if ent_col and ent_col in df.columns and metric_col and metric_col in df.columns:
        ent_group = df.groupby(ent_col)[metric_col].sum().sort_values(ascending=False)
        total_entities = len(ent_group)
        if total_entities > 0:
            # 80/20 concentration check
            top_20_pct_count = max(1, int(np.ceil(0.20 * total_entities)))
            top_20_val = ent_group.iloc[:top_20_pct_count].sum()
            pareto_share = (top_20_val / curr_total * 100.0) if curr_total > 0 else 0
            
            tp1 = f"The top {top_20_pct_count} {ent_col.replace('_', ' ')} accounts ({min(20, round(top_20_pct_count/total_entities*100))}%) generate {pareto_share:.1f}% of aggregate {metric_name}."
            categorized_insights["Top Performers"].append(tp1)
            
            lead_entity = ent_group.index[0]
            lead_val = ent_group.iloc[0]
            tp2 = f"Leading {ent_col.replace('_', ' ')} '{lead_entity}' generated {format_currency(lead_val)}."
            categorized_insights["Top Performers"].append(tp2)
            key_takeaways.append(tp1)

    # 5. Low Performers Insights
    if ent_col and ent_col in df.columns and metric_col and metric_col in df.columns:
        ent_group = df.groupby(ent_col)[metric_col].sum().sort_values(ascending=True)
        if len(ent_group) >= 4:
            q1_threshold = ent_group.quantile(0.25)
            bottom_quartile = ent_group[ent_group <= q1_threshold]
            lp1 = f"{len(bottom_quartile)} {ent_col.replace('_', ' ')} accounts sit in the bottom quartile, each generating under {format_currency(q1_threshold)}."
            categorized_insights["Low Performers"].append(lp1)

            # Performance spread between top and bottom
            top_val = ent_group.max()
            low_val = ent_group.min()
            spread_factor = round(top_val / low_val, 1) if low_val > 0 else 0
            if spread_factor > 1:
                lp2 = f"The spread between the highest and lowest performer is {spread_factor}x."
                categorized_insights["Low Performers"].append(lp2)

    # 6. Anomalies & Outliers
    if metric_col and metric_col in df.columns and pd.api.types.is_numeric_dtype(df[metric_col]):
        series = df[metric_col].dropna()
        if len(series) >= 10:
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                upper_bound = q75 + (1.5 * iqr)
                outliers = series[series > upper_bound]
                if len(outliers) > 0:
                    a1 = f"Detected {len(outliers):,} high-value transaction outliers exceeding the statistical threshold of {format_currency(upper_bound)}."
                    categorized_insights["Anomalies"].append(a1)
                    outlier_val = outliers.sum()
                    outlier_share = (outlier_val / curr_total * 100.0) if curr_total > 0 else 0
                    a2 = f"These high-value outliers account for {format_currency(outlier_val)} ({outlier_share:.1f}% of total {metric_name})."
                    categorized_insights["Anomalies"].append(a2)
                else:
                    categorized_insights["Anomalies"].append("Transaction values follow standard distribution with no severe statistical outliers.")

    # 7. Opportunity Areas
    if category_col and category_col in df.columns and metric_col and metric_col in df.columns:
        cat_agg = df.groupby(category_col)[metric_col].agg(['mean', 'count'])
        if len(cat_agg) > 1:
            highest_avg_cat = cat_agg['mean'].idxmax()
            highest_avg_val = cat_agg.loc[highest_avg_cat, 'mean']
            lowest_avg_cat = cat_agg['mean'].idxmin()
            lowest_avg_val = cat_agg.loc[lowest_avg_cat, 'mean']
            
            op1 = f"Average transaction value in '{lowest_avg_cat}' ({format_currency(lowest_avg_val)}) trails '{highest_avg_cat}' ({format_currency(highest_avg_val)}) by {format_percentage((lowest_avg_val - highest_avg_val)/highest_avg_val*100, show_sign=False)}."
            categorized_insights["Opportunity Areas"].append(op1)
            
            gap_realization = (highest_avg_val - lowest_avg_val) * cat_agg.loc[lowest_avg_cat, 'count']
            op2 = f"Lifting '{lowest_avg_cat}' average realization to parity represents an addressable upside of ~{format_currency(gap_realization, compact=True)}."
            categorized_insights["Opportunity Areas"].append(op2)
            key_takeaways.append(op1)

    # Ensure key takeaways has 4-6 concise items
    if len(key_takeaways) < 4:
        for cat, items in categorized_insights.items():
            for item in items:
                if item not in key_takeaways:
                    key_takeaways.append(item)
                if len(key_takeaways) >= 5:
                    break
            if len(key_takeaways) >= 5:
                break

    return {
        "key_takeaways": key_takeaways[:6],
        "categorized": categorized_insights
    }
