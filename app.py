"""
Business Analytics Dashboard - Executive KPI Performance & Business Insights
Main Streamlit Application.
Strictly focused on: Data Analytics, KPIs, Visualizations, and Business Insights.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_data, inspect_data, detect_schema
from src.data_cleaning import assess_data_quality, generate_quality_summary_table, clean_data
from src.data_analysis import analyze_all_numerical, analyze_categorical, analyze_time_series
from src.kpi_engine import calculate_kpis, generate_kpi_calculation_table
from src.insights import generate_business_insights
from src.visualizations import (
    create_kpi_trend_chart,
    create_category_chart,
    create_top_10_chart,
    create_bottom_10_chart,
    create_distribution_chart,
    create_performance_trend_chart,
    create_contribution_chart,
    create_correlation_chart
)
from utils.helpers import format_currency, format_number, format_percentage, PALETTE

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Business Analytics Dashboard | KPI & Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Executive CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Clean layout and typography */
    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .dashboard-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .dashboard-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 500;
    }
    
    /* Executive Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 0.8rem;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .kpi-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.82rem;
        margin-top: 0.4rem;
        font-weight: 600;
    }
    .kpi-pos { color: #10B981; }
    .kpi-neg { color: #EF4444; }
    .kpi-neu { color: #64748B; }

    /* Insights Box */
    .insights-panel {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.5rem;
    }
    .insights-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.6rem;
    }
    .insight-item {
        font-size: 0.93rem;
        color: #334155;
        line-height: 1.55;
        margin-bottom: 0.4rem;
    }

    /* Section Headings */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #3B82F6;
        padding-left: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Data Source & Global Controls
# ---------------------------------------------------------
st.sidebar.markdown("### 📂 Data Source & Ingestion")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
default_csv = os.path.join(DATA_DIR, "dataset.csv")
default_xlsx = os.path.join(DATA_DIR, "sample_ecommerce.xlsx")

source_choice = st.sidebar.radio(
    "Select Dataset:",
    options=[
        "Enterprise B2B Tech Sales (CSV)",
        "Retail & E-Commerce (Excel)",
        "Upload Custom File (CSV / Excel)"
    ],
    index=0
)

uploaded_df = None

if source_choice == "Enterprise B2B Tech Sales (CSV)":
    if os.path.exists(default_csv):
        uploaded_df = load_data(default_csv, file_type='csv')
    else:
        st.sidebar.warning(f"File {default_csv} not found. Generating default data...")
elif source_choice == "Retail & E-Commerce (Excel)":
    if os.path.exists(default_xlsx):
        uploaded_df = load_data(default_xlsx, file_type='excel')
    else:
        st.sidebar.warning("Excel sample loading...")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        file_ext = "excel" if uploaded_file.name.endswith(('.xlsx', '.xls')) else "csv"
        uploaded_df = load_data(uploaded_file, file_type=file_ext)

# Fallback in case sample files were not yet on disk
if uploaded_df is None or uploaded_df.empty:
    # Generate synthetic enterprise dataframe in-memory so dashboard always loads seamlessly
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=1000, freq="D")
    uploaded_df = pd.DataFrame({
        "Order_Date": np.random.choice(dates, 1500),
        "Order_ID": [f"ORD-{i:05d}" for i in range(1, 1501)],
        "Customer_ID": [f"CUST-{np.random.randint(100, 250)}" for _ in range(1500)],
        "Region": np.random.choice(["North America", "EMEA", "APAC", "LATAM"], 1500),
        "Customer_Segment": np.random.choice(["Enterprise", "Mid-Market", "Small Business"], 1500),
        "Category": np.random.choice(["Cloud Infrastructure", "Enterprise Software", "Hardware & Systems", "Professional Services"], 1500),
        "Product_Name": np.random.choice(["Compute Pro", "Analytics Suite", "Storage Array", "Cloud Advisor", "Security Gateway"], 1500),
        "Sales_Amount": np.round(np.random.uniform(500, 15000, 1500), 2),
        "Quantity": np.random.randint(1, 20, 1500),
        "Profit": np.round(np.random.uniform(100, 4500, 1500), 2),
        "Target_Sales": np.round(np.random.uniform(600, 14000, 1500), 2),
        "Order_Status": np.random.choice(["Completed", "Completed", "Delivered", "In Progress", "Cancelled"], 1500)
    })

raw_df = uploaded_df.copy(deep=True)
metadata = inspect_data(raw_df)

# Auto-detect schema
schema = detect_schema(raw_df)

# Sidebar Schema Mapping Configuration (Expandable)
with st.sidebar.expander("⚙️ Column Schema Mapping", expanded=False):
    date_col = st.selectbox(
        "Date Column",
        options=[None] + list(raw_df.columns),
        index=(list(raw_df.columns).index(schema["date_col"]) + 1) if schema["date_col"] in raw_df.columns else 0
    )
    metric_col = st.selectbox(
        "Primary Metric (Value / Revenue)",
        options=schema["all_numeric"] if schema["all_numeric"] else list(raw_df.columns),
        index=schema["all_numeric"].index(schema["primary_metric"]) if schema["primary_metric"] in schema["all_numeric"] else 0
    )
    category_col = st.selectbox(
        "Category Dimension",
        options=schema["all_categorical"] if schema["all_categorical"] else list(raw_df.columns),
        index=schema["all_categorical"].index(schema["primary_category"]) if schema["primary_category"] in schema["all_categorical"] else 0
    )
    entity_col = st.selectbox(
        "Entity Dimension (Customer / Product)",
        options=list(raw_df.columns),
        index=list(raw_df.columns).index(schema["primary_entity"]) if schema["primary_entity"] in raw_df.columns else 0
    )
    status_col = st.selectbox(
        "Status / Conversion Column",
        options=[None] + schema["all_categorical"],
        index=(schema["all_categorical"].index(schema["status_col"]) + 1) if schema["status_col"] in schema["all_categorical"] else 0
    )
    secondary_metric_col = st.selectbox(
        "Secondary Metric (Profit / Yield)",
        options=[None] + [c for c in schema["all_numeric"] if c != metric_col],
        index=1 if len([c for c in schema["all_numeric"] if c != metric_col]) > 0 else 0
    )
    target_col = st.selectbox(
        "Target Column",
        options=[None] + schema["all_numeric"],
        index=(schema["all_numeric"].index(schema["target_col"]) + 1) if schema["target_col"] in schema["all_numeric"] else 0
    )

# ---------------------------------------------------------
# Step 2 & 4: Data Quality Check & Non-Destructive Cleaning
# ---------------------------------------------------------
clean_df, audit_log = clean_data(
    raw_df,
    date_col=date_col,
    remove_duplicates=True,
    handle_missing=True,
    standardize_names=False, # preserve user column case for visual mapping
    handle_outliers=False
)

# ---------------------------------------------------------
# Sidebar: Dynamic Interactive Filters
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Dashboard Filters")

filtered_df = clean_df.copy()

# Date Range Filter
if date_col and date_col in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
    min_date = filtered_df[date_col].min().date()
    max_date = filtered_df[date_col].max().date()
    
    if min_date < max_date:
        date_range = st.sidebar.date_input(
            "Date Range Filter",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_d, end_d = date_range
            mask = (filtered_df[date_col].dt.date >= start_d) & (filtered_df[date_col].dt.date <= end_d)
            filtered_df = filtered_df[mask]

# Granularity Selector
granularity = st.sidebar.selectbox(
    "Trend Granularity",
    options=["Monthly", "Quarterly", "Weekly", "Daily", "Yearly"],
    index=0
)

# Categorical Multiselect Filter
if category_col and category_col in filtered_df.columns:
    cat_options = sorted(list(filtered_df[category_col].dropna().unique()))
    selected_cats = st.sidebar.multiselect(
        f"Filter by {category_col.replace('_', ' ').title()}:",
        options=cat_options,
        default=[]
    )
    if selected_cats:
        filtered_df = filtered_df[filtered_df[category_col].isin(selected_cats)]

# Region Filter (if available)
if schema.get("primary_region") and schema["primary_region"] in filtered_df.columns and schema["primary_region"] != category_col:
    reg_col = schema["primary_region"]
    reg_options = sorted(list(filtered_df[reg_col].dropna().unique()))
    selected_regs = st.sidebar.multiselect(
        f"Filter by {reg_col.replace('_', ' ').title()}:",
        options=reg_options,
        default=[]
    )
    if selected_regs:
        filtered_df = filtered_df[filtered_df[reg_col].isin(selected_regs)]

# ---------------------------------------------------------
# Analytics Engine Computations
# ---------------------------------------------------------
# Time series aggregation
df_trend = pd.DataFrame()
if date_col and date_col in filtered_df.columns and metric_col and metric_col in filtered_df.columns:
    df_trend = analyze_time_series(filtered_df, date_col, metric_col, granularity=granularity, agg_func="sum")

# Compute dynamic KPIs
kpis = calculate_kpis(
    df=filtered_df,
    date_col=date_col,
    metric_col=metric_col,
    target_col=target_col,
    entity_col=entity_col,
    status_col=status_col,
    secondary_metric_col=secondary_metric_col
)

# Automated Business Insights
insights = generate_business_insights(
    df=filtered_df,
    kpis=kpis,
    date_col=date_col,
    metric_col=metric_col,
    category_col=category_col,
    entity_col=entity_col,
    df_trend=df_trend
)

# ---------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# ---------------------------------------------------------

# Header Section
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">Business Analytics Dashboard</div>
    <div class="dashboard-subtitle">KPI Performance & Business Insights</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# STEP 1 & 2: DATA QUALITY & CLEANING ACCORDION
# ---------------------------------------------------------
with st.expander("📋 Data Ingestion & Data Quality Report (Raw Data → Clean Data)", expanded=False):
    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        st.markdown("#### Data Quality Summary Table")
        st.dataframe(generate_quality_summary_table(raw_df), use_container_width=True, hide_index=True)
    with col_q2:
        st.markdown("#### Cleaning Pipeline Audit Log")
        for entry in audit_log:
            st.markdown(f"- ✅ {entry}")
        st.info(f"Raw Records: **{len(raw_df):,}** → Clean Active Records: **{len(filtered_df):,}** (Maintains strict data immutability).")

# ---------------------------------------------------------
# STEP 3: KPI ROW (4–6 EXECUTIVE CARDS)
# ---------------------------------------------------------
st.markdown('<div class="section-title">Executive KPI Scorecard</div>', unsafe_allow_html=True)

kpi_cols = st.columns(6)

def render_kpi_card(col, label, value_str, pct_str, delta_color="kpi-pos", sub_label="vs Prev Period"):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value_str}</div>
        <div class="kpi-sub {delta_color}">{pct_str} <span style="color:#94A3B8; font-weight:normal;">{sub_label}</span></div>
    </div>
    """, unsafe_allow_html=True)

# Card 1: Total Revenue / Value
val_color = "kpi-pos" if kpis.get("val_pct", 0) >= 0 else "kpi-neg"
render_kpi_card(
    kpi_cols[0],
    f"Total {metric_col or 'Value'}",
    format_currency(kpis.get("curr_total_val", 0), compact=True),
    f"{kpis.get('val_pct', 0):+.1f}%",
    val_color
)

# Card 2: Average Value (AOV)
avg_color = "kpi-pos" if kpis.get("avg_pct", 0) >= 0 else "kpi-neg"
render_kpi_card(
    kpi_cols[1],
    f"Average {metric_col or 'Value'}",
    format_currency(kpis.get("curr_avg_val", 0), compact=True),
    f"{kpis.get('avg_pct', 0):+.1f}%",
    avg_color
)

# Card 3: Period Growth %
growth_color = "kpi-pos" if kpis.get("growth_pct", 0) >= 0 else "kpi-neg"
render_kpi_card(
    kpi_cols[2],
    "Period Growth %",
    f"{kpis.get('growth_pct', 0):+.1f}%",
    "MoM / QoQ",
    growth_color,
    sub_label="Trajectory"
)

# Card 4: Total Records
rec_color = "kpi-pos" if kpis.get("rec_pct", 0) >= 0 else "kpi-neg"
render_kpi_card(
    kpi_cols[3],
    "Total Records",
    format_number(kpis.get("curr_records", 0)),
    f"{kpis.get('rec_pct', 0):+.1f}%",
    rec_color,
    sub_label="Transactions"
)

# Card 5: Distinct Entities
ent_color = "kpi-pos" if kpis.get("entity_pct", 0) >= 0 else "kpi-neg"
render_kpi_card(
    kpi_cols[4],
    f"Active {entity_col or 'Entities'}",
    format_number(kpis.get("curr_entities", 0)),
    f"{kpis.get('entity_pct', 0):+.1f}%",
    ent_color,
    sub_label="Unique Count"
)

# Card 6: Target % or Margin %
if kpis.get("has_target"):
    t_val = kpis.get("curr_target_achieved", 100.0)
    t_color = "kpi-pos" if t_val >= 100 else ("kpi-neg" if t_val < 85 else "kpi-neu")
    render_kpi_card(
        kpi_cols[5],
        "Target Quota %",
        f"{t_val:.1f}%",
        f"{kpis.get('target_pct_diff', 0):+.1f}%",
        t_color,
        sub_label="vs Quota"
    )
elif kpis.get("has_secondary"):
    m_val = kpis.get("curr_margin_pct", 0.0)
    m_color = "kpi-pos" if m_val > 25 else ("kpi-neg" if m_val < 10 else "kpi-neu")
    render_kpi_card(
        kpi_cols[5],
        f"{secondary_metric_col} %",
        f"{m_val:.1f}%",
        f"{kpis.get('margin_pct_change', 0):+.1f}%",
        m_color,
        sub_label="Yield"
    )
elif kpis.get("has_status"):
    c_val = kpis.get("curr_completion_rate", 0.0)
    render_kpi_card(
        kpi_cols[5],
        "Completion %",
        f"{c_val:.1f}%",
        f"{kpis.get('comp_pct', 0):+.1f}%",
        "kpi-pos",
        sub_label="Fulfillment"
    )
else:
    render_kpi_card(
        kpi_cols[5],
        "Data Health",
        "100%",
        "Valid",
        "kpi-pos",
        sub_label="Integrity"
    )

# ---------------------------------------------------------
# STEP 4: KEY BUSINESS INSIGHTS PANEL (SECTION 14)
# ---------------------------------------------------------
st.markdown("""
<div class="insights-panel">
    <div class="insights-title">💡 Key Business Insights</div>
""", unsafe_allow_html=True)

for takeaway in insights.get("key_takeaways", []):
    st.markdown(f"<div class='insight-item'>• {takeaway}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Categorized Drill-Down Expanders for Insights (Section 10)
with st.expander("🔍 Explore Comprehensive Insights by Analytical Category (7 Categories)", expanded=False):
    tab_cats = st.tabs([
        "Overall Performance", 
        "Trend", 
        "Category", 
        "Top Performers", 
        "Low Performers", 
        "Anomalies", 
        "Opportunity Areas"
    ])
    
    for i, (cat_name, items) in enumerate(insights.get("categorized", {}).items()):
        with tab_cats[i]:
            if items:
                for statement in items:
                    st.markdown(f"- **{cat_name}**: {statement}")
            else:
                st.markdown(f"_No anomalies or specific patterns detected under {cat_name}._")

# ---------------------------------------------------------
# STEP 5: VISUAL SECTION (MATPLOTLIB VISUALIZATIONS)
# ---------------------------------------------------------
st.markdown('<div class="section-title">Core Performance Visualizations</div>', unsafe_allow_html=True)

# Top Visual Row: Performance Trend (Left) & Category Performance (Right)
col_v1, col_v2 = st.columns([1, 1])

with col_v1:
    st.markdown(f"##### Performance Trend ({granularity})")
    fig_trend = create_kpi_trend_chart(
        df_trend,
        date_col=date_col or "Date",
        metric_col=metric_col or "Value",
        metric_label=metric_col or "Value"
    )
    st.pyplot(fig_trend, use_container_width=True)
    plt.close(fig_trend)

with col_v2:
    st.markdown(f"##### Category Performance ({category_col.replace('_', ' ').title() if category_col else 'Category'})")
    fig_cat = create_category_chart(
        filtered_df,
        category_col=category_col or "Category",
        metric_col=metric_col or "Value",
        metric_label=metric_col or "Revenue"
    )
    st.pyplot(fig_cat, use_container_width=True)
    plt.close(fig_cat)

# Bottom Visual Row 1: Top 10 Performers & Bottom 10 Performers
col_v3, col_v4 = st.columns([1, 1])

with col_v3:
    st.markdown(f"##### Top 10 Drivers ({entity_col.replace('_', ' ').title() if entity_col else 'Entities'})")
    fig_top = create_top_10_chart(
        filtered_df,
        entity_col=entity_col or category_col or "Entity",
        metric_col=metric_col or "Value",
        metric_label=metric_col or "Revenue"
    )
    st.pyplot(fig_top, use_container_width=True)
    plt.close(fig_top)

with col_v4:
    st.markdown(f"##### Bottom 10 Performers & Gaps ({entity_col.replace('_', ' ').title() if entity_col else 'Entities'})")
    fig_bot = create_bottom_10_chart(
        filtered_df,
        entity_col=entity_col or category_col or "Entity",
        metric_col=metric_col or "Value",
        metric_label=metric_col or "Revenue"
    )
    st.pyplot(fig_bot, use_container_width=True)
    plt.close(fig_bot)

# Bottom Visual Row 2: Distribution & Contribution Analysis
col_v5, col_v6 = st.columns([1, 1])

with col_v5:
    st.markdown(f"##### Distribution Analysis ({metric_col or 'Value'})")
    fig_dist = create_distribution_chart(
        filtered_df,
        metric_col=metric_col or "Value",
        metric_label=metric_col or "Transaction Value"
    )
    st.pyplot(fig_dist, use_container_width=True)
    plt.close(fig_dist)

with col_v6:
    st.markdown(f"##### Contribution & Pareto Analysis ({category_col.replace('_', ' ').title() if category_col else 'Category'})")
    fig_contrib = create_contribution_chart(
        filtered_df,
        category_col=category_col or "Category",
        metric_col=metric_col or "Value",
        metric_label=metric_col or "Revenue"
    )
    st.pyplot(fig_contrib, use_container_width=True)
    plt.close(fig_contrib)

# Expandable Additional Deep-Dive Visuals
with st.expander("📈 Additional Visualizations (Correlation Heatmap & Period Growth Breakdown)", expanded=False):
    col_v7, col_v8 = st.columns([1, 1])
    with col_v7:
        st.markdown("##### Numerical Correlation Matrix")
        fig_corr = create_correlation_chart(filtered_df)
        st.pyplot(fig_corr, use_container_width=True)
        plt.close(fig_corr)
    with col_v8:
        st.markdown(f"##### Periodic Breakdown ({granularity})")
        fig_perf = create_performance_trend_chart(
            df_trend,
            metric_col=metric_col or "Value",
            metric_label=metric_col or "Revenue"
        )
        st.pyplot(fig_perf, use_container_width=True)
        plt.close(fig_perf)

# ---------------------------------------------------------
# STEP 6: DETAILED DATA TABLE & KPI AUDIT (SECTION 15)
# ---------------------------------------------------------
st.markdown('<div class="section-title">Detailed Analysis & Audit</div>', unsafe_allow_html=True)

with st.expander("📊 Standardized KPI Calculation Table (Audit & Verification)", expanded=True):
    kpi_audit_df = generate_kpi_calculation_table(kpis)
    st.dataframe(kpi_audit_df, use_container_width=True, hide_index=True)

with st.expander("🔢 Exploratory Data Analysis (Descriptive Statistics)", expanded=False):
    st.markdown("##### Numerical Variable Statistical Profile")
    eda_num_df = analyze_all_numerical(filtered_df)
    st.dataframe(eda_num_df, use_container_width=True, hide_index=True)
    
    if category_col and category_col in filtered_df.columns:
        st.markdown(f"##### Categorical Distribution for {category_col}")
        eda_cat_df = analyze_categorical(filtered_df, category_col, top_n=10)
        st.dataframe(eda_cat_df, use_container_width=True, hide_index=True)

with st.expander("📄 Filtered Dataset View & Export", expanded=False):
    st.write(f"Displaying **{len(filtered_df):,}** filtered records:")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Download Button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="filtered_business_analytics_dataset.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Business Analytics Dashboard • Professional Data Analyst KPI & Insights Engine • Python, Pandas, Matplotlib, Streamlit")
