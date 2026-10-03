"""
Matplotlib Visualization Module.
Produces 8 professional, executive-ready analytical charts.
"""
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from utils.helpers import setup_matplotlib_style, PALETTE, format_currency, format_number

# Apply default styling
setup_matplotlib_style()


def create_kpi_trend_chart(
    df_trend: pd.DataFrame, 
    date_col: str, 
    metric_col: str, 
    metric_label: str = "Revenue"
) -> plt.Figure:
    """
    Visualization 1 — KPI Trend over time with 3-period moving average and area shading.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    if df_trend.empty or metric_col not in df_trend.columns:
        ax.text(0.5, 0.5, "Insufficient time-series data for trend analysis", ha='center', va='center', color='#64748B')
        ax.axis('off')
        return fig

    x_labels = df_trend["Period_Label"] if "Period_Label" in df_trend.columns else df_trend.index.astype(str)
    x = np.arange(len(df_trend))
    y = df_trend[metric_col].values

    # Plot baseline line and filled area
    ax.plot(x, y, color=PALETTE['secondary'], marker='o', markersize=4, linewidth=2, label=f"Actual {metric_label}", zorder=4)
    ax.fill_between(x, y, color=PALETTE['secondary'], alpha=0.12, zorder=2)

    # Plot rolling average if available
    if "Rolling_3P_Avg" in df_trend.columns and len(df_trend) >= 3:
        ax.plot(x, df_trend["Rolling_3P_Avg"], color=PALETTE['accent'], linestyle='--', linewidth=1.8, label="3-Period Moving Avg", zorder=5)

    # Annotate max and min points
    if len(y) > 0:
        max_idx = int(np.argmax(y))
        min_idx = int(np.argmin(y))
        
        ax.scatter([x[max_idx]], [y[max_idx]], color=PALETTE['success'], s=60, zorder=6)
        ax.annotate(
            f"Peak: {format_currency(y[max_idx], compact=True)}",
            (x[max_idx], y[max_idx]),
            textcoords="offset points", xytext=(0, 10), ha='center',
            fontweight='bold', fontsize=8, color='#065F46',
            bbox=dict(boxstyle="round,pad=0.2", fc="#ECFDF5", ec="#A7F3D0", lw=0.8)
        )
        
        if max_idx != min_idx and len(y) > 2:
            ax.scatter([x[min_idx]], [y[min_idx]], color=PALETTE['danger'], s=60, zorder=6)
            ax.annotate(
                f"Low: {format_currency(y[min_idx], compact=True)}",
                (x[min_idx], y[min_idx]),
                textcoords="offset points", xytext=(0, -16), ha='center',
                fontweight='bold', fontsize=8, color='#991B1B',
                bbox=dict(boxstyle="round,pad=0.2", fc="#FEF2F2", ec="#FECACA", lw=0.8)
            )

    # Format Axes
    ax.set_title(f"{metric_label} Performance Over Time", pad=12, loc='left', color=PALETTE['neutral_dark'])
    ax.set_xticks(x)
    
    # Handle label density
    step = max(1, len(x) // 10)
    visible_labels = [label if (i % step == 0 or i == len(x)-1) else "" for i, label in enumerate(x_labels)]
    ax.set_xticklabels(visible_labels, rotation=35, ha='right')
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_currency(val, compact=True)))
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def create_category_chart(
    df: pd.DataFrame, 
    category_col: str, 
    metric_col: str, 
    metric_label: str = "Revenue"
) -> plt.Figure:
    """
    Visualization 2 — Category Comparison (Sorted horizontal bar chart with exact value labels).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    if df.empty or category_col not in df.columns or metric_col not in df.columns:
        ax.text(0.5, 0.5, "Data unavailable for Category Comparison", ha='center', va='center', color='#64748B')
        ax.axis('off')
        return fig

    cat_agg = df.groupby(category_col)[metric_col].sum().sort_values(ascending=True).tail(10)
    y_pos = np.arange(len(cat_agg))
    values = cat_agg.values
    categories = [str(c) for c in cat_agg.index]

    colors = [PALETTE['secondary'] if i < len(values) - 1 else PALETTE['primary'] for i in range(len(values))]
    bars = ax.barh(y_pos, values, color=colors, height=0.6, zorder=3)

    # Direct value labels on ends of bars
    max_val = max(values) if len(values) > 0 else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + (max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            format_currency(val, compact=True),
            va='center', ha='left', fontsize=8.5, fontweight='bold', color=PALETTE['neutral_dark']
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, max_val * 1.22)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_currency(val, compact=True)))
    ax.set_title(f"Total {metric_label} by {category_col.replace('_', ' ').title()}", pad=12, loc='left', color=PALETTE['neutral_dark'])
    
    plt.tight_layout()
    return fig


def create_top_10_chart(
    df: pd.DataFrame, 
    entity_col: str, 
    metric_col: str, 
    metric_label: str = "Revenue"
) -> plt.Figure:
    """
    Visualization 3 — Top 10 Performers with % contribution share.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    if df.empty or entity_col not in df.columns or metric_col not in df.columns:
        ax.text(0.5, 0.5, "Data unavailable for Top 10 Performers", ha='center', va='center', color='#64748B')
        ax.axis('off')
        return fig

    agg = df.groupby(entity_col)[metric_col].sum().sort_values(ascending=False).head(10)
    total_metric = df[metric_col].sum()
    
    agg = agg.iloc[::-1]  # reverse for bottom-to-top rendering
    y_pos = np.arange(len(agg))
    values = agg.values
    entities = [str(e)[:24] for e in agg.index]

    bars = ax.barh(y_pos, values, color=PALETTE['accent'], height=0.6, zorder=3)
    max_val = max(values) if len(values) > 0 else 1

    for bar, val in zip(bars, values):
        pct = (val / total_metric * 100.0) if total_metric > 0 else 0
        ax.text(
            bar.get_width() + (max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{format_currency(val, compact=True)} ({pct:.1f}%)",
            va='center', ha='left', fontsize=8, fontweight='bold', color=PALETTE['neutral_dark']
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(entities)
    ax.set_xlim(0, max_val * 1.30)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_currency(val, compact=True)))
    ax.set_title(f"Top 10 {entity_col.replace('_', ' ').title()} by {metric_label}", pad=12, loc='left', color=PALETTE['neutral_dark'])
    
    plt.tight_layout()
    return fig


def create_bottom_10_chart(
    df: pd.DataFrame, 
    entity_col: str, 
    metric_col: str, 
    metric_label: str = "Revenue"
) -> plt.Figure:
    """
    Visualization 4 — Bottom 10 Performers (Identifying underperforming areas and gap to median).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    if df.empty or entity_col not in df.columns or metric_col not in df.columns:
        ax.text(0.5, 0.5, "Data unavailable for Bottom 10 Performers", ha='center', va='center', color='#64748B')
        ax.axis('off')
        return fig

    # Filter out exact zeros if desired or show bottom active
    agg_all = df.groupby(entity_col)[metric_col].sum()
    median_val = float(agg_all.median())
    
    agg = agg_all.sort_values(ascending=True).head(10)
    y_pos = np.arange(len(agg))
    values = agg.values
    entities = [str(e)[:24] for e in agg.index]

    bars = ax.barh(y_pos, values, color=PALETTE['warning'], height=0.6, zorder=3)
    max_val = max(values) if len(values) > 0 else 1

    for bar, val in zip(bars, values):
        gap = median_val - val
        gap_text = f"Gap: -{format_currency(gap, compact=True)}" if gap > 0 else "At/Above Med"
        ax.text(
            bar.get_width() + (max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{format_currency(val, compact=True)} ({gap_text})",
            va='center', ha='left', fontsize=8, color='#78350F'
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(entities)
    ax.set_xlim(0, max(max_val * 1.35, 1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_currency(val, compact=True)))
    ax.set_title(f"Lowest 10 {entity_col.replace('_', ' ').title()} by {metric_label} (Performance Gaps)", pad=12, loc='left', color=PALETTE['neutral_dark'])
    
    plt.tight_layout()
    return fig


def create_distribution_chart(
    df: pd.DataFrame, 
    metric_col: str, 
    metric_label: str = "Transaction Value"
) -> plt.Figure:
    """
    Visualization 5 — Distribution Histogram with Median & Mean reference lines.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    if df.empty or metric_col not in df.columns:
        ax.text(0.5, 0.5, "Data unavailable for Distribution Analysis", ha='center', va='center', color='#64748B')
        ax.axis('off')
        return fig

    data = df[metric_col].dropna()
    # Clip extreme 1% tails for readable plotting if high skew
    p99 = data.quantile(0.99)
    filtered_data = data[data <= p99] if p99 > 0 else data

    counts, bins, patches = ax.hist(filtered_data, bins=30, color=PALETTE['secondary'], alpha=0.75, edgecolor='#FFFFFF', zorder=3)

    mean_val = float(filtered_data.mean())
    median_val = float(filtered_data.median())

    ax.axvline(mean_val, color=PALETTE['danger'], linestyle='--', linewidth=1.8, label=f"Mean: {format_currency(mean_val, compact=True)}")
    ax.axvline(median_val, color=PALETTE['success'], linestyle='-', linewidth=2, label=f"Median: {format_currency(median_val, compact=True)}")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_currency(val, compact=True)))
    ax.set_title(f"Distribution of {metric_label} (Up to 99th Percentile)", pad=12, loc='left', color=PALETTE['neutral_dark'])
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Frequency (Record Count)")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    return fig


def create_performance_trend_chart(
    df_trend: pd.DataFrame, 
    metric_col: str, 
    metric_label: str = "Revenue"
) -> plt.Figure:
    """
    Visualization 6 — Performance Trend bar chart with period-over-period growth indicators.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    if df_trend.empty or metric_col not in df_trend.columns:
        ax.text(0.5, 0.5, "Data unavailable for Performance Trend", ha='center', va='center', color='#64748B')
        ax.axis('off')
        return fig

    x_labels = df_trend["Period_Label"] if "Period_Label" in df_trend.columns else df_trend.index.astype(str)
    x = np.arange(len(df_trend))
    y = df_trend[metric_col].values

    # Color bars by positive or negative MoM growth
    if "Growth_Pct" in df_trend.columns:
        colors = [PALETTE['success'] if g >= 0 else PALETTE['danger'] for g in df_trend["Growth_Pct"]]
    else:
        colors = [PALETTE['primary']] * len(x)

    bars = ax.bar(x, y, color=colors, width=0.6, alpha=0.85, zorder=3)

    # Top value annotations
    max_val = max(y) if len(y) > 0 else 1
    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_val * 0.02),
            format_currency(val, compact=True),
            ha='center', va='bottom', fontsize=8, color=PALETTE['neutral_dark'], fontweight='bold'
        )

    ax.set_xticks(x)
    step = max(1, len(x) // 10)
    visible_labels = [label if (i % step == 0 or i == len(x)-1) else "" for i, label in enumerate(x_labels)]
    ax.set_xticklabels(visible_labels, rotation=35, ha='right')
    
    ax.set_ylim(0, max_val * 1.18)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_currency(val, compact=True)))
    ax.set_title(f"Periodic {metric_label} Breakdown (Green = Positive Growth, Red = Contraction)", pad=12, loc='left', color=PALETTE['neutral_dark'])
    
    plt.tight_layout()
    return fig


def create_contribution_chart(
    df: pd.DataFrame, 
    category_col: str, 
    metric_col: str, 
    metric_label: str = "Revenue"
) -> plt.Figure:
    """
    Visualization 7 — Contribution Analysis (Pareto Chart: Sorted Volume Bars + Cumulative Share Line).
    """
    setup_matplotlib_style()
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    
    if df.empty or category_col not in df.columns or metric_col not in df.columns:
        ax1.text(0.5, 0.5, "Data unavailable for Contribution Analysis", ha='center', va='center', color='#64748B')
        ax1.axis('off')
        return fig

    agg = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False).head(8)
    total_val = df[metric_col].sum()
    
    categories = [str(c)[:15] for c in agg.index]
    values = agg.values
    cum_pct = (np.cumsum(values) / total_val * 100.0) if total_val > 0 else np.zeros_like(values)

    x = np.arange(len(categories))
    ax1.bar(x, values, color=PALETTE['primary'], width=0.55, zorder=3, label=f"{metric_label} Volume")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: format_currency(val, compact=True)))
    ax1.set_ylabel(metric_label)

    # Secondary axis for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct, color=PALETTE['danger'], marker='D', markersize=5, linewidth=2, label="Cumulative % Share", zorder=5)
    ax2.axhline(80, color='#94A3B8', linestyle=':', linewidth=1.2, label="80% Threshold")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative Share %")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.grid(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=25, ha='right')
    ax1.set_title(f"Contribution & Pareto Analysis by {category_col.replace('_', ' ').title()}", pad=12, loc='left', color=PALETTE['neutral_dark'])
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    plt.tight_layout()
    return fig


def create_correlation_chart(
    df: pd.DataFrame, 
    numeric_cols: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualization 8 — Correlation Heatmap across numeric metrics with annotated coefficients.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    if numeric_cols is None:
        numeric_cols = list(df.select_dtypes(include=['number']).columns)

    # Take up to 6 most relevant numeric columns
    cols_to_use = [c for c in numeric_cols if not c.lower().endswith('_id')][:6]
    
    if len(cols_to_use) < 2:
        ax.text(0.5, 0.5, "Requires at least 2 numerical variables for correlation matrix", ha='center', va='center', color='#64748B')
        ax.axis('off')
        return fig

    corr = df[cols_to_use].corr().round(2)
    data = corr.values
    
    cax = ax.matshow(data, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    labels = [c.replace('_', ' ').title()[:12] for c in cols_to_use]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='left')
    ax.set_yticklabels(labels)

    # Annotate coefficients
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=8.5)

    ax.grid(False)
    ax.set_title("Numerical Variable Correlation Matrix", pad=20, loc='left', color=PALETTE['neutral_dark'])
    
    plt.tight_layout()
    return fig
