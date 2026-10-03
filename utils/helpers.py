"""
Helper functions for formatting, aesthetics, and date handling.
"""
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt


def detect_currency_symbol(metric_name: Optional[str] = None) -> str:
    """Infer appropriate currency symbol from metric column name."""
    if not metric_name:
        return "₹"
    name = str(metric_name).lower()
    if any(k in name for k in ['inr', 'rs', 'rupee']):
        return "₹"
    if any(k in name for k in ['eur', 'euro', '€']):
        return "€"
    if any(k in name for k in ['gbp', 'pound', '£']):
        return "£"
    if any(k in name for k in ['usd', 'dollar', '$']):
        return "$"
    return "₹"


def format_currency(value: Union[int, float], compact: bool = False, currency_symbol: str = "₹") -> str:
    """Format a numerical value as currency, optionally compact (K, M, B)."""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "N/A"
    
    val = float(value)
    negative = val < 0
    val_abs = abs(val)
    
    if compact:
        if val_abs >= 1_000_000_000:
            formatted = f"{val_abs / 1_000_000_000:.2f}B"
        elif val_abs >= 1_000_000:
            formatted = f"{val_abs / 1_000_000:.2f}M"
        elif val_abs >= 1_000:
            formatted = f"{val_abs / 1_000:.1f}K"
        else:
            formatted = f"{val_abs:,.2f}"
    else:
        formatted = f"{val_abs:,.2f}"
        
    return f"-{currency_symbol}{formatted}" if negative else f"{currency_symbol}{formatted}"


def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """Format a standard number with thousands separator."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    if decimals > 0:
        return f"{float(value):,.{decimals}f}"
    return f"{int(round(value)):,}"


def format_percentage(value: Union[int, float], show_sign: bool = True, decimals: int = 1) -> str:
    """Format a ratio or percentage."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    val = float(value)
    sign = "+" if (show_sign and val > 0) else ""
    return f"{sign}{val:.{decimals}f}%"


def calculate_percentage_change(current: float, previous: float) -> Tuple[float, str]:
    """
    Calculate safe percentage change and direction indicator.
    Returns (pct_change, status_indicator).
    """
    if previous is None or previous == 0:
        return 0.0, "neutral"
    
    pct = ((current - previous) / abs(previous)) * 100.0
    indicator = "positive" if pct > 0 else ("negative" if pct < 0 else "neutral")
    return round(pct, 2), indicator


def setup_matplotlib_style():
    """
    Configures Matplotlib runtime parameters for clean, executive-ready presentation.
    Uses crisp vector-ready DPI, modern typography, muted grid lines, and legible ticks.
    """
    plt.rcParams.update({
        'figure.autolayout': True,
        'figure.dpi': 140,
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial', 'Helvetica'],
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.labelweight': 'bold',
        'axes.edgecolor': '#E2E8F0',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#EDF2F7',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#E2E8F0',
    })


# Executive Color Palette
PALETTE = {
    'primary': '#1E3A8A',       # Deep Navy Blue
    'secondary': '#3B82F6',     # Slate Blue
    'accent': '#0D9488',        # Teal Cyan
    'success': '#10B981',       # Emerald Green
    'warning': '#F59E0B',       # Warm Amber
    'danger': '#EF4444',        # Coral Crimson
    'neutral_dark': '#1F2937',  # Charcoal
    'neutral_light': '#F8FAFC', # Off-white soft
    'border': '#E2E8F0',        # Subtle border
    'bars': ['#2563EB', '#0D9488', '#F59E0B', '#6366F1', '#EC4899', '#14B8A6', '#8B5CF6', '#F97316']
}
