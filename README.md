# Professional Data Analyst KPI & Business Insights Dashboard

An end-to-end, production-grade **Data Analytics Dashboard** built in Python using **Pandas, NumPy, Matplotlib, and Streamlit**.

This project demonstrates the complete real-world workflow of a Senior Data Analyst, BI Analyst, and Dashboard Developer:
$$\text{Raw Data} \longrightarrow \text{Data Cleaning} \longrightarrow \text{Data Transformation} \longrightarrow \text{Exploratory Data Analysis} \longrightarrow \text{KPI Engine} \longrightarrow \text{Matplotlib Visualizations} \longrightarrow \text{Business Insights} \longrightarrow \text{Interactive Dashboard}$$

Strictly focused on **quantitative business analytics, financial/operational KPIs, visualizations, and data-driven insights**.

---

## 1. Key Features

- **Multi-Source Data Ingestion**:
  - Out-of-the-box support for **CSV** and **Excel (`.xlsx`, `.xls`)** files.
  - Preloaded with multi-year enterprise datasets:
    1. *Enterprise B2B Technology Sales & Financials (`data/dataset.csv`)*
    2. *Retail & E-Commerce Transactions (`data/sample_ecommerce.xlsx`)*
  - User file uploader with schema auto-detection.
  - Extensible interface for future database / SQL connections (`sqlalchemy`).

- **Automated Data Quality & Non-Destructive Cleaning**:
  - Audits missing values, duplicate records, data type classifications, cardinality, and outliers via the IQR method.
  - Generates a standardized **Data Quality Summary Table**.
  - Non-destructive transformation pipeline maintaining strict data immutability:
    $$\text{Raw Data (Read-Only)} \longrightarrow \text{Clean Active Dataset}$$
  - Transparent audit trail detailing all cleaning actions.

- **Dynamic KPI Engine**:
  - Dynamically calculates 8 core operational and financial KPIs with period-over-period comparison:
    1. **Total Revenue / Value**: $\sum \text{Value}$
    2. **Average Ticket / AOV**: $\frac{1}{N}\sum \text{Value}$
    3. **Period Growth %**: $\frac{\text{Current} - \text{Previous}}{|\text{Previous}|} \times 100$
    4. **Total Records**: $\text{COUNT}(\text{Rows})$
    5. **Distinct Entities / Customers**: $\text{COUNT}(\text{DISTINCT } \text{Entity\_ID})$
    6. **Conversion / Completion Rate %**: $\frac{\text{Completed}}{\text{Total}} \times 100$
    7. **Target Achievement %**: $\frac{\text{Actual Value}}{\text{Target Quota}} \times 100$
    8. **Average Yield / Margin %**: $\frac{\sum \text{Profit}}{\sum \text{Revenue}} \times 100$
  - Generates the standardized **KPI Calculation & Audit Table** with transparent formulas, prior-period benchmarks, change %, and color-coded status badges.

- **8 Bespoke Matplotlib Executive Visualizations**:
  - **KPI Trend**: High-resolution time-series line chart with 3-period moving average, shaded confidence band, and peak/trough callouts.
  - **Category Comparison**: Sorted horizontal bar chart with direct currency value labels.
  - **Top 10 Performers**: Ranked driver analysis showing absolute values and cumulative percentage contributions.
  - **Bottom 10 Performers**: Highlights lagging segments and quantifies performance gaps to the median.
  - **Distribution Analysis**: Value histogram with median and mean reference lines.
  - **Performance Trend Breakdown**: Periodic volume bars with green/red MoM growth indicators.
  - **Contribution & Pareto Analysis**: Dual-axis Pareto chart (Volume bars + Cumulative share curve with 80% threshold).
  - **Correlation Matrix**: Numerical heatmap with annotated Pearson correlation coefficients.

- **Automated Business Insights Engine**:
  - Synthesizes calculations into concise, executive-ready analytical statements across 7 core categories:
    1. *Overall Performance*
    2. *Trend & Trajectory*
    3. *Category Performance*
    4. *Top Performers & Concentration (Pareto)*
    5. *Low Performers & Performance Gaps*
    6. *Anomalies & Statistical Outliers*
    7. *Opportunity Areas & Addressable Upside*
  - Uses objective, calculation-traceable language without speculative assumptions.

- **Interactive Executive Dashboard**:
  - Dynamic sidebar controls: date range sliders, multi-select dimension filters (Region, Category, Segment), and time granularity selector (Daily, Weekly, Monthly, Quarterly, Yearly).
  - Clean executive CSS design system with responsive KPI scorecard.
  - Filtered data table preview with one-click **CSV Data Export**.

---

## 2. Project Structure

```text
data_analytics_dashboard/
│
├── app.py                     # Main Streamlit Executive Dashboard
│
├── data/                      # Dataset repository
│   ├── dataset.csv            # Enterprise B2B Sales & Financials dataset
│   └── sample_ecommerce.xlsx  # Multi-channel E-Commerce Excel dataset
│
├── src/                       # Core Analytics Modules
│   ├── __init__.py
│   ├── data_loader.py         # Ingestion, schema inference, and SQL connector interface
│   ├── data_cleaning.py       # Quality assessment, audit logs, and non-destructive cleaning
│   ├── data_analysis.py       # Descriptive statistics, frequency distributions, time-series
│   ├── kpi_engine.py          # Dynamic KPI framework & calculation audit table
│   ├── visualizations.py      # 8 Bespoke Matplotlib analytical charts
│   └── insights.py            # Automated Business Insights engine (7 categories)
│
├── utils/                     # Formatting & Styling Utilities
│   ├── __init__.py
│   └── helpers.py             # Currency/number formatters, color palette, matplotlib styling
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation & methodology guide
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**:
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

---

## 4. Analytical Methodology

### Non-Destructive Data Pipeline
Data ingestion strictly isolates raw inputs. The cleaning pipeline standardizes categorical strings, imputes missing values with median/mode strategies, validates timestamps, and strips whitespace without mutating the underlying raw file.

### Period-over-Period KPI Formulation
When a date dimension is available, the selected window $[T_0, T_1]$ defines the current period length $\Delta T = T_1 - T_0$. The baseline comparison period is automatically computed as the preceding window $[T_0 - \Delta T, T_0]$.

$$\text{Growth \%} = \frac{\text{Metric}_{\text{Current}} - \text{Metric}_{\text{Previous}}}{|\text{Metric}_{\text{Previous}}|} \times 100$$

### Outlier Detection (IQR Method)
Numerical outliers are evaluated using the standard Tukey Interquartile Range:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
Values exceeding these boundaries are flagged in the Data Quality report and synthesized into the Anomalies section of the Insights Engine.
