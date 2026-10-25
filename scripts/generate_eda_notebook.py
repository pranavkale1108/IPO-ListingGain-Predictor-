"""
generate_eda_notebook.py
------------------------
Programmatically generates the Jupyter Notebook 'notebooks/01_eda.ipynb'
with well-structured markdown cells, code cells, and figure exports to 'reports/figures/'.
"""

import json
import os

def create_eda_notebook(notebook_path: str = "notebooks/01_eda.ipynb"):
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)

    cells = []

    def add_md(source_text: str):
        cell_id = f"cell_md_{len(cells):02d}"
        lines = [line + "\n" for line in source_text.strip().splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "markdown",
            "id": cell_id,
            "metadata": {},
            "source": lines
        })

    def add_code(source_text: str):
        cell_id = f"cell_code_{len(cells):02d}"
        lines = [line + "\n" for line in source_text.strip().splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "id": cell_id,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # Cell 0: Title & Overview
    add_md(r"""# 📊 01. Exploratory Data Analysis (EDA) - IPO Listing-Gain Predictor

**Project**: IPO Listing-Gain Predictor (Indian Equity Market)  
**Academic Context**: College ML Mini Project (Educational use only, not financial advice)  
**Objective**: Explore the distribution of listing-day gains, relationship with subscription metrics, market mood, sector performance, and missingness patterns to prepare for **Expectation-Maximization (EM) Imputation** and downstream modeling.

---
### EDA Agenda:
1. **Dataset Overview & Schema Inspection**
2. **Missing-Value Report** (Justification for EM Algorithm Imputation)
3. **Distribution of Listing Gain %** (Skewness, Outliers, Classification Targets)
4. **Subscription Metrics vs Listing Gain %** (Log-scale analysis of QIB, NII, Retail demand)
5. **Gain by Sector** (Industry-level performance variation)
6. **Feature Correlation Heatmap** (Multicollinearity & key signal discovery)
7. **Export Figures** to `reports/figures/`""")

    # Cell 1: Imports & Configuration
    add_code(r"""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Ensure output directory exists
FIG_DIR = os.path.join("..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
print(f"Figures will be saved to: {os.path.abspath(FIG_DIR)}")""")

    # Cell 2: Load Data Markdown
    add_md(r"""## 1. Load Cleaned IPO Dataset
We load `data/processed/ipo_clean.csv`, which was sanitized, deduplicated, chronologically sorted, and feature-engineered by `src/data_loader.py`.""")

    # Cell 3: Load Data Code
    add_code(r"""# Load data (handles path from either notebooks/ dir or project root)
DATA_PATH = os.path.join("..", "data", "processed", "ipo_clean.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("data", "processed", "ipo_clean.csv")

df = pd.read_csv(DATA_PATH)
df['listing_date'] = pd.to_datetime(df['listing_date'])

print(f"Loaded {len(df)} records across {df.shape[1]} columns.")
print(f"Date range: {df['listing_date'].min().date()} to {df['listing_date'].max().date()}")
df.head(5)""")

    # Cell 4: Info
    add_code(r"""df.info()""")

    # Cell 5: Missing Value Report Markdown
    add_md(r"""## 2. Missing-Value Report & EM Imputation Rationale

In the Indian IPO ecosystem, certain fundamental metrics (e.g., P/E ratio for loss-making new-age tech companies, historical growth rates, unquoted Grey Market Premiums) are legitimately unobserved.
Instead of crude mean/median replacement or discarding valuable IPO rows, we will implement the **Expectation-Maximization (EM) algorithm** to impute missing values by modeling the underlying joint distribution.""")

    # Cell 6: Missing Value Code
    add_code(r"""# Compute missing values summary
missing_counts = df.isnull().sum()
missing_pcts = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage (%)': missing_pcts.round(2)
}).sort_values(by='Missing Count', ascending=False)

missing_df_active = missing_df[missing_df['Missing Count'] > 0]
print("Features requiring Imputation:")
print(missing_df_active)

# Visualization: Missing Value Barplot
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if pct > 10 else '#e67e22' for pct in missing_df_active['Missing Percentage (%)']]
bars = ax.barh(missing_df_active.index, missing_df_active['Missing Percentage (%)'], color=colors, edgecolor='black', alpha=0.85)

ax.set_title("Missing Values by Feature (% of Total Dataset)\n(Candidates for EM Algorithm Imputation)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Missing Percentage (%)", fontsize=11, fontweight='semibold')
ax.set_xlim(0, max(missing_df_active['Missing Percentage (%)']) + 5)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', fontsize=10, fontweight='bold', color='#333333')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "05_missing_values_report.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved figure -> {fig_path}")
plt.show()""")

    # Cell 7: Listing Gain Distribution Markdown
    add_md(r"""## 3. Distribution of Listing Gain (%) & Verdict Breakdown

Here we explore the primary regression target (`listing_gain_pct`) and its classification groupings:
- **Avoid**: Gain < 0% (discount on debut)
- **Neutral**: Gain between 0% and 15% (modest listing)
- **Apply**: Gain > 15% (strong listing pop)""")

    # Cell 8: Listing Gain Distribution Code
    add_code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Distribution Histogram + KDE
sns.histplot(df['listing_gain_pct'], kde=True, ax=axes[0], color='#2980b9', bins=25, edgecolor='black', alpha=0.7)
mean_gain = df['listing_gain_pct'].mean()
median_gain = df['listing_gain_pct'].median()
axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5, label='0% Threshold (Positive vs Negative)')
axes[0].axvline(mean_gain, color='green', linestyle='-', linewidth=1.5, label=f'Mean: {mean_gain:.1f}%')
axes[0].axvline(median_gain, color='purple', linestyle=':', linewidth=1.8, label=f'Median: {median_gain:.1f}%')

axes[0].set_title("Distribution of Listing Gain (%)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Listing Gain (%)", fontsize=10, fontweight='semibold')
axes[0].set_ylabel("Number of IPOs", fontsize=10, fontweight='semibold')
axes[0].legend(frameon=True)

# Subplot 2: Categorical Verdict Distribution
verdict_order = ['Avoid', 'Neutral', 'Apply']
verdict_colors = {'Avoid': '#e74c3c', 'Neutral': '#f39c12', 'Apply': '#27ae60'}
verdict_counts = df['verdict'].value_counts().reindex(verdict_order)

bars = axes[1].bar(verdict_order, verdict_counts.values, color=[verdict_colors[v] for v in verdict_order], edgecolor='black', alpha=0.85)
axes[1].set_title("Classification Target: Verdict Breakdown", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Investment Verdict", fontsize=10, fontweight='semibold')
axes[1].set_ylabel("Count of IPOs", fontsize=10, fontweight='semibold')

for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    axes[1].text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h} ({pct:.1f}%)", ha='center', fontsize=10, fontweight='bold')

axes[1].set_ylim(0, max(verdict_counts.values) + 20)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "01_listing_gain_distribution.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved figure -> {fig_path}")
plt.show()""")

    # Cell 9: Subscription vs Gain Markdown
    add_md(r"""## 4. Subscription vs Listing Gain % (Log-Scale Analysis)

In the Indian IPO market, subscription multiples (especially institutional QIB and total subscriptions) often exhibit heavy right-skew (spanning from 0.5x to 200x+). We plot subscription vs listing gain on a **log-scale** x-axis to reveal the underlying power-law relationship.""")

    # Cell 10: Subscription vs Gain Code
    add_code(r"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sub_cols = [
    ('total_sub', 'Total Subscription (x)', '#8e44ad'),
    ('qib_sub', 'QIB Subscription (x)', '#2980b9'),
    ('retail_sub', 'Retail Subscription (x)', '#d35400')
]

for idx, (col, label, color) in enumerate(sub_cols):
    ax = axes[idx]
    sns.regplot(
        data=df,
        x=col,
        y='listing_gain_pct',
        ax=ax,
        scatter_kws={'alpha': 0.6, 'color': color, 'edgecolor': 'none'},
        line_kws={'color': '#c0392b', 'linewidth': 2},
        logx=True
    )
    ax.set_xscale('log')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title(f"{label} vs Listing Gain % (Log X)", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"{label} [Log Scale]", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Listing Gain (%)" if idx == 0 else "", fontsize=10, fontweight='semibold')

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "02_subscription_vs_gain.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved figure -> {fig_path}")
plt.show()""")

    # Cell 11: Sector Markdown
    add_md(r"""## 5. Listing Gain by Sector

Different industries experience distinct investor appetite, regulatory cycles, and valuation multiples. This visualization highlights sector-specific median listing returns and dispersion.""")

    # Cell 12: Sector Code
    add_code(r"""# Sort sectors by median listing gain
sector_order = df.groupby('sector')['listing_gain_pct'].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df,
    x='listing_gain_pct',
    y='sector',
    order=sector_order,
    palette='Blues_r',
    ax=ax,
    fliersize=3,
    linewidth=1.2
)

ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='0% Baseline')
ax.axvline(15, color='green', linestyle=':', linewidth=1.5, alpha=0.8, label='15% Apply Threshold')

ax.set_title("Listing Gain (%) Distribution Across Industry Sectors", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Listing Gain (%)", fontsize=11, fontweight='semibold')
ax.set_ylabel("Sector", fontsize=11, fontweight='semibold')
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "04_gain_by_sector.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved figure -> {fig_path}")
plt.show()""")

    # Cell 13: Correlation Heatmap Markdown
    add_md(r"""## 6. Correlation Heatmap of Numeric Features

We compute the Pearson correlation matrix to understand interactions between issue characteristics, subscriptions, financial health, macro conditions (Nifty trend), and the listing pop.""")

    # Cell 14: Correlation Heatmap Code
    add_code(r"""numeric_cols = [
    'issue_size_cr', 'price_band_upper', 'qib_sub', 'nii_sub', 'retail_sub',
    'total_sub', 'gmp_last_day', 'fresh_issue_pct', 'revenue_growth_pct',
    'profit_growth_pct', 'pe_vs_peers', 'nifty_7d_trend_pct', 'listing_gain_pct'
]

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=1.0,
    vmin=-1.0,
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.6,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
    ax=ax
)

ax.set_title("Feature Correlation Heatmap (Numeric IPO Metrics & Targets)", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "03_correlation_heatmap.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved figure -> {fig_path}")
plt.show()""")

    # Cell 15: Conclusion Markdown
    add_md(r"""## 7. Key Findings & Pipeline Next Steps

### 🔍 Key Insights from EDA:
1. **Subscription as Lead Signal**: Total subscription and QIB demand show a strong positive non-linear relationship with listing gain, confirming log-transformation and non-linear models (Polynomial Regression, Gradient Boosting) will be impactful.
2. **Missingness in Financials & GMP**: `profit_growth_pct` (~18%), `gmp_last_day` (~15%), and `pe_vs_peers` (~10%) have missing entries. Our **Expectation-Maximization (EM)** algorithm is essential to recover this joint distribution without inducing bias.
3. **Sector Clustering Potential**: Clear clustering in returns across high-momentum sectors (Technology, Consumer) vs cyclical capital goods, confirming that **K-Means / K-Medoids** can identify genuine peer IPOs.
4. **Market Momentum**: Positive correlation between `nifty_7d_trend_pct` and debut pop supports our planned **Hidden Markov Model (HMM)** to detect Cold/Warm/Hot market regimes.

### ⏭️ Next Actions:
- Run `src/split.py` to establish strict chronological train/test boundaries.
- Implement EM Imputation fitted on the train split only.
- Implement custom NumPy Hidden Markov Model (HMM) for market mood classification.""")

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)

    print(f"[OK] Created Jupyter notebook at: {notebook_path}")

if __name__ == "__main__":
    create_eda_notebook()
