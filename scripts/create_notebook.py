import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Hyperlocal Delivery SLA Breach Analysis
### Quick-Commerce Operations Analytics & Predictive Dispatch Optimization (Mumbai, 15 Dark Stores)

**Scenario**: A fast-growing quick-commerce startup (Zepto / Blinkit model) operates 15 dark stores across Mumbai promising **10-minute instant delivery**.
**Business Problem**: The operations leadership needs to diagnose:
1. Which dark stores are breaching the 10-minute SLA most frequently?
2. Where is time being lost (Store packing vs Rider dispatch wait vs Last-mile transit)?
3. What is the financial impact (Refund exposure, Customer churn, Revenue at risk)?
4. Can we proactively predict SLA breaches *at the moment of order placement* to trigger corrective interventions?

---
### Workflow Architecture
- **Step 1**: Environment Setup & Visual Palette Configuration
- **Step 2**: Data Ingestion (Orders, Stores, Riders, Stockouts)
- **Step 3**: Feature Engineering & Bottleneck Attribution
- **Step 4**: Executive SLA Breach Diagnostics
- **Step 5**: Granular Time Component Decomposition
- **Step 6**: Store Leaderboard & Operational Tiers
- **Step 7**: Rider Fleet Dynamics & Productivity
- **Step 8**: Demand Surges & Temporal Heatmaps
- **Step 9**: Monsoon Weather Impact & Statistical Significance
- **Step 10**: Financial Exposure & Churn Quantification
- **Step 11**: Predictive Machine Learning (Gradient Boosting Dispatch Scorer)
- **Step 12**: BI Data Pipeline Export (Star Schema Datasets)
"""))

# Step 1 Code
cells.append(nbf.v4.new_markdown_cell("## Step 1: Imports & Visual Styling Setup"))
cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)

# Paths
BASE_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_DIR = BASE_DIR / 'data'
OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette
BLUE, RED, AMBER, TEAL, PURPLE, GRAY, DARKGRAY, GREEN = (
    '#185FA5', '#E24B4A', '#EF9F27', '#1D9E75', '#534AB7', '#888780', '#444444', '#27AE60'
)
ZONE_COLORS = {'West': BLUE, 'Central': TEAL, 'South': PURPLE, 'East': AMBER}
TIER_COLORS = {'A': TEAL, 'B': AMBER, 'C': RED}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
})
print('Setup complete. Dynamic paths resolved to:', BASE_DIR)
"""))

# Step 2
cells.append(nbf.v4.new_markdown_cell("## Step 2: Load & Inspect Operations Datasets"))
cells.append(nbf.v4.new_code_cell("""orders = pd.read_csv(DATA_DIR / 'orders.csv', parse_dates=['order_datetime', 'order_date'])
stores = pd.read_csv(DATA_DIR / 'dark_stores.csv')
riders = pd.read_csv(DATA_DIR / 'riders.csv')
inv    = pd.read_csv(DATA_DIR / 'inventory_alerts.csv', parse_dates=['date'])

print(f'Orders           : {len(orders):,} rows x {orders.shape[1]} cols')
print(f'Dark Stores      : {len(stores)} stores across {stores["zone"].nunique()} zones')
print(f'Riders           : {len(riders)} active delivery partners')
print(f'Inventory Alerts : {len(inv):,} stockout occurrences')
print(f'Date Range       : {orders["order_date"].min().date()} to {orders["order_date"].max().date()}')
print(f'Total GMV        : INR {orders["order_value"].sum():,.2f}')
"""))

# Step 3
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Data Cleaning & Feature Engineering
- Filter for completed delivered orders
- Engineer temporal blocks and breach severity classes
- Classify root-cause operational bottlenecks
- Quantify refund exposure and churn loss
"""))
cells.append(nbf.v4.new_code_cell("""df = orders[orders['status'] == 'Delivered'].copy()
df['is_sla_breach'] = df['is_sla_breach'].astype(int)

# Merge store tier metadata
df = df.merge(stores[['store_id', 'performance_tier', 'capacity', 'avg_riders']], on='store_id', how='left')

# Hour blocks & temporal tags
df['hour_block'] = pd.cut(
    df['order_hour'],
    bins=[-1, 5, 9, 11, 14, 17, 21, 23],
    labels=['Late Night', 'Morning Peak', 'Mid Morning', 'Lunch Peak', 'Afternoon', 'Evening Peak', 'Night']
)
df['week_number'] = df['order_date'].dt.isocalendar().week.astype(int)
df['day_num']     = df['order_date'].dt.dayofweek

# Breach severity
df['breach_margin_min'] = (df['total_time_min'] - df['sla_target_min']).round(2)
df['breach_severity'] = pd.cut(
    df['total_time_min'],
    bins=[0, 10, 12, 15, 20, 999],
    labels=['On Time (<=10)', 'Slight (10-12)', 'Moderate (12-15)', 'Severe (15-20)', 'Critical (>20)']
)

# Root-cause bottleneck identification
def bottleneck(row):
    if row['is_sla_breach'] == 0:
        return 'No Breach'
    if row['prep_time_min'] > 3.0:
        return 'Slow Packing'
    if row['rider_wait_min'] > 2.0:
        return 'Rider Unavailable'
    if row['delivery_time_min'] > 6.0:
        return 'Last Mile Delay'
    return 'Cumulative Delay'

df['bottleneck'] = df.apply(bottleneck, axis=1)

# Revenue impact metrics
df['refund_exposure']      = np.where(df['is_sla_breach'] == 1, df['order_value'] * 0.15, 0.0)
df['churn_loss_est']       = np.where(df['is_sla_breach'] == 1, df['order_value'] * 0.08, 0.0)
df['total_rev_impact']     = df['refund_exposure'] + df['churn_loss_est']
df['breached_order_value'] = np.where(df['is_sla_breach'] == 1, df['order_value'], 0.0)

print(f'Delivered orders analysed: {len(df):,}')
print(f'Total Revenue at Risk    : INR {df["total_rev_impact"].sum():,.2f}')
"""))

# Step 4
cells.append(nbf.v4.new_markdown_cell("## Step 4: Executive SLA Breach Overview"))
cells.append(nbf.v4.new_code_cell("""overall_breach = df['is_sla_breach'].mean() * 100
total_orders   = len(df)
total_breaches = df['is_sla_breach'].sum()
avg_time       = df['total_time_min'].mean()

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Hyperlocal Delivery SLA Breach - Executive Overview', fontsize=16, fontweight='bold', y=0.98)

# KPI Cards
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
kpis = [
    ('Total Orders', f'{total_orders:,}', BLUE),
    ('SLA Breach Rate', f'{overall_breach:.1f}%', RED),
    ('Avg Delivery Time', f'{avg_time:.1f} min', AMBER),
    ('Revenue at Risk', f'INR {df["total_rev_impact"].sum()/1e5:.1f}L', RED),
]
for i, (label, value, color) in enumerate(kpis):
    y_pos = 0.85 - i * 0.22
    ax1.add_patch(FancyBboxPatch((0.03, y_pos-0.08), 0.94, 0.17, boxstyle='round,pad=0.01',
                  facecolor=color+'22', edgecolor=color, linewidth=1.5))
    ax1.text(0.50, y_pos+0.03, value, ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    ax1.text(0.50, y_pos-0.04, label, ha='center', va='center', fontsize=9, color=DARKGRAY)
ax1.set_title('Key Performance Indicators')

# Severity Donut
ax2 = fig.add_subplot(gs[0, 1])
sev_counts = df['breach_severity'].value_counts().sort_index()
sev_colors = [TEAL, AMBER, '#E8B84B', '#E8593C', RED]
ax2.pie(sev_counts.values, labels=[s.split('(')[0].strip() for s in sev_counts.index],
        colors=sev_colors[:len(sev_counts)], autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}, pctdistance=0.80)
ax2.set_title('Delivery Time Distribution')

# Root Causes
ax3 = fig.add_subplot(gs[0, 2])
btl_counts = df[df['bottleneck'] != 'No Breach']['bottleneck'].value_counts()
ax3.bar(btl_counts.index, btl_counts.values, color=[RED, AMBER, PURPLE, BLUE][:len(btl_counts)], width=0.55)
ax3.tick_params(axis='x', rotation=20)
ax3.set_ylabel('Breach Count')
ax3.set_title('Root Cause of SLA Breaches')

# Monthly Trend
ax4 = fig.add_subplot(gs[1, 0])
monthly = df.groupby(['month','month_name']).agg(
    total=('is_sla_breach','count'), breaches=('is_sla_breach','sum')
).reset_index().sort_values('month')
monthly['breach_rate'] = monthly['breaches'] / monthly['total'] * 100
ax4.bar(monthly['month_name'], monthly['total']/1000, color=BLUE+'66', width=0.6, label='Orders (K)')
ax4_twin = ax4.twinx()
ax4_twin.plot(monthly['month_name'], monthly['breach_rate'], color=RED, marker='o', linewidth=2.5, label='Breach %')
ax4_twin.set_ylabel('Breach Rate (%)', color=RED)
ax4.set_ylabel('Orders (thousands)')
ax4.set_title('Monthly Orders vs SLA Breach Rate')

# Zone Breach Rate
ax5 = fig.add_subplot(gs[1, 1])
zone_stats = df.groupby('zone').agg(breach_rate=('is_sla_breach','mean')).reset_index()
zone_stats['breach_rate'] *= 100
ax5.bar(zone_stats['zone'], zone_stats['breach_rate'], color=[ZONE_COLORS[z] for z in zone_stats['zone']], width=0.55)
ax5.axhline(overall_breach, color=GRAY, linestyle='--', linewidth=1.5, label=f'Avg: {overall_breach:.1f}%')
ax5.set_ylabel('SLA Breach Rate (%)')
ax5.set_title('Breach Rate by Zone')
ax5.legend()

# Weekday vs Weekend
ax6 = fig.add_subplot(gs[1, 2])
wd_stats = df.groupby('is_weekend').agg(breach_rate=('is_sla_breach','mean')).reset_index()
wd_stats['breach_rate'] *= 100
ax6.bar(['Weekday', 'Weekend'], wd_stats['breach_rate'], color=[BLUE, PURPLE], width=0.45)
ax6.set_ylabel('SLA Breach Rate (%)')
ax6.set_title('Weekday vs Weekend Breach Rate')

plt.tight_layout()
plt.show()
"""))

# Step 5
cells.append(nbf.v4.new_markdown_cell("## Step 5: Time Component Breakdown (Prep vs Rider Wait vs Last Mile)"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Time Component Analysis - Where Is Time Being Lost?', fontsize=14, fontweight='bold')

# Component Comparison
comp_data = df.groupby('is_sla_breach')[['prep_time_min','rider_wait_min','delivery_time_min']].mean()
comp_data.index = ['On Time', 'Breached']
comp_data.columns = ['Prep', 'Rider Wait', 'Last Mile']
comp_data.plot(kind='bar', ax=axes[0], color=[TEAL, AMBER, BLUE], edgecolor='white', width=0.5)
axes[0].axhline(10, color=RED, linestyle='--', linewidth=1.5, label='10-min SLA Target')
axes[0].set_xticklabels(['On Time', 'Breached'], rotation=0)
axes[0].set_ylabel('Minutes')
axes[0].set_title('Avg Components (On Time vs Breached)')
axes[0].legend()

# Store Tier Bottlenecks
tier_btl = df[df['bottleneck']!='No Breach'].groupby(['performance_tier','bottleneck']).size().unstack(fill_value=0)
tier_btl_pct = tier_btl.div(tier_btl.sum(axis=1), axis=0) * 100
tier_btl_pct.plot(kind='bar', ax=axes[1], edgecolor='white', width=0.6, color=[RED, AMBER, PURPLE, BLUE])
axes[1].set_xticklabels(['Tier A (Best)','Tier B','Tier C (Worst)'], rotation=0)
axes[1].set_ylabel('% of Breaches')
axes[1].set_title('Root Causes by Store Tier')
axes[1].legend(title='Cause', fontsize=8)

# Prep Time by Category
cat_prep = df.groupby('category')[['prep_time_min','is_sla_breach']].mean()
cat_prep['is_sla_breach'] *= 100
cat_prep = cat_prep.sort_values('prep_time_min', ascending=True)
axes[2].barh(cat_prep.index, cat_prep['prep_time_min'], color=BLUE, height=0.55)
axes[2].axvline(3.0, color=AMBER, linestyle='--', linewidth=1.5, label='3 min Alert Threshold')
axes[2].set_xlabel('Avg Prep Time (min)')
axes[2].set_title('Prep Time by Product Category')
axes[2].legend()

plt.tight_layout()
plt.show()
"""))

# Step 6
cells.append(nbf.v4.new_markdown_cell("## Step 6: Store-Level Performance Ranking"))
cells.append(nbf.v4.new_code_cell("""store_perf = df.groupby(['store_id','store_name','zone','performance_tier']).agg(
    total_orders   = ('is_sla_breach', 'count'),
    sla_breaches   = ('is_sla_breach', 'sum'),
    breach_rate    = ('is_sla_breach', 'mean'),
    avg_total_time = ('total_time_min', 'mean'),
    avg_prep       = ('prep_time_min',  'mean'),
    avg_rider_wait = ('rider_wait_min', 'mean'),
    avg_last_mile  = ('delivery_time_min', 'mean'),
    rev_at_risk    = ('total_rev_impact','sum'),
).reset_index()
store_perf['breach_rate'] *= 100
store_perf = store_perf.sort_values('breach_rate', ascending=False).reset_index(drop=True)
store_perf['rank'] = range(1, len(store_perf)+1)

display(store_perf[['rank','store_name','zone','performance_tier','breach_rate','avg_total_time','rev_at_risk']].round(2))
"""))

# Step 7 & 8
cells.append(nbf.v4.new_markdown_cell("## Step 7 & 8: Rider Analytics & Demand Pattern Heatmaps"))
cells.append(nbf.v4.new_code_cell("""hourly = df.groupby('order_hour').agg(
    orders=('is_sla_breach','count'), breach_rate=('is_sla_breach','mean')
).reset_index()
hourly['breach_rate'] *= 100

heatmap_data = df.groupby(['day_of_week','order_hour'])['is_sla_breach'].mean().unstack() * 100
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heatmap_data = heatmap_data.reindex(day_order)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Demand Surges and SLA Breach Heatmap', fontsize=14, fontweight='bold')

# Hourly Orders vs Breach Rate
axes[0].bar(hourly['order_hour'], hourly['orders'],
            color=[RED if h in [8,9,12,13,19,20,21] else BLUE for h in hourly['order_hour']], width=0.7, alpha=0.75)
ax_twin = axes[0].twinx()
ax_twin.plot(hourly['order_hour'], hourly['breach_rate'], 'o-', color=RED, linewidth=2.5)
ax_twin.set_ylabel('Breach Rate (%)', color=RED)
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Order Volume')
axes[0].set_title('Hourly Volume vs Breach Rate (Red = Peaks)')

# Heatmap
sns.heatmap(heatmap_data, ax=axes[1], cmap='RdYlGn_r', fmt='.0f', annot=False, linewidths=0.2, cbar_kws={'label': 'Breach %'})
axes[1].set_xlabel('Hour of Day')
axes[1].set_title('Breach Heatmap (Day x Hour)')

plt.tight_layout()
plt.show()
"""))

# Step 9 & 10
cells.append(nbf.v4.new_markdown_cell("## Step 9 & 10: Monsoon Weather Impact & Revenue Exposure"))
cells.append(nbf.v4.new_code_cell("""rain_stats = df.groupby('is_rain').agg(
    breach_rate=('is_sla_breach','mean'), avg_time=('total_time_min','mean'), avg_last_mile=('delivery_time_min','mean')
).reset_index()
rain_stats['breach_rate'] *= 100
rain_stats.index = ['Clear Day', 'Rainy Day']
display(rain_stats.round(2))

total_gmv          = df['order_value'].sum()
breached_gmv       = df[df['is_sla_breach']==1]['order_value'].sum()
total_refund_exp   = df['refund_exposure'].sum()
total_churn_loss   = df['churn_loss_est'].sum()
total_impact       = df['total_rev_impact'].sum()

print(f'Total GMV              : INR {total_gmv/1e7:.2f} Cr')
print(f'GMV on Breached Orders : INR {breached_gmv/1e7:.2f} Cr ({breached_gmv/total_gmv*100:.1f}%)')
print(f'Estimated Refund Exp.  : INR {total_refund_exp/1e5:.1f} Lakhs')
print(f'Estimated Churn Loss   : INR {total_churn_loss/1e5:.1f} Lakhs')
print(f'Total Revenue at Risk  : INR {total_impact/1e5:.1f} Lakhs')
"""))

# Step 11
cells.append(nbf.v4.new_markdown_cell("## Step 11: Machine Learning Predictive Model (Pre-Dispatch SLA Risk Scorer)"))
cells.append(nbf.v4.new_code_cell("""model_features = ['order_hour', 'is_weekend', 'is_peak_hour', 'is_rain', 'month', 'n_items', 'order_value']

df_model = df.copy()
for col in ['category', 'zone', 'performance_tier']:
    le = LabelEncoder()
    df_model[f'{col}_enc'] = le.fit_transform(df_model[col].astype(str))
    model_features.append(f'{col}_enc')

le_block = LabelEncoder()
df_model['hour_block_enc'] = le_block.fit_transform(df_model['hour_block'].astype(str))
model_features.append('hour_block_enc')

X = df_model[model_features].fillna(0)
y = df_model['is_sla_breach']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

gb = GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, subsample=0.85, random_state=42)
gb.fit(X_train, y_train)

gb_proba = gb.predict_proba(X_test)[:, 1]
gb_preds = (gb_proba >= 0.50).astype(int)
gb_auc   = roc_auc_score(y_test, gb_proba)

print(f'Gradient Boosting Pre-Dispatch ROC-AUC: {gb_auc:.4f}')
print(classification_report(y_test, gb_preds, target_names=['On Time', 'Breach']))
"""))

# Step 12
cells.append(nbf.v4.new_markdown_cell("""## Step 12: Summary of Generated Power BI Star Schema Tables
All processed fact and dimension tables are exported to `outputs/`:
1. `orders_scored.csv` - Fact table with ML pre-dispatch breach scores and risk bands
2. `store_performance.csv` - Dark store dimension and aggregated leaderboard
3. `daily_kpi.csv` - Daily time series for trendline analysis
4. `rider_performance.csv` - Fleet productivity and performance ratings
5. `inventory_alerts_clean.csv` - Stockout log for availability correlation
"""))

nb['cells'] = cells

output_path = Path('notebooks/hyperlocal_analysis.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Generated Jupyter Notebook successfully at:', output_path)
