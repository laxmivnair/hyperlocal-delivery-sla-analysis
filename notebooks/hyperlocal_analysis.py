"""
=============================================================================
PROJECT 4: HYPERLOCAL DELIVERY SLA BREACH ANALYSIS
=============================================================================
Company scenario : A quick-commerce startup (Zepto/Blinkit style) running
                   15 dark stores across Mumbai, promising 10-minute delivery.
                   The ops head needs to know: which stores breach SLA most,
                   why it happens, and what it costs the business.

Your role        : Operations Data Analyst — find the root causes, quantify
                   the revenue impact, and build a dashboard the ops team
                   uses every morning.

Unique angle     : Quick-commerce is the hottest sector in India (2024-25).
                   Zepto, Blinkit, Swiggy Instamart all hire ops analysts.
                   This project speaks DIRECTLY to their business problem.

Tools            : Python (Pandas, Scikit-learn, Scipy, Seaborn), SQL, Power BI
Dataset          : 81,494 orders | 15 dark stores | 350 riders | 6 months
=============================================================================

WORKFLOW OVERVIEW:
  Step 2  -> Load & inspect all 4 tables
  Step 3  -> Data cleaning & feature engineering
  Step 4  -> SLA breach overview (the headline KPIs)
  Step 5  -> Time component deep-dive (prep vs wait vs last mile)
  Step 6  -> Store-level performance ranking
  Step 7  -> Rider performance analysis
  Step 8  -> Peak hour & demand pattern analysis
  Step 9  -> Rain/weather impact analysis
  Step 10 -> Revenue & business impact quantification
  Step 11 -> Predictive model (will this order breach SLA?)
  Step 12 -> Export 5 CSVs for Power BI
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
from pathlib import Path

# Ensure UTF-8 output encoding across Windows / Linux consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# Define dynamic paths based on repository root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette — clean corporate style ────────────────────────────────────
BLUE     = "#185FA5"
RED      = "#E24B4A"
AMBER    = "#EF9F27"
TEAL     = "#1D9E75"
PURPLE   = "#534AB7"
GRAY     = "#888780"
DARKGRAY = "#444444"
GREEN    = "#27AE60"

ZONE_COLORS = {"West": BLUE, "Central": TEAL, "South": PURPLE, "East": AMBER}
TIER_COLORS = {"A": TEAL, "B": AMBER, "C": RED}
RISK_COLORS = {"Low": TEAL, "Medium": AMBER, "High": RED}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titleweight":  "bold",
})


def run_pipeline():
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2 — LOAD & INSPECT ALL 4 TABLES
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 2: Loading all tables")
    print("="*65)

    orders_path = DATA_DIR / "orders.csv"
    stores_path = DATA_DIR / "dark_stores.csv"
    riders_path = DATA_DIR / "riders.csv"
    inv_path    = DATA_DIR / "inventory_alerts.csv"

    if not orders_path.exists():
        raise FileNotFoundError(f"Missing required dataset: {orders_path}. Run generate_data.py first.")

    orders = pd.read_csv(orders_path, parse_dates=["order_datetime", "order_date"])
    stores = pd.read_csv(stores_path)
    riders = pd.read_csv(riders_path)
    inv    = pd.read_csv(inv_path, parse_dates=["date"])

    print(f"orders           : {len(orders):,} rows x {orders.shape[1]} cols")
    print(f"dark_stores      : {len(stores)} rows")
    print(f"riders           : {len(riders)} rows")
    print(f"inventory_alerts : {len(inv):,} rows")
    print(f"\nDate range: {orders['order_date'].min().date()} -> {orders['order_date'].max().date()}")
    print(f"Total GMV  : INR {orders['order_value'].sum():,.0f}")

    # Quick status breakdown
    print(f"\nOrder status:")
    print(orders["status"].value_counts().to_string())

    print(f"\nOverall SLA breach rate: {orders['is_sla_breach'].mean()*100:.1f}%")
    print(f"Stores x Zones:\n{stores.groupby('zone')['store_id'].count().to_string()}")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3 — DATA CLEANING & FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 3: Cleaning & Feature Engineering")
    print("="*65)

    # ── 3A: Work only with completed delivered orders ─────────────────────────────
    df = orders[orders["status"] == "Delivered"].copy()
    df["is_sla_breach"] = df["is_sla_breach"].astype(int)
    print(f"Working dataset (Delivered only): {len(df):,} orders")

    # ── 3B: Merge store performance tier ─────────────────────────────────────────
    df = df.merge(
        stores[["store_id","performance_tier","capacity","avg_riders"]],
        on="store_id", how="left"
    )

    # ── 3C: Time-based features ───────────────────────────────────────────────────
    df["hour_block"] = pd.cut(
        df["order_hour"],
        bins=[-1, 5, 9, 11, 14, 17, 21, 23],
        labels=["Late Night","Morning Peak","Mid Morning",
                "Lunch Peak","Afternoon","Evening Peak","Night"]
    )

    df["week_number"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["day_num"]     = df["order_date"].dt.dayofweek   # 0=Mon

    # ── 3D: SLA breach severity ───────────────────────────────────────────────────
    df["breach_margin_min"] = (df["total_time_min"] - df["sla_target_min"]).round(2)

    df["breach_severity"] = pd.cut(
        df["total_time_min"],
        bins=[0, 10, 12, 15, 20, 999],
        labels=["On Time (<=10)", "Slight (10-12)", "Moderate (12-15)",
                "Severe (15-20)", "Critical (>20)"]
    )

    # ── 3E: Bottleneck label — which component caused the breach ──────────────────
    def bottleneck(row):
        if row["is_sla_breach"] == 0:
            return "No Breach"
        if row["prep_time_min"] > 3.0:
            return "Slow Packing"
        if row["rider_wait_min"] > 2.0:
            return "Rider Unavailable"
        if row["delivery_time_min"] > 6.0:
            return "Last Mile Delay"
        return "Cumulative Delay"

    df["bottleneck"] = df.apply(bottleneck, axis=1)

    # ── 3F: Revenue impact per order ─────────────────────────────────────────────
    df["refund_exposure"]      = np.where(df["is_sla_breach"]==1, df["order_value"]*0.15, 0.0)
    df["churn_loss_est"]       = np.where(df["is_sla_breach"]==1, df["order_value"]*0.08, 0.0)
    df["total_rev_impact"]     = df["refund_exposure"] + df["churn_loss_est"]
    df["breached_order_value"] = np.where(df["is_sla_breach"]==1, df["order_value"], 0.0)

    print(f"\nFeatures engineered:")
    print(f"  breach_severity : {df['breach_severity'].value_counts().to_dict()}")
    print(f"  bottleneck types: {df['bottleneck'].value_counts().to_dict()}")
    print(f"\nTotal refund exposure : INR {df['refund_exposure'].sum():,.0f}")
    print(f"Total churn loss est  : INR {df['churn_loss_est'].sum():,.0f}")
    print(f"Total revenue at risk : INR {df['total_rev_impact'].sum():,.0f}")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 4 — SLA BREACH OVERVIEW (THE HEADLINE CHARTS)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 4: SLA Breach Overview")
    print("="*65)

    overall_breach = df["is_sla_breach"].mean() * 100
    total_orders   = len(df)
    total_breaches = df["is_sla_breach"].sum()
    avg_time       = df["total_time_min"].mean()

    print(f"Overall breach rate  : {overall_breach:.1f}%")
    print(f"Total orders         : {total_orders:,}")
    print(f"Total breaches       : {total_breaches:,}")
    print(f"Avg delivery time    : {avg_time:.1f} min")

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)
    fig.suptitle("Hyperlocal Delivery SLA Breach - Executive Overview",
                 fontsize=16, fontweight="bold", y=1.01)

    # ── Plot 1: KPI summary as text cards ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    kpis = [
        ("Total Orders",       f"{total_orders:,}",            BLUE),
        ("SLA Breach Rate",    f"{overall_breach:.1f}%",       RED),
        ("Avg Delivery Time",  f"{avg_time:.1f} min",          AMBER),
        ("Revenue at Risk",    f"INR {df['total_rev_impact'].sum()/1e5:.1f}L", RED),
    ]
    for i, (label, value, color) in enumerate(kpis):
        y_pos = 0.85 - i * 0.22
        ax1.add_patch(FancyBboxPatch((0.03, y_pos-0.08), 0.94, 0.17,
                      boxstyle="round,pad=0.01", facecolor=color+"22",
                      edgecolor=color, linewidth=1.5))
        ax1.text(0.50, y_pos+0.03, value, ha="center", va="center",
                 fontsize=14, fontweight="bold", color=color)
        ax1.text(0.50, y_pos-0.04, label, ha="center", va="center",
                 fontsize=9, color=DARKGRAY)
    ax1.set_title("Key Performance Indicators")

    # ── Plot 2: Breach severity donut ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sev_counts = df["breach_severity"].value_counts().sort_index()
    sev_colors = [TEAL, AMBER, "#E8B84B", "#E8593C", RED]
    wedges, texts, autotexts = ax2.pie(
        sev_counts.values,
        labels=[s.split("(")[0].strip() for s in sev_counts.index],
        colors=sev_colors[:len(sev_counts)],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.80
    )
    for at in autotexts: at.set_fontsize(9)
    ax2.set_title("Delivery Time Distribution")

    # ── Plot 3: Bottleneck analysis ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    btl_counts = df[df["bottleneck"] != "No Breach"]["bottleneck"].value_counts()
    btl_colors = [RED, AMBER, PURPLE, BLUE][:len(btl_counts)]
    bars = ax3.bar(btl_counts.index, btl_counts.values,
                   color=btl_colors, width=0.55)
    ax3.tick_params(axis="x", rotation=20)
    ax3.set_ylabel("Breach Count")
    ax3.set_title("Root Cause of SLA Breaches")
    for bar, val in zip(bars, btl_counts.values):
        pct = val / total_breaches * 100
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                 f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=9)

    # ── Plot 4: Monthly breach trend ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    monthly = df.groupby(["month","month_name"]).agg(
        total=("is_sla_breach","count"),
        breaches=("is_sla_breach","sum"),
        gmv=("order_value","sum")
    ).reset_index().sort_values("month")
    monthly["breach_rate"] = monthly["breaches"] / monthly["total"] * 100
    ax4.bar(monthly["month_name"], monthly["total"]/1000, color=BLUE+"66",
            width=0.6, label="Total Orders (K)")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(monthly["month_name"], monthly["breach_rate"],
                  color=RED, marker="o", linewidth=2.5, markersize=7, label="Breach %")
    ax4_twin.spines["right"].set_visible(True)
    ax4_twin.spines["top"].set_visible(False)
    ax4_twin.set_ylabel("Breach Rate (%)", color=RED)
    ax4_twin.tick_params(axis="y", colors=RED)
    ax4.set_ylabel("Orders (thousands)")
    ax4.set_title("Monthly Orders vs SLA Breach Rate")
    ax4.tick_params(axis="x", rotation=30)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")

    # ── Plot 5: SLA breach by zone ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    zone_stats = df.groupby("zone").agg(
        breach_rate=("is_sla_breach","mean"),
        orders=("is_sla_breach","count"),
        avg_time=("total_time_min","mean")
    ).reset_index()
    zone_stats["breach_rate"] *= 100
    z_colors = [ZONE_COLORS[z] for z in zone_stats["zone"]]
    bars = ax5.bar(zone_stats["zone"], zone_stats["breach_rate"],
                   color=z_colors, width=0.55)
    ax5.axhline(overall_breach, color=GRAY, linestyle="--", linewidth=1.5,
                label=f"Overall avg: {overall_breach:.1f}%")
    for bar, val in zip(bars, zone_stats["breach_rate"]):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax5.set_ylabel("SLA Breach Rate (%)")
    ax5.set_title("Breach Rate by Zone")
    ax5.legend(fontsize=9)

    # ── Plot 6: Weekend vs weekday ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    wd_stats = df.groupby("is_weekend").agg(
        breach_rate=("is_sla_breach","mean"),
        avg_time=("total_time_min","mean"),
        orders=("is_sla_breach","count")
    ).reset_index()
    wd_stats["breach_rate"] *= 100
    wd_labels = ["Weekday", "Weekend"]
    wd_colors = [BLUE, PURPLE]
    bars = ax6.bar(wd_labels, wd_stats["breach_rate"], color=wd_colors, width=0.45)
    for bar, val, cnt in zip(bars, wd_stats["breach_rate"], wd_stats["orders"]):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val:.1f}%\n({cnt:,} orders)", ha="center", fontsize=11)
    ax6.set_ylabel("SLA Breach Rate (%)")
    ax6.set_title("Weekday vs Weekend Breach Rate")
    ax6.set_ylim(0, wd_stats["breach_rate"].max()*1.3)

    plt.tight_layout()
    chart1_path = OUTPUTS_DIR / "01_sla_overview.png"
    plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Chart saved: {chart1_path}")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 5 — TIME COMPONENT DEEP DIVE
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 5: Time Component Deep-Dive")
    print("="*65)

    print("Average time breakdown (ALL orders):")
    print(f"  Prep time        : {df['prep_time_min'].mean():.2f} min")
    print(f"  Rider wait       : {df['rider_wait_min'].mean():.2f} min")
    print(f"  Last mile        : {df['delivery_time_min'].mean():.2f} min")
    print(f"  TOTAL            : {df['total_time_min'].mean():.2f} min")
    print(f"  SLA target       : {df['sla_target_min'].iloc[0]} min")

    print("\nBreached orders time breakdown:")
    breached = df[df["is_sla_breach"]==1]
    print(f"  Prep time        : {breached['prep_time_min'].mean():.2f} min (+{breached['prep_time_min'].mean()-df['prep_time_min'].mean():.2f})")
    print(f"  Rider wait       : {breached['rider_wait_min'].mean():.2f} min (+{breached['rider_wait_min'].mean()-df['rider_wait_min'].mean():.2f})")
    print(f"  Last mile        : {breached['delivery_time_min'].mean():.2f} min (+{breached['delivery_time_min'].mean()-df['delivery_time_min'].mean():.2f})")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Time Component Analysis - Where Is Time Being Lost?",
                 fontsize=14, fontweight="bold")

    # ── Plot 1: Stacked time components — breached vs on-time ────────────────────
    comp_data = df.groupby("is_sla_breach")[
        ["prep_time_min","rider_wait_min","delivery_time_min"]
    ].mean()
    comp_data.index = ["On Time", "Breached"]
    comp_data.columns = ["Prep", "Rider Wait", "Last Mile"]
    comp_data.plot(kind="bar", ax=axes[0], color=[TEAL, AMBER, BLUE],
                   edgecolor="white", width=0.5)
    axes[0].axhline(10, color=RED, linestyle="--", linewidth=1.5, label="10-min SLA")
    axes[0].set_xticklabels(["On Time", "Breached"], rotation=0)
    axes[0].set_ylabel("Minutes")
    axes[0].set_title("Avg Time Components\n(Breached vs On Time)")
    axes[0].legend(fontsize=9)

    # ── Plot 2: Bottleneck distribution by store tier ─────────────────────────────
    tier_btl = df[df["bottleneck"]!="No Breach"].groupby(
        ["performance_tier","bottleneck"]
    ).size().unstack(fill_value=0)
    tier_btl_pct = tier_btl.div(tier_btl.sum(axis=1), axis=0) * 100
    tier_btl_pct.plot(kind="bar", ax=axes[1], edgecolor="white", width=0.6,
                      color=[RED, AMBER, PURPLE, BLUE])
    axes[1].set_xticklabels(["Tier A (Best)","Tier B","Tier C (Worst)"], rotation=0)
    axes[1].set_ylabel("% of Breaches")
    axes[1].set_title("Breach Root Cause by Store Tier")
    axes[1].legend(title="Cause", fontsize=8, bbox_to_anchor=(1,1))

    # ── Plot 3: Prep time by category (explains category delays) ─────────────────
    cat_prep = df.groupby("category")[["prep_time_min","is_sla_breach"]].mean()
    cat_prep["is_sla_breach"] *= 100
    cat_prep = cat_prep.sort_values("prep_time_min", ascending=True)
    ax_c = axes[2]
    bars = ax_c.barh(cat_prep.index, cat_prep["prep_time_min"],
                      color=BLUE, height=0.55, alpha=0.85)
    ax_c_twin = ax_c.twinx()
    ax_c_twin.plot(cat_prep["is_sla_breach"], range(len(cat_prep)),
                   "o--", color=RED, markersize=8, linewidth=1.5)
    ax_c_twin.set_xlabel("Breach Rate (%)", color=RED)
    ax_c_twin.tick_params(colors=RED)
    ax_c_twin.spines["right"].set_visible(True)
    ax_c_twin.spines["top"].set_False = False
    ax_c.axvline(3.0, color=AMBER, linestyle="--", linewidth=1.5,
                  label="Threshold: 3 min")
    ax_c.set_xlabel("Avg Prep Time (min)")
    ax_c.set_title("Prep Time & Breach Rate by Category")
    ax_c.legend(fontsize=9)

    plt.tight_layout()
    chart2_path = OUTPUTS_DIR / "02_time_components.png"
    plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Chart saved: {chart2_path}")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 6 — STORE-LEVEL PERFORMANCE RANKING
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 6: Store Performance Ranking")
    print("="*65)

    store_perf = df.groupby(["store_id","store_name","zone","performance_tier"]).agg(
        total_orders      = ("is_sla_breach", "count"),
        sla_breaches      = ("is_sla_breach", "sum"),
        breach_rate       = ("is_sla_breach", "mean"),
        avg_total_time    = ("total_time_min", "mean"),
        avg_prep          = ("prep_time_min",  "mean"),
        avg_rider_wait    = ("rider_wait_min", "mean"),
        avg_last_mile     = ("delivery_time_min", "mean"),
        total_gmv         = ("order_value",    "sum"),
        rev_at_risk       = ("total_rev_impact","sum"),
    ).reset_index()
    store_perf["breach_rate"] *= 100
    store_perf = store_perf.sort_values("breach_rate", ascending=False).reset_index(drop=True)
    store_perf["rank"] = range(1, len(store_perf)+1)

    print("Store Leaderboard (worst -> best):")
    print(store_perf[["rank","store_name","zone","performance_tier",
                       "breach_rate","avg_total_time","rev_at_risk"]].round(2).to_string(index=False))

    # ── Store leaderboard chart ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Dark Store Performance Leaderboard", fontsize=14, fontweight="bold")

    # Left: breach rate bar chart
    colors = [TIER_COLORS[t] for t in store_perf["performance_tier"]]
    bars = axes[0].barh(store_perf["store_name"][::-1],
                        store_perf["breach_rate"][::-1],
                        color=colors[::-1], height=0.65)
    axes[0].axvline(overall_breach, color=DARKGRAY, linestyle="--",
                    linewidth=1.5, label=f"Overall avg: {overall_breach:.1f}%")
    for bar, val in zip(bars, store_perf["breach_rate"][::-1]):
        axes[0].text(val+0.3, bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=9)
    axes[0].set_xlabel("SLA Breach Rate (%)")
    axes[0].set_title("Breach Rate per Store (Red=Tier C, Amber=Tier B, Green=Tier A)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, store_perf["breach_rate"].max()+8)

    # Right: time components per store (stacked bar)
    store_perf_sorted = store_perf.sort_values("avg_total_time", ascending=True)
    axes[1].barh(store_perf_sorted["store_name"], store_perf_sorted["avg_prep"],
                 label="Prep", color=TEAL, height=0.6)
    axes[1].barh(store_perf_sorted["store_name"], store_perf_sorted["avg_rider_wait"],
                 left=store_perf_sorted["avg_prep"], label="Rider Wait", color=AMBER, height=0.6)
    axes[1].barh(store_perf_sorted["store_name"], store_perf_sorted["avg_last_mile"],
                 left=store_perf_sorted["avg_prep"]+store_perf_sorted["avg_rider_wait"],
                 label="Last Mile", color=BLUE, height=0.6)
    axes[1].axvline(10, color=RED, linestyle="--", linewidth=2, label="10-min SLA")
    axes[1].set_xlabel("Average Minutes")
    axes[1].set_title("Time Component Breakdown by Store")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    chart3_path = OUTPUTS_DIR / "03_store_performance.png"
    plt.savefig(chart3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Chart saved: {chart3_path}")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 7 — RIDER PERFORMANCE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 7: Rider Performance Analysis")
    print("="*65)

    rider_perf = df.merge(
        riders[["rider_id","experience_days","avg_rating","vehicle_type","shift"]],
        on="rider_id", how="left"
    )

    rider_stats = rider_perf.groupby("rider_id").agg(
        deliveries        = ("is_sla_breach", "count"),
        breaches          = ("is_sla_breach", "sum"),
        breach_rate       = ("is_sla_breach", "mean"),
        avg_delivery_time = ("delivery_time_min", "mean"),
        experience_days   = ("experience_days", "first"),
        avg_rating        = ("avg_rating", "first"),
        vehicle_type      = ("vehicle_type", "first"),
        shift             = ("shift", "first"),
        store_id          = ("store_id", "first"),
    ).reset_index()

    rider_stats["breach_rate"] *= 100
    rider_stats = rider_stats[rider_stats["deliveries"] >= 30].copy()

    # Performance label
    def rider_label(r):
        if r < 30:   return "High Performer"
        if r < 50:   return "Average"
        return "Needs Training"
    rider_stats["performance_label"] = rider_stats["breach_rate"].apply(rider_label)

    print(f"Riders with >=30 deliveries : {len(rider_stats)}")
    print(f"\nPerformance distribution:")
    print(rider_stats["performance_label"].value_counts().to_string())
    print(f"\nVehicle type vs breach rate:")
    print(rider_stats.groupby("vehicle_type")["breach_rate"].mean().round(2).to_string())

    # ── Rider analysis charts ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Rider Performance Analysis", fontsize=14, fontweight="bold")

    # Experience vs breach rate scatter
    colors_map = {"High Performer": TEAL, "Average": AMBER, "Needs Training": RED}
    for label, grp in rider_stats.groupby("performance_label"):
        axes[0].scatter(grp["experience_days"], grp["breach_rate"],
                        c=colors_map[label], alpha=0.6, s=40, label=label)
    # Regression line
    m, b, r, p, _ = stats.linregress(rider_stats["experience_days"],
                                       rider_stats["breach_rate"])
    x_line = np.linspace(0, rider_stats["experience_days"].max(), 100)
    axes[0].plot(x_line, m*x_line+b, "--", color=GRAY, linewidth=1.5,
                 label=f"Trend (r={r:.2f})")
    axes[0].set_xlabel("Experience (days)")
    axes[0].set_ylabel("Breach Rate (%)")
    axes[0].set_title(f"Experience vs Breach Rate\n(r={r:.2f}, p={p:.4f})")
    axes[0].legend(fontsize=8)

    # Vehicle type comparison
    veh_perf = rider_stats.groupby("vehicle_type").agg(
        breach_rate=("breach_rate","mean"),
        riders=("rider_id","count"),
        avg_exp=("experience_days","mean")
    ).reset_index()
    v_colors = [BLUE, TEAL, AMBER][:len(veh_perf)]
    bars = axes[1].bar(veh_perf["vehicle_type"], veh_perf["breach_rate"],
                       color=v_colors, width=0.5)
    for bar, val, cnt in zip(bars, veh_perf["breach_rate"], veh_perf["riders"]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{val:.1f}%\n(n={cnt})", ha="center", fontsize=10)
    axes[1].set_ylabel("Avg Breach Rate (%)")
    axes[1].set_title("Breach Rate by Vehicle Type")
    axes[1].set_ylim(0, veh_perf["breach_rate"].max()*1.3)

    # Shift analysis
    shift_perf = rider_stats.groupby("shift").agg(
        breach_rate=("breach_rate","mean"),
        riders=("rider_id","count")
    ).reset_index()
    s_colors = [BLUE, AMBER, PURPLE, TEAL][:len(shift_perf)]
    bars = axes[2].bar(shift_perf["shift"], shift_perf["breach_rate"],
                       color=s_colors, width=0.5)
    for bar, val in zip(bars, shift_perf["breach_rate"]):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{val:.1f}%", ha="center", fontsize=11)
    axes[2].set_ylabel("Avg Breach Rate (%)")
    axes[2].set_title("Breach Rate by Rider Shift")

    plt.tight_layout()
    chart4_path = OUTPUTS_DIR / "04_rider_performance.png"
    plt.savefig(chart4_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Chart saved: {chart4_path}")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 8 — PEAK HOUR & DEMAND PATTERN ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 8: Peak Hour & Demand Pattern Analysis")
    print("="*65)

    hourly = df.groupby("order_hour").agg(
        orders=("is_sla_breach","count"),
        breaches=("is_sla_breach","sum"),
        breach_rate=("is_sla_breach","mean"),
        avg_time=("total_time_min","mean"),
        avg_wait=("rider_wait_min","mean"),
    ).reset_index()
    hourly["breach_rate"] *= 100

    # Day x hour heatmap data
    heatmap_data = df.groupby(["day_of_week","order_hour"])["is_sla_breach"].mean().unstack() * 100
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heatmap_data = heatmap_data.reindex(day_order)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Demand & SLA Pattern Analysis", fontsize=14, fontweight="bold")

    # Hourly demand vs breach rate
    axes[0].bar(hourly["order_hour"], hourly["orders"],
                color=[RED if h in [8,9,12,13,19,20,21] else BLUE
                       for h in hourly["order_hour"]],
                width=0.7, alpha=0.75, label="Orders")
    ax_twin = axes[0].twinx()
    ax_twin.plot(hourly["order_hour"], hourly["breach_rate"],
                 "o-", color=RED, linewidth=2.5, markersize=7, label="Breach Rate %")
    ax_twin.set_ylabel("Breach Rate (%)", color=RED)
    ax_twin.tick_params(colors=RED)
    ax_twin.spines["right"].set_visible(True)
    ax_twin.spines["top"].set_visible(False)
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Number of Orders")
    axes[0].set_title("Hourly Order Volume vs Breach Rate\n(Red bars = peak hours)")
    axes[0].set_xticks(range(0,24,2))
    h_lines1, h_labels1 = axes[0].get_legend_handles_labels()
    h_lines2, h_labels2 = ax_twin.get_legend_handles_labels()
    axes[0].legend(h_lines1+h_lines2, h_labels1+h_labels2, fontsize=8)

    # Day x Hour heatmap
    sns.heatmap(heatmap_data, ax=axes[1], cmap="RdYlGn_r",
                fmt=".0f", annot=False, linewidths=0.2,
                cbar_kws={"label": "Breach Rate (%)"})
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_title("SLA Breach Rate Heatmap\n(Day x Hour) - Red = High Risk")

    plt.tight_layout()
    chart5_path = OUTPUTS_DIR / "05_demand_patterns.png"
    plt.savefig(chart5_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Chart saved: {chart5_path}")
    print(f"\nPeak hour breach rates:")
    print(hourly[hourly["order_hour"].isin([8,9,12,13,19,20,21])][
        ["order_hour","breach_rate","avg_time","avg_wait"]].to_string(index=False))


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 9 — WEATHER IMPACT ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 9: Weather Impact Analysis")
    print("="*65)

    rain_stats = df.groupby("is_rain").agg(
        orders=("is_sla_breach","count"),
        breach_rate=("is_sla_breach","mean"),
        avg_time=("total_time_min","mean"),
        avg_last_mile=("delivery_time_min","mean"),
        avg_prep=("prep_time_min","mean"),
    ).reset_index()
    rain_stats["breach_rate"] *= 100
    rain_stats.index = ["Clear", "Rainy"]

    print("Rain vs Clear day comparison:")
    print(rain_stats[["breach_rate","avg_time","avg_last_mile","avg_prep"]].round(2))

    rain_lift = rain_stats.loc["Rainy","breach_rate"] - rain_stats.loc["Clear","breach_rate"]
    time_lift  = rain_stats.loc["Rainy","avg_time"] - rain_stats.loc["Clear","avg_time"]
    print(f"\nRain impact:")
    print(f"  Breach rate increase : +{rain_lift:.1f} percentage points")
    print(f"  Avg time increase    : +{time_lift:.1f} min")

    # T-test for significance
    rain_times  = df[df["is_rain"]==1]["total_time_min"]
    clear_times = df[df["is_rain"]==0]["total_time_min"]
    t_stat, p_val = stats.ttest_ind(rain_times, clear_times)
    print(f"  Statistical test     : t={t_stat:.2f}, p={p_val:.4f} ({'Significant' if p_val<0.05 else 'Not significant'})")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 10 — REVENUE & BUSINESS IMPACT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 10: Revenue & Business Impact")
    print("="*65)

    total_gmv          = df["order_value"].sum()
    breached_gmv       = df[df["is_sla_breach"]==1]["order_value"].sum()
    total_refund_exp   = df["refund_exposure"].sum()
    total_churn_loss   = df["churn_loss_est"].sum()
    total_impact       = df["total_rev_impact"].sum()

    print(f"Total GMV             : INR {total_gmv/1e7:.2f} Crore")
    print(f"GMV on breached orders: INR {breached_gmv/1e7:.2f} Crore  ({breached_gmv/total_gmv*100:.1f}% of GMV)")
    print(f"\nRevenue impact:")
    print(f"  Refund exposure (15%): INR {total_refund_exp/1e5:.1f} Lakhs")
    print(f"  Churn loss est  (8%) : INR {total_churn_loss/1e5:.1f} Lakhs")
    print(f"  TOTAL at risk        : INR {total_impact/1e5:.1f} Lakhs")

    # Store-level revenue impact (vectorized calculation)
    store_impact = df.groupby(["store_name","zone"]).agg(
        total_gmv       = ("order_value","sum"),
        breached_gmv    = ("breached_order_value","sum"),
        refund_exp      = ("refund_exposure","sum"),
        churn_loss      = ("churn_loss_est","sum"),
        total_impact    = ("total_rev_impact","sum"),
        breach_rate     = ("is_sla_breach","mean"),
    ).reset_index().sort_values("total_impact", ascending=False)
    store_impact["breach_rate"] *= 100

    print(f"\nTop 5 stores by revenue at risk:")
    print(store_impact[["store_name","zone","breach_rate","total_impact"]].head(5).round(0).to_string(index=False))

    # Revenue impact chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Business Impact of SLA Breaches", fontsize=14, fontweight="bold")

    # Waterfall chart for revenue impact
    categories = ["Total GMV", "GMV\nBreached", "Refund\nExposure", "Churn\nLoss", "Total\nImpact"]
    values = [total_gmv/1e5, breached_gmv/1e5,
              -total_refund_exp/1e5, -total_churn_loss/1e5,
              -total_impact/1e5]
    colors_wf = [TEAL, AMBER, RED, RED, DARKGRAY]
    bars = axes[0].bar(categories, [abs(v) for v in values], color=colors_wf, width=0.55)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                     f"INR {abs(val):.0f}L", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Amount (INR Lakhs)")
    axes[0].set_title("Revenue Impact Breakdown")

    # Store-level impact bar
    top_stores = store_impact.head(10)
    bars = axes[1].barh(top_stores["store_name"][::-1],
                        top_stores["total_impact"][::-1]/1e3,
                        color=[ZONE_COLORS[z] for z in top_stores["zone"][::-1]],
                        height=0.6)
    for bar, val in zip(bars, top_stores["total_impact"][::-1]/1e3):
        axes[1].text(val+0.5, bar.get_y()+bar.get_height()/2,
                     f"INR {val:.0f}K", va="center", fontsize=9)
    axes[1].set_xlabel("Revenue at Risk (INR thousands)")
    axes[1].set_title("Revenue at Risk by Store\n(Colour = Zone)")
    legend_patches = [plt.Rectangle((0,0),1,1, color=c, label=z)
                      for z, c in ZONE_COLORS.items()]
    axes[1].legend(handles=legend_patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    chart6_path = OUTPUTS_DIR / "06_revenue_impact.png"
    plt.savefig(chart6_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Chart saved: {chart6_path}")


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 11 — SLA BREACH PREDICTION MODEL
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 11: SLA Breach Prediction Model")
    print("="*65)
    print("""
WHY BUILD A PREDICTION MODEL?
-----------------------------
Current state  : SLA breach is detected AFTER delivery (reactive).
Future state   : Predict breach BEFORE dispatch (proactive).
Business value : Ops team can add a rider OR notify customer in advance.
                 "Your order may take 12-14 mins today due to peak demand"
                 -> reduces complaint rate even if SLA is breached.
""")

    # Feature set — only features known AT ORDER TIME (before dispatch)
    model_features = [
        "order_hour",
        "is_weekend",
        "is_peak_hour",
        "is_rain",
        "month",
        "n_items",
        "order_value",
    ]

    # Add encoded categorical features
    df_model = df.copy()
    for col in ["category", "zone", "performance_tier"]:
        le = LabelEncoder()
        df_model[f"{col}_enc"] = le.fit_transform(df_model[col].astype(str))
        model_features.append(f"{col}_enc")

    le_block = LabelEncoder()
    df_model["hour_block_enc"] = le_block.fit_transform(df_model["hour_block"].astype(str))
    model_features.append("hour_block_enc")

    X = df_model[model_features].fillna(0)
    y = df_model["is_sla_breach"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Train Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42
    )
    gb.fit(X_train, y_train)

    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_preds = (gb_proba >= 0.50).astype(int)
    gb_auc   = roc_auc_score(y_test, gb_proba)

    print(f"Gradient Boosting AUC : {gb_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, gb_preds, target_names=["On Time","Breach"]))

    # Cross-validation (3-fold for speed and stability)
    print("Running cross-validation...")
    cv_scores = cross_val_score(gb, X, y, cv=3, scoring="roc_auc", n_jobs=1)
    print(f"3-Fold CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Feature importance
    fi = pd.Series(gb.feature_importances_, index=model_features).sort_values(ascending=True)

    # Model evaluation chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("SLA Breach Prediction Model", fontsize=14, fontweight="bold")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, gb_proba)
    axes[0].plot(fpr, tpr, color=BLUE, linewidth=2.5, label=f"GB (AUC={gb_auc:.3f})")
    axes[0].plot([0,1],[0,1],"--", color=GRAY, linewidth=1)
    axes[0].fill_between(fpr, tpr, alpha=0.10, color=BLUE)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve - SLA Breach Prediction")
    axes[0].legend(fontsize=10)

    # Confusion matrix
    cm = confusion_matrix(y_test, gb_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On Time","Breach"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix (threshold=0.50)")

    # Feature importance
    fi.tail(10).plot(kind="barh", ax=axes[2], color=BLUE)
    axes[2].set_title("Top Feature Importances")
    axes[2].set_xlabel("Importance Score")

    plt.tight_layout()
    chart7_path = OUTPUTS_DIR / "07_prediction_model.png"
    plt.savefig(chart7_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Chart saved: {chart7_path}")

    # Score all orders with breach probability
    df["breach_probability"]   = gb.predict_proba(X.fillna(0))[:, 1]
    df["predicted_breach"]     = (df["breach_probability"] >= 0.50).astype(int)
    df["breach_risk_band"]     = pd.cut(
        df["breach_probability"],
        bins=[0, 0.30, 0.55, 0.75, 1.01],
        labels=["Low Risk","Medium Risk","High Risk","Very High Risk"],
        include_lowest=True
    )


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 12 — EXPORT FOR POWER BI
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STEP 12: Exporting data for Power BI")
    print("="*65)

    # ── Table 1: Orders (main fact table) ─────────────────────────────────────────
    order_export_cols = [
        "order_id","order_date","order_hour","day_of_week","is_weekend",
        "month","month_name","store_id","store_name","zone","pincode",
        "rider_id","category","n_items","order_value","payment_mode",
        "prep_time_min","rider_wait_min","delivery_time_min","total_time_min",
        "sla_target_min","is_sla_breach","is_rain","is_peak_hour","status",
        "hour_block","breach_severity","bottleneck","performance_tier",
        "refund_exposure","churn_loss_est","total_rev_impact",
        "breach_probability","breach_risk_band",
    ]
    orders_export_path = OUTPUTS_DIR / "orders_scored.csv"
    df[order_export_cols].to_csv(orders_export_path, index=False)
    print(f"[OK] orders_scored.csv ({len(df):,} rows) -> {orders_export_path}")

    # ── Table 2: Store performance summary ────────────────────────────────────────
    store_perf_export = store_perf.merge(
        stores[["store_id","lat","lng","capacity","avg_riders"]], on="store_id", how="left"
    )
    store_export_path = OUTPUTS_DIR / "store_performance.csv"
    store_perf_export.to_csv(store_export_path, index=False)
    print(f"[OK] store_performance.csv ({len(store_perf_export)} stores) -> {store_export_path}")

    # ── Table 3: Daily KPI trend ──────────────────────────────────────────────────
    daily_kpi = df.groupby("order_date").agg(
        total_orders    = ("is_sla_breach","count"),
        sla_breaches    = ("is_sla_breach","sum"),
        breach_rate_pct = ("is_sla_breach","mean"),
        gmv             = ("order_value","sum"),
        avg_time        = ("total_time_min","mean"),
        rev_at_risk     = ("total_rev_impact","sum"),
        rain_day        = ("is_rain","max"),
    ).reset_index()
    daily_kpi["breach_rate_pct"] *= 100
    daily_export_path = OUTPUTS_DIR / "daily_kpi.csv"
    daily_kpi.to_csv(daily_export_path, index=False)
    print(f"[OK] daily_kpi.csv ({len(daily_kpi)} days) -> {daily_export_path}")

    # ── Table 4: Rider performance ────────────────────────────────────────────────
    rider_export_path = OUTPUTS_DIR / "rider_performance.csv"
    rider_stats.to_csv(rider_export_path, index=False)
    print(f"[OK] rider_performance.csv ({len(rider_stats)} riders) -> {rider_export_path}")

    # ── Table 5: Inventory alerts ─────────────────────────────────────────────────
    inv_clean_path = OUTPUTS_DIR / "inventory_alerts_clean.csv"
    inv.to_csv(inv_clean_path, index=False)
    print(f"[OK] inventory_alerts_clean.csv ({len(inv):,} alerts) -> {inv_clean_path}")

    # ── Final business summary ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("[REPORT] FINAL BUSINESS IMPACT SUMMARY")
    print("="*65)
    print(f"Total orders analysed  : {len(df):,}")
    print(f"SLA breach rate        : {df['is_sla_breach'].mean()*100:.1f}%")
    print(f"Total SLA breaches     : {df['is_sla_breach'].sum():,}")
    print(f"Total GMV              : INR {df['order_value'].sum()/1e7:.2f} Crore")
    print(f"Revenue at risk        : INR {df['total_rev_impact'].sum()/1e5:.1f} Lakhs")
    print(f"Worst store            : {store_perf.iloc[0]['store_name']} ({store_perf.iloc[0]['breach_rate']:.1f}% breach)")
    print(f"Best store             : {store_perf.iloc[-1]['store_name']} ({store_perf.iloc[-1]['breach_rate']:.1f}% breach)")
    print(f"Peak breach hour       : {hourly.loc[hourly['breach_rate'].idxmax(),'order_hour']}:00 ({hourly['breach_rate'].max():.1f}%)")
    print(f"Rain impact            : +{rain_lift:.1f}% breach rate on rainy days")
    print(f"Prediction model AUC   : {gb_auc:.4f}")
    print(f"\nTop 3 actionable recommendations:")
    print(f"  1. Add 2 riders at {store_perf.iloc[0]['store_name']} during 19:00-21:00 (worst store + peak)")
    print(f"  2. Improve packing SOP for Meats & Seafood (highest prep time)")
    print(f"  3. Monsoon rider surge pricing - +2 riders per store June-September")

    print("\n" + "="*65)
    print("ALL STEPS COMPLETE [OK]")
    print("="*65)
    print("\nOutputs generated in:", OUTPUTS_DIR)


if __name__ == "__main__":
    run_pipeline()
