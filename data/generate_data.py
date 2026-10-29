"""
=============================================================================
HYPERLOCAL DELIVERY SLA BREACH ANALYSIS — DATASET GENERATOR
=============================================================================
Simulates 6 months of operations for a quick-commerce startup (Zepto/Blinkit
style) operating across Mumbai with 15 dark stores.

Generates 4 tables that mirror what a real ops analyst works with:
  1. orders          — every delivery order with timestamps
  2. dark_stores     — store metadata (location, capacity, staff)
  3. riders          — delivery partner profiles
  4. inventory_alerts — stockout events per store per day

Run this FIRST before the analysis notebook.
=============================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Resolve data directory relative to this script
DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_datasets():
    np.random.seed(42)
    random.seed(42)

    # ── Config ─────────────────────────────────────────────────────────────────────
    START_DATE   = datetime(2024, 1, 1)
    END_DATE     = datetime(2024, 6, 30)
    N_DAYS       = (END_DATE - START_DATE).days + 1
    SLA_MINUTES  = 10          # promised delivery time
    N_ORDERS     = 80000       # ~6 months of a mid-size Mumbai operation

    # ── Mumbai dark store locations (real Mumbai areas) ───────────────────────────
    DARK_STORES = [
        {"store_id":"DS01","name":"Andheri West",    "zone":"West",  "lat":19.1369,"lng":72.8260,"capacity":150,"pincode":"400058"},
        {"store_id":"DS02","name":"Bandra West",     "zone":"West",  "lat":19.0596,"lng":72.8295,"capacity":120,"pincode":"400050"},
        {"store_id":"DS03","name":"Juhu",            "zone":"West",  "lat":19.1075,"lng":72.8263,"capacity":100,"pincode":"400049"},
        {"store_id":"DS04","name":"Malad West",      "zone":"West",  "lat":19.1866,"lng":72.8484,"capacity":130,"pincode":"400064"},
        {"store_id":"DS05","name":"Borivali West",   "zone":"West",  "lat":19.2322,"lng":72.8561,"capacity":110,"pincode":"400092"},
        {"store_id":"DS06","name":"Powai",           "zone":"Central","lat":19.1176,"lng":72.9060,"capacity":140,"pincode":"400076"},
        {"store_id":"DS07","name":"Ghatkopar West",  "zone":"Central","lat":19.0858,"lng":72.9079,"capacity":125,"pincode":"400086"},
        {"store_id":"DS08","name":"Kurla West",      "zone":"Central","lat":19.0665,"lng":72.8799,"capacity":115,"pincode":"400070"},
        {"store_id":"DS09","name":"Thane West",      "zone":"Central","lat":19.1974,"lng":72.9660,"capacity":160,"pincode":"400601"},
        {"store_id":"DS10","name":"Mulund West",     "zone":"Central","lat":19.1727,"lng":72.9470,"capacity":105,"pincode":"400080"},
        {"store_id":"DS11","name":"Dadar West",      "zone":"South", "lat":19.0178,"lng":72.8478,"capacity":135,"pincode":"400028"},
        {"store_id":"DS12","name":"Lower Parel",     "zone":"South", "lat":18.9954,"lng":72.8319,"capacity":120,"pincode":"400013"},
        {"store_id":"DS13","name":"Worli",           "zone":"South", "lat":19.0176,"lng":72.8162,"capacity":110,"pincode":"400018"},
        {"store_id":"DS14","name":"Chembur",         "zone":"East",  "lat":19.0522,"lng":72.9005,"capacity":130,"pincode":"400071"},
        {"store_id":"DS15","name":"Vikhroli West",   "zone":"East",  "lat":19.1057,"lng":72.9246,"capacity":100,"pincode":"400083"},
    ]

    stores_df = pd.DataFrame(DARK_STORES)

    # Rider count per store (1 rider per 10 capacity, with variance)
    stores_df["avg_riders"] = (stores_df["capacity"] / 10).astype(int) + np.random.randint(-2, 3, len(stores_df))
    stores_df["avg_riders"] = stores_df["avg_riders"].clip(8, 20)

    # Store performance tier (affects SLA breach rate)
    tiers = ["A"] * 4 + ["B"] * 6 + ["C"] * 5
    np.random.shuffle(tiers)
    stores_df["performance_tier"] = tiers

    dark_stores_path = DATA_DIR / "dark_stores.csv"
    stores_df.to_csv(dark_stores_path, index=False)
    print(f"[OK] Dark stores: {len(stores_df)} stores across {stores_df['zone'].nunique()} zones -> {dark_stores_path}")

    # ── Rider profiles ─────────────────────────────────────────────────────────────
    N_RIDERS = 350
    rider_ids = [f"RDR{str(i).zfill(4)}" for i in range(1, N_RIDERS+1)]

    riders_df = pd.DataFrame({
        "rider_id":       rider_ids,
        "store_id":       np.random.choice(stores_df["store_id"], N_RIDERS),
        "experience_days":np.random.randint(10, 730, N_RIDERS),
        "avg_rating":     np.round(np.random.normal(4.2, 0.4, N_RIDERS).clip(1,5), 1),
        "vehicle_type":   np.random.choice(["Bicycle","EV Bike","Petrol Bike"],
                                            N_RIDERS, p=[0.15, 0.45, 0.40]),
        "shift":          np.random.choice(["Morning","Afternoon","Evening","Night"],
                                            N_RIDERS, p=[0.30,0.25,0.30,0.15]),
    })
    riders_path = DATA_DIR / "riders.csv"
    riders_df.to_csv(riders_path, index=False)
    print(f"[OK] Riders: {N_RIDERS} delivery partners -> {riders_path}")

    # ── Order generation ───────────────────────────────────────────────────────────
    print("Generating orders (this takes ~20 seconds)...")

    CATEGORIES = {
        "Grocery & Staples":  {"weight":0.28, "prep_min":1.5, "prep_std":0.5},
        "Fruits & Vegetables":{"weight":0.15, "prep_min":2.0, "prep_std":0.8},
        "Dairy & Breakfast":  {"weight":0.12, "prep_min":1.0, "prep_std":0.3},
        "Snacks & Drinks":    {"weight":0.13, "prep_min":1.0, "prep_std":0.3},
        "Personal Care":      {"weight":0.09, "prep_min":1.5, "prep_std":0.5},
        "Household Items":    {"weight":0.08, "prep_min":2.5, "prep_std":1.0},
        "Meats & Seafood":    {"weight":0.06, "prep_min":3.5, "prep_std":1.2},
        "Baby Products":      {"weight":0.05, "prep_min":1.5, "prep_std":0.5},
        "Medicines":          {"weight":0.04, "prep_min":2.0, "prep_std":0.8},
    }

    cat_names    = list(CATEGORIES.keys())
    cat_weights  = [v["weight"] for v in CATEGORIES.values()]

    PAYMENT_MODES = ["UPI","Credit Card","Debit Card","Cash on Delivery","Wallet"]

    # Demand profile — realistic hourly pattern for Mumbai
    # Peak: 8-10am (breakfast), 12-2pm (lunch), 7-10pm (dinner/evening)
    HOURLY_DEMAND = {
        0:0.8, 1:0.3, 2:0.2, 3:0.2, 4:0.3, 5:0.8,
        6:2.0, 7:3.5, 8:5.0, 9:5.5, 10:4.0,
        11:3.5,12:5.0,13:5.5,14:4.5,15:3.5,
        16:3.0,17:4.0,18:5.0,19:7.0,20:8.0,
        21:7.5,22:5.0,23:3.0
    }
    total_demand = sum(HOURLY_DEMAND.values())
    hour_probs   = [HOURLY_DEMAND[h]/total_demand for h in range(24)]

    records = []
    order_num = 1

    for day_offset in range(N_DAYS):
        date = START_DATE + timedelta(days=day_offset)
        weekday   = date.weekday()         # 0=Mon, 6=Sun
        is_weekend = weekday >= 5
        is_monday  = weekday == 0

        # Seasonal demand multiplier (India: festive Oct-Nov, summer June)
        month = date.month
        month_mult = {1:0.90, 2:0.92, 3:0.95, 4:0.98, 5:1.02, 6:1.05}[month]

        # Daily order volume
        daily_orders = int(
            N_ORDERS / N_DAYS *
            (1.25 if is_weekend else 1.0) *
            (0.85 if is_monday else 1.0) *
            month_mult *
            np.random.normal(1.0, 0.08)
        )

        for _ in range(daily_orders):
            # Time slot
            hour   = np.random.choice(range(24), p=hour_probs)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            order_time = date.replace(hour=hour, minute=minute, second=second)

            # Store (weighted by zone demand at this hour)
            # Evening = South Mumbai more popular; morning = suburbs
            if hour in range(7, 11):       # morning suburbs higher
                store_weights = [1.3 if z in ["West","Central"] else 0.8
                                 for z in stores_df["zone"]]
            elif hour in range(18, 23):    # evening south popular
                store_weights = [1.4 if z == "South" else 1.0
                                 for z in stores_df["zone"]]
            else:
                store_weights = [1.0] * len(stores_df)

            sw_total = sum(store_weights)
            store_probs = [w/sw_total for w in store_weights]
            store_row = stores_df.sample(1, weights=store_probs).iloc[0]
            store_id  = store_row["store_id"]
            perf_tier = store_row["performance_tier"]

            # Category
            cat = np.random.choice(cat_names, p=cat_weights)
            cat_info = CATEGORIES[cat]

            # Order value
            order_value = round(np.random.lognormal(mean=5.2, sigma=0.6), 2)
            order_value = np.clip(order_value, 89, 4999)
            n_items     = np.random.randint(1, 12)

            # ── SLA calculation ─────────────────────────────────────────────────────
            # Prep time (packing at dark store)
            prep_time = max(0.5, np.random.normal(cat_info["prep_min"], cat_info["prep_std"]))

            # Rider wait time (higher during peak, lower off-peak)
            is_peak = hour in [8,9,12,13,19,20,21]
            rider_wait_base = 1.5 if is_peak else 0.8
            rider_wait = max(0.2, np.random.normal(rider_wait_base, 0.5))

            # Delivery time — depends on performance tier, hour, weather
            # Base delivery time (distance from dark store)
            base_delivery = np.random.normal(5.5, 1.5)

            # Modifiers
            hour_mult = 1.4 if is_peak else 1.0           # traffic
            tier_mult = {"A":0.85, "B":1.0, "C":1.20}[perf_tier]
            rain_event = (month in [6] and np.random.random() < 0.20)  # June monsoon
            rain_mult  = np.random.uniform(1.3, 1.8) if rain_event else 1.0

            delivery_time = max(2.0, base_delivery * hour_mult * tier_mult * rain_mult)

            # Total delivery time (minutes from order placed)
            total_time = prep_time + rider_wait + delivery_time

            # Add random outliers (address issues, customer not home, etc.)
            if np.random.random() < 0.04:
                total_time += np.random.uniform(5, 20)

            total_time = round(total_time, 2)
            is_sla_breach = int(total_time > SLA_MINUTES)

            # Timestamps
            delivered_time = order_time + timedelta(minutes=total_time)

            # Order status
            if np.random.random() < 0.015:    # 1.5% cancellations
                status = "Cancelled"
                total_time = None
                is_sla_breach = None
            elif np.random.random() < 0.005:   # 0.5% returns
                status = "Returned"
            else:
                status = "Delivered"

            # Assign rider
            store_riders = riders_df[riders_df["store_id"] == store_id]
            if len(store_riders) > 0:
                rider_id = store_riders.sample(1).iloc[0]["rider_id"]
            else:
                rider_id = "RDR0001"

            records.append({
                "order_id":         f"ORD{str(order_num).zfill(7)}",
                "order_datetime":   order_time.strftime("%Y-%m-%d %H:%M:%S"),
                "order_date":       order_time.strftime("%Y-%m-%d"),
                "order_hour":       hour,
                "day_of_week":      date.strftime("%A"),
                "is_weekend":       int(is_weekend),
                "month":            month,
                "month_name":       date.strftime("%B"),
                "store_id":         store_id,
                "store_name":       store_row["name"],
                "zone":             store_row["zone"],
                "pincode":          store_row["pincode"],
                "rider_id":         rider_id,
                "category":         cat,
                "n_items":          n_items,
                "order_value":      order_value,
                "payment_mode":     np.random.choice(PAYMENT_MODES, p=[0.52,0.18,0.15,0.10,0.05]),
                "prep_time_min":    round(prep_time, 2),
                "rider_wait_min":   round(rider_wait, 2),
                "delivery_time_min":round(delivery_time, 2),
                "total_time_min":   total_time,
                "sla_target_min":   SLA_MINUTES,
                "is_sla_breach":    is_sla_breach,
                "is_rain":          int(rain_event),
                "is_peak_hour":     int(is_peak),
                "status":           status,
            })
            order_num += 1

    orders_df = pd.DataFrame(records)
    orders_path = DATA_DIR / "orders.csv"
    orders_df.to_csv(orders_path, index=False)
    print(f"[OK] Orders: {len(orders_df):,} -> {orders_path}")
    print(f"  Delivered : {(orders_df['status']=='Delivered').sum():,}")
    print(f"  Cancelled : {(orders_df['status']=='Cancelled').sum():,}")
    delivered_mask = orders_df["is_sla_breach"].notna()
    print(f"  SLA breach: {orders_df['is_sla_breach'].dropna().sum():,.0f}  "
          f"({orders_df.loc[delivered_mask, 'is_sla_breach'].mean()*100:.1f}%)")

    # ── Inventory stockout events ──────────────────────────────────────────────────
    print("Generating inventory alerts...")
    stockout_records = []

    for day_offset in range(N_DAYS):
        date = START_DATE + timedelta(days=day_offset)
        for _, store in stores_df.iterrows():
            # Higher-capacity stores stockout less
            base_stockout_prob = 0.4 if store["performance_tier"] == "C" else \
                                 0.25 if store["performance_tier"] == "B" else 0.15
            n_stockouts = np.random.binomial(n=len(cat_names),
                                              p=base_stockout_prob * 0.3)
            if n_stockouts > 0:
                stockout_cats = np.random.choice(cat_names, size=n_stockouts, replace=False)
                for cat in stockout_cats:
                    duration_hours = round(np.random.uniform(0.5, 6.0), 1)
                    stockout_records.append({
                        "date":           date.strftime("%Y-%m-%d"),
                        "store_id":       store["store_id"],
                        "store_name":     store["name"],
                        "zone":           store["zone"],
                        "category":       cat,
                        "duration_hours": duration_hours,
                        "orders_impacted":int(duration_hours * np.random.uniform(2, 8)),
                    })

    inv_df = pd.DataFrame(stockout_records)
    inv_path = DATA_DIR / "inventory_alerts.csv"
    inv_df.to_csv(inv_path, index=False)
    print(f"[OK] Inventory alerts: {len(inv_df):,} stockout events -> {inv_path}")

    print("\n" + "="*55)
    print("ALL DATASETS GENERATED [OK]")
    print("="*55)
    print("Files created:")
    print(f"  orders.csv            {len(orders_df):>7,} rows")
    print(f"  dark_stores.csv       {len(stores_df):>7,} rows")
    print(f"  riders.csv            {len(riders_df):>7,} rows")
    print(f"  inventory_alerts.csv  {len(inv_df):>7,} rows")

if __name__ == "__main__":
    generate_datasets()
