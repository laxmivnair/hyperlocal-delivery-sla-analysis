"""
=============================================================================
HYPERLOCAL OPERATIONS PIPELINE VERIFICATION SUITE
=============================================================================
Performs end-to-end data integrity checks, output validations, and model score
audits for the Hyperlocal Delivery SLA Breach Analysis project.
=============================================================================
"""

import sys
from pathlib import Path
import pandas as pd

# Safe terminal encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


def test_integrity():
    print("=" * 65)
    print("RUNNING HYPERLOCAL PIPELINE VERIFICATION")
    print("=" * 65)

    passed_checks = 0
    total_checks = 0

    # 1. Raw Data Files
    raw_files = [
        ("dark_stores.csv", 15),
        ("riders.csv", 300),
        ("orders.csv", 70000),
        ("inventory_alerts.csv", 1000),
    ]

    print("\n[1/4] Checking Raw Datasets in data/...")
    for filename, min_rows in raw_files:
        total_checks += 1
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  [FAIL] Missing file: {path}")
            continue
        df = pd.read_csv(path)
        if len(df) < min_rows:
            print(f"  [FAIL] {filename} has {len(df)} rows (expected >= {min_rows})")
        else:
            print(f"  [PASS] {filename:<22} : {len(df):>7,} rows, {len(df.columns):>2} cols")
            passed_checks += 1

    # 2. Output Figures
    expected_figures = [
        "01_sla_overview.png",
        "02_time_components.png",
        "03_store_performance.png",
        "04_rider_performance.png",
        "05_demand_patterns.png",
        "06_revenue_impact.png",
        "07_prediction_model.png",
    ]

    print("\n[2/4] Checking Visualizations in outputs/...")
    for fig_name in expected_figures:
        total_checks += 1
        fig_path = OUTPUTS_DIR / fig_name
        if not fig_path.exists():
            print(f"  [FAIL] Missing plot: {fig_name}")
        elif fig_path.stat().st_size < 10000:
            print(f"  [FAIL] Plot too small: {fig_name} ({fig_path.stat().st_size} bytes)")
        else:
            print(f"  [PASS] {fig_name:<28} : {fig_path.stat().st_size / 1024:>6.1f} KB")
            passed_checks += 1

    # 3. Output CSVs
    output_csvs = [
        ("orders_scored.csv", 75000),
        ("store_performance.csv", 15),
        ("daily_kpi.csv", 180),
        ("rider_performance.csv", 300),
        ("inventory_alerts_clean.csv", 1000),
    ]

    print("\n[3/4] Checking Export Datasets in outputs/...")
    for csv_name, min_rows in output_csvs:
        total_checks += 1
        csv_path = OUTPUTS_DIR / csv_name
        if not csv_path.exists():
            print(f"  [FAIL] Missing export: {csv_name}")
            continue
        df = pd.read_csv(csv_path)
        if len(df) < min_rows:
            print(f"  [FAIL] {csv_name} has {len(df)} rows (expected >= {min_rows})")
        else:
            print(f"  [PASS] {csv_name:<28} : {len(df):>7,} rows")
            passed_checks += 1

    # 4. Scored Data Sanity & ML Integrity
    print("\n[4/4] Auditing Scored Order Fact Table...")
    scored_path = OUTPUTS_DIR / "orders_scored.csv"
    if scored_path.exists():
        scored_df = pd.read_csv(scored_path)

        # Check probability bounds
        total_checks += 1
        probs = scored_df["breach_probability"]
        if (probs >= 0.0).all() and (probs <= 1.0).all():
            print(f"  [PASS] ML Probabilities in [0.0, 1.0] : min={probs.min():.3f}, max={probs.max():.3f}")
            passed_checks += 1
        else:
            print("  [FAIL] Breach probabilities out of bounds!")

        # Check risk bands
        total_checks += 1
        risk_bands = scored_df["breach_risk_band"].value_counts().to_dict()
        if len(risk_bands) >= 3 and not scored_df["breach_risk_band"].isna().any():
            print(f"  [PASS] Risk Bands Populated        : {risk_bands}")
            passed_checks += 1
        else:
            print("  [FAIL] Risk bands missing or null!")

        # Check financial metrics non-negative
        total_checks += 1
        rev_impact = scored_df["total_rev_impact"]
        if (rev_impact >= 0.0).all() and rev_impact.sum() > 0:
            print(f"  [PASS] Financial Exposure Valid     : Total at risk = INR {rev_impact.sum():,.2f}")
            passed_checks += 1
        else:
            print("  [FAIL] Invalid revenue impact values!")

    # Summary
    print("\n" + "=" * 65)
    print(f"VERIFICATION SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED")
    print("=" * 65)

    if passed_checks == total_checks:
        print("[SUCCESS] All pipeline deliverables and data artifacts verified!\n")
        return 0
    else:
        print(f"[WARNING] {total_checks - passed_checks} check(s) did not meet requirements.\n")
        return 1


if __name__ == "__main__":
    sys.exit(test_integrity())
