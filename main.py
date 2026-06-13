# ============================================================
# Lights Out and Away — F1 ML Project
# main.py
#
# Single entry point to run the complete ML pipeline:
#   1. Collect F1 data from 2022, 2023, 2024 seasons
#   2. Preprocess and engineer features
#   3. Train winner prediction models (LR + KNN)
#   4. Train pit stop prediction models (LR + KNN)
#   5. Generate all F1-themed visualizations
#   6. Generate the professional PDF report
#
# Usage: python main.py
# ============================================================

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Ensure the project root is in the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def print_banner():
    """Print the F1-themed project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🏎️  LIGHTS OUT AND AWAY                                  ║
    ║                                                              ║
    ║     Predicting F1 Race Winners & Pit Stop Durations          ║
    ║     Using Supervised Machine Learning                        ║
    ║                                                              ║
    ║     Algorithms: Linear Regression & K-Nearest Neighbors      ║
    ║     Data: FastF1 (2022, 2023, 2024 Seasons)                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_step(step_num, emoji, message):
    """Print a formatted step message."""
    print(f"\n{'='*65}")
    print(f"   {emoji}  STEP {step_num}/6 — {message}")
    print(f"{'='*65}")


def main():
    """
    Run the complete Lights Out and Away ML pipeline.

    Steps:
        1. Collect data from FastF1 API
        2. Preprocess and engineer features
        3. Train winner prediction models
        4. Train pit stop prediction models
        5. Generate all F1-themed visualizations
        6. Generate professional PDF report
    """
    start_time = time.time()
    print_banner()

    # Create required directories
    dirs = [
        os.path.join(PROJECT_ROOT, "cache"),
        os.path.join(PROJECT_ROOT, "data", "raw"),
        os.path.join(PROJECT_ROOT, "data", "processed"),
        os.path.join(PROJECT_ROOT, "models"),
        os.path.join(PROJECT_ROOT, "outputs", "plots"),
        os.path.join(PROJECT_ROOT, "outputs", "reports"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # STEP 1: DATA COLLECTION
    # ══════════════════════════════════════════════════════════

    print_step(1, "🏎️", "LIGHTS OUT AND AWAY — Starting data collection...")

    from src.data_collection import collect_all_seasons

    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    race_data_exists = os.path.exists(os.path.join(raw_data_dir, "race_data_raw.csv"))

    if race_data_exists:
        print("\n   📂 Raw data already exists! Skipping data collection.")
        print("   💡 Delete data/raw/ folder to force re-collection.")
    else:
        total, failed = collect_all_seasons()
        print(f"\n   ✅ Data collection complete: {total} sessions, {failed} failed")

    # ══════════════════════════════════════════════════════════
    # STEP 2: PREPROCESSING
    # ══════════════════════════════════════════════════════════

    print_step(2, "🔧", "Preprocessing data...")

    from src.preprocessing import preprocess_all
    import pandas as pd

    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    race_processed_exists = os.path.exists(os.path.join(processed_dir, "race_winner_data.csv"))

    if race_processed_exists:
        print("\n   📂 Processed data already exists! Skipping preprocessing.")
        print("   💡 Delete data/processed/ folder to force re-processing.")
    else:
        preprocess_result = preprocess_all()
        if preprocess_result is None:
            print("\n   ❌ Preprocessing failed. Check raw data files.")
            print("   💡 Run data collection first (Step 1).")
            sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # STEP 3: WINNER PREDICTION
    # ══════════════════════════════════════════════════════════

    print_step(3, "🏆", "Training winner prediction models...")

    from src.winner_prediction import train_winner_prediction

    winner_results = train_winner_prediction()

    if not winner_results:
        print("\n   ⚠ Winner prediction returned no results.")
        winner_results = {}

    # ══════════════════════════════════════════════════════════
    # STEP 4: PIT STOP PREDICTION
    # ══════════════════════════════════════════════════════════

    print_step(4, "⏱️", "Training pit stop prediction models...")

    from src.pitstop_prediction import train_pitstop_prediction

    pitstop_results = train_pitstop_prediction()

    if not pitstop_results:
        print("\n   ⚠ Pit stop prediction returned no results.")
        pitstop_results = {}

    # ══════════════════════════════════════════════════════════
    # STEP 5: VISUALIZATIONS
    # ══════════════════════════════════════════════════════════

    print_step(5, "📊", "Generating F1 visualizations...")

    from src.visualizations import generate_all_visualizations
    import pandas as pd

    # Load full processed data for visualizations
    race_full_path = os.path.join(processed_dir, "race_full_processed.csv")
    pitstop_full_path = os.path.join(processed_dir, "pitstop_full_processed.csv")
    tyre_raw_path = os.path.join(raw_data_dir, "tyre_data_raw.csv")

    race_full = pd.read_csv(race_full_path) if os.path.exists(race_full_path) else pd.DataFrame()
    pitstop_full = pd.read_csv(pitstop_full_path) if os.path.exists(pitstop_full_path) else pd.DataFrame()
    tyre_df = pd.read_csv(tyre_raw_path) if os.path.exists(tyre_raw_path) else pd.DataFrame()

    # Load processed ML data
    race_p1_path = os.path.join(processed_dir, "race_winner_data.csv")
    pitstop_p2_path = os.path.join(processed_dir, "pitstop_duration_data.csv")

    race_p1 = pd.read_csv(race_p1_path) if os.path.exists(race_p1_path) else pd.DataFrame()
    pitstop_p2 = pd.read_csv(pitstop_p2_path) if os.path.exists(pitstop_p2_path) else pd.DataFrame()

    # Build the visualization data dict
    viz_data = {
        "race_full": race_full,
        "pitstop_full": pitstop_full,
        "tyre_df": tyre_df,
        "race_p1": race_p1,
        "pitstop_p2": pitstop_p2,
        "winner_results": winner_results,
        "pitstop_results": pitstop_results,
    }

    # Add learning curves from winner prediction
    if "learning_curves" in winner_results:
        viz_data["learning_curves"] = winner_results["learning_curves"]

    # Add cross-validation results from both problems
    cv_results = {}
    if "lr_cv_scores" in winner_results:
        cv_results["Winner LR (RMSE)"] = winner_results["lr_cv_scores"]
    if "knn_cv_scores" in winner_results:
        cv_results["Winner KNN (Accuracy)"] = winner_results["knn_cv_scores"]
    if "lr_cv_scores" in pitstop_results:
        cv_results["PitStop LR (RMSE)"] = pitstop_results["lr_cv_scores"]
    if "knn_cv_scores" in pitstop_results:
        cv_results["PitStop KNN (RMSE)"] = pitstop_results["knn_cv_scores"]
    if cv_results:
        viz_data["cv_results"] = cv_results

    # Add feature importance from winner prediction
    if "feature_importance" in winner_results:
        viz_data["feature_importance"] = winner_results["feature_importance"]

    # Add season metrics
    if "season_metrics" in winner_results:
        viz_data["season_metrics"] = winner_results["season_metrics"]

    generate_all_visualizations(viz_data)

    # ══════════════════════════════════════════════════════════
    # STEP 6: PDF REPORT
    # ══════════════════════════════════════════════════════════

    print_step(6, "📄", "Generating report...")

    from src.report_generator import generate_report

    report_path = generate_report(winner_results, pitstop_results)

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*65}")
    print(f"   ✅  PROJECT COMPLETE! — Lights Out and Away")
    print(f"{'='*65}")
    print(f"   ⏱️  Total time: {minutes}m {seconds}s")
    print(f"   📂  Data:    data/raw/ and data/processed/")
    print(f"   🤖  Models:  models/")
    print(f"   📊  Plots:   outputs/plots/")
    print(f"   📄  Report:  outputs/reports/lights_out_and_away_report.pdf")
    print(f"{'='*65}")
    print(f"   🏎️  Check outputs/ folder for all results!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
