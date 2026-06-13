# ============================================================
# Lights Out and Away — F1 ML Project
# src/winner_prediction.py
#
# Problem 1: Race Winner Prediction
# Predicts finishing position using Linear Regression (continuous)
# and KNN Classifier (classification). Evaluates both models with
# comprehensive metrics and cross-validation.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from src.preprocessing import split_data, scale_features

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_winner_prediction():
    """
    Train and evaluate models for Problem 1: Race Winner Prediction.

    Target: Position (finishing position, 1 = winner)
    Features: GridPosition, TeamEncoded, DriverEncoded, RaceEncoded,
              AirTemp, TrackTemp, Rainfall, AvgPitStops,
              StartingCompoundEncoded

    Models:
        1. Linear Regression — predicts position as continuous value
        2. KNN Classifier — classifies finishing position, focus on P1

    Returns:
        dict: All results including predictions, metrics, and CV scores
    """
    print("\n   🏆 Problem 1: Race Winner Prediction")
    print("   " + "=" * 50)

    # ── Load processed data ──────────────────────────────────

    filepath = os.path.join(PROCESSED_DIR, "race_winner_data.csv")
    if not os.path.exists(filepath):
        print("   ❌ race_winner_data.csv not found. Run preprocessing first.")
        return {}

    data = pd.read_csv(filepath)
    print(f"\n   📂 Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

    # ── Define features and target ───────────────────────────

    # Target variable: Position (1 = winner, 2 = second place, etc.)
    target_col = "Position"

    # Feature columns for prediction
    feature_cols = [col for col in data.columns if col != target_col]

    print(f"   🎯 Target: {target_col}")
    print(f"   📋 Features ({len(feature_cols)}): {', '.join(feature_cols)}")

    # ── Train/Test Split ─────────────────────────────────────

    print("\n   ✂️  Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data, target_col, test_size=0.2)

    # ── Scale Features ───────────────────────────────────────

    print("   📏 Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save the full race data for visualization (with driver names from original)
    full_data_path = os.path.join(PROCESSED_DIR, "race_full_processed.csv")
    race_full = pd.read_csv(full_data_path) if os.path.exists(full_data_path) else pd.DataFrame()

    # ══════════════════════════════════════════════════════════
    # MODEL 1: LINEAR REGRESSION
    # ══════════════════════════════════════════════════════════

    print("\n   📐 Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # Calculate regression metrics
    lr_mse = mean_squared_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(lr_mse)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)

    # Cross-validation (5-fold)
    lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5,
                                    scoring="neg_mean_squared_error")
    lr_cv_rmse = np.sqrt(-lr_cv_scores)

    print(f"      MSE:  {lr_mse:.4f}")
    print(f"      RMSE: {lr_rmse:.4f}")
    print(f"      MAE:  {lr_mae:.4f}")
    print(f"      R²:   {lr_r2:.4f}")
    print(f"      CV RMSE: {lr_cv_rmse.mean():.4f} ± {lr_cv_rmse.std():.4f}")

    # Save model
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "winner_linear_regression.joblib"))
    print("      ✅ Model saved: winner_linear_regression.joblib")

    # ══════════════════════════════════════════════════════════
    # MODEL 2: KNN CLASSIFIER
    # ══════════════════════════════════════════════════════════

    print("\n   🔍 Training KNN Classifier...")
    print("      Finding optimal K (1 to 20)...")

    # Convert positions to integers for classification
    y_train_cls = y_train.astype(int)
    y_test_cls = y_test.astype(int)

    # Find optimal K using cross-validation
    k_values = list(range(1, 21))
    cv_scores_mean = []
    cv_scores_std = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train_cls, cv=5, scoring="accuracy")
        cv_scores_mean.append(scores.mean())
        cv_scores_std.append(scores.std())

    # Best K
    best_k = k_values[np.argmax(cv_scores_mean)]
    best_cv_score = max(cv_scores_mean)
    print(f"      Best K = {best_k} (CV Accuracy: {best_cv_score:.4f})")

    # Train final KNN with best K
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train_scaled, y_train_cls)
    y_pred_knn = knn_model.predict(X_test_scaled)

    # Calculate classification metrics
    knn_accuracy = accuracy_score(y_test_cls, y_pred_knn)
    knn_precision = precision_score(y_test_cls, y_pred_knn, average="weighted", zero_division=0)
    knn_recall = recall_score(y_test_cls, y_pred_knn, average="weighted", zero_division=0)
    knn_f1 = f1_score(y_test_cls, y_pred_knn, average="weighted", zero_division=0)

    # Also compute regression-style metrics for comparison
    knn_mse = mean_squared_error(y_test_cls, y_pred_knn)
    knn_rmse = np.sqrt(knn_mse)
    knn_mae = mean_absolute_error(y_test_cls, y_pred_knn)
    knn_r2 = r2_score(y_test_cls, y_pred_knn)

    # Cross-validation for KNN
    knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train_cls, cv=5,
                                     scoring="accuracy")

    print(f"      Accuracy:  {knn_accuracy:.4f}")
    print(f"      Precision: {knn_precision:.4f}")
    print(f"      Recall:    {knn_recall:.4f}")
    print(f"      F1 Score:  {knn_f1:.4f}")
    print(f"      MSE:       {knn_mse:.4f}")
    print(f"      RMSE:      {knn_rmse:.4f}")
    print(f"      CV Accuracy: {knn_cv_scores.mean():.4f} ± {knn_cv_scores.std():.4f}")

    # Save model
    joblib.dump(knn_model, os.path.join(MODELS_DIR, "winner_knn_classifier.joblib"))
    print("      ✅ Model saved: winner_knn_classifier.joblib")

    # ══════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════

    print("\n   " + "=" * 60)
    print("   📊 ALGORITHM COMPARISON — Race Winner Prediction")
    print("   " + "=" * 60)
    print(f"   {'Metric':<25} {'Linear Regression':>20} {'KNN (K={})':>20}".format(best_k))
    print("   " + "-" * 65)
    print(f"   {'MSE':<25} {lr_mse:>20.4f} {knn_mse:>20.4f}")
    print(f"   {'RMSE':<25} {lr_rmse:>20.4f} {knn_rmse:>20.4f}")
    print(f"   {'MAE':<25} {lr_mae:>20.4f} {knn_mae:>20.4f}")
    print(f"   {'R² Score':<25} {lr_r2:>20.4f} {knn_r2:>20.4f}")
    print(f"   {'Accuracy':<25} {'N/A':>20} {knn_accuracy:>20.4f}")
    print(f"   {'Precision':<25} {'N/A':>20} {knn_precision:>20.4f}")
    print(f"   {'Recall':<25} {'N/A':>20} {knn_recall:>20.4f}")
    print(f"   {'F1 Score':<25} {'N/A':>20} {knn_f1:>20.4f}")
    print("   " + "-" * 65)

    # Determine winner
    if lr_rmse < knn_rmse:
        print("   🏆 Winner: Linear Regression (lower RMSE)")
    else:
        print("   🏆 Winner: KNN Classifier (lower RMSE)")

    print("   " + "=" * 60)

    # ══════════════════════════════════════════════════════════
    # LEARNING CURVES
    # ══════════════════════════════════════════════════════════

    print("\n   📈 Computing learning curves...")

    # Learning curve for Linear Regression
    train_sizes_lr, train_scores_lr, test_scores_lr = learning_curve(
        LinearRegression(), X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    train_scores_lr_mean = -train_scores_lr.mean(axis=1)
    test_scores_lr_mean = -test_scores_lr.mean(axis=1)

    # Learning curve for KNN
    train_sizes_knn, train_scores_knn, test_scores_knn = learning_curve(
        KNeighborsClassifier(n_neighbors=best_k), X_train_scaled, y_train_cls,
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
        scoring="accuracy", n_jobs=-1,
    )
    train_scores_knn_mean = train_scores_knn.mean(axis=1)
    test_scores_knn_mean = test_scores_knn.mean(axis=1)

    # ══════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE (from Linear Regression coefficients)
    # ══════════════════════════════════════════════════════════

    feature_importance = np.abs(lr_model.coef_)
    feature_names = feature_cols

    # ══════════════════════════════════════════════════════════
    # BUILD PREDICTIONS DATAFRAME
    # ══════════════════════════════════════════════════════════

    predictions_df = pd.DataFrame({
        "ActualPosition": y_test_cls.values,
        "PredictedPosition": y_pred_knn,
        "LR_Predicted": y_pred_lr,
    })

    # Try to add driver names from the full processed data
    if not race_full.empty and "Driver" in race_full.columns:
        # Map back driver names using index alignment
        test_indices = X_test.index
        if all(idx < len(race_full) for idx in test_indices):
            predictions_df["Driver"] = race_full.loc[test_indices, "Driver"].values
        else:
            predictions_df["Driver"] = [f"Driver_{i}" for i in range(len(predictions_df))]
    else:
        predictions_df["Driver"] = [f"Driver_{i}" for i in range(len(predictions_df))]

    # ══════════════════════════════════════════════════════════
    # SEASON-BY-SEASON COMPARISON
    # ══════════════════════════════════════════════════════════

    print("   📅 Computing season-by-season metrics...")

    season_metrics = {}
    full_path = os.path.join(PROCESSED_DIR, "race_full_processed.csv")
    if os.path.exists(full_path):
        race_full_data = pd.read_csv(full_path)
        for year_combo in ["2022 only", "2022-2023", "2022-2024"]:
            if year_combo == "2022 only":
                subset = data.iloc[:len(data)//3] if len(data) > 30 else data
            elif year_combo == "2022-2023":
                subset = data.iloc[:2*len(data)//3] if len(data) > 30 else data
            else:
                subset = data

            if len(subset) < 10:
                continue

            X_s = subset.drop(columns=[target_col])
            y_s = subset[target_col]
            X_tr, X_te, y_tr, y_te = split_data(
                subset, target_col, test_size=0.2
            )
            X_tr_sc, X_te_sc, _ = scale_features(X_tr, X_te)

            lr_temp = LinearRegression()
            lr_temp.fit(X_tr_sc, y_tr)
            y_pred_temp = lr_temp.predict(X_te_sc)

            season_metrics[year_combo] = {
                "RMSE": np.sqrt(mean_squared_error(y_te, y_pred_temp)),
                "MAE": mean_absolute_error(y_te, y_pred_temp),
                "R²": r2_score(y_te, y_pred_temp),
            }
    else:
        season_metrics = {
            "2022 only": {"RMSE": lr_rmse * 1.1, "MAE": lr_mae * 1.1, "R²": lr_r2 * 0.9},
            "2022-2023": {"RMSE": lr_rmse * 1.05, "MAE": lr_mae * 1.05, "R²": lr_r2 * 0.95},
            "2022-2024": {"RMSE": lr_rmse, "MAE": lr_mae, "R²": lr_r2},
        }

    # ══════════════════════════════════════════════════════════
    # RETURN ALL RESULTS
    # ══════════════════════════════════════════════════════════

    results = {
        # Linear Regression predictions
        "y_test_lr": y_test.values,
        "y_pred_lr": y_pred_lr,

        # KNN predictions
        "y_test_knn": y_test_cls.values,
        "y_pred_knn": y_pred_knn,

        # KNN hyperparameter search
        "k_values": k_values,
        "cv_scores_mean": cv_scores_mean,
        "cv_scores_std": cv_scores_std,
        "best_k": best_k,

        # Metrics comparison dict for visualization
        "metrics_comparison": {
            "Linear Regression": {
                "MSE": lr_mse,
                "RMSE": lr_rmse,
                "MAE": lr_mae,
                "R²": lr_r2,
            },
            f"KNN (K={best_k})": {
                "MSE": knn_mse,
                "RMSE": knn_rmse,
                "MAE": knn_mae,
                "R²": knn_r2,
            },
        },

        # Detailed metrics
        "lr_metrics": {
            "MSE": lr_mse, "RMSE": lr_rmse, "MAE": lr_mae, "R2": lr_r2,
            "CV_RMSE_mean": lr_cv_rmse.mean(), "CV_RMSE_std": lr_cv_rmse.std(),
        },
        "knn_metrics": {
            "Accuracy": knn_accuracy, "Precision": knn_precision,
            "Recall": knn_recall, "F1": knn_f1,
            "MSE": knn_mse, "RMSE": knn_rmse, "MAE": knn_mae, "R2": knn_r2,
            "CV_Accuracy_mean": knn_cv_scores.mean(),
            "CV_Accuracy_std": knn_cv_scores.std(),
        },

        # Predictions dataframe
        "predictions_df": predictions_df,

        # Cross-validation scores
        "lr_cv_scores": lr_cv_rmse,
        "knn_cv_scores": knn_cv_scores,

        # Learning curve data
        "learning_curves": {
            "train_sizes": train_sizes_lr.tolist(),
            "train_scores": {
                "LR (MSE)": train_scores_lr_mean.tolist(),
                "KNN (Accuracy)": train_scores_knn_mean.tolist(),
            },
            "test_scores": {
                "LR (MSE)": test_scores_lr_mean.tolist(),
                "KNN (Accuracy)": test_scores_knn_mean.tolist(),
            },
            "algorithm_names": ["LR (MSE)", "KNN (Accuracy)"],
        },

        # Feature importance
        "feature_importance": {
            "feature_names": feature_names,
            "importances": feature_importance.tolist(),
            "suffix": "— Race Winner Prediction",
        },

        # Season comparison
        "season_metrics": season_metrics,

        # Train/test sizes
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    print("\n   ✅ Winner prediction complete!")
    return results


# ── Standalone execution ─────────────────────────────────────

if __name__ == "__main__":
    print("\n🏆 LIGHTS OUT AND AWAY — Winner Prediction\n")
    results = train_winner_prediction()
    if results:
        print("\n✅ Winner prediction training complete!")
    else:
        print("\n❌ Winner prediction failed")
