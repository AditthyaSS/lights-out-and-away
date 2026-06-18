# ============================================================
# Lights Out and Away — F1 ML Project
# src/pitstop_prediction.py
#
# Problem 2: Pit Stop Duration Prediction
# Predicts pit stop duration in seconds using Linear Regression
# and KNN Regressor. Evaluates both models with comprehensive
# metrics, cross-validation, and residual analysis.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.preprocessing import split_data, scale_features

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_pitstop_prediction():
    """
    Train and evaluate models for Problem 2: Pit Stop Duration Prediction.

    Target: PitStopDuration (seconds)
    Features: TeamEncoded, TyreCompoundEncoded, LapNumber, RaceEncoded,
              AirTemp, TrackTemp, TyreAge, RaceYear, DriverEncoded

    Models:
        1. Linear Regression — predicts pit stop duration
        2. KNN Regressor — predicts pit stop duration

    Returns:
        dict: All results including predictions, metrics, and CV scores
    """
    print("\n   [P2] Problem 2: Pit Stop Duration Prediction")
    print("   " + "=" * 50)

    # ── Load processed data ──────────────────────────────────

    filepath = os.path.join(PROCESSED_DIR, "pitstop_duration_data.csv")
    if not os.path.exists(filepath):
        print("   ❌ pitstop_duration_data.csv not found. Run preprocessing first.")
        return {}

    data = pd.read_csv(filepath)
    print(f"\n   📂 Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

    if len(data) < 20:
        print("   ⚠ Not enough data for pit stop prediction (need at least 20 rows)")
        return {}

    # ── Define features and target ───────────────────────────

    # Target variable: PitStopDuration (in seconds)
    target_col = "PitStopDuration"

    # Feature columns for prediction
    feature_cols = [col for col in data.columns if col != target_col]

    print(f"   🎯 Target: {target_col}")
    print(f"   📋 Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"   📊 Target stats: mean={data[target_col].mean():.2f}s, "
          f"std={data[target_col].std():.2f}s, "
          f"min={data[target_col].min():.2f}s, max={data[target_col].max():.2f}s")

    # ── Train/Test Split ─────────────────────────────────────

    print("\n   ✂️  Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data, target_col, test_size=0.2)

    # ── Scale Features ───────────────────────────────────────

    print("   📏 Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ══════════════════════════════════════════════════════════
    # MODEL 1: LINEAR REGRESSION
    # ══════════════════════════════════════════════════════════

    print("\n   📐 Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # Calculate metrics
    lr_mse = mean_squared_error(y_test, y_pred_lr)
    lr_rmse = np.sqrt(lr_mse)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)

    # Cross-validation (5-fold)
    lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5,
                                    scoring="neg_mean_squared_error")
    lr_cv_rmse = np.sqrt(-lr_cv_scores)

    # Residual analysis
    lr_residuals = y_test.values - y_pred_lr
    lr_residual_mean = np.mean(lr_residuals)
    lr_residual_std = np.std(lr_residuals)

    print(f"      MSE:  {lr_mse:.4f}")
    print(f"      RMSE: {lr_rmse:.4f}")
    print(f"      MAE:  {lr_mae:.4f}")
    print(f"      R²:   {lr_r2:.4f}")
    print(f"      CV RMSE: {lr_cv_rmse.mean():.4f} ± {lr_cv_rmse.std():.4f}")
    print(f"      Residual Mean: {lr_residual_mean:.4f}, Std: {lr_residual_std:.4f}")

    # Save model
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "pitstop_linear_regression.joblib"))
    print("      ✅ Model saved: pitstop_linear_regression.joblib")

    # ══════════════════════════════════════════════════════════
    # MODEL 2: KNN REGRESSOR
    # ══════════════════════════════════════════════════════════

    print("\n   🔍 Training KNN Regressor...")
    print("      Finding optimal K (1 to 20)...")

    # Find optimal K using cross-validation
    k_values = list(range(1, 21))
    cv_scores_mean = []
    cv_scores_std = []

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5,
                                  scoring="neg_mean_squared_error")
        rmse_scores = np.sqrt(-scores)
        cv_scores_mean.append(rmse_scores.mean())
        cv_scores_std.append(rmse_scores.std())

    # Best K (lowest RMSE)
    best_k = k_values[np.argmin(cv_scores_mean)]
    best_cv_rmse = min(cv_scores_mean)
    print(f"      Best K = {best_k} (CV RMSE: {best_cv_rmse:.4f})")

    # Train final KNN with best K
    knn_model = KNeighborsRegressor(n_neighbors=best_k)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)

    # Calculate metrics
    knn_mse = mean_squared_error(y_test, y_pred_knn)
    knn_rmse = np.sqrt(knn_mse)
    knn_mae = mean_absolute_error(y_test, y_pred_knn)
    knn_r2 = r2_score(y_test, y_pred_knn)

    # Cross-validation for final KNN
    knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5,
                                     scoring="neg_mean_squared_error")
    knn_cv_rmse = np.sqrt(-knn_cv_scores)

    # Residual analysis
    knn_residuals = y_test.values - y_pred_knn
    knn_residual_mean = np.mean(knn_residuals)
    knn_residual_std = np.std(knn_residuals)

    print(f"      MSE:  {knn_mse:.4f}")
    print(f"      RMSE: {knn_rmse:.4f}")
    print(f"      MAE:  {knn_mae:.4f}")
    print(f"      R²:   {knn_r2:.4f}")
    print(f"      CV RMSE: {knn_cv_rmse.mean():.4f} ± {knn_cv_rmse.std():.4f}")
    print(f"      Residual Mean: {knn_residual_mean:.4f}, Std: {knn_residual_std:.4f}")

    # Save model
    joblib.dump(knn_model, os.path.join(MODELS_DIR, "pitstop_knn_regressor.joblib"))
    print("      ✅ Model saved: pitstop_knn_regressor.joblib")

    # ══════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════

    print("\n   " + "=" * 60)
    print("   📊 ALGORITHM COMPARISON — Pit Stop Duration Prediction")
    print("   " + "=" * 60)
    print(f"   {'Metric':<25} {'Linear Regression':>20} {'KNN (K={})':>20}".format(best_k))
    print("   " + "-" * 65)
    print(f"   {'MSE':<25} {lr_mse:>20.4f} {knn_mse:>20.4f}")
    print(f"   {'RMSE':<25} {lr_rmse:>20.4f} {knn_rmse:>20.4f}")
    print(f"   {'MAE':<25} {lr_mae:>20.4f} {knn_mae:>20.4f}")
    print(f"   {'R² Score':<25} {lr_r2:>20.4f} {knn_r2:>20.4f}")
    print(f"   {'CV RMSE Mean':<25} {lr_cv_rmse.mean():>20.4f} {knn_cv_rmse.mean():>20.4f}")
    print(f"   {'Residual Mean':<25} {lr_residual_mean:>20.4f} {knn_residual_mean:>20.4f}")
    print(f"   {'Residual Std':<25} {lr_residual_std:>20.4f} {knn_residual_std:>20.4f}")
    print("   " + "-" * 65)

    # Determine winner
    if lr_rmse < knn_rmse:
        print("   🏆 Winner: Linear Regression (lower RMSE)")
    else:
        print("   🏆 Winner: KNN Regressor (lower RMSE)")

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
    train_rmse_lr = np.sqrt(-train_scores_lr.mean(axis=1))
    test_rmse_lr = np.sqrt(-test_scores_lr.mean(axis=1))

    # Learning curve for KNN
    train_sizes_knn, train_scores_knn, test_scores_knn = learning_curve(
        KNeighborsRegressor(n_neighbors=best_k), X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    train_rmse_knn = np.sqrt(-train_scores_knn.mean(axis=1))
    test_rmse_knn = np.sqrt(-test_scores_knn.mean(axis=1))

    # ══════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════

    feature_importance = np.abs(lr_model.coef_)
    feature_names = feature_cols

    # ══════════════════════════════════════════════════════════
    # RETURN ALL RESULTS
    # ══════════════════════════════════════════════════════════

    results = {
        # Linear Regression predictions
        "y_test_lr": y_test.values,
        "y_pred_lr": y_pred_lr,

        # KNN predictions
        "y_test_knn": y_test.values,
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
            "Residual_Mean": lr_residual_mean, "Residual_Std": lr_residual_std,
        },
        "knn_metrics": {
            "MSE": knn_mse, "RMSE": knn_rmse, "MAE": knn_mae, "R2": knn_r2,
            "CV_RMSE_mean": knn_cv_rmse.mean(), "CV_RMSE_std": knn_cv_rmse.std(),
            "Residual_Mean": knn_residual_mean, "Residual_Std": knn_residual_std,
        },

        # Cross-validation scores
        "lr_cv_scores": lr_cv_rmse,
        "knn_cv_scores": knn_cv_rmse,

        # Learning curves
        "learning_curves": {
            "train_sizes": train_sizes_lr.tolist(),
            "train_scores": {
                "LR (RMSE)": train_rmse_lr.tolist(),
                "KNN (RMSE)": train_rmse_knn.tolist(),
            },
            "test_scores": {
                "LR (RMSE)": test_rmse_lr.tolist(),
                "KNN (RMSE)": test_rmse_knn.tolist(),
            },
            "algorithm_names": ["LR (RMSE)", "KNN (RMSE)"],
        },

        # Feature importance
        "feature_importance": {
            "feature_names": feature_names,
            "importances": feature_importance.tolist(),
            "suffix": "— Pit Stop Duration",
        },

        # Train/test sizes
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    print("\n   ✅ Pit stop prediction complete!")
    return results


# ── Standalone execution ─────────────────────────────────────

if __name__ == "__main__":
    print("\n⏱️  LIGHTS OUT AND AWAY — Pit Stop Prediction\n")
    results = train_pitstop_prediction()
    if results:
        print("\n✅ Pit stop prediction training complete!")
    else:
        print("\n❌ Pit stop prediction failed")
