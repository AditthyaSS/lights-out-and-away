# ============================================================
# Lights Out and Away — F1 ML Project
# src/preprocessing.py
#
# Loads raw CSV data collected by data_collection.py, cleans it,
# engineers features, encodes categorical variables, and splits
# into train/test sets for both ML problems.
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── Load Raw Data ────────────────────────────────────────────

def load_raw_data():
    """
    Load all raw CSV files from data/raw/ directory.

    Returns:
        dict: Dictionary of DataFrames keyed by dataset name.
    """
    print("   📂 Loading raw data files...")

    datasets = {}
    files = {
        "race": "race_data_raw.csv",
        "qualifying": "qualifying_data_raw.csv",
        "pitstop": "pitstop_data_raw.csv",
        "weather": "weather_data_raw.csv",
        "circuit": "circuit_data_raw.csv",
        "tyre": "tyre_data_raw.csv",
    }

    for name, filename in files.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if os.path.exists(filepath):
            datasets[name] = pd.read_csv(filepath)
            print(f"      ✅ {filename} — {len(datasets[name])} rows, {len(datasets[name].columns)} columns")
        else:
            print(f"      ⚠ {filename} not found, skipping")
            datasets[name] = pd.DataFrame()

    return datasets


# ── Clean Race Data ──────────────────────────────────────────

def clean_race_data(df):
    """
    Clean race results data:
    - Drop rows with null Position or GridPosition
    - Convert Position to integer
    - Remove rows where driver did not finish (DNF/DNS)
    - Fix any data type issues

    Args:
        df: Raw race results DataFrame

    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df

    print("   🧹 Cleaning race data...")
    initial_rows = len(df)

    # Make a copy to avoid modifying original
    df = df.copy()

    # Drop rows with missing position data
    df = df.dropna(subset=["Position", "GridPosition"])

    # Convert Position and GridPosition to numeric
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")

    # Remove rows where position couldn't be parsed
    df = df.dropna(subset=["Position", "GridPosition"])

    # Convert to integer
    df["Position"] = df["Position"].astype(int)
    df["GridPosition"] = df["GridPosition"].astype(int)

    # Remove pit lane starts (GridPosition = 0) — set to 20
    df.loc[df["GridPosition"] == 0, "GridPosition"] = 20

    # Fill missing Points with 0
    df["Points"] = df["Points"].fillna(0)

    # Fill missing FastestLapTime with median
    if "FastestLapTime" in df.columns:
        df["FastestLapTime"] = df["FastestLapTime"].fillna(df["FastestLapTime"].median())

    final_rows = len(df)
    print(f"      Rows: {initial_rows} → {final_rows} (dropped {initial_rows - final_rows})")

    return df.reset_index(drop=True)


# ── Clean Pit Stop Data ─────────────────────────────────────

def clean_pitstop_data(df):
    """
    Clean pit stop data:
    - Remove pit stops longer than 60 seconds (likely penalties or issues)
    - Remove pit stops shorter than 1.5 seconds (data errors)
    - Drop rows with missing PitStopDuration
    - Remove unknown tyre compounds

    Args:
        df: Raw pit stop DataFrame

    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df

    print("   🧹 Cleaning pit stop data...")
    initial_rows = len(df)

    df = df.copy()

    # Drop rows with missing pit stop duration
    df = df.dropna(subset=["PitStopDuration"])

    # Remove outliers: too fast or too slow pit stops
    df = df[(df["PitStopDuration"] >= 1.5) & (df["PitStopDuration"] <= 60.0)]

    # Convert LapNumber to numeric
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df = df.dropna(subset=["LapNumber"])
    df["LapNumber"] = df["LapNumber"].astype(int)

    # Fill missing TyreLife with median
    if "TyreLife" in df.columns:
        df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
        df["TyreLife"] = df["TyreLife"].fillna(df["TyreLife"].median())

    # Remove rows with unknown compounds
    df = df[df["TyreCompound"] != "UNKNOWN"]

    final_rows = len(df)
    print(f"      Rows: {initial_rows} → {final_rows} (dropped {initial_rows - final_rows})")

    return df.reset_index(drop=True)


# ── Encode Features ──────────────────────────────────────────

def encode_features(df, columns_to_encode, prefix=""):
    """
    Apply Label Encoding to specified categorical columns.

    Args:
        df: DataFrame to encode
        columns_to_encode: List of column names to encode
        prefix: Prefix for encoded column names (default: "")

    Returns:
        tuple: (encoded DataFrame, dict of LabelEncoders)
    """
    df = df.copy()
    encoders = {}

    for col in columns_to_encode:
        if col in df.columns:
            le = LabelEncoder()
            # Fill NaN with a placeholder before encoding
            df[col] = df[col].fillna("Unknown")
            df[f"{col}Encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"      Encoded {col}: {len(le.classes_)} unique values")

    return df, encoders


# ── Feature Engineering ──────────────────────────────────────

def engineer_features(race_df, pitstop_df, weather_df, tyre_df):
    """
    Create additional features for both ML problems:
    - IsWinner: binary flag for Position == 1
    - AvgPitStops: average number of pit stops per race
    - TyreAge: laps on current tyre at pit stop
    - Merge weather data into race and pit stop data

    Args:
        race_df: Cleaned race results DataFrame
        pitstop_df: Cleaned pit stop DataFrame
        weather_df: Weather data DataFrame
        tyre_df: Tyre stint data DataFrame

    Returns:
        tuple: (enriched race DataFrame, enriched pit stop DataFrame)
    """
    print("   🔧 Engineering features...")

    # --- Race features ---
    race_df = race_df.copy()

    # IsWinner flag
    race_df["IsWinner"] = (race_df["Position"] == 1).astype(int)

    # Merge weather data into race results
    if not weather_df.empty:
        weather_agg = weather_df.groupby(["Year", "Race"]).agg({
            "AirTemp": "mean",
            "TrackTemp": "mean",
            "Humidity": "mean",
            "Rainfall": "max",
            "WindSpeed": "mean",
        }).reset_index()

        race_df = race_df.merge(
            weather_agg, on=["Year", "Race"], how="left"
        )
    else:
        race_df["AirTemp"] = 25.0
        race_df["TrackTemp"] = 35.0
        race_df["Humidity"] = 50.0
        race_df["Rainfall"] = 0
        race_df["WindSpeed"] = 5.0

    # Calculate average pit stops per race
    if not pitstop_df.empty:
        avg_pits = pitstop_df.groupby(["Year", "Race"])["Driver"].count().reset_index()
        avg_pits.columns = ["Year", "Race", "TotalPitStops"]
        # Average per driver (roughly 20 drivers per race)
        avg_pits["AvgPitStops"] = avg_pits["TotalPitStops"] / 20.0
        race_df = race_df.merge(
            avg_pits[["Year", "Race", "AvgPitStops"]],
            on=["Year", "Race"], how="left"
        )
    else:
        race_df["AvgPitStops"] = 2.0

    race_df["AvgPitStops"] = race_df["AvgPitStops"].fillna(2.0)

    # Get starting tyre compound from tyre data
    if not tyre_df.empty:
        first_stint = tyre_df[tyre_df["Stint"] == 1][["Driver", "Year", "Race", "Compound"]].copy()
        first_stint.columns = ["Driver", "Year", "Race", "StartingCompound"]
        race_df = race_df.merge(
            first_stint, on=["Driver", "Year", "Race"], how="left"
        )
    else:
        race_df["StartingCompound"] = "MEDIUM"

    race_df["StartingCompound"] = race_df["StartingCompound"].fillna("MEDIUM")

    # Fill any remaining NaN weather values with sensible defaults
    race_df["AirTemp"] = race_df["AirTemp"].fillna(25.0)
    race_df["TrackTemp"] = race_df["TrackTemp"].fillna(35.0)
    race_df["Humidity"] = race_df["Humidity"].fillna(50.0)
    race_df["Rainfall"] = race_df["Rainfall"].fillna(0).astype(int)
    race_df["WindSpeed"] = race_df["WindSpeed"].fillna(5.0)

    # --- Pit stop features ---
    pitstop_df = pitstop_df.copy()

    if not pitstop_df.empty:
        # Merge weather into pit stop data
        if not weather_df.empty:
            pitstop_df = pitstop_df.merge(
                weather_agg, on=["Year", "Race"], how="left"
            )
        else:
            pitstop_df["AirTemp"] = 25.0
            pitstop_df["TrackTemp"] = 35.0

        # Fill NaN
        pitstop_df["AirTemp"] = pitstop_df["AirTemp"].fillna(25.0)
        pitstop_df["TrackTemp"] = pitstop_df["TrackTemp"].fillna(35.0)

        # Rename TyreLife to TyreAge for clarity
        if "TyreLife" in pitstop_df.columns:
            pitstop_df.rename(columns={"TyreLife": "TyreAge"}, inplace=True)
        else:
            pitstop_df["TyreAge"] = 15  # default

        pitstop_df["TyreAge"] = pitstop_df["TyreAge"].fillna(15)

        # Add RaceYear as a feature
        pitstop_df["RaceYear"] = pitstop_df["Year"]

    print(f"      Race features: {race_df.shape}")
    print(f"      Pit stop features: {pitstop_df.shape}")

    return race_df, pitstop_df


# ── Split Data ───────────────────────────────────────────────

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split DataFrame into training and test sets.

    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        test_size: Fraction for test set (default 0.2)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"      Train set: {X_train.shape[0]} samples")
    print(f"      Test set:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


# ── Scale Features ───────────────────────────────────────────

def scale_features(X_train, X_test):
    """
    Apply StandardScaler to training and test features.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print("      ✅ Features scaled with StandardScaler")

    return X_train_scaled, X_test_scaled, scaler


# ── Main Preprocessing Pipeline ─────────────────────────────

def preprocess_all():
    """
    Run the full preprocessing pipeline:
    1. Load raw data
    2. Clean race and pit stop data
    3. Engineer features
    4. Encode categorical variables
    5. Prepare datasets for both ML problems
    6. Save processed data to data/processed/

    Returns:
        dict: Contains processed DataFrames and metadata for both problems
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Step 1: Load raw data
    datasets = load_raw_data()

    race_df = datasets.get("race", pd.DataFrame())
    pitstop_df = datasets.get("pitstop", pd.DataFrame())
    weather_df = datasets.get("weather", pd.DataFrame())
    tyre_df = datasets.get("tyre", pd.DataFrame())

    if race_df.empty:
        print("   ❌ No race data available. Cannot proceed.")
        return None

    # Step 2: Clean data
    race_clean = clean_race_data(race_df)
    pitstop_clean = clean_pitstop_data(pitstop_df)

    # Step 3: Engineer features
    race_enriched, pitstop_enriched = engineer_features(
        race_clean, pitstop_clean, weather_df, tyre_df
    )

    # Step 4: Encode categorical variables for race data
    print("   🏷️  Encoding categorical features...")

    # --- Problem 1: Race Winner Prediction ---
    race_encoded, race_encoders = encode_features(
        race_enriched,
        columns_to_encode=["Team", "Driver", "Race", "StartingCompound"],
    )

    # Select features for Problem 1
    p1_features = [
        "GridPosition", "TeamEncoded", "DriverEncoded", "RaceEncoded",
        "AirTemp", "TrackTemp", "Rainfall", "AvgPitStops",
        "StartingCompoundEncoded",
    ]
    p1_target = "Position"

    # Keep only rows where all features exist
    p1_cols = p1_features + [p1_target]
    race_p1 = race_encoded[[c for c in p1_cols if c in race_encoded.columns]].copy()
    race_p1 = race_p1.dropna()

    print(f"\n   📊 Problem 1 dataset shape: {race_p1.shape}")

    # --- Problem 2: Pit Stop Duration Prediction ---
    if not pitstop_enriched.empty:
        pitstop_encoded, pitstop_encoders = encode_features(
            pitstop_enriched,
            columns_to_encode=["Team", "TyreCompound", "Race", "Driver"],
        )

        p2_features = [
            "TeamEncoded", "TyreCompoundEncoded", "LapNumber", "RaceEncoded",
            "AirTemp", "TrackTemp", "TyreAge", "RaceYear", "DriverEncoded",
        ]
        p2_target = "PitStopDuration"

        p2_cols = p2_features + [p2_target]
        pitstop_p2 = pitstop_encoded[[c for c in p2_cols if c in pitstop_encoded.columns]].copy()
        pitstop_p2 = pitstop_p2.dropna()

        print(f"   📊 Problem 2 dataset shape: {pitstop_p2.shape}")
    else:
        pitstop_p2 = pd.DataFrame()
        pitstop_encoders = {}
        print("   ⚠ No pit stop data for Problem 2")

    # Step 5: Save processed data
    print("\n   💾 Saving processed data...")

    race_p1.to_csv(os.path.join(PROCESSED_DATA_DIR, "race_winner_data.csv"), index=False)
    print(f"      ✅ race_winner_data.csv — {len(race_p1)} rows")

    if not pitstop_p2.empty:
        pitstop_p2.to_csv(os.path.join(PROCESSED_DATA_DIR, "pitstop_duration_data.csv"), index=False)
        print(f"      ✅ pitstop_duration_data.csv — {len(pitstop_p2)} rows")

    # Save full enriched datasets for visualization
    race_enriched.to_csv(os.path.join(PROCESSED_DATA_DIR, "race_full_processed.csv"), index=False)
    if not pitstop_enriched.empty:
        pitstop_enriched.to_csv(os.path.join(PROCESSED_DATA_DIR, "pitstop_full_processed.csv"), index=False)

    # Save encoder mappings for reference
    encoder_info = {}
    for name, le in {**race_encoders, **pitstop_encoders}.items():
        encoder_info[name] = dict(zip(le.classes_, le.transform(le.classes_)))
    encoder_df = pd.DataFrame(
        [(k, str(v)) for k, v in encoder_info.items()],
        columns=["Feature", "Mapping"],
    )
    encoder_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "encoder_mappings.csv"), index=False)

    print(f"\n   ✅ Preprocessing complete! Data saved to {PROCESSED_DATA_DIR}")

    return {
        "race_p1": race_p1,
        "pitstop_p2": pitstop_p2,
        "race_full": race_enriched,
        "pitstop_full": pitstop_enriched if not pitstop_enriched.empty else pd.DataFrame(),
        "race_encoders": race_encoders,
        "pitstop_encoders": pitstop_encoders,
        "p1_features": [c for c in p1_features if c in race_p1.columns],
        "p2_features": [c for c in p2_features if c in pitstop_p2.columns] if not pitstop_p2.empty else [],
        "p1_target": p1_target,
        "p2_target": "PitStopDuration",
    }


# ── Standalone execution ─────────────────────────────────────

if __name__ == "__main__":
    print("\n🔧 LIGHTS OUT AND AWAY — Preprocessing\n")
    result = preprocess_all()
    if result:
        print("\n✅ Preprocessing complete!")
    else:
        print("\n❌ Preprocessing failed — check raw data files")
