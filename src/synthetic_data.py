# ============================================================
# Lights Out and Away — F1 ML Project
# src/synthetic_data.py
#
# Generates realistic synthetic F1 data when FastF1 API data
# is unavailable. Uses real team/driver/circuit statistics from
# the 2022-2024 seasons to produce statistically accurate mock
# data for all six CSV files required by the pipeline.
#
# This module is called automatically by data_collection.py
# when FastF1 session loading fails, ensuring the pipeline
# always produces usable output even without internet access.
# ============================================================

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── Random seed for reproducibility ─────────────────────────
RNG = np.random.default_rng(42)

# ── Real-world constants ──────────────────────────────────────

SEASONS = [2022, 2023, 2024]

RACES = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "United States", "Mexico",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]

# Real 2022-2024 driver lineup with team and relative performance
DRIVERS = {
    "VER": {"name": "Max Verstappen",       "team": "Red Bull Racing", "skill": 0.98},
    "PER": {"name": "Sergio Perez",         "team": "Red Bull Racing", "skill": 0.84},
    "LEC": {"name": "Charles Leclerc",      "team": "Ferrari",         "skill": 0.91},
    "SAI": {"name": "Carlos Sainz",         "team": "Ferrari",         "skill": 0.88},
    "HAM": {"name": "Lewis Hamilton",       "team": "Mercedes",        "skill": 0.93},
    "RUS": {"name": "George Russell",       "team": "Mercedes",        "skill": 0.87},
    "NOR": {"name": "Lando Norris",         "team": "McLaren",         "skill": 0.90},
    "PIA": {"name": "Oscar Piastri",        "team": "McLaren",         "skill": 0.85},
    "ALO": {"name": "Fernando Alonso",      "team": "Aston Martin",    "skill": 0.89},
    "STR": {"name": "Lance Stroll",         "team": "Aston Martin",    "skill": 0.78},
    "GAS": {"name": "Pierre Gasly",         "team": "Alpine",          "skill": 0.81},
    "OCO": {"name": "Esteban Ocon",         "team": "Alpine",          "skill": 0.80},
    "ALB": {"name": "Alexander Albon",      "team": "Williams",        "skill": 0.79},
    "SAR": {"name": "Logan Sargeant",       "team": "Williams",        "skill": 0.72},
    "TSU": {"name": "Yuki Tsunoda",         "team": "RB",              "skill": 0.80},
    "RIC": {"name": "Daniel Ricciardo",     "team": "RB",              "skill": 0.82},
    "BOT": {"name": "Valtteri Bottas",      "team": "Kick Sauber",     "skill": 0.80},
    "ZHO": {"name": "Zhou Guanyu",          "team": "Kick Sauber",     "skill": 0.75},
    "HUL": {"name": "Nico Hulkenberg",      "team": "Haas F1 Team",    "skill": 0.80},
    "MAG": {"name": "Kevin Magnussen",      "team": "Haas F1 Team",    "skill": 0.79},
}

DRIVER_CODES = list(DRIVERS.keys())
NUM_DRIVERS = len(DRIVER_CODES)

# Team baseline pit stop speed in seconds (mean, std)
TEAM_PIT_PARAMS = {
    "Red Bull Racing": (2.35, 0.18),
    "Ferrari":         (2.45, 0.22),
    "Mercedes":        (2.50, 0.20),
    "McLaren":         (2.40, 0.19),
    "Aston Martin":    (2.65, 0.25),
    "Alpine":          (2.70, 0.28),
    "Williams":        (2.75, 0.30),
    "RB":              (2.68, 0.26),
    "Kick Sauber":     (2.80, 0.32),
    "Haas F1 Team":    (2.72, 0.29),
}

# Points system
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Circuit weather profiles (air_temp_mean, track_temp_mean, rainfall_prob)
CIRCUIT_WEATHER = {
    "Bahrain":        (28, 38, 0.02),
    "Saudi Arabia":   (26, 34, 0.01),
    "Australia":      (22, 30, 0.12),
    "Japan":          (18, 24, 0.15),
    "China":          (20, 27, 0.10),
    "Miami":          (30, 44, 0.08),
    "Emilia Romagna": (19, 26, 0.14),
    "Monaco":         (22, 32, 0.10),
    "Canada":         (20, 28, 0.18),
    "Spain":          (26, 38, 0.05),
    "Austria":        (20, 30, 0.20),
    "Great Britain":  (17, 22, 0.25),
    "Hungary":        (28, 40, 0.08),
    "Belgium":        (16, 21, 0.30),
    "Netherlands":    (18, 24, 0.18),
    "Italy":          (25, 35, 0.06),
    "Azerbaijan":     (24, 32, 0.04),
    "Singapore":      (30, 38, 0.20),
    "United States":  (22, 30, 0.10),
    "Mexico":         (20, 28, 0.08),
    "Brazil":         (25, 32, 0.22),
    "Las Vegas":      (18, 22, 0.02),
    "Qatar":          (32, 48, 0.01),
    "Abu Dhabi":      (28, 36, 0.01),
}

# Tyre compound probabilities per race
COMPOUND_PROBS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_WEIGHTS = [0.40, 0.35, 0.15, 0.06, 0.04]

# Circuit lap count
CIRCUIT_LAPS = {
    "Bahrain": 57, "Saudi Arabia": 50, "Australia": 58, "Japan": 53,
    "China": 56, "Miami": 57, "Emilia Romagna": 63, "Monaco": 78,
    "Canada": 70, "Spain": 66, "Austria": 71, "Great Britain": 52,
    "Hungary": 70, "Belgium": 44, "Netherlands": 72, "Italy": 53,
    "Azerbaijan": 51, "Singapore": 62, "United States": 56,
    "Mexico": 71, "Brazil": 71, "Las Vegas": 50, "Qatar": 57, "Abu Dhabi": 58,
}


def _generate_race_result(year, race):
    """Generate one race result with 20 drivers."""
    weather_params = CIRCUIT_WEATHER.get(race, (25, 35, 0.05))
    air_temp = float(RNG.normal(weather_params[0], 2))
    track_temp = float(RNG.normal(weather_params[1], 3))
    rainfall = int(RNG.random() < weather_params[2])

    # Build performance scores: skill + team year bonus + noise
    year_bonus = {"Red Bull Racing": 0.05 if year == 2022 else (0.08 if year == 2023 else 0.06),
                  "McLaren": 0.0 if year == 2022 else (0.03 if year == 2023 else 0.07),
                  "Mercedes": 0.03 if year == 2022 else 0.02,
                  "Ferrari": 0.02}.get

    driver_scores = []
    for code, info in DRIVERS.items():
        team_bonus = year_bonus(info["team"], 0.0)
        rain_factor = 1.0
        if rainfall and code in ["HAM", "VER", "ALO"]:
            rain_factor = 0.04  # extra bonus in rain
        score = (
            info["skill"] + team_bonus + rain_factor +
            float(RNG.normal(0, 0.08))
        )
        driver_scores.append((code, info, score))

    # Sort by score descending to get finishing order
    driver_scores.sort(key=lambda x: x[2], reverse=True)

    # Qualifying: similar sort with slightly more noise
    qual_scores = [
        (code, info, info["skill"] + float(RNG.normal(0, 0.06)))
        for code, info in DRIVERS.items()
    ]
    qual_scores.sort(key=lambda x: x[2], reverse=True)
    grid_map = {code: pos + 1 for pos, (code, _, _) in enumerate(qual_scores)}

    rows = []
    for pos, (code, info, _) in enumerate(driver_scores, 1):
        grid_pos = grid_map[code]
        points = POINTS_MAP.get(pos, 0)
        rows.append({
            "Driver": code,
            "DriverNumber": DRIVER_CODES.index(code) + 1,
            "FullName": info["name"],
            "Team": info["team"],
            "Position": pos,
            "GridPosition": grid_pos,
            "Points": float(points),
            "Status": "Finished" if pos <= 16 else "DNF",
            "FastestLapTime": float(RNG.normal(95.0, 3.0)),
            "Year": year,
            "Race": race,
            "AirTemp": round(air_temp, 1),
            "TrackTemp": round(track_temp, 1),
            "Rainfall": rainfall,
        })

    return pd.DataFrame(rows)


def _generate_qualifying(year, race):
    """Generate qualifying session results."""
    qual_scores = [
        (code, DRIVERS[code], DRIVERS[code]["skill"] + float(RNG.normal(0, 0.07)))
        for code in DRIVER_CODES
    ]
    qual_scores.sort(key=lambda x: x[2], reverse=True)

    rows = []
    for pos, (code, info, _) in enumerate(qual_scores, 1):
        base_time = 80.0 + (pos - 1) * 0.12 + float(RNG.normal(0, 0.05))
        rows.append({
            "Driver": code,
            "Team": info["team"],
            "QualifyingPosition": pos,
            "Q1Time": base_time + 0.4 if pos <= 20 else np.nan,
            "Q2Time": base_time + 0.2 if pos <= 15 else np.nan,
            "Q3Time": base_time if pos <= 10 else np.nan,
            "Year": year,
            "Race": race,
        })
    return pd.DataFrame(rows)


def _generate_pitstops(year, race):
    """Generate pit stop records for a race."""
    total_laps = CIRCUIT_LAPS.get(race, 57)
    rows = []

    for code, info in DRIVERS.items():
        team = info["team"]
        mean_dur, std_dur = TEAM_PIT_PARAMS[team]

        # Number of pit stops: 1-3, weighted toward 2
        n_stops = int(RNG.choice([1, 2, 3], p=[0.15, 0.65, 0.20]))
        stop_laps = sorted(
            RNG.choice(range(8, total_laps - 5), size=n_stops, replace=False)
        )

        compounds = RNG.choice(COMPOUND_PROBS[:3], size=n_stops + 1, p=[0.45, 0.40, 0.15])
        tyre_age = 0

        for i, lap in enumerate(stop_laps):
            duration = float(RNG.normal(mean_dur, std_dur))
            # Occasional slow stop
            if RNG.random() < 0.04:
                duration += float(RNG.uniform(5, 20))
            duration = max(1.8, min(duration, 55.0))

            weather_params = CIRCUIT_WEATHER.get(race, (25, 35, 0.05))
            rows.append({
                "Driver": code,
                "Team": team,
                "LapNumber": int(lap),
                "PitStopDuration": round(duration, 3),
                "TyreCompound": str(compounds[i]),
                "TyreLife": int(tyre_age + lap if i == 0 else lap - stop_laps[i - 1]),
                "Year": year,
                "Race": race,
            })
            tyre_age = 0

    return pd.DataFrame(rows)


def _generate_weather(year, race):
    """Generate aggregated weather data for a race session."""
    wp = CIRCUIT_WEATHER.get(race, (25, 35, 0.05))
    rainfall = int(RNG.random() < wp[2])
    return pd.DataFrame([{
        "AirTemp": round(float(RNG.normal(wp[0], 2)), 1),
        "TrackTemp": round(float(RNG.normal(wp[1], 3)), 1),
        "Humidity": round(float(RNG.normal(55, 10)), 1),
        "Rainfall": rainfall,
        "WindSpeed": round(float(RNG.normal(8, 4)), 1),
        "Year": year,
        "Race": race,
    }])


def _generate_circuit_info(year, race):
    """Generate circuit info row."""
    return {
        "Circuit": race,
        "Country": race,
        "TotalLaps": CIRCUIT_LAPS.get(race, 57),
        "Year": year,
        "Race": race,
    }


def _generate_tyres(year, race):
    """Generate tyre stint data for a race."""
    total_laps = CIRCUIT_LAPS.get(race, 57)
    rows = []
    for code, info in DRIVERS.items():
        n_stints = int(RNG.choice([2, 3, 4], p=[0.25, 0.55, 0.20]))
        compounds = RNG.choice(COMPOUND_PROBS[:3], size=n_stints, p=[0.45, 0.40, 0.15])
        stint_laps = np.diff([0] + sorted(
            RNG.choice(range(5, total_laps - 3), size=n_stints - 1, replace=False).tolist()
        ) + [total_laps])

        for stint_idx, (compound, length) in enumerate(zip(compounds, stint_laps), 1):
            rows.append({
                "Driver": code,
                "Stint": stint_idx,
                "Compound": str(compound),
                "StintLength": int(length),
                "StartLap": 1 if stint_idx == 1 else int(sum(stint_laps[:stint_idx - 1])),
                "EndLap": int(sum(stint_laps[:stint_idx])),
                "Team": info["team"],
                "Year": year,
                "Race": race,
                "NumPitStops": n_stints - 1,
            })
    return pd.DataFrame(rows)


def generate_all_synthetic_data(raw_data_dir: str) -> None:
    """
    Generate synthetic F1 data for all 3 seasons and 24 races.

    Saves the following CSVs to raw_data_dir:
        - race_data_raw.csv
        - qualifying_data_raw.csv
        - pitstop_data_raw.csv
        - weather_data_raw.csv
        - circuit_data_raw.csv
        - tyre_data_raw.csv

    Args:
        raw_data_dir: Path to the data/raw/ directory.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    print("   🔄 Generating synthetic F1 data (API unavailable)...")

    all_race, all_qual, all_pit, all_weather, all_circuit, all_tyre = [], [], [], [], [], []

    for year in SEASONS:
        for race in RACES:
            # 2022 didn't have Las Vegas, Qatar, or China
            if year == 2022 and race in ("Las Vegas", "Qatar", "China"):
                continue

            all_race.append(_generate_race_result(year, race))
            all_qual.append(_generate_qualifying(year, race))
            all_pit.append(_generate_pitstops(year, race))
            all_weather.append(_generate_weather(year, race))
            all_circuit.append(_generate_circuit_info(year, race))
            all_tyre.append(_generate_tyres(year, race))

    datasets = {
        "race_data_raw.csv":       pd.concat(all_race, ignore_index=True),
        "qualifying_data_raw.csv": pd.concat(all_qual, ignore_index=True),
        "pitstop_data_raw.csv":    pd.concat(all_pit,  ignore_index=True),
        "weather_data_raw.csv":    pd.concat(all_weather, ignore_index=True),
        "circuit_data_raw.csv":    pd.DataFrame(all_circuit),
        "tyre_data_raw.csv":       pd.concat(all_tyre, ignore_index=True),
    }

    for filename, df in datasets.items():
        path = os.path.join(raw_data_dir, filename)
        df.to_csv(path, index=False)
        print(f"      ✅ {filename} — {len(df):,} rows")

    print("   ✅ Synthetic data generation complete.")


# ── Standalone execution ──────────────────────────────────────

if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base, "data", "raw")
    generate_all_synthetic_data(raw_dir)
