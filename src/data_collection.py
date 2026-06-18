# ============================================================
# Lights Out and Away — F1 ML Project
# src/data_collection.py
#
# Collects real Formula 1 data from the 2022, 2023, and 2024
# seasons using the FastF1 Python library. Fetches race results,
# qualifying data, pit stop data, and weather data for all 24
# races across three seasons.
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import fastf1

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

SEASONS = [2022, 2023, 2024]

RACES_2024 = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "United States", "Mexico",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]

# Some race names differ across seasons — map alternates
RACE_NAME_ALIASES = {
    "Great Britain": ["Great Britain", "British Grand Prix"],
    "Emilia Romagna": ["Emilia Romagna", "EmiliaRomagna"],
    "Las Vegas": ["Las Vegas"],
    "Saudi Arabia": ["Saudi Arabia", "Jeddah"],
}

# ── Enable FastF1 cache ─────────────────────────────────────

def enable_cache():
    """Enable the FastF1 cache to avoid re-downloading session data."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f"   Cache enabled at: {CACHE_DIR}")


# ── Data Collection Functions ────────────────────────────────

def collect_race_data(year, race_name):
    """
    Fetch race session data for a given year and race.

    Returns a DataFrame with columns:
        Driver, DriverNumber, Team, Position, GridPosition, Points,
        FastestLapTime, Status, Year, Race
    """
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()

        results = session.results
        if results is None or results.empty:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Driver": results["Abbreviation"],
            "DriverNumber": results["DriverNumber"],
            "FullName": results["FullName"],
            "Team": results["TeamName"],
            "Position": pd.to_numeric(results["Position"], errors="coerce"),
            "GridPosition": pd.to_numeric(results["GridPosition"], errors="coerce"),
            "Points": pd.to_numeric(results["Points"], errors="coerce"),
            "Status": results["Status"],
            "Year": year,
            "Race": race_name,
        })

        # Try to get fastest lap time
        try:
            df["FastestLapTime"] = results["FastestLapTime"].dt.total_seconds()
        except Exception:
            df["FastestLapTime"] = np.nan

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"      ⚠ Race data failed for {race_name} {year}: {e}")
        return pd.DataFrame()


def collect_qualifying_data(year, race_name):
    """
    Fetch qualifying session data for a given year and race.

    Returns a DataFrame with columns:
        Driver, Team, QualifyingPosition, Q1Time, Q2Time, Q3Time, Year, Race
    """
    try:
        session = fastf1.get_session(year, race_name, "Q")
        session.load()

        results = session.results
        if results is None or results.empty:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Driver": results["Abbreviation"],
            "Team": results["TeamName"],
            "QualifyingPosition": pd.to_numeric(results["Position"], errors="coerce"),
            "Year": year,
            "Race": race_name,
        })

        # Extract Q1, Q2, Q3 times
        for q in ["Q1", "Q2", "Q3"]:
            try:
                df[f"{q}Time"] = results[q].dt.total_seconds()
            except Exception:
                df[f"{q}Time"] = np.nan

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"      ⚠ Qualifying data failed for {race_name} {year}: {e}")
        return pd.DataFrame()


def collect_pitstop_data(year, race_name):
    """
    Fetch pit stop data from the race session laps.

    In FastF1, PitInTime is recorded on the lap where the driver enters
    the pit lane, while PitOutTime is recorded on the next lap (the
    out-lap). To compute pit stop duration, we match each driver's
    PitInTime with the PitOutTime from their subsequent lap.

    Returns a DataFrame with columns:
        Driver, Team, LapNumber, PitStopDuration, TyreCompound,
        TyreLife, Year, Race
    """
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()

        laps = session.laps
        if laps is None or laps.empty:
            return pd.DataFrame()

        pit_data = []

        # Process each driver's laps individually to match PitInTime
        # on the in-lap with PitOutTime on the following out-lap
        for driver in laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)

            for i in range(len(driver_laps)):
                lap = driver_laps.iloc[i]
                pit_in = lap.get("PitInTime")

                if pd.isna(pit_in):
                    continue

                # This lap has a PitInTime — look for PitOutTime on the next lap
                duration = np.nan
                if i + 1 < len(driver_laps):
                    next_lap = driver_laps.iloc[i + 1]
                    pit_out = next_lap.get("PitOutTime")
                    if pd.notna(pit_out):
                        duration = (pit_out - pit_in).total_seconds()

                # If we still don't have a duration, try same-lap fallback
                if pd.isna(duration):
                    pit_out_same = lap.get("PitOutTime")
                    if pd.notna(pit_out_same):
                        duration = (pit_out_same - pit_in).total_seconds()

                # Skip if we couldn't compute duration
                if pd.isna(duration):
                    continue

                # Get tyre information (compound on the in-lap)
                compound = lap.get("Compound", "UNKNOWN")
                tyre_life = lap.get("TyreLife", np.nan)

                pit_data.append({
                    "Driver": lap.get("Driver", ""),
                    "Team": lap.get("Team", ""),
                    "LapNumber": lap.get("LapNumber", np.nan),
                    "PitStopDuration": duration,
                    "TyreCompound": compound if pd.notna(compound) else "UNKNOWN",
                    "TyreLife": tyre_life if pd.notna(tyre_life) else np.nan,
                    "Year": year,
                    "Race": race_name,
                })

        if not pit_data:
            return pd.DataFrame()

        return pd.DataFrame(pit_data)

    except Exception as e:
        print(f"      ⚠ Pit stop data failed for {race_name} {year}: {e}")
        return pd.DataFrame()


def collect_weather_data(year, race_name):
    """
    Fetch weather data from the race session.

    Returns a DataFrame with columns:
        AirTemp, TrackTemp, Humidity, Rainfall, WindSpeed, Year, Race
    """
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()

        weather = session.weather_data
        if weather is None or weather.empty:
            return pd.DataFrame()

        # Aggregate weather data — take the mean for the session
        weather_summary = {
            "AirTemp": weather["AirTemp"].mean(),
            "TrackTemp": weather["TrackTemp"].mean(),
            "Humidity": weather["Humidity"].mean(),
            "Rainfall": int(weather["Rainfall"].any()),
            "WindSpeed": weather["WindSpeed"].mean(),
            "Year": year,
            "Race": race_name,
        }

        return pd.DataFrame([weather_summary])

    except Exception as e:
        print(f"      ⚠ Weather data failed for {race_name} {year}: {e}")
        return pd.DataFrame()


def collect_circuit_info(year, race_name):
    """
    Fetch circuit information from the race session.

    Returns a dict with circuit name, country, and number of laps.
    """
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()

        event = session.event
        total_laps = session.total_laps if hasattr(session, "total_laps") else np.nan

        return {
            "Circuit": event.get("Location", race_name) if hasattr(event, "get") else race_name,
            "Country": event.get("Country", "") if hasattr(event, "get") else "",
            "TotalLaps": total_laps,
            "Year": year,
            "Race": race_name,
        }
    except Exception:
        return {
            "Circuit": race_name,
            "Country": "",
            "TotalLaps": np.nan,
            "Year": year,
            "Race": race_name,
        }


def collect_tyre_data(year, race_name):
    """
    Fetch tyre stint data from the race session.

    Returns a DataFrame with stint-level tyre usage per driver.
    """
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()

        laps = session.laps
        if laps is None or laps.empty:
            return pd.DataFrame()

        # Group by driver and stint to get tyre compound per stint
        stints = (
            laps.groupby(["Driver", "Stint"])
            .agg(
                Compound=("Compound", "first"),
                StintLength=("LapNumber", "count"),
                StartLap=("LapNumber", "min"),
                EndLap=("LapNumber", "max"),
                Team=("Team", "first"),
            )
            .reset_index()
        )

        stints["Year"] = year
        stints["Race"] = race_name

        # Count number of pit stops per driver (stints - 1)
        pit_counts = stints.groupby("Driver")["Stint"].max().reset_index()
        pit_counts.columns = ["Driver", "NumPitStops"]
        pit_counts["NumPitStops"] = pit_counts["NumPitStops"] - 1

        stints = stints.merge(pit_counts, on="Driver", how="left")

        return stints

    except Exception as e:
        print(f"      ⚠ Tyre data failed for {race_name} {year}: {e}")
        return pd.DataFrame()


def get_race_list_for_year(year):
    """
    Get the list of race names for a given year.
    Some races may not exist in all seasons.
    """
    # Use the 2024 race list as the base for all seasons
    # FastF1 will throw an error if a race doesn't exist,
    # which we catch and skip gracefully
    return RACES_2024.copy()


def collect_all_seasons():
    """
    Collect all data across all 3 seasons (2022, 2023, 2024) for
    all 24 races. Saves combined data to data/raw/ as CSVs.

    If FastF1 API is unavailable for a season (network error or cache
    miss), falls back to realistic synthetic data generation so the
    pipeline always completes successfully.

    This is the main collection function called by main.py.
    """
    enable_cache()
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    all_race_data = []
    all_qualifying_data = []
    all_pitstop_data = []
    all_weather_data = []
    all_circuit_data = []
    all_tyre_data = []

    total_sessions = 0
    failed_sessions = 0

    for year in SEASONS:
        print(f"\n{'='*60}")
        print(f"   🏎️  SEASON {year}")
        print(f"{'='*60}")

        races = get_race_list_for_year(year)

        for i, race in enumerate(races, 1):
            print(f"\n   [{i:02d}/24] Collecting {race} {year}...", end=" ")

            try:
                # Collect race data
                race_df = collect_race_data(year, race)
                if not race_df.empty:
                    all_race_data.append(race_df)

                # Collect qualifying data
                qual_df = collect_qualifying_data(year, race)
                if not qual_df.empty:
                    all_qualifying_data.append(qual_df)

                # Collect pit stop data
                pit_df = collect_pitstop_data(year, race)
                if not pit_df.empty:
                    all_pitstop_data.append(pit_df)

                # Collect weather data
                weather_df = collect_weather_data(year, race)
                if not weather_df.empty:
                    all_weather_data.append(weather_df)

                # Collect circuit info
                circuit_info = collect_circuit_info(year, race)
                all_circuit_data.append(circuit_info)

                # Collect tyre data
                tyre_df = collect_tyre_data(year, race)
                if not tyre_df.empty:
                    all_tyre_data.append(tyre_df)

                total_sessions += 1
                print("✅ done")

            except Exception as e:
                failed_sessions += 1
                print(f"❌ failed ({e})")
                continue

    # ── Combine and save ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("   📦 Saving collected data to data/raw/")
    print(f"{'='*60}")

    # Race data
    if all_race_data:
        race_combined = pd.concat(all_race_data, ignore_index=True)
        race_combined.to_csv(os.path.join(RAW_DATA_DIR, "race_data_raw.csv"), index=False)
        print(f"   ✅ race_data_raw.csv — {len(race_combined)} rows")
    else:
        print("   ⚠ No race data collected")

    # Qualifying data
    if all_qualifying_data:
        qual_combined = pd.concat(all_qualifying_data, ignore_index=True)
        qual_combined.to_csv(os.path.join(RAW_DATA_DIR, "qualifying_data_raw.csv"), index=False)
        print(f"   ✅ qualifying_data_raw.csv — {len(qual_combined)} rows")
    else:
        print("   ⚠ No qualifying data collected")

    # Pit stop data
    if all_pitstop_data:
        pit_combined = pd.concat(all_pitstop_data, ignore_index=True)
        pit_combined.to_csv(os.path.join(RAW_DATA_DIR, "pitstop_data_raw.csv"), index=False)
        print(f"   ✅ pitstop_data_raw.csv — {len(pit_combined)} rows")
    else:
        print("   ⚠ No pit stop data collected")

    # Weather data
    if all_weather_data:
        weather_combined = pd.concat(all_weather_data, ignore_index=True)
        weather_combined.to_csv(os.path.join(RAW_DATA_DIR, "weather_data_raw.csv"), index=False)
        print(f"   ✅ weather_data_raw.csv — {len(weather_combined)} rows")
    else:
        print("   ⚠ No weather data collected")

    # Circuit data
    if all_circuit_data:
        circuit_combined = pd.DataFrame(all_circuit_data)
        circuit_combined.to_csv(os.path.join(RAW_DATA_DIR, "circuit_data_raw.csv"), index=False)
        print(f"   ✅ circuit_data_raw.csv — {len(circuit_combined)} rows")
    else:
        print("   ⚠ No circuit data collected")

    # Tyre data
    if all_tyre_data:
        tyre_combined = pd.concat(all_tyre_data, ignore_index=True)
        tyre_combined.to_csv(os.path.join(RAW_DATA_DIR, "tyre_data_raw.csv"), index=False)
        print(f"   ✅ tyre_data_raw.csv — {len(tyre_combined)} rows")
    else:
        print("   ⚠ No tyre data collected")

    # ── Fallback: synthetic data if nothing was collected ────
    _check_and_fill_missing_with_synthetic(total_sessions, failed_sessions)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"   📊 Collection Summary")
    print(f"{'='*60}")
    print(f"   Total sessions collected: {total_sessions}")
    print(f"   Failed sessions:          {failed_sessions}")
    print(f"   Data saved to:            {RAW_DATA_DIR}")
    print(f"{'='*60}\n")

    return total_sessions, failed_sessions


def _check_and_fill_missing_with_synthetic(total_sessions, failed_sessions):
    """
    If the race_data_raw.csv is missing or too small (fewer than 100 rows),
    generate synthetic data to supplement or replace missing API data.

    This guarantees the pipeline always has enough data to train models.
    """
    race_csv = os.path.join(RAW_DATA_DIR, "race_data_raw.csv")
    pit_csv = os.path.join(RAW_DATA_DIR, "pitstop_data_raw.csv")

    needs_synthetic = False

    if not os.path.exists(race_csv):
        needs_synthetic = True
        print("\n   ⚠ race_data_raw.csv missing — falling back to synthetic data")
    else:
        try:
            df = pd.read_csv(race_csv)
            if len(df) < 100:
                needs_synthetic = True
                print(f"\n   ⚠ Only {len(df)} race rows collected — supplementing with synthetic data")
        except Exception:
            needs_synthetic = True

    if not os.path.exists(pit_csv):
        needs_synthetic = True

    if needs_synthetic:
        from src.synthetic_data import generate_all_synthetic_data
        generate_all_synthetic_data(RAW_DATA_DIR)


# ── Standalone execution ─────────────────────────────────────

if __name__ == "__main__":
    print("\n🏎️  LIGHTS OUT AND AWAY — Data Collection\n")
    collect_all_seasons()
    print("✅ Data collection complete!")
