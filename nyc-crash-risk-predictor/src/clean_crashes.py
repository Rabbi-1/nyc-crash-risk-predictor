"""Clean raw NYC crash records for modeling."""

from pathlib import Path

import pandas as pd


USEFUL_COLUMNS = [
    "crash_date",
    "crash_time",
    "borough",
    "zip_code",
    "latitude",
    "longitude",
]


def _normalize_zip_code(value):
    """Convert ZIP code values from the CSV into clean five-character strings."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]

    if text.isdigit() and len(text) < 5:
        text = text.zfill(5)

    return text if text else pd.NA


def clean_crash_data(
    input_path="data/raw/nyc_crashes_raw.csv",
    output_path="data/processed/nyc_crashes_clean.csv",
):
    """Load raw crash data, clean key fields, and save a modeling-ready CSV.

    The cleaned data contains the original selected crash fields plus:
    ``crash_datetime``, ``date``, ``hour``, and ``day_of_week``.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Raw CSV file downloaded from NYC Open Data.
    output_path : str or pathlib.Path
        Location for the cleaned CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned crash records.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    missing_columns = sorted(set(USEFUL_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing columns: {missing_columns}")

    df = df[USEFUL_COLUMNS].copy()
    df = df.dropna(subset=["crash_date", "crash_time", "zip_code"])

    df["zip_code"] = df["zip_code"].apply(_normalize_zip_code)
    df = df.dropna(subset=["zip_code"])

    crash_date = pd.to_datetime(df["crash_date"], errors="coerce").dt.normalize()
    crash_time_text = df["crash_time"].astype(str).str.strip()
    crash_time_hm = pd.to_datetime(crash_time_text, format="%H:%M", errors="coerce")
    crash_time_hms = pd.to_datetime(
        crash_time_text,
        format="%H:%M:%S",
        errors="coerce",
    )
    crash_time = crash_time_hm.fillna(crash_time_hms)
    crash_time_delta = (
        pd.to_timedelta(crash_time.dt.hour, unit="h")
        + pd.to_timedelta(crash_time.dt.minute, unit="m")
        + pd.to_timedelta(crash_time.dt.second, unit="s")
    )
    df["crash_datetime"] = crash_date + crash_time_delta
    df = df.dropna(subset=["crash_datetime"])

    df["date"] = df["crash_datetime"].dt.date
    df["hour"] = df["crash_datetime"].dt.hour.astype(int)
    df["day_of_week"] = df["crash_datetime"].dt.day_name()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
