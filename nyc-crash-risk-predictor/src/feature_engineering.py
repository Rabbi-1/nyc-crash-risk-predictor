"""Feature engineering for hourly ZIP-code crash rates."""

from pathlib import Path

import pandas as pd


GROUP_COLUMNS = ["zip_code", "day_of_week", "hour"]


def build_crash_rate_table(
    cleaned_path="data/processed/nyc_crashes_clean.csv",
    output_path="data/processed/crash_rate_table.csv",
):
    """Build a table of estimated hourly crash rates by ZIP, day, and hour.

    Exposure is estimated as the number of distinct observed dates for each
    day of week. For example, if the cleaned dataset contains 20 Fridays, then
    Friday 5 PM for each ZIP has 20 observed one-hour periods.

    Parameters
    ----------
    cleaned_path : str or pathlib.Path
        Cleaned crash CSV produced by ``clean_crash_data``.
    output_path : str or pathlib.Path
        Location for the crash rate table.

    Returns
    -------
    pandas.DataFrame
        Rate table with crash_count, observed_hours, and estimated_lambda.
    """
    cleaned_path = Path(cleaned_path)
    output_path = Path(output_path)

    df = pd.read_csv(cleaned_path, dtype={"zip_code": str})
    required_columns = {"zip_code", "day_of_week", "hour", "date"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Cleaned data is missing columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["zip_code", "day_of_week", "hour", "date"])
    df["hour"] = df["hour"].astype(int)

    observed_dates = df[["date", "day_of_week"]].drop_duplicates()
    observed_hours_by_day = observed_dates.groupby("day_of_week").size()

    rate_table = (
        df.groupby(GROUP_COLUMNS)
        .size()
        .reset_index(name="crash_count")
        .sort_values(GROUP_COLUMNS)
    )

    rate_table["observed_hours"] = rate_table["day_of_week"].map(observed_hours_by_day)
    rate_table["estimated_lambda"] = (
        rate_table["crash_count"] / rate_table["observed_hours"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rate_table.to_csv(output_path, index=False)
    return rate_table

