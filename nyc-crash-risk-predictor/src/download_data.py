"""Download NYC motor vehicle crash data from NYC Open Data."""

from pathlib import Path

import requests


DATASET_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"
SELECTED_COLUMNS = [
    "crash_date",
    "crash_time",
    "borough",
    "zip_code",
    "latitude",
    "longitude",
]


def download_crash_data(
    limit=50000,
    output_path="data/raw/nyc_crashes_raw.csv",
):
    """Download a sample of NYC crash records and save it as a CSV file.

    Parameters
    ----------
    limit : int, default=50000
        Maximum number of rows to request from NYC Open Data.
    output_path : str or pathlib.Path
        Location where the raw CSV should be saved.

    Returns
    -------
    pathlib.Path
        Path to the downloaded CSV file.
    """
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "$limit": int(limit),
        "$select": ",".join(SELECTED_COLUMNS),
        "$order": "crash_date DESC, crash_time DESC",
    }

    response = requests.get(DATASET_URL, params=params, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    return output_path

