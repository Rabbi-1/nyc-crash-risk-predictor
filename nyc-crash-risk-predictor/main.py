"""End-to-end demo for the NYC crash risk predictor project."""

from pathlib import Path

import pandas as pd

from src.clean_crashes import clean_crash_data
from src.download_data import download_crash_data
from src.feature_engineering import build_crash_rate_table
from src.monte_carlo import predict_for_zip_hour
from src.validation import convergence_test
from src.visualization import (
    create_choropleth_map,
    plot_convergence,
    plot_empirical_poisson_check,
    plot_simulated_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_crashes_raw.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "nyc_crashes_clean.csv"
RATE_PATH = PROJECT_ROOT / "data" / "processed" / "crash_rate_table.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def choose_example_row(rate_table, zip_code, day_of_week, hour):
    """Use the requested row when possible, otherwise fall back to a valid row."""
    rate_table = rate_table.copy()
    rate_table["zip_code"] = rate_table["zip_code"].astype(str)
    preferred = rate_table[
        (rate_table["zip_code"] == zip_code)
        & (rate_table["day_of_week"] == day_of_week)
        & (rate_table["hour"].astype(int) == int(hour))
    ]

    if not preferred.empty:
        return preferred.iloc[0], False

    if zip_code not in set(rate_table["zip_code"]):
        print(f"ZIP {zip_code} was not found in this downloaded sample.")
    else:
        print(
            f"ZIP {zip_code} exists, but not for {day_of_week} hour {hour} "
            "in this sample."
        )

    print("Using the first valid ZIP/day/hour row from the rate table instead.")
    return rate_table.iloc[0], True


def main():
    """Run the complete project workflow."""
    print("Downloading NYC crash data...")
    download_crash_data(limit=50000, output_path=RAW_PATH)

    print("Cleaning crash data...")
    cleaned_df = clean_crash_data(input_path=RAW_PATH, output_path=CLEAN_PATH)
    print(f"Cleaned rows: {len(cleaned_df):,}")

    print("Building hourly crash rate table...")
    rate_table = build_crash_rate_table(cleaned_path=CLEAN_PATH, output_path=RATE_PATH)
    print(f"Rate-table rows: {len(rate_table):,}")

    zip_code = "10001"
    day_of_week = "Friday"
    hour = 17
    weather_condition = "rain"

    selected_row, used_fallback = choose_example_row(
        rate_table=rate_table,
        zip_code=zip_code,
        day_of_week=day_of_week,
        hour=hour,
    )
    if used_fallback:
        zip_code = str(selected_row["zip_code"])
        day_of_week = str(selected_row["day_of_week"])
        hour = int(selected_row["hour"])

    result = predict_for_zip_hour(
        rate_table=rate_table,
        zip_code=zip_code,
        day_of_week=day_of_week,
        hour=hour,
        weather_condition=weather_condition,
        num_trials=10000,
        random_seed=42,
    )

    print("\nMonte Carlo crash risk estimate")
    print("--------------------------------")
    print(f"ZIP code: {result['zip_code']}")
    print(f"Day/hour: {result['day_of_week']} at {result['hour']:02d}:00")
    print(f"Weather condition: {result['weather_condition']}")
    print(f"Observed crashes in bucket: {result['crash_count']}")
    print(f"Observed hours: {result['observed_hours']}")
    print(f"Estimated lambda: {result['estimated_lambda']:.4f}")
    print(
        "Probability of at least one crash: "
        f"{result['probability_at_least_one']:.2%}"
    )
    print(f"Mean simulated crashes: {result['mean_simulated_crashes']:.3f}")
    print(
        "5th to 95th percentile simulated crashes: "
        f"{result['percentile_5']:.0f} to {result['percentile_95']:.0f}"
    )

    histogram_path = FIGURE_DIR / "simulated_crash_counts.png"
    plot_simulated_counts(result["simulated_counts"], histogram_path)
    print(f"Saved histogram: {histogram_path}")

    convergence_df = convergence_test(
        crash_count=result["crash_count"],
        observed_hours=result["observed_hours"],
        weather_condition=weather_condition,
        random_seed=42,
    )
    convergence_path = FIGURE_DIR / "convergence_plot.png"
    plot_convergence(convergence_df, convergence_path)
    print(f"Saved convergence plot: {convergence_path}")

    poisson_check_path = FIGURE_DIR / "empirical_vs_poisson_daily_counts.png"
    poisson_info = plot_empirical_poisson_check(
        cleaned_df,
        zip_code=zip_code,
        day_of_week=day_of_week,
        hour=hour,
        output_path=poisson_check_path,
    )
    print(f"Saved Poisson check (daily counts): {poisson_info['output_path']}")
    print(
        f"  var/mean = {poisson_info['variance_to_mean']:.3f} over "
        f"{poisson_info['n_days']} days (1.0 is Poisson with constant rate)"
    )

    # Create interactive choropleth map
    MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    map_path = MAP_DIR / f"crash_map_{day_of_week.lower()}_{hour:02d}.html"

    print(f"\nCreating interactive crash probability map...")
    map_result = create_choropleth_map(
        rate_table=rate_table,
        day_of_week=day_of_week,
        hour=hour,
        weather_condition=weather_condition,
        num_trials=10000,
        random_seed=42,
        output_path=map_path,
    )
    print(f"Saved interactive map: {map_result['output_path']}")
    print(f"  Mapped {len(map_result['results_df'])} ZIP codes")

    print("\nConvergence check")
    print(convergence_df.to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.max_columns", 20)
    main()

