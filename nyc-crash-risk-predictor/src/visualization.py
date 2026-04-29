"""Visualization helpers for simulation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .validation import daily_crash_counts_for_bucket


def _poisson_pmf_up_to(k_max, mu):
    """P(X=k) for k=0..k_max, X ~ Poisson(mu). No SciPy required."""
    k_max = int(k_max)
    if k_max < 0:
        raise ValueError("k_max must be nonnegative.")
    if mu < 0:
        raise ValueError("mu must be nonnegative.")

    pmf = np.zeros(k_max + 1)
    pmf[0] = np.exp(-mu)
    for j in range(1, k_max + 1):
        pmf[j] = pmf[j - 1] * mu / j
    return np.arange(0, k_max + 1, dtype=int), pmf


def plot_simulated_counts(simulated_counts, output_path):
    """Save a histogram of Monte Carlo simulated crash counts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = np.asarray(simulated_counts)
    max_count = int(counts.max()) if counts.size else 0
    bins = np.arange(-0.5, max_count + 1.5, 1)

    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=bins, color="#2f80ed", edgecolor="white")
    plt.title("Monte Carlo Simulated Crash Counts")
    plt.xlabel("Crashes in selected one-hour block")
    plt.ylabel("Number of trials")
    plt.xticks(range(max_count + 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_empirical_poisson_check(
    cleaned_df,
    zip_code,
    day_of_week,
    hour,
    output_path,
    *,
    min_days=1,
):
    """Save a bar chart of daily bucket counts with a Poisson(λ) PMF overlay.

    The empirical distribution is the count of crashes on each *calendar day*
    in the (ZIP, day of week, hour) bucket, including **zero** on exposure days
    when the bucket is empty. ``λ`` is set to the **sample mean** of those
    daily counts, matching a Poisson GOF visual for teaching.

    Parameters
    ----------
    cleaned_df : pandas.DataFrame
        Data from ``clean_crash_data``.
    zip_code, day_of_week, hour
        Same meaning as in ``daily_crash_counts_for_bucket``.
    output_path : str or pathlib.Path
        File path for the PNG.
    min_days : int, default=1
        Forwarded to ``daily_crash_counts_for_bucket``.

    Returns
    -------
    dict
        ``output_path`` (``pathlib.Path``), ``daily_counts`` (``pandas.Series``),
        ``sample_mean``, ``sample_variance``, ``variance_to_mean``,
        ``n_days`` for use in printouts or notes.
    """
    daily = daily_crash_counts_for_bucket(
        cleaned_df,
        zip_code,
        day_of_week,
        hour,
        min_days=min_days,
    )
    values = daily.values.astype(int)
    n = len(values)
    mean = float(np.mean(values))
    var = float(np.var(values, ddof=0))
    ratio = float(var / mean) if mean > 0 else float("nan")

    max_obs = int(values.max())
    k_max = min(max(15, max_obs + 1), 40)
    ks, pmf = _poisson_pmf_up_to(k_max, mean)

    empirical = np.bincount(values, minlength=k_max + 1)[: k_max + 1]
    y_line = n * pmf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = min(8 + 0.25 * k_max, 14)
    plt.figure(figsize=(fig_w, 5.5))
    x = np.arange(0, k_max + 1)
    w = 0.42
    shift = 0.2
    plt.bar(
        x - shift,
        empirical,
        width=w,
        color="#2f80ed",
        edgecolor="white",
        label="Observed (days in bucket)",
        align="center",
    )
    plt.plot(
        ks,
        y_line,
        color="#1f7a4d",
        marker="D",
        markersize=4,
        linewidth=1.5,
        label=f"Expected under Poisson(λ), λ̂ = {mean:.3f} (n × P(X=k))",
    )
    plt.title(
        f"Daily crash counts — ZIP {zip_code}, {day_of_week} {hour:02d}:00\n"
        f"var/mean = {ratio:.2f} (1.0 if Poisson with constant λ); n = {n} days"
    )
    plt.xlabel("Crashes that day in this one-hour block")
    plt.ylabel("Number of days (bars) / expected days under Poisson (line)")
    plt.xticks(x)
    plt.xlim(-0.6, k_max + 0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "output_path": output_path,
        "daily_counts": daily,
        "sample_mean": mean,
        "sample_variance": var,
        "variance_to_mean": ratio,
        "n_days": n,
    }


def plot_convergence(convergence_df, output_path):
    """Save a line plot showing probability estimates by trial count."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        convergence_df["num_trials"],
        convergence_df["probability_at_least_one"],
        marker="o",
        color="#1f7a4d",
    )
    plt.title("Monte Carlo Convergence")
    plt.xlabel("Number of trials")
    plt.ylabel("Estimated P(at least one crash)")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def create_zip_probability_map(results_df, output_path):
    """Create a simple Folium circle-marker map of ZIP-level probabilities.

    ``results_df`` should contain ``zip_code``, ``latitude``, ``longitude``,
    and ``probability_at_least_one`` columns. If it contains multiple rows per
    ZIP, the function averages latitude, longitude, and probability by ZIP.
    """
    import folium  # optional dependency; not needed for other plots in this module

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "zip_code",
        "latitude",
        "longitude",
        "probability_at_least_one",
    }
    missing_columns = sorted(required_columns - set(results_df.columns))
    if missing_columns:
        raise ValueError(f"Map results are missing columns: {missing_columns}")

    map_df = results_df.copy()
    map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
    map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
    map_df["probability_at_least_one"] = pd.to_numeric(
        map_df["probability_at_least_one"],
        errors="coerce",
    )
    map_df = map_df.dropna(
        subset=["zip_code", "latitude", "longitude", "probability_at_least_one"]
    )

    if map_df.empty:
        raise ValueError("No valid map rows after dropping missing coordinates.")

    zip_summary = (
        map_df.groupby("zip_code", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            probability_at_least_one=("probability_at_least_one", "mean"),
        )
    )

    center = [zip_summary["latitude"].mean(), zip_summary["longitude"].mean()]
    crash_map = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    for _, row in zip_summary.iterrows():
        probability = float(row["probability_at_least_one"])
        color = "#d73027" if probability >= 0.25 else "#fc8d59"
        color = "#1a9850" if probability < 0.10 else color
        radius = 5 + 20 * probability

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=(
                f"ZIP {row['zip_code']}<br>"
                f"P(at least one crash): {probability:.1%}"
            ),
        ).add_to(crash_map)

    crash_map.save(output_path)
    return output_path


def create_choropleth_map(
    rate_table,
    day_of_week,
    hour,
    weather_condition="clear",
    num_trials=10000,
    random_seed=42,
    geojson_url=None,
    output_path="outputs/maps/crash_probability_map.html",
):
    import folium
    import requests

    from .monte_carlo import predict_for_zip_hour

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Default NYC ZIP code boundaries GeoJSON from GitHub
    if geojson_url is None:
        geojson_url = (
            "https://raw.githubusercontent.com/nycehs/NYC_geography/master/"
            "MODZCTA_2010_WGS1984.geo.json"
        )

    # Compute probabilities for each ZIP
    print(f"Computing crash probabilities for {day_of_week} at {hour:02d}:00...")
    results = []

    rate_table = rate_table.copy()
    rate_table["zip_code"] = rate_table["zip_code"].astype(str)

    # Filter to requested day/hour
    bucket = rate_table[
        (rate_table["day_of_week"] == day_of_week)
        & (rate_table["hour"].astype(int) == hour)
    ]

    if bucket.empty:
        raise ValueError(
            f"No data found for {day_of_week} at hour {hour}. "
            f"Available days: {rate_table['day_of_week'].unique()}"
        )

    for _, row in bucket.iterrows():
        try:
            result = predict_for_zip_hour(
                rate_table=rate_table,
                zip_code=row["zip_code"],
                day_of_week=day_of_week,
                hour=hour,
                weather_condition=weather_condition,
                num_trials=num_trials,
                random_seed=random_seed,
            )
            results.append(
                {
                    "zip_code": row["zip_code"],
                    "probability_at_least_one": result["probability_at_least_one"],
                    "crash_count": result["crash_count"],
                    "observed_hours": result["observed_hours"],
                    "estimated_lambda": result["estimated_lambda"],
                }
            )
        except Exception as e:
            print(f"Warning: Skipping ZIP {row['zip_code']}: {e}")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No results to map.")

    # Fetch GeoJSON
    print("Fetching NYC ZIP code boundaries...")
    response = requests.get(geojson_url, timeout=30)
    response.raise_for_status()
    geojson_data = response.json()

    # Normalize ZIP codes in GeoJSON to match rate table
    for feature in geojson_data["features"]:
        props = feature.get("properties", {})
        # Try common ZIP code field names (MODZCTA is the primary one)
        zip_code = props.get("MODZCTA") or props.get("modzcta") or props.get("zipcode") or props.get("ZIP")
        if zip_code:
            props["zip_code"] = str(zip_code).strip()

    # Create map centered on NYC
    crash_map = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    # Add choropleth layer
    folium.Choropleth(
        geo_data=geojson_data,
        name="Crash Probability",
        data=results_df,
        columns=["zip_code", "probability_at_least_one"],
        key_on="feature.properties.zip_code",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_weight=1,
        line_color="white",
        legend_name="Probability of at least one crash",
        highlight=True,
    ).add_to(crash_map)

    # Add tooltips showing ZIP and probability
    folium.GeoJson(
        geojson_data,
        name="ZIP Codes",
        tooltip=folium.GeoJsonTooltip(
            fields=["zip_code"],
            aliases=["ZIP Code:"],
            localize=True,
        ),
        style_function=lambda x: {
            "fillColor": "#ffffff",
            "color": "#000000",
            "fillOpacity": 0,
            "weight": 0,
        },
    ).add_to(crash_map)

    # Add title
    title_html = f"""
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: auto;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px;
                border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <b>NYC Crash Risk Map</b><br>
        <span style="font-size:12px;">
        {day_of_week} at {hour:02d}:00<br>
        Weather: {weather_condition}<br>
        Probability of at least one crash
        </span>
    </div>
    """
    crash_map.get_root().html.add_child(folium.Element(title_html))

    # Add layer control
    folium.LayerControl().add_to(crash_map)

    crash_map.save(output_path)
    print(f"Saved interactive map: {output_path}")

    return {
        "output_path": output_path,
        "results_df": results_df,
        "map": crash_map,
    }

