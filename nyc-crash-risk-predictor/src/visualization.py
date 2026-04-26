"""Visualization helpers for simulation results."""

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

