"""Monte Carlo simulation tools for hourly crash risk."""

from os import PathLike

import numpy as np
import pandas as pd


WEATHER_MULTIPLIERS = {
    "clear": 1.00,
    "rain": 1.20,
    "snow": 1.35,
    "fog": 1.15,
    "high_wind": 1.10,
}


def simulate_crash_probability(
    crash_count,
    observed_hours,
    weather_condition="clear",
    num_trials=10000,
    alpha_prior=1,
    beta_prior=1,
    random_seed=None,
):
    """Estimate the chance of at least one crash in a one-hour period.

    A Gamma-Poisson model is used. The Gamma posterior represents uncertainty
    about the hourly crash rate lambda, and each Monte Carlo trial samples a
    possible lambda before sampling a Poisson crash count.

    Parameters
    ----------
    crash_count : int
        Number of crashes observed in the selected ZIP/day/hour bucket.
    observed_hours : int
        Number of one-hour periods observed for that day of week.
    weather_condition : str, default="clear"
        One of clear, rain, snow, fog, or high_wind.
    num_trials : int, default=10000
        Number of Monte Carlo trials to run.
    alpha_prior : float, default=1
        Shape parameter for the Gamma prior.
    beta_prior : float, default=1
        Rate parameter for the Gamma prior.
    random_seed : int or None
        Optional seed for reproducible results.

    Returns
    -------
    dict
        Simulation summary and the raw simulated count array.
    """
    if crash_count < 0:
        raise ValueError("crash_count must be nonnegative.")
    if observed_hours < 0:
        raise ValueError("observed_hours must be nonnegative.")
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if alpha_prior <= 0 or beta_prior <= 0:
        raise ValueError("Gamma prior parameters must be positive.")

    weather_key = weather_condition.lower().strip()
    if weather_key not in WEATHER_MULTIPLIERS:
        valid_weather = ", ".join(sorted(WEATHER_MULTIPLIERS))
        raise ValueError(f"Unknown weather condition. Choose one of: {valid_weather}.")

    alpha_posterior = alpha_prior + crash_count
    beta_posterior = beta_prior + observed_hours

    rng = np.random.default_rng(random_seed)
    sampled_lambda = rng.gamma(
        shape=alpha_posterior,
        scale=1 / beta_posterior,
        size=num_trials,
    )
    adjusted_lambda = sampled_lambda * WEATHER_MULTIPLIERS[weather_key]
    simulated_counts = rng.poisson(lam=adjusted_lambda)

    return {
        "probability_at_least_one": float(np.mean(simulated_counts >= 1)),
        "mean_simulated_crashes": float(np.mean(simulated_counts)),
        "median_simulated_crashes": float(np.median(simulated_counts)),
        "percentile_5": float(np.percentile(simulated_counts, 5)),
        "percentile_95": float(np.percentile(simulated_counts, 95)),
        "simulated_counts": simulated_counts,
    }


def predict_for_zip_hour(
    rate_table,
    zip_code,
    day_of_week,
    hour,
    weather_condition="clear",
    num_trials=10000,
    random_seed=None,
):
    """Look up a ZIP/day/hour row and run the Monte Carlo simulation.

    Parameters
    ----------
    rate_table : pandas.DataFrame or str
        Crash rate table, or a path to a CSV containing it.
    zip_code : str
        NYC ZIP code to evaluate.
    day_of_week : str
        Day name such as "Friday".
    hour : int
        Hour of day from 0 to 23.
    weather_condition : str, default="clear"
        Weather scenario to apply.
    num_trials : int, default=10000
        Number of Monte Carlo trials.
    random_seed : int or None
        Optional seed for reproducible results.

    Returns
    -------
    dict
        Simulation output plus the selected input metadata.
    """
    if isinstance(rate_table, (str, bytes, PathLike)):
        rate_table = pd.read_csv(rate_table, dtype={"zip_code": str})
    else:
        rate_table = rate_table.copy()

    rate_table["zip_code"] = rate_table["zip_code"].astype(str)
    hour = int(hour)
    day = str(day_of_week).strip()
    zip_text = str(zip_code).strip()

    matches = rate_table[
        (rate_table["zip_code"] == zip_text)
        & (rate_table["day_of_week"] == day)
        & (rate_table["hour"].astype(int) == hour)
    ]

    if matches.empty:
        raise ValueError(
            f"No rate-table row found for ZIP {zip_text}, {day}, hour {hour}."
        )

    row = matches.iloc[0]
    result = simulate_crash_probability(
        crash_count=int(row["crash_count"]),
        observed_hours=int(row["observed_hours"]),
        weather_condition=weather_condition,
        num_trials=num_trials,
        random_seed=random_seed,
    )

    result.update(
        {
            "zip_code": zip_text,
            "day_of_week": day,
            "hour": hour,
            "weather_condition": weather_condition,
            "crash_count": int(row["crash_count"]),
            "observed_hours": int(row["observed_hours"]),
            "estimated_lambda": float(row["estimated_lambda"]),
        }
    )
    return result
