"""Beginner-friendly validation helpers for crash risk simulations."""

import pandas as pd

from .monte_carlo import simulate_crash_probability


def convergence_test(
    crash_count,
    observed_hours,
    weather_condition="clear",
    trial_sizes=None,
    alpha_prior=1,
    beta_prior=1,
    random_seed=None,
):
    """Run the same simulation with increasing numbers of trials.

    Parameters
    ----------
    crash_count : int
        Observed crashes in the selected bucket.
    observed_hours : int
        Observed one-hour periods for the selected day of week.
    weather_condition : str, default="clear"
        Weather scenario passed into the simulator.
    trial_sizes : list[int] or None
        Trial counts to test. Defaults to 100, 500, 1000, 5000, 10000.
    alpha_prior : float, default=1
        Gamma prior shape.
    beta_prior : float, default=1
        Gamma prior rate.
    random_seed : int or None
        Optional seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: num_trials and probability_at_least_one.
    """
    if trial_sizes is None:
        trial_sizes = [100, 500, 1000, 5000, 10000]

    rows = []
    for index, num_trials in enumerate(trial_sizes):
        seed = None if random_seed is None else random_seed + index
        result = simulate_crash_probability(
            crash_count=crash_count,
            observed_hours=observed_hours,
            weather_condition=weather_condition,
            num_trials=num_trials,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            random_seed=seed,
        )
        rows.append(
            {
                "num_trials": int(num_trials),
                "probability_at_least_one": result["probability_at_least_one"],
            }
        )

    return pd.DataFrame(rows)


def brier_score(predicted_probability, actual_outcome):
    """Calculate the Brier score for one probability prediction.

    A Brier score is the squared error between a predicted probability and the
    actual binary outcome. Lower is better.
    """
    probability = float(predicted_probability)
    outcome = int(actual_outcome)

    if not 0 <= probability <= 1:
        raise ValueError("predicted_probability must be between 0 and 1.")
    if outcome not in (0, 1):
        raise ValueError("actual_outcome must be 0 or 1.")

    return (probability - outcome) ** 2


def simple_train_test_split_by_date(cleaned_df, test_fraction=0.2):
    """Split cleaned crash data into earlier training dates and later test dates.

    This is intentionally simple for an introductory statistics project. It
    avoids random row splitting so that future dates are held out from the
    training period.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    df = cleaned_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values("date")

    unique_dates = sorted(df["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("Need at least two unique dates to create a date split.")

    split_index = max(1, int(len(unique_dates) * (1 - test_fraction)))
    split_index = min(split_index, len(unique_dates) - 1)
    test_dates = set(unique_dates[split_index:])

    train_df = df[~df["date"].isin(test_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, test_df
