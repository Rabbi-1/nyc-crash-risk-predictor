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


def daily_crash_counts_for_bucket(
    cleaned_df, zip_code, day_of_week, hour, *, min_days=1
):
    """Count crashes per calendar day in one ZIP / day-of-week / hour bucket.

    Uses the same exposure idea as :func:`build_crash_rate_table`: days where
    that *day of week* appears anywhere in the cleaned data define the
    one-hour “slots” for the bucket, including days with **zero** crashes in
    that slot (reindexed with zeros).

    Parameters
    ----------
    cleaned_df : pandas.DataFrame
        Output of ``clean_crash_data``; must include ``date``, ``zip_code``,
        ``day_of_week``, and ``hour``.
    zip_code : str
        Five-character ZIP (string to match the rate table).
    day_of_week : str
        Full English day name, e.g. ``"Friday"``, matching ``day_of_week``.
    hour : int
        Integer hour 0--23.
    min_days : int, default=1
        If the resulting series has fewer than this many days, a ``ValueError``
        is raised (avoids empty or trivial plots).

    Returns
    -------
    pandas.Series
        Index: calendar ``date``; values: integer crash count that day in the
        bucket. Sorted by date. Name ``crash_count``.

    See Also
    --------
    The mean of this series equals ``estimated_lambda`` from
    ``crash_count / len(series)`` when the exposure count matches
    ``observed_hours`` from the rate table for that bucket.
    """
    if min_days < 1:
        raise ValueError("min_days must be at least 1.")

    required = {"date", "zip_code", "day_of_week", "hour"}
    missing = sorted(required - set(cleaned_df.columns))
    if missing:
        raise ValueError(f"cleaned_df is missing columns: {missing}")

    df = cleaned_df.copy()
    df["zip_code"] = df["zip_code"].astype(str)
    z = str(zip_code).strip()
    h = int(hour)
    dow = str(day_of_week).strip()

    exposure = df[df["day_of_week"] == dow].dropna(subset=["date"])
    exposure_dates = (
        pd.to_datetime(exposure["date"], errors="coerce")
        .dt.date.dropna()
        .drop_duplicates()
        .sort_values()
    )
    if exposure_dates.empty:
        raise ValueError(f"No rows found for day_of_week {dow!r} in cleaned_df.")

    in_bucket = df[
        (df["zip_code"] == z) & (df["day_of_week"] == dow) & (df["hour"].astype(int) == h)
    ]
    by_date = in_bucket.assign(
        _d=pd.to_datetime(in_bucket["date"], errors="coerce").dt.date
    ).groupby("_d", dropna=True).size()

    counts = by_date.reindex(exposure_dates, fill_value=0)
    counts = counts.astype(int)
    counts.name = "crash_count"

    if len(counts) < min_days:
        raise ValueError(
            f"Not enough exposure days for bucket ZIP {z}, {dow}, hour {h}: "
            f"need at least {min_days}, got {len(counts)}."
        )

    return counts
