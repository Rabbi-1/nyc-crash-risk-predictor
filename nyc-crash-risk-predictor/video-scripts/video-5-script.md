# NYC Motor Vehicle Collisions Data: Dataset, Cleaning, Bucketing, Poisson Modeling, and NOAA Weather

This note describes the NYC Open Data **Motor Vehicle Collisions – Crashes** dataset, how to clean and aggregate it by ZIP code, hour, and day of week (aligned with this project’s `src/` pipeline), why a **Poisson** arrival model is often used, what **empirical checks** support or challenge that choice, and how to **merge hourly NOAA weather** for a stronger version of the analysis.

---

## 1. The NYC Open Data crash dataset

**Official name:** Motor Vehicle Collisions – Crashes (NYPD reported collisions).

**Socrata API CSV endpoint (this repo):**

[https://data.cityofnewyork.us/resource/h9gi-nx95.csv](https://data.cityofnewyork.us/resource/h9gi-nx95.csv)

Each row is typically **one crash event** (not one injury; injury/person tables exist in related datasets). Fields vary over time; this project selects a small subset for teaching and speed:


| Field                   | Role                                                           |
| ----------------------- | -------------------------------------------------------------- |
| `crash_date`            | Calendar date of the crash                                     |
| `crash_time`            | Time of day (often `H:M` or `H:M:S`)                           |
| `borough`               | NYC borough                                                    |
| `zip_code`              | ZIP tabulation area / postal code (used for spatial bucketing) |
| `latitude`, `longitude` | Coordinates when geocoded (optional for maps / QC)             |


The raw feed is large. Downloads commonly use Socrata query parameters named `limit`, `select`, and `order` (in HTTP query strings these are often spelled with a leading dollar sign before each name; see `src/download_data.py` in this repo).

**Important caveats:** reporting lag, revised records, missing ZIP or time, geocode errors, and the fact that **ZIP is an imperfect geography** for traffic risk (it mixes many streets and land uses).

---

## 2. How to clean it

Cleaning goals: **parseable datetime**, **consistent ZIP strings**, **valid hour and day-of-week labels**, and **numeric coordinates** where present.

### 2.1 Row and column subset

- Keep only columns needed for modeling and QC.
- Drop rows missing **date**, **time**, or **ZIP** (they cannot be placed in a ZIP/hour/dow bucket).

### 2.2 ZIP code normalization

Common issues:

- Floats from CSV (`10001.0`)
- Short numeric strings (`1001` → should be `01001` only where appropriate; NYC ZIPs are often 5 digits starting with `0` for some areas—**zero-pad to 5 characters** when the value is all digits and shorter than 5.)

This project implements normalization in `_normalize_zip_code` in `src/clean_crashes.py`.

### 2.3 Build `crash_datetime`, `date`, `hour`, `day_of_week`

1. Parse `crash_date` with `errors='coerce'` and normalize to midnight date.
2. Parse `crash_time` after stripping whitespace; try `%H:%M` then fall back to `%H:%M:%S`.
3. Combine date + time into **`crash_datetime`**, then derive:
  - **`date`**: calendar date (for counting distinct exposure days)
  - **`hour`**: integer 0–23
  - **`day_of_week`**: full English name (e.g. `Friday`) to match grouping keys

Drop rows where datetime parsing fails.

### 2.4 Coordinates

Cast `latitude` / `longitude` with `errors='coerce'` for mapping; they are not required for the ZIP/hour/dow rate table in this project.

**Reference implementation:** `clean_crash_data` in `src/clean_crashes.py` writes `data/processed/nyc_crashes_clean.csv`.

---

## 3. Bucketing by ZIP code, hour, and day of week

After cleaning, each crash row sits in a bucket **(zip_code, day_of_week, hour)**.

### 3.1 Crash counts per bucket

Group the cleaned dataframe:

- `groupby(["zip_code", "day_of_week", "hour"]).size()` → **`crash_count`**

### 3.2 Exposure: `observed_hours`

For a **teaching exposure** definition consistent with this repo: count **distinct calendar dates** that appear for each **day of week** in the cleaned sample. For each bucket on that day of week, **observed hours** is the number of distinct dates labeled that weekday in the dataset—the same quantity as the **`observed_hours`** column in the rate table.

Example: if the cleaned data spans **20 different Fridays**, then every Friday bucket for a given ZIP and hour is assumed to have **20** one-hour periods of exposure (see docstring in `src/feature_engineering.py`).

Then the **empirical crashes per hour** for that ZIP / day / hour (lambda-hat, **λ̂**) is:

**λ̂** = `crash_count` / `observed_hours`

The numerator and denominator are the same values as **`crash_count`** and **`observed_hours`** in `crash_rate_table.csv` (and match **`estimated_lambda`** in that table).

**Reference implementation:** `build_crash_rate_table` in `src/feature_engineering.py` outputs `data/processed/crash_rate_table.csv` with columns including `crash_count`, `observed_hours`, and `estimated_lambda`.

**Modeling note:** this exposure rule ignores **ZIP-specific date coverage** (every ZIP gets the same weekday count). A stricter pipeline would count, for each `(zip, dow, hour)`, the distinct dates on which that ZIP actually had any data—or use a full calendar of NYC hours.

---

## 4. Poisson assumption for crash “arrivals”

### 4.1 What the model assumes

For a fixed bucket (ZIP, day, hour) and a **fixed hour-long window**, let **N** be the random number of crashes in that window. A **Poisson** model assumes:

1. **Counts:** *N* ~ Poisson(**λ**), with **λ** the expected number of crashes in that hour (possibly scaled for weather or other covariates).
2. **Independence** across disjoint time windows (after accounting for systematic effects)—so **no residual clustering** beyond what **λ** explains.
3. **Rare events** in a fine enough partition: classical Poisson limits (from Bernoulli trials) motivate treating many small independent risks as Poisson.

In this repo’s Monte Carlo layer, after drawing an uncertain **λ** from a **Gamma** posterior, each trial samples **`Poisson(adjusted_lambda)`** (see `simulate_crash_probability` in `src/monte_carlo.py`). That is a **Gamma–Poisson** (negative-binomial marginal over **λ**) hierarchy for **parameter uncertainty**, not a claim that the raw world is exactly Poisson.

### 4.2 What Poisson is *not* claiming

- It does not deny **traffic volume, events, or weather**; those usually enter by making **λ** **vary by hour** or by covariates.
- Multi-vehicle pileups can break strict independence; **overdispersion** (variance > mean) is common when **λ** is heterogeneous or clustered.

---

## 5. Empirical distributions that justify (or test) Poisson

Poisson should be evaluated on **comparable windows**—e.g. **one-hour counts** for the same (ZIP, dow, hour) across many days—so the support of the count distribution is meaningful.

### 5.1 Mean–variance relationship

For Poisson(**λ**), **Var(N) = E(N) = λ**.

Empirical check: for each bucket with many days *d* = 1, …, *D*, let *n<sub>d</sub>* be the crash count in that ZIP/dow/hour on day *d*. Compare:

- sample mean *n̄* (n-bar)
- sample variance *s*²

If *s*²/*n̄* is often **substantially above 1**, counts are **overdispersed** relative to Poisson (a **Negative Binomial** or random-**λ** model may fit better). Near 1 supports Poisson *as a first-order* model.

### 5.2 Histogram vs Poisson PMF

For buckets with **large D** and moderate *n̄*:

- Plot the empirical frequency of *n<sub>d</sub>* ∈ {0, 1, 2, …}.
- Overlay Poisson(**λ̂**) with **λ̂** = *n̄*.

Good visual agreement for small counts (lots of zeros and ones) is common for rare events; systematic **fat tails** or excess zeros suggest alternatives (zero-inflated Poisson, hurdle, or NB).

### 5.3 Poissonness plot / deviance

- **Poissonness plot** (count metameter): checks curvature against the expected pattern under Poisson.
- **Deviance / Pearson χ²** in a **GLM** with log link and Poisson family: **overdispersion** shows up as residual deviance >> residual degrees of freedom.

### 5.4 Time series diagnostics (optional)

If using **consecutive hours** in one ZIP, check **autocorrelation** in residuals; strong ARIMA structure violates independence across hours and pushes toward **time-series count models** (e.g. integer-valued GARCH) or richer covariates.

**Bottom line:** Poisson is a **standard, interpretable starting point** for rare counts per fixed window; **mean–variance ratio** and **PMF overlays** are the clearest undergraduate-friendly empirical justifications, with explicit discussion when the data **reject** simple Poisson.

---

## 6. Weather merge: NOAA hourly data

The starter project uses **hand-set weather multipliers** (`WEATHER_MULTIPLIERS` in `src/monte_carlo.py`). A data-driven extension merges **observed hourly weather** onto each crash (or onto each calendar hour for each ZIP).

### 6.1 Typical NOAA sources

- **NOAA ISD (Integrated Surface Database)** – global hourly (and sub-hourly) station observations; widely used for research.
- **NCEI** access portals and APIs for station data (URLs and APIs change; search NCEI for “ISD” or “hourly station data” for current endpoints).

NYC-relevant stations often include airports and climate stations (e.g. **JFK**, **LGA**, **Central Park**), each with an identifier in the ISD station catalog.

### 6.2 Constructing an hourly weather table

For a chosen station (or **nearest station** to a ZIP centroid):

1. Pull **UTC or local timestamps** and hourly fields such as:
  - **precipitation** (e.g. hourly totals),
  - **temperature**, **dewpoint**,
  - **wind speed / gust**,
  - **visibility** (for fog),
  - **weather codes** / present weather strings (for rain/snow).
2. Build a regular table keyed by **`datetime_hour`** (e.g. floor `crash_datetime` to the hour in **America/New_York** to match NYPD local reporting).

### 6.3 Merge keys

- **Crash-level merge:** `pd.merge_asof` on sorted `crash_datetime` with weather **at or before** the crash hour, or an **exact** merge on `date` + `hour` if both tables are hourly buckets.
- **ZIP-level teaching merge:** assign each ZIP a **representative station** (nearest Euclidean distance in projected coordinates, or fixed “NYC blend”), then use the same hourly series for all ZIPs (acknowledge **spatial error**).

### 6.4 Using weather in the model

Instead of a discrete `rain` / `clear` label, you can:

- Fit a **Poisson or NB regression**: log **λ** = *β*₀ + *β*ᵀ*x*<sub>weather</sub> + …
- Or bin weather into quantiles and estimate **multipliers from data** (replacing the teaching table in `monte_carlo.py`).

Always document **timezone alignment** and **station vs ZIP mismatch** in limitations.

---

## 7. Map to this repository


| Step                | Artifact / module                                                    |
| ------------------- | -------------------------------------------------------------------- |
| Download            | `src/download_data.py` → `data/raw/nyc_crashes_raw.csv`              |
| Clean               | `src/clean_crashes.py` → `data/processed/nyc_crashes_clean.csv`      |
| Bucket / rates      | `src/feature_engineering.py` → `data/processed/crash_rate_table.csv` |
| Poisson simulation  | `src/monte_carlo.py` (`rng.poisson`)                                 |
| Weather (teaching)  | `WEATHER_MULTIPLIERS` in `src/monte_carlo.py`                        |
| Weather (extension) | *Not implemented in repo*—merge NOAA hourly as in §6                 |


---

## References (external)

- NYC Open Data – Motor Vehicle Collisions – Crashes: `https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95`
- NOAA / NCEI – Integrated Surface Database (ISD): search `https://www.ncei.noaa.gov/` for current ISD documentation and access.

