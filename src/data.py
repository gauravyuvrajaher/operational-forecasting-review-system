"""
Synthetic support-operations dataset generator.

Produces a daily contact-centre style series with the structural features that
make real operational forecasting hard:

  - multiplicative growth trend
  - strong day-of-week seasonality (weekday peak, weekend trough)
  - annual seasonality
  - public-holiday suppression
  - a step change from a product launch
  - incident spikes (short, sharp, non-repeating)
  - AHT (average handle time) drifting independently of volume

The generator is seeded, so every figure in the README is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Irish public holidays, 2023-2025. Volume is suppressed on these dates.
HOLIDAYS = [
    "2023-01-01", "2023-03-17", "2023-04-10", "2023-05-01", "2023-06-05",
    "2023-08-07", "2023-10-30", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-02-05", "2024-03-17", "2024-04-01", "2024-05-06",
    "2024-06-03", "2024-08-05", "2024-10-28", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-02-03", "2025-03-17", "2025-04-21", "2025-05-05",
    "2025-06-02", "2025-08-04", "2025-10-27", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-02-02", "2026-03-17", "2026-04-06", "2026-05-04",
    "2026-06-01", "2026-08-03", "2026-10-26", "2026-12-25", "2026-12-26",
]


def make_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Holiday regressors for any date range, past or future.

    Holidays are the single largest source of avoidable forecast error in
    operational series: they are large, they move year to year, and — unlike
    incidents — they are known in advance. A model without them has to learn a
    weekly pattern that breaks predictably several times a year, and it cannot.

    Three features, because a holiday does not only suppress its own day:
        is_holiday   - the day itself
        pre_holiday  - demand pulled forward
        post_holiday - deferred demand landing the next working day
    """
    hol = pd.DatetimeIndex(pd.to_datetime(HOLIDAYS))
    is_hol = index.isin(hol)
    return pd.DataFrame(
        {
            "is_holiday": is_hol.astype(float),
            "pre_holiday": index.isin(hol - pd.Timedelta(days=1)).astype(float),
            "post_holiday": index.isin(hol + pd.Timedelta(days=1)).astype(float),
        },
        index=index,
    )

DOW_FACTORS = {
    0: 1.28,  # Monday - post-weekend backlog
    1: 1.15,
    2: 1.08,
    3: 1.05,
    4: 0.98,
    5: 0.52,  # Saturday
    6: 0.44,  # Sunday
}


def generate(
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    base_volume: float = 4200.0,
    annual_growth: float = 0.14,
    launch_date: str = "2024-09-16",
    launch_uplift: float = 0.18,
    seed: int = 20260824,
) -> pd.DataFrame:
    """Generate the daily operational series.

    Returns a frame indexed by date with columns:
        volume      - contacts received
        aht_seconds - average handle time
        is_holiday  - holiday flag
        is_incident - incident-spike flag (ground truth, for evaluation only)
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    n = len(idx)
    t = np.arange(n)

    # --- Trend: compounding daily growth ---------------------------------
    daily_growth = (1 + annual_growth) ** (1 / 365.25)
    trend = base_volume * daily_growth**t

    # --- Day-of-week seasonality -----------------------------------------
    dow = np.array([DOW_FACTORS[d.weekday()] for d in idx])

    # --- Annual seasonality: Q4 peak, summer trough ----------------------
    doy = idx.dayofyear.to_numpy()
    annual = 1 + 0.12 * np.sin(2 * np.pi * (doy - 80) / 365.25) \
               + 0.07 * np.sin(4 * np.pi * (doy - 40) / 365.25)

    # --- Product launch step change ---------------------------------------
    launch = pd.Timestamp(launch_date)
    ramp = np.clip((idx - launch).days / 30.0, 0, 1)
    step = 1 + launch_uplift * ramp

    # --- Holidays ---------------------------------------------------------
    hol = idx.isin(pd.to_datetime(HOLIDAYS))
    hol_factor = np.where(hol, 0.35, 1.0)

    # Days either side of a holiday run slightly hot (deferred demand)
    shoulder = np.zeros(n)
    hol_pos = np.flatnonzero(hol)
    for p in hol_pos:
        if p + 1 < n:
            shoulder[p + 1] += 0.15
        if p - 1 >= 0:
            shoulder[p - 1] += 0.08

    signal = trend * dow * annual * step * hol_factor * (1 + shoulder)

    # --- Incident spikes ---------------------------------------------------
    # Roughly one meaningful incident every five weeks.
    is_incident = rng.random(n) < (1 / 35)
    spike = np.where(is_incident, rng.uniform(1.45, 2.30, n), 1.0)
    # Spikes decay over the following two days.
    decay = np.ones(n)
    for p in np.flatnonzero(is_incident):
        if p + 1 < n:
            decay[p + 1] = max(decay[p + 1], 1.22)
        if p + 2 < n:
            decay[p + 2] = max(decay[p + 2], 1.08)
    signal = signal * spike * decay

    # --- Observation noise -------------------------------------------------
    noise = rng.normal(1.0, 0.045, n)
    volume = np.maximum(np.round(signal * noise), 0).astype(int)

    # --- AHT: independent slow drift + weekday effect ----------------------
    aht = (
        430
        + 55 * np.sin(2 * np.pi * (doy - 150) / 365.25)
        + 0.045 * t                      # complexity creep as product matures
        + np.where(idx.weekday >= 5, 28, 0)   # weekend staff less experienced
        + rng.normal(0, 11, n)
    )
    # Incidents drive handle time up as well as volume.
    aht = aht * np.where(is_incident, 1.16, 1.0)

    return pd.DataFrame(
        {
            "volume": volume,
            "aht_seconds": np.round(aht, 1),
            "is_holiday": hol,
            "is_incident": is_incident,
        },
        index=idx,
    ).rename_axis("date")


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\n{len(df)} days | mean volume {df.volume.mean():,.0f} "
          f"| mean AHT {df.aht_seconds.mean():.0f}s "
          f"| {df.is_incident.sum()} incidents")
