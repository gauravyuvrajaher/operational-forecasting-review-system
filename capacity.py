"""
Capacity planning: translating a volume forecast into a staffing requirement.

A forecast that stops at "how many contacts" is only half an answer. The
question operations actually asks is "how many people, on which days, to hold
the service level" — and the two are not linearly related. Erlang C is
non-linear in a way that matters: the last few points of service level cost far
more headcount than the first eighty.

Model assumptions, stated plainly because they are the model's real weakness:
    - Poisson arrivals within the interval (bursty traffic breaks this)
    - exponential handle times
    - no abandonment (so staffing is conservative — real queues shed load)
    - infinite queue capacity

Erlang A (Palm) relaxes the abandonment assumption and is the natural next
step if this were carrying real headcount decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ServiceTarget:
    """A service-level agreement, expressed the way WFM teams express it."""
    answer_pct: float = 0.80        # answer 80% of contacts...
    within_seconds: float = 20.0    # ...within 20 seconds
    max_occupancy: float = 0.85     # cap: sustained higher burns agents out
    shrinkage: float = 0.30         # leave, breaks, training, absence


def erlang_b(agents: int, intensity: float) -> float:
    """Erlang B blocking probability, computed by the stable recursion.

    The closed form overflows for realistic agent counts; the recursion does
    not, which is why it is used here.
    """
    inv = 1.0
    for n in range(1, agents + 1):
        inv = 1.0 + inv * n / intensity
    return 1.0 / inv


def erlang_c(agents: int, intensity: float) -> float:
    """Probability that an arriving contact has to wait."""
    if agents <= intensity:
        return 1.0
    b = erlang_b(agents, intensity)
    return b / (1 - (intensity / agents) * (1 - b))


def service_level(agents: int, intensity: float, aht_seconds: float,
                  target_seconds: float) -> float:
    """Fraction of contacts answered within the target wait."""
    if agents <= intensity:
        return 0.0
    c = erlang_c(agents, intensity)
    return 1 - c * math.exp(-(agents - intensity) * target_seconds / aht_seconds)


def agents_required(
    contacts: float,
    aht_seconds: float,
    interval_seconds: float = 1800.0,
    target: ServiceTarget = ServiceTarget(),
) -> dict:
    """Minimum agents to hold the service target for one interval.

    Returns both the raw Erlang requirement and the rostered requirement after
    shrinkage — the second is the number that goes into a hiring plan, and the
    gap between them is routinely the thing that gets forgotten.
    """
    intensity = (contacts * aht_seconds) / interval_seconds  # Erlangs
    if intensity <= 0:
        return {"intensity": 0.0, "agents_raw": 0, "agents_rostered": 0,
                "service_level": 1.0, "occupancy": 0.0}

    n = max(1, math.floor(intensity) + 1)
    while True:
        sl = service_level(n, intensity, aht_seconds, target.within_seconds)
        occ = intensity / n
        if sl >= target.answer_pct and occ <= target.max_occupancy:
            break
        n += 1
        if n > 10000:
            break

    rostered = math.ceil(n / (1 - target.shrinkage))
    return {
        "intensity": round(intensity, 2),
        "agents_raw": n,
        "agents_rostered": rostered,
        "service_level": round(service_level(n, intensity, aht_seconds,
                                             target.within_seconds), 4),
        "occupancy": round(intensity / n, 4),
    }


def staffing_plan(
    forecast: pd.Series,
    aht_seconds: pd.Series | float,
    operating_hours: float = 12.0,
    target: ServiceTarget = ServiceTarget(),
) -> pd.DataFrame:
    """Convert a daily volume forecast into a daily rostered-agent plan.

    Daily volume is spread evenly across operating hours, which understates the
    intraday peak. Real WFM applies an intraday arrival curve; the flat
    assumption is called out here because it makes this plan optimistic at peak
    and the reader should know that.
    """
    if np.isscalar(aht_seconds):
        aht = pd.Series(float(aht_seconds), index=forecast.index)
    else:
        aht = aht_seconds.reindex(forecast.index).ffill()

    interval = 1800.0
    intervals_per_day = operating_hours * 3600 / interval

    rows = []
    for date, vol in forecast.items():
        per_interval = max(vol / intervals_per_day, 0.0)
        r = agents_required(per_interval, float(aht.loc[date]), interval, target)
        r["date"] = date
        r["forecast_volume"] = round(float(vol))
        r["aht_seconds"] = round(float(aht.loc[date]), 1)
        rows.append(r)

    df = pd.DataFrame(rows).set_index("date")
    return df[["forecast_volume", "aht_seconds", "intensity", "agents_raw",
               "agents_rostered", "service_level", "occupancy"]]


def cost_of_error(actual: pd.Series, forecast: pd.Series,
                  aht_seconds: float, target: ServiceTarget = ServiceTarget(),
                  operating_hours: float = 12.0) -> pd.DataFrame:
    """Restate forecast error in agent-days rather than percentages.

    This is the translation that makes accuracy legible to a budget holder: a
    4% WAPE is abstract, "we rostered 11 agent-days we did not need and were
    short 6 on the days we did" is not.
    """
    plan_f = staffing_plan(forecast, aht_seconds, operating_hours, target)
    plan_a = staffing_plan(actual, aht_seconds, operating_hours, target)

    out = pd.DataFrame({
        "actual_volume": actual.round(0),
        "forecast_volume": forecast.round(0),
        "agents_needed": plan_a.agents_rostered,
        "agents_planned": plan_f.agents_rostered,
    })
    out["agent_gap"] = out.agents_planned - out.agents_needed
    out["status"] = np.where(out.agent_gap > 0, "over-staffed",
                     np.where(out.agent_gap < 0, "under-staffed", "matched"))
    return out
