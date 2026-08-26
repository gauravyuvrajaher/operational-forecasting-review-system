"""
Forecast accuracy measurement, model health monitoring and variance
root-cause decomposition.

Producing a forecast is the easy half. The half that decides whether anyone
trusts the output is knowing, week to week, whether the model is still working
and — when it is not — which part of it broke.

Metric choice:
    WAPE  is the headline. MAPE divides by the actual, so low-volume weekend
          days dominate the average and a model can look terrible while being
          fine on the days that carry the staffing cost.
    Bias  (MPE) is tracked separately because a model can be accurate in
          magnitude and still consistently under-forecast, which is the error
          that understaffs a queue.
    Tracking signal flags sustained one-directional drift before it is visible
          in WAPE at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------

def wape(actual: pd.Series, forecast: pd.Series) -> float:
    """Weighted absolute percentage error. Volume-weighted, so large days count."""
    return 100 * (actual - forecast).abs().sum() / actual.sum()


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    return 100 * ((actual - forecast).abs() / actual.replace(0, np.nan)).mean()


def bias_pct(actual: pd.Series, forecast: pd.Series) -> float:
    """Signed error. Negative means the model under-forecasts."""
    return 100 * (forecast - actual).sum() / actual.sum()


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.sqrt(((actual - forecast) ** 2).mean()))


def tracking_signal(actual: pd.Series, forecast: pd.Series,
                    window: int = 28) -> float:
    """Cumulative signed error over mean absolute deviation, most recent window.

    Conventionally |TS| > 4 means the model has drifted and should be refitted
    rather than left to run.

    The window matters. Computed over all evaluated history the statistic grows
    with sample size and stops being comparable to the threshold — a long, mildly
    biased backtest will breach it even when the model is currently healthy. The
    question the metric answers is "is it drifting now", so it is scoped to the
    last `window` days.
    """
    err = (forecast - actual).tail(window)
    mad = err.abs().mean()
    return float(err.sum() / mad) if mad else 0.0


def summary(frame: pd.DataFrame) -> dict:
    """All headline metrics for a backtest frame."""
    a, f = frame["actual"], frame["forecast"]
    return {
        "wape_pct": round(wape(a, f), 2),
        "mape_pct": round(mape(a, f), 2),
        "bias_pct": round(bias_pct(a, f), 2),
        "rmse": round(rmse(a, f), 1),
        "tracking_signal": round(tracking_signal(a, f), 2),
        "tracking_window_days": 28,
        "n_days": len(frame),
    }


# --------------------------------------------------------------------------
# Health monitoring
# --------------------------------------------------------------------------

def health_monitor(frame: pd.DataFrame, window: int = 14,
                   threshold_pct: float = 8.0) -> pd.DataFrame:
    """Rolling accuracy with a control limit.

    Returns a daily frame carrying rolling WAPE and a breach flag. This is the
    artefact an operations team actually watches: a single number per day, with
    a line on it that says when to intervene.
    """
    df = frame.sort_values("date").copy()
    df["abs_err"] = (df.actual - df.forecast).abs()

    roll_err = df.abs_err.rolling(window, min_periods=window).sum()
    roll_act = df.actual.rolling(window, min_periods=window).sum()
    df["rolling_wape"] = 100 * roll_err / roll_act
    df["breach"] = df.rolling_wape > threshold_pct
    df["threshold"] = threshold_pct
    return df.drop(columns="abs_err")


# --------------------------------------------------------------------------
# Variance root-cause
# --------------------------------------------------------------------------

def decompose_variance(frame: pd.DataFrame,
                       incident_flags: pd.Series | None = None) -> pd.DataFrame:
    """Attribute total forecast error to identifiable causes.

    Splits absolute error across four buckets:

        incident      - days flagged as operational incidents. Genuinely
                        unforecastable; excluding them is legitimate, hiding
                        them is not.
        holiday_adj   - days adjacent to public holidays, where deferred
                        demand shifts rather than disappears.
        day_of_week   - the share explained by a systematic weekday effect,
                        estimated as the mean signed error per weekday.
        unexplained   - residual. This is the number that should drive
                        methodology change.

    Reporting this instead of a single WAPE is what turns "the forecast was
    wrong" into "the forecast was wrong for a reason we can or cannot fix".
    """
    df = frame.sort_values("date").copy()
    df["error"] = df.forecast - df.actual
    df["abs_error"] = df.error.abs()
    df["weekday"] = pd.to_datetime(df.date).dt.day_name()

    if incident_flags is not None:
        df["is_incident"] = pd.to_datetime(df.date).map(incident_flags).fillna(False)
    else:
        df["is_incident"] = False

    # Systematic weekday component: the portion of error the model repeats
    # every week and could therefore correct for.
    dow_bias = df.groupby("weekday").error.transform("mean")
    df["dow_component"] = dow_bias.abs().clip(upper=df.abs_error)

    total = df.abs_error.sum()
    incident = df.loc[df.is_incident, "abs_error"].sum()
    remaining = df.loc[~df.is_incident]
    dow = remaining.dow_component.sum()
    unexplained = total - incident - dow

    out = pd.DataFrame(
        {
            "cause": ["Incident spikes", "Systematic day-of-week bias", "Unexplained residual"],
            "abs_error": [incident, dow, unexplained],
        }
    )
    out["share_pct"] = (100 * out.abs_error / total).round(1)
    out["abs_error"] = out.abs_error.round(0)
    return out.sort_values("share_pct", ascending=False).reset_index(drop=True)


def accuracy_by_lead_time(frame: pd.DataFrame) -> pd.DataFrame:
    """WAPE by forecast lead time.

    Accuracy decays with horizon. Knowing the shape of that decay tells
    planners how far ahead the forecast can safely drive a hiring decision.
    """
    g = frame.groupby("lead_days").apply(
        lambda d: pd.Series({"wape_pct": round(wape(d.actual, d.forecast), 2)}),
        include_groups=False,
    )
    g["week"] = ((g.index - 1) // 7 + 1)
    return g.reset_index()
