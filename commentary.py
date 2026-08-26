"""
Automated forecast commentary.

The recurring manual task in any forecasting function is not building the model
— it is writing the same weekly narrative: what changed, whether the model is
still healthy, what the planner needs to act on. That task is high-volume,
low-variance, and template-shaped, which makes it the right thing to automate.

This module assembles a structured evidence pack from the pipeline artefacts,
runs a deterministic rule layer over it to decide which exceptions are worth
raising, and then uses an LLM only for the final translation into prose.

The ordering is deliberate. The rules decide *what is true*; the model decides
only *how to say it*. Letting the model do the reasoning would put unverifiable
numbers in front of a planner, which is exactly the failure mode that stops
teams trusting automated reporting.

Runs without an API key — it falls back to templated prose from the same rules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import pandas as pd

MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"


# --------------------------------------------------------------------------
# Evidence assembly
# --------------------------------------------------------------------------

@dataclass
class Exception_:
    severity: str      # "action" | "watch" | "info"
    code: str
    detail: str


def build_evidence(
    metrics: dict,
    health: pd.DataFrame,
    variance: pd.DataFrame,
    lead_time: pd.DataFrame,
    plan: pd.DataFrame,
    model_ranking: pd.DataFrame,
) -> dict:
    """Collapse the pipeline outputs into a compact, quotable evidence pack."""
    recent = health.dropna(subset=["rolling_wape"]).tail(28)
    return {
        "headline_wape_pct": metrics["wape_pct"],
        "bias_pct": metrics["bias_pct"],
        "tracking_signal": metrics["tracking_signal"],
        "days_evaluated": metrics["n_days"],
        "rolling_wape_latest": round(float(recent.rolling_wape.iloc[-1]), 2) if len(recent) else None,
        "control_threshold_pct": float(health.threshold.iloc[0]),
        "breach_days_last_28": int(recent.breach.sum()) if len(recent) else 0,
        "variance_causes": variance.to_dict("records"),
        "wape_week_1": float(lead_time.loc[lead_time.week == 1, "wape_pct"].mean().round(2)),
        "wape_week_4": float(lead_time.loc[lead_time.week == 4, "wape_pct"].mean().round(2)),
        "best_model": model_ranking.iloc[0]["model"],
        "best_model_wape": float(model_ranking.iloc[0]["wape_pct"]),
        "baseline_wape": float(
            model_ranking.loc[model_ranking.model == "SeasonalNaive", "wape_pct"].iloc[0]
        ),
        "horizon_days": int(len(plan)),
        "peak_rostered_agents": int(plan.agents_rostered.max()),
        "trough_rostered_agents": int(plan.agents_rostered.min()),
        "mean_rostered_agents": round(float(plan.agents_rostered.mean()), 1),
        "peak_date": str(plan.agents_rostered.idxmax().date()),
    }


# --------------------------------------------------------------------------
# Deterministic rule layer
# --------------------------------------------------------------------------

def raise_exceptions(ev: dict) -> list[Exception_]:
    """Decide what is worth a planner's attention. No model involved."""
    out: list[Exception_] = []

    if ev["rolling_wape_latest"] and ev["rolling_wape_latest"] > ev["control_threshold_pct"]:
        out.append(Exception_(
            "action", "ACCURACY_BREACH",
            f"Rolling 14-day WAPE is {ev['rolling_wape_latest']}%, above the "
            f"{ev['control_threshold_pct']}% control limit."))

    if abs(ev["tracking_signal"]) > 4:
        direction = "over" if ev["tracking_signal"] > 0 else "under"
        out.append(Exception_(
            "action", "MODEL_DRIFT",
            f"Tracking signal is {ev['tracking_signal']}, indicating sustained "
            f"{direction}-forecasting. Refit rather than leave running."))

    if abs(ev["bias_pct"]) > 2:
        direction = "over" if ev["bias_pct"] > 0 else "under"
        out.append(Exception_(
            "watch", "BIAS",
            f"Model {direction}-forecasts by {abs(ev['bias_pct'])}% overall; "
            f"{'surplus roster cost' if ev['bias_pct'] > 0 else 'service-level risk'} "
            f"is the exposure."))

    lift = round(100 * (1 - ev["best_model_wape"] / ev["baseline_wape"]), 1)
    if lift < 15:
        out.append(Exception_(
            "watch", "WEAK_LIFT",
            f"{ev['best_model']} beats the seasonal-naive baseline by only "
            f"{lift}%. The added complexity is not yet earning its maintenance cost."))
    else:
        out.append(Exception_(
            "info", "MODEL_LIFT",
            f"{ev['best_model']} improves on the seasonal-naive baseline by {lift}%."))

    top = ev["variance_causes"][0]
    out.append(Exception_(
        "info", "VARIANCE_DRIVER",
        f"{top['cause']} accounts for {top['share_pct']}% of total absolute error."))

    decay = round(ev["wape_week_4"] - ev["wape_week_1"], 2)
    out.append(Exception_(
        "info", "HORIZON_DECAY",
        f"Accuracy decays {decay} percentage points from week 1 "
        f"({ev['wape_week_1']}%) to week 4 ({ev['wape_week_4']}%)."))

    return out


# --------------------------------------------------------------------------
# Narrative generation
# --------------------------------------------------------------------------

_SYSTEM = """You write the weekly forecast review note for a support operations \
planning team.

Rules:
- Use ONLY the figures in the evidence pack. Never estimate, round differently, \
or introduce a number that is not present.
- Lead with anything marked severity "action". If there is none, say the model \
is within tolerance and keep it short.
- Write for a planner who owns a roster, not for a data scientist.
- Three short paragraphs maximum. No headings, no bullet points, no preamble.
- State what to do, not what was computed."""


def _fallback(ev: dict, exceptions: list[Exception_]) -> str:
    """Templated narrative used when no API key is configured."""
    actions = [e for e in exceptions if e.severity == "action"]
    watches = [e for e in exceptions if e.severity == "watch"]

    if actions:
        p1 = ("Forecast requires intervention this cycle. "
              + " ".join(e.detail for e in actions))
    else:
        p1 = (f"Forecast is within tolerance. Headline WAPE across "
              f"{ev['days_evaluated']} evaluated days is {ev['headline_wape_pct']}%, "
              f"inside the {ev['control_threshold_pct']}% control limit.")

    p2 = " ".join(e.detail for e in watches) if watches else (
        f"No secondary concerns. {ev['variance_causes'][0]['cause']} remains the "
        f"largest error contributor at {ev['variance_causes'][0]['share_pct']}%.")

    p3 = (f"Roster plan over the next {ev['horizon_days']} days averages "
          f"{ev['mean_rostered_agents']} agents, peaking at "
          f"{ev['peak_rostered_agents']} on {ev['peak_date']} and falling to "
          f"{ev['trough_rostered_agents']} at trough. Weekend troughs are the "
          f"flex capacity if cover is needed elsewhere.")

    return "\n\n".join([p1, p2, p3])


def generate(ev: dict, exceptions: list[Exception_]) -> tuple[str, str]:
    """Return (narrative, source) where source is 'llm' or 'rules'."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _fallback(ev, exceptions), "rules"

    try:
        import urllib.request

        payload = {
            "model": MODEL,
            "max_tokens": 700,
            "system": _SYSTEM,
            "messages": [{
                "role": "user",
                "content": (
                    "Evidence pack:\n"
                    + json.dumps(ev, indent=2)
                    + "\n\nExceptions raised by the rule layer:\n"
                    + json.dumps([asdict(e) for e in exceptions], indent=2)
                    + "\n\nWrite the review note."
                ),
            }],
        }
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
        text = "".join(b.get("text", "") for b in body.get("content", [])
                       if b.get("type") == "text").strip()
        return (text, "llm") if text else (_fallback(ev, exceptions), "rules")

    except Exception:
        # Never let the commentary layer break the pipeline. The numbers are
        # the deliverable; the prose is a convenience.
        return _fallback(ev, exceptions), "rules"
