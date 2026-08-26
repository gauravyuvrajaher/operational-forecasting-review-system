"""
End-to-end pipeline.

    python run_pipeline.py

Runs: data -> model selection -> backtest -> accuracy -> variance ->
capacity plan -> automated commentary, and writes artefacts to outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src import accuracy, capacity, commentary, data, forecast

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

HORIZON = 28
FOLDS = 8


def main() -> None:
    pd.set_option("display.width", 120)

    print("=" * 72)
    print("SUPPORT VOLUME FORECASTING & CAPACITY PLANNING")
    print("=" * 72)

    # 1. Data ---------------------------------------------------------------
    df = data.generate()
    y = df.volume
    print(f"\n[1] Data: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")
    print(f"    mean daily volume {y.mean():,.0f} | {df.is_incident.sum()} incident days")

    # 2. Model selection ----------------------------------------------------
    print(f"\n[2] Model comparison ({FOLDS} folds x {HORIZON}-day horizon, rolling origin)")
    ranking = forecast.compare_models(y, horizon=HORIZON, folds=FOLDS)
    print(ranking.to_string(index=False))
    best = ranking.iloc[0]["model"]

    # 3. Backtest the winner ------------------------------------------------
    bt = forecast.rolling_origin_backtest(y, best, horizon=HORIZON, folds=FOLDS)
    metrics = accuracy.summary(bt.frame)
    print(f"\n[3] Backtest accuracy — {best}")
    for k, v in metrics.items():
        print(f"    {k:>18}: {v}")

    # 4. Health & variance --------------------------------------------------
    health = accuracy.health_monitor(bt.frame)
    variance = accuracy.decompose_variance(bt.frame, df.is_incident)
    lead = accuracy.accuracy_by_lead_time(bt.frame)

    print("\n[4] Variance root-cause")
    print(variance.to_string(index=False))
    print("\n    Accuracy by lead-time week")
    print(lead.groupby("week").wape_pct.mean().round(2).to_string())

    # 5. Forward forecast & staffing ---------------------------------------
    fc = forecast.fit_and_forecast(y, best, horizon=HORIZON)
    aht_recent = float(df.aht_seconds.tail(28).mean())
    plan = capacity.staffing_plan(fc, aht_recent)

    print(f"\n[5] Capacity plan — next {HORIZON} days (AHT {aht_recent:.0f}s, "
          f"shrinkage 30%, SLA 80/20)")
    print(plan.head(10).to_string())
    print(f"    ...peak {plan.agents_rostered.max()} agents on "
          f"{plan.agents_rostered.idxmax().date()}")

    # Cost of forecast error, in agent-days
    tail = bt.frame[bt.frame.origin == bt.frame.origin.max()].set_index("date")
    cost = capacity.cost_of_error(tail.actual, tail.forecast, aht_recent)
    over = int(cost.loc[cost.agent_gap > 0, "agent_gap"].sum())
    under = int(-cost.loc[cost.agent_gap < 0, "agent_gap"].sum())
    print(f"\n    Cost of error, most recent fold: {over} agent-days over-staffed, "
          f"{under} under-staffed")

    # 6. Automated commentary ----------------------------------------------
    ev = commentary.build_evidence(metrics, health, variance, lead, plan, ranking)
    exceptions = commentary.raise_exceptions(ev)
    note, source = commentary.generate(ev, exceptions)

    print(f"\n[6] Automated review note (source: {source})")
    print("-" * 72)
    print(note)
    print("-" * 72)

    # 7. Artefacts ----------------------------------------------------------
    bt.frame.to_csv(OUT / "backtest.csv", index=False)
    variance.to_csv(OUT / "variance_decomposition.csv", index=False)
    plan.to_csv(OUT / "staffing_plan.csv")
    ranking.to_csv(OUT / "model_ranking.csv", index=False)
    (OUT / "review_note.md").write_text(note)

    _charts(y, bt.frame, health, fc, plan)
    print(f"\n[7] Artefacts written to {OUT}/")


def _charts(y, bt_frame, health, fc, plan) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))

    hist = y.tail(180)
    ax[0].plot(hist.index, hist.values, lw=1.1, color="#1B3B6F", label="actual")
    ax[0].plot(fc.index, fc.values, lw=1.6, color="#C1440E", ls="--", label="forecast")
    ax[0].set_title("Daily support volume — history and 28-day forecast")
    ax[0].legend(frameon=False)

    h = health.dropna(subset=["rolling_wape"])
    ax[1].plot(h.date, h.rolling_wape, lw=1.2, color="#1B3B6F", label="rolling 14d WAPE")
    ax[1].axhline(h.threshold.iloc[0], color="#C1440E", ls="--", lw=1,
                  label="control limit")
    ax[1].set_ylabel("%")
    ax[1].set_title("Forecast health monitor")
    ax[1].legend(frameon=False)

    ax[2].bar(plan.index, plan.agents_rostered, color="#1B3B6F", width=0.8)
    ax[2].set_title("Rostered agent requirement (Erlang C, 80/20 SLA, 30% shrinkage)")
    ax[2].set_ylabel("agents")

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT / "dashboard.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
