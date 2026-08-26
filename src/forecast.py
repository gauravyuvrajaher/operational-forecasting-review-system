"""
Forecasting models and rolling-origin backtesting.

Three models, deliberately ordered by complexity so that the added value of
each is measurable rather than assumed:

    SeasonalNaive     - last same-weekday value. The baseline any model must beat.
    HoltWinters       - triple exponential smoothing, additive trend + weekly season.
    SARIMA            - (p,d,q)(P,D,Q,7) on log volume, with holiday regressors.

The holiday regressors are the difference between SARIMA winning and losing.
Without them the seasonal-naive baseline is competitive, because both models
are wrong on the same days and the naive one is free to run.

Outlier handling matters more than model choice here. Incident spikes are
non-repeating, so leaving them in the training window teaches the model a
seasonality that does not exist. `winsorise` caps them before fitting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import make_exog

warnings.filterwarnings("ignore")

SEASON = 7


def winsorise(y: pd.Series, window: int = 28, z: float = 3.0) -> pd.Series:
    """Cap incident spikes against a rolling median/MAD band.

    Uses median absolute deviation rather than standard deviation because the
    spikes themselves inflate the standard deviation and mask each other.
    """
    med = y.rolling(window, min_periods=7, center=True).median()
    mad = (y - med).abs().rolling(window, min_periods=7, center=True).median()
    scale = 1.4826 * mad.replace(0, np.nan)
    upper = med + z * scale
    return y.where(y <= upper, upper).fillna(y)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class SeasonalNaive:
    name = "SeasonalNaive"

    def fit(self, y: pd.Series):
        self._tail = y.iloc[-SEASON:].to_numpy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / SEASON))
        return np.tile(self._tail, reps)[:horizon]


class HoltWinters:
    name = "HoltWinters"

    def fit(self, y: pd.Series):
        self._model = ExponentialSmoothing(
            y.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=SEASON,
            initialization_method="estimated",
        ).fit(optimized=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._model.forecast(horizon))


class Sarima:
    name = "SARIMA"

    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, SEASON)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, y: pd.Series):
        # Log transform stabilises variance: absolute swings scale with level.
        self._last = y.index[-1]
        exog = make_exog(y.index)
        self._model = SARIMAX(
            np.log(y.astype(float).clip(lower=1)),
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future = pd.date_range(self._last + pd.Timedelta(days=1),
                               periods=horizon, freq="D")
        return np.exp(np.asarray(
            self._model.forecast(horizon, exog=make_exog(future))))


MODELS = {m.name: m for m in (SeasonalNaive(), HoltWinters(), Sarima())}


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    model: str
    horizon: int
    frame: pd.DataFrame = field(repr=False)

    @property
    def wape(self) -> float:
        e = (self.frame.actual - self.frame.forecast).abs().sum()
        return 100 * e / self.frame.actual.sum()


def rolling_origin_backtest(
    y: pd.Series,
    model_name: str = "SARIMA",
    horizon: int = 28,
    folds: int = 8,
    step: int = 28,
    min_train: int = 365,
    clean_outliers: bool = True,
) -> BacktestResult:
    """Walk the forecast origin forward through history.

    Each fold trains only on data available at that origin, so the evaluation
    reflects what the model would actually have produced in production. A
    single holdout would not — it hides how performance varies across the year.
    """
    rows = []
    end = len(y) - horizon
    origins = [end - i * step for i in range(folds)][::-1]
    origins = [o for o in origins if o >= min_train]

    for origin in origins:
        train = y.iloc[:origin]
        test = y.iloc[origin: origin + horizon]
        fit_series = winsorise(train) if clean_outliers else train

        model = MODELS[model_name].__class__() if model_name != "SARIMA" else Sarima()
        yhat = model.fit(fit_series).predict(horizon)

        rows.append(
            pd.DataFrame(
                {
                    "date": test.index,
                    "actual": test.to_numpy(),
                    "forecast": np.round(yhat, 1),
                    "origin": y.index[origin - 1],
                    "lead_days": np.arange(1, horizon + 1),
                }
            )
        )

    return BacktestResult(model_name, horizon, pd.concat(rows, ignore_index=True))


def compare_models(y: pd.Series, horizon: int = 28, folds: int = 8) -> pd.DataFrame:
    """Backtest every model on identical folds and rank by WAPE."""
    out = []
    for nm in MODELS:
        r = rolling_origin_backtest(y, nm, horizon=horizon, folds=folds)
        out.append({"model": nm, "wape_pct": round(r.wape, 2),
                    "folds": r.frame.origin.nunique()})
    return pd.DataFrame(out).sort_values("wape_pct").reset_index(drop=True)


def fit_and_forecast(y: pd.Series, model_name: str = "SARIMA",
                     horizon: int = 28) -> pd.Series:
    """Fit on all history and produce the forward forecast."""
    model = Sarima() if model_name == "SARIMA" else MODELS[model_name].__class__()
    yhat = model.fit(winsorise(y)).predict(horizon)
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.Series(np.round(yhat, 1), index=idx, name="forecast")
