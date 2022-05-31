from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Protocol
import numpy as np

import pandas as pd
from tqdm import tqdm

from seasonal_time_series.models import PredictiveModel
from seasonal_time_series.splitters import SPLITTERS_MAP
from seasonal_time_series.ts_models import MODELS_FACTORY_MAP


class Splitter(Protocol):
    def split(self, df: pd.DataFrame | pd.Series) -> list[tuple[pd.Index, pd.Index]]:
        ...


def split_trains_test(
    y: pd.DataFrame, splitter_type: str, splitter_kwargs: dict[str, Any]
) -> dict[str, Any]:

    y = y["Passengers"]

    splitter = SPLITTERS_MAP.get(splitter_type)(**splitter_kwargs)

    if not splitter:
        raise ValueError(f"Splitter {splitter_type} not found")

    slices = splitter.split(y)

    train_indexes_list = [s[0] for s in slices]
    test_indexes_list = [s[1] for s in slices]

    indexes = [idx[0] for idx in test_indexes_list]

    trains = [y.loc[idx] for idx in train_indexes_list]
    test = y.loc[indexes]

    return dict(trains=trains, test=test)


def fit_trains_and_predict_next(
    trains: list[pd.Series],
    test: pd.Series,
    model: str,
    model_kwargs: dict[str, Any],
) -> pd.Series:

    model_factory = MODELS_FACTORY_MAP.get(model)
    if not model_factory:
        raise ValueError(f"Model {model} not found")

    partial_predict = partial(_predict, model_factory, model_kwargs)
    with Pool() as pool:
        preds = pool.map(partial_predict, trains)
        return pd.Series(preds, index=test.index)


def _predict(
    model_factory: Callable[..., PredictiveModel],
    model_kwargs: dict[str, Any],
    train: pd.Series,
):
    model = model_factory(**(model_kwargs if model_kwargs else {}))
    model.fit(train)
    pred = model.predict()

    return pred


def _mse(y_preds: pd.Series, y_true: pd.Series) -> float:
    se = (y_preds - y_true) ** 2
    return se.mean()


def _rmse(y_preds: pd.Series, y_true: pd.Series) -> float:
    return np.sqrt(_mse(y_preds, y_true))


def _smape(y_preds: pd.Series, y_true: pd.Series) -> float:
    abs_diff = np.abs(y_preds - y_true)
    sum_of_abs = np.abs(y_preds) + np.abs(y_true)
    return 100 / len(y_preds) * np.sum(2 * abs_diff / (sum_of_abs))


def _mae(y_preds: pd.Series, y_true: pd.Series) -> float:
    return np.abs(y_preds - y_true).mean()


def _mape(y_preds: pd.Series, y_true: pd.Series) -> float:
    y_preds, y_true = np.array(y_preds), np.array(y_true)
    return np.mean(np.abs((y_preds - y_true) / y_preds)) * 100


def generate_metrics(y_preds: pd.Series, y_true: pd.Series) -> dict[str, Any]:
    metrics = dict(
        rmse=_rmse(y_preds, y_true),
        smape=_smape(y_preds, y_true),
        mape=_mape(y_preds, y_true),
        mae=_mae(y_preds, y_true),
    )
    metrics = {k: float(v) for k, v in metrics.items()}
    return metrics
