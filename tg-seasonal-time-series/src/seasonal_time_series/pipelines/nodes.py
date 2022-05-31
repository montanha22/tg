from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Protocol

import pandas as pd
from tqdm import tqdm

from seasonal_time_series.models import PredictiveModel
from seasonal_time_series.splitters import SPLITTERS_MAP


class Splitter(Protocol):
    def split(self, df: pd.DataFrame | pd.Series) -> list[tuple[pd.Index, pd.Index]]:
        ...


def split_trains_test(y: pd.DataFrame, splitter_type: str) -> dict[str, Any]:

    y = y["Passengers"]

    splitter = SPLITTERS_MAP.get(splitter_type)()

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
    model_factory: Callable[..., PredictiveModel],
) -> pd.Series:

    partial_predict = partial(_predict, model_factory)
    with Pool() as pool:
        preds = pool.map(partial_predict, tqdm(trains, total=len(trains)))
        return pd.Series(preds, index=test.index)


def _predict(
    model_factory: Callable[..., PredictiveModel],
    train: pd.Series,
):
    model = model_factory()
    model.fit(train)
    pred = model.predict()

    return pred
