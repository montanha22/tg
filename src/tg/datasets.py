from typing import Tuple

import pandas as pd

from tg import get_data_path
from tg.utils import stack_lags


class Dataset:
    train_size: int
    tuning_train_size: int
    period: int


def _read_passengers_dataset() -> Dataset:
    df = pd.read_csv(get_data_path("raw/air_passengers.csv"),
                     parse_dates=["Month"],
                     index_col="Month")
    s = df["Passengers"]
    setattr(s, "period", 12)
    setattr(s, "train_size", 110)
    setattr(s, "tuning_train_size", 94)
    setattr(s, "stack_size", 12)
    return s


def _read_perfect_sine30() -> Dataset:
    df = pd.read_csv(get_data_path("raw/perfect_sine.csv"), index_col="index")
    s = df["sine"]
    setattr(s, "period", 30)
    setattr(s, "train_size", 110)
    setattr(s, "tuning_train_size", 70)
    setattr(s, "stack_size", 30)
    return s


def _read_noisy_sine30() -> Dataset:
    df = pd.read_csv(get_data_path("raw/noisy_sine.csv"), index_col="index")
    s = df["sine"]
    setattr(s, "period", 30)
    setattr(s, "train_size", 110)
    setattr(s, "tuning_train_size", 70)
    setattr(s, "stack_size", 30)
    return s


def _read_homicides() -> Dataset:
    df = pd.read_csv(get_data_path("raw/homicides.csv"),
                     parse_dates=["Month"],
                     index_col="Month")
    s = df["Homicides"]
    setattr(s, "period", 12)
    setattr(s, "train_size", 110)
    setattr(s, "tuning_train_size", 94)
    setattr(s, "stack_size", 12)
    return s


def _read_random_walk() -> Dataset:
    df = pd.read_csv(get_data_path("raw/random_walk.csv"), index_col="index")
    s = df["random_walk"]
    setattr(s, "period", 1)
    setattr(s, "train_size", 110)
    setattr(s, "tuning_train_size", 94)
    setattr(s, "stack_size", 12)
    return s


DATASET_FACTORY_LOOKUP = {
    "AIR_PASSENGERS": _read_passengers_dataset,
    "PERFECT_SINE30": _read_perfect_sine30,
    "NOISY_SINE30": _read_noisy_sine30,
    "HOMICIDES": _read_homicides,
    "RANDOM_WALK": _read_random_walk,
}


def _get_default_input(dataset: Dataset) -> Tuple[pd.Series, None]:
    return dataset, None


def _get_lagged_input(dataset: Dataset) -> Tuple[pd.Series, pd.DataFrame]:
    timesteps = dataset.period
    stack_size = dataset.stack_size
    X = pd.DataFrame(stack_lags(dataset, stack_size))
    y = dataset[stack_size:]
    return y, X


INPUT_FACTORY_LOOKUP = {
    'NAIVE': _get_default_input,
    'ARIMA': _get_default_input,
    'SARIMA': _get_default_input,
    'RNN': _get_lagged_input,
    'SVR': _get_lagged_input,
    'ELM': _get_lagged_input,
    'STL': _get_default_input,
    'ES': _get_default_input,
    'LSTM': _get_lagged_input,
    'ARIMA_RNN': _get_default_input,
    'SARIMA_SVR': _get_default_input,
    'STL_ELM': _get_default_input,
    'ES_LSTM': _get_default_input,
    'ES_ELM': _get_default_input,
}


class DatasetFactoryLookupCallback:

    def __init__(self, dataset_name: str):
        if dataset_name not in DATASET_FACTORY_LOOKUP.keys():
            raise KeyError("Invalid dataset (or not implemented yet)")
        self.dataset_name = dataset_name
        self.dataset = DATASET_FACTORY_LOOKUP[dataset_name]

    def __call__(self,
                 model_name: str = None) -> Tuple[pd.Series, pd.DataFrame]:

        if not model_name:
            return self.dataset, None

        if model_name not in INPUT_FACTORY_LOOKUP.keys():
            raise KeyError("Invalid model (or not implemented yet)")

        return INPUT_FACTORY_LOOKUP[model_name](self.dataset())
