import pandas as pd
from tg import get_data_path

class Dataset:
    train_size: int
    period: int


def _read_passengers_dataset() -> pd.Series:
    df = pd.read_csv(get_data_path("raw/air_passengers.csv"),
                     parse_dates=["Month"],
                     index_col="Month")
    s = df["Passengers"]
    setattr(s, "period", 12)
    setattr(s, "train_size", 110)
    return s


def _read_perfect_sine30() -> pd.Series:
    df = pd.read_csv(get_data_path("raw/perfect_sine.csv"), index_col="index")
    s = df["sine"]
    setattr(s, "period", 30)
    setattr(s, "train_size", 110)
    return s


def _read_noisy_sine30() -> pd.Series:
    df = pd.read_csv(get_data_path("raw/noisy_sine.csv"), index_col="index")
    s = df["sine"]
    setattr(s, "period", 30)
    setattr(s, "train_size", 110)
    return s


DATASET_FACTORY_LOOKUP = {
    "AIR_PASSENGERS": _read_passengers_dataset,
    "PERFECT_SINE30": _read_perfect_sine30,
    "NOISY_SINE30": _read_noisy_sine30,
}


class DatasetFactoryLookupCallback:

    def __init__(self, dataset_name: str):
        if dataset_name not in DATASET_FACTORY_LOOKUP.keys():
            raise KeyError("Invalid dataset (or not implemented yet)")
        self.dataset_name = dataset_name

    def __call__(self):
        return DATASET_FACTORY_LOOKUP[self.dataset_name]()
