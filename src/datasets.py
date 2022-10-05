import pandas as pd


class Dataset(pd.DataFrame):
    train_size: int
    period: int


def _read_passengers_dataset() -> Dataset:
    df = pd.read_csv(
        "src/data/raw/air_passengers.csv", parse_dates=["Month"], index_col="Month"
    )
    s = df["Passengers"]
    setattr(s, "period", 12)
    setattr(s, "train_size", 110)
    return s


def _read_perfect_sine30() -> Dataset:
    df = pd.read_csv("src/data/raw/perfect_sine.csv", index_col="index")
    s = df["sine"]
    setattr(s, "period", 30)
    setattr(s, "train_size", 110)
    return s


def _read_noisy_sine30() -> Dataset:
    df = pd.read_csv("src/data/raw/noisy_sine.csv", index_col="index")
    s = df["sine"]
    setattr(s, "period", 30)
    setattr(s, "train_size", 110)
    return s


DATASET_FACTORY_LOOKUP = {
    "AIR_PASSENGERS": _read_passengers_dataset,
    "PERFECT_SINE30": _read_perfect_sine30,
    "NOISY_SINE30": _read_noisy_sine30,
}
