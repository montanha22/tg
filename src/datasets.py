import pandas as pd


def _read_passengers_dataset() -> pd.Series:
    df = pd.read_csv(
        "src/data/raw/air_passengers.csv", parse_dates=["Month"], index_col="Month"
    )
    return df["Passengers"]


DATASET_FACTORY_LOOKUP = {
    "AIR_PASSENGERS": _read_passengers_dataset,
}

