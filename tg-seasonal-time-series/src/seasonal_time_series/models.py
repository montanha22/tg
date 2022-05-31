from typing import Protocol

import pandas as pd


class PredictiveModel(Protocol):
    def fit(self, y: pd.Series) -> None:
        ...

    def predict(self) -> pd.Series:
        ...
