from typing import Protocol

import optuna
import pandas as pd


class OneAheadModel(Protocol):
    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> None:
        ...

    def predict_one_ahead(self) -> float:
        ...

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        ...
