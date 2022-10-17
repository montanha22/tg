from abc import abstractmethod

import optuna
import pandas as pd


class OneAheadModel:

    def __init__(self):
        self.model = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame = pd.DataFrame()) -> None:
        ...

    @abstractmethod
    def predict_one_ahead(self) -> float:
        ...

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        ...


class HybridModel(OneAheadModel):
    pass
