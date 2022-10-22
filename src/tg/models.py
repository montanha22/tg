from abc import abstractmethod
from typing import Type

import optuna
import pandas as pd


class OneAheadModel:

    single_input = True
    min_fit_size = None
    tunable = False

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

    def __init__(self,
                 first_model_class: Type[OneAheadModel],
                 second_model_class: Type[OneAheadModel],
                 first_model_params: dict = {},
                 second_model_params: dict = {}) -> None:
        super().__init__()
        self.first_model = first_model_class(**first_model_params)
        self.second_model = second_model_class(**second_model_params)

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> None:

        if self.second_model.single_input:
            raise ValueError("Second model not compatible with hybrid model")
