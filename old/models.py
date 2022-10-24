from abc import abstractmethod
from typing import Type

import numpy as np
import optuna
import pandas as pd

from tg.utils import stack_lags


class OneAheadModel:

    single_input = True
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

        first_model_residuals = self.first_model.predict_residuals()[1:]
        y_residuals = stack_lags(first_model_residuals, 12)
        y_lagged = stack_lags(y[1:], 12)
        X = pd.DataFrame(np.hstack([y_residuals, y_lagged]))
