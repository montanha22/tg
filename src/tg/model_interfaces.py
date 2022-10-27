from abc import abstractmethod
from typing import Literal, Type

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
    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps: int = None) -> None:
        ...

    @abstractmethod
    def predict_one_ahead(self) -> float:
        ...

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        ...


class HybridModel(OneAheadModel):

    def __init__(
            self,
            first_model_class: Type[OneAheadModel],
            second_model_class: Type[OneAheadModel],
            method: Literal['residue', 'decomposition'] = 'residue') -> None:
        super().__init__()
        self.first_model_class = first_model_class
        self.second_model_class = second_model_class
        self.method = method

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps: int = None) -> None:

        if self.second_model.single_input:
            raise ValueError("Second model not compatible with hybrid model")

        if self.method == 'residue':
            self.first_model.fit(y=y, X=X, timesteps=timesteps)
            first_model_residuals = self.first_model.predict_residuals()[1:]
            y_residuals = stack_lags(first_model_residuals, timesteps)
            y_lagged = stack_lags(y[1:], timesteps)
            X = pd.DataFrame(np.hstack([y_residuals, y_lagged]))
            y = y[timesteps + 1:]
            self.second_model.fit(y=y, X=X, timesteps=timesteps)

        elif self.method == 'decomposition':
            self.first_model.fit(y=y, X=X, timesteps=timesteps)
            seasonal_pred = self.first_model.predict_seasonal_one_ahead()
            trend_component = self.first_model.get_trend()
            residual_component = self.first_model.get_residuals()
            
            y = y - first_model_forecast
            self.second_model.fit(y=y, X=X, timesteps=timesteps)

    def predict_one_ahead(self) -> float:
        return self.second_model.predict_one_ahead()
