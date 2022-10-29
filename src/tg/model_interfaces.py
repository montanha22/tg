from abc import abstractmethod
from functools import partial
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
            timesteps: int = None,
            stack_size: int = None) -> None:
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
            timesteps: int = None,
            stack_size: int = None) -> None:

        if self.second_model.single_input:
            raise ValueError("Second model not compatible with hybrid model")

        if self.method == 'residue':

            self.first_model.fit(y=y,
                                 X=X,
                                 timesteps=timesteps,
                                 stack_size=stack_size)
            first_model_residuals = self.first_model.predict_residuals()[1:]
            y_residuals = stack_lags(first_model_residuals, stack_size)
            y_lagged = stack_lags(y[1:], stack_size)
            X = pd.DataFrame(np.hstack([y_residuals, y_lagged]))
            y = y[stack_size + 1:]
            self.second_model.fit(y=y,
                                  X=X,
                                  timesteps=timesteps,
                                  stack_size=stack_size)

        elif self.method == 'decomposition':

            self.first_model.fit(y=y,
                                 X=X,
                                 timesteps=timesteps,
                                 stack_size=stack_size)

            trend_component = self.first_model.get_trend()
            residual_component = self.first_model.get_residuals()

            X_trend = pd.DataFrame(
                stack_lags(x=trend_component, lags=stack_size))
            y_trend = trend_component[timesteps:]

            X_residual = pd.DataFrame(
                stack_lags(x=residual_component, lags=stack_size))
            y_residual = residual_component[timesteps:]

            self.trend_model.fit(y=y_trend,
                                 X=X_trend,
                                 timesteps=timesteps,
                                 stack_size=stack_size)
            self.residual_model.fit(y=y_residual,
                                    X=X_residual,
                                    timesteps=timesteps,
                                    stack_size=stack_size)

    def predict_one_ahead(self) -> float:
        if self.method == 'decomposition':
            seasonal_pred = self.first_model.predict_seasonal_one_ahead()
            trend_pred = self.trend_model.predict_one_ahead()
            residual_pred = self.residual_model.predict_one_ahead()
            return seasonal_pred + trend_pred + residual_pred
        return self.second_model.predict_one_ahead()
