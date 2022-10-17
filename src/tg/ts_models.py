import os
from typing import Dict, Type

import optuna
import pandas as pd
import pmdarima as pm
import tensorflow as tf
import keras
from keras import layers
from tg.models import OneAheadModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NaiveModel(OneAheadModel):

    def __init__(self, constant: float = 0.0):
        super().__init__()
        self.constant = constant

    def fit(self, y: pd.Series, X: pd.DataFrame = pd.DataFrame()) -> None:
        self.last_value = y.values[-1]
        self._is_fitted = True

    def predict_one_ahead(self) -> float:
        if not self._is_fitted or self.last_value is None:
            raise ValueError("Model not fitted yet.")
        return self.last_value * (1 + self.constant)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict:
        return {"constant": trial.suggest_uniform("constant", -1, 1)}


class SARIMAModel(OneAheadModel):

    def __init__(self, m: int):
        super().__init__()
        self.m = m

    def fit(self, y: pd.Series, X: pd.DataFrame = pd.DataFrame()) -> None:
        if not X.empty:
            raise ValueError("SARIMA does not support exogenous variables")
        self.model = pm.auto_arima(y.values, seasonal=True, m=self.m)
        self._is_fitted = True

    def predict_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(n_periods=1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {}


class RNNModel(OneAheadModel):

    def __init__(self, n_layers: int, n_units: int, lr: float):
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.lr = lr
        self.model = None

    def fit(self, y: pd.Series, X: pd.DataFrame = pd.DataFrame()) -> None:
        if not X.empty:
            raise ValueError("RNN does not support exogenous variables")
        self.model = keras.Sequential()
        for _ in range(self.n_layers):
            self.model.add(layers.LSTM(self.n_units, return_sequences=True))
        self.model.add(layers.LSTM(self.n_units))
        self.model.add(layers.Dense(1))
        self.model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
        )
        self.model.fit(
            y.values.reshape(-1, 1, 1),
            y.values.reshape(-1, 1),
            epochs=100,
            verbose=0,
        )
        self._is_fitted = True

    def predict_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(y.values.reshape(-1, 1, 1))[-1][0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "n_units": trial.suggest_int("n_units", 1, 10),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-1),
        }


_MODEL_CLASS_LOOKUP: Dict[str, Type[OneAheadModel]] = {
    "SARIMA": SARIMAModel,
    "NAIVE": NaiveModel,
    "RNN": RNNModel
}


class ModelClassLookupCallback:

    def __init__(self, model_name: str):
        if model_name not in _MODEL_CLASS_LOOKUP.keys():
            raise KeyError("Invalid model (or not implemented yet)")
        self.model_name = model_name

    def __call__(self, **kwargs) -> OneAheadModel:
        return _MODEL_CLASS_LOOKUP[self.model_name](**kwargs)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return _MODEL_CLASS_LOOKUP[self.model_name].suggest_params(trial)
