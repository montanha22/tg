import os
from typing import Dict, Type

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import numpy as np
import optuna
import pandas as pd
import pmdarima as pm
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras import layers

from src.models import OneAheadModel


class SARIMAModel:
    def __init__(self, m: int):
        self.m = m
        self.model = None

    def fit(self, y: pd.Series) -> None:
        self.model = pm.auto_arima(y, seasonal=True, m=self.m)

    def predict_one_ahead(self) -> float:
        return self.model.predict(n_periods=1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {"m": trial.suggest_categorical("m", [1, 4, 6, 12])}


class ARIMAModel:
    def __init__(self):
        self.model = None

    def fit(self, y: pd.Series) -> None:
        self.model = pm.auto_arima(y, seasonal=False)

    def predict_one_ahead(self) -> float:
        return self.model.predict(n_periods=1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {}


def stack_lags(x: np.ndarray, lags: int):
    return np.vstack([np.roll(x, -i) for i in range(lags)]).T[:-lags]


class RNNModel:
    def __init__(self, timesteps: int, hidden_units: int, epochs: int):
        self.model = None
        self.timesteps = timesteps
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.x_input = None

    def _create_model(self) -> keras.Sequential:
        model = keras.Sequential()
        model.add(
            layers.SimpleRNN(
                units=self.hidden_units,
                activation="relu",
                input_shape=(self.timesteps, 1),
            )
        )
        model.add(layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, y: pd.Series) -> None:
        y = np.array(y.values.astype("float32"))
        self.x_input = y[-self.timesteps :].reshape(1, self.timesteps)
        stacked = stack_lags(y, self.timesteps + 1)
        x, y = stacked[:, :-1], stacked[:, -1]

        self.model = self._create_model()
        self.model.fit(x, y, epochs=self.epochs, verbose=0)

    def predict_one_ahead(self) -> float:
        return self.model.predict(self.x_input).reshape(-1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        timesteps = trial.suggest_categorical("timesteps", [2, 4, 6, 12, 24, 36])
        hidden_units = trial.suggest_int("hidden_units", 5, 25, 5)
        epochs = trial.suggest_int("epochs", 300, 800, 100)

        return {"timesteps": timesteps, "hidden_units": hidden_units, "epochs": epochs}


class NaiveModel:
    def __init__(self, constant: float):
        self.constant = constant
        self.last_value = None

    def fit(self, y: pd.Series) -> None:
        self.last_value = y.values[-1]

    def predict_one_ahead(self) -> float:
        return self.last_value * (1 + self.constant)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {"constant": trial.suggest_uniform("constant", -1, 1)}


MODEL_CLASS_LOOKUP: Dict[str, Type[OneAheadModel]] = {
    "SARIMA": SARIMAModel,
    "ARIMA": ARIMAModel,
    "RNN": RNNModel,
    "NAIVE": NaiveModel,
}
