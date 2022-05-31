import os
from typing import Callable

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import numpy as np
import pandas as pd
import pmdarima as pm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .models import PredictiveModel


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras import layers


class SARIMAModel:
    def __init__(self, m: int):
        self.m = m
        self.model = None

    def fit(self, y: pd.Series) -> None:
        self.model = pm.auto_arima(y, seasonal=True, m=self.m)

    def predict(self) -> float:
        return self.model.predict(n_periods=1)[0]


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

    def predict(self) -> float:
        return self.model.predict(self.x_input).reshape(-1)[0]


class NaiveModel:
    def __init__(self):
        self.last_value = None

    def fit(self, y: pd.Series) -> None:
        self.last_value = y.values[-1]

    def predict(self) -> float:
        return self.last_value


MODELS_FACTORY_MAP: dict[str, Callable[..., PredictiveModel]] = {
    "SARIMA": SARIMAModel,
    "RNN": RNNModel,
    "NAIVE": NaiveModel,
}
