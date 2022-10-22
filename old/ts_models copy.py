import os
from typing import Dict, Literal, Type

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from functools import partial
from typing import Dict, Type

import keras
import numpy as np
import optuna
import pandas as pd
import pmdarima as pm
import tensorflow as tf
from statsmodels.tsa.tsatools import lagmat
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras import layers
from sklearn.svm import SVR
from tg.models import OneAheadModel


class SARIMAModel:

    def __init__(self, m: int):
        self.m = m
        self.model = None

    def fit(self, y: pd.Series) -> None:
        self.model = pm.auto_arima(y, seasonal=True, m=self.m)
        partial_predict = partial(_predict, model_class, parameters)
        preds = []
        for train in tqdm(trains, ):
            preds.append(partial_predict(train))

        return pd.Series(preds, index=test.index, name=test.name)

    def predict_one_ahead(self) -> float:
        return self.model.predict(n_periods=1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {}


class SARIMASVRModel:

    def __init__(
        self,
        svr_kernel: Literal["linear", "poly", "rbf", "sigmoid"],
        svr_C: float,
        m: int,
    ):
        self.arima_m = m
        self.svr_kernel = svr_kernel
        self.svr_C = svr_C

    def fit(self, y: pd.Series) -> None:
        y = y.values
        y_len = len(y)
        self.sarima = pm.auto_arima(y, seasonal=True, m=self.arima_m)
        self.svr = SVR(kernel=self.svr_kernel, C=self.svr_C)

        arima_y_pred = self.sarima.predict_in_sample()
        arima_errors = np.array(y - arima_y_pred)[1:]
        last_arima_errors = lagmat(arima_errors, self.arima_m, "both")
        lagged_y = lagmat(y[1:], self.arima_m, "both")
        svm_features = np.hstack([last_arima_errors, lagged_y])
        svm_y_train = y[-(y_len - self.arima_m) + 1:].reshape(-1)

        self.svr.fit(svm_features, svm_y_train)

        last_arima_errors = lagmat(arima_errors, self.arima_m - 1, "both",
                                   "in")
        lagged_y = lagmat(y, self.arima_m - 1, "both", "in")

        self.svr_input = np.hstack([last_arima_errors[-1],
                                    lagged_y[-1]]).reshape(1, -1)

    def predict_one_ahead(self) -> float:
        return self.svr.predict(self.svr_input)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {
            "svr_kernel":
            trial.suggest_categorical("svr_kernel",
                                      ("linear", "poly", "rbf", "sigmoid")),
            "svr_C":
            trial.suggest_loguniform("svr_C", low=1e-2, high=1e2),
        }


class ARIMAModel:

    def __init__(self, m: int = None):
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

    def __init__(self,
                 timesteps: int,
                 hidden_units: int,
                 epochs: int,
                 m: int = None):
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
            ))
        model.add(layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, y: pd.Series) -> None:
        y = y.values
        self.x_input = y[-self.timesteps:].reshape(1, self.timesteps)
        stacked = stack_lags(y, self.timesteps + 1)
        x, y = stacked[:, :-1], stacked[:, -1]

        self.model = self._create_model()
        self.model.fit(x, y, epochs=self.epochs, verbose=0)

    def predict_one_ahead(self) -> float:
        return self.model.predict(self.x_input).reshape(-1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        timesteps = trial.suggest_categorical("timesteps",
                                              [2, 4, 6, 12, 24, 36])
        hidden_units = trial.suggest_int("hidden_units", 5, 25, 5)
        epochs = trial.suggest_int("epochs", 300, 800, 100)

        return {
            "timesteps": timesteps,
            "hidden_units": hidden_units,
            "epochs": epochs
        }


class NaiveModel:

    def __init__(self, constant: float, m: int = None):
        self.constant = constant
        self.last_value = None

    def fit(self, y: pd.Series) -> None:
        self.last_value = y.values[-1]

    def predict_one_ahead(self) -> float:
        return self.last_value * (1 + self.constant)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {"constant": trial.suggest_uniform("constant", -1, 1)}


_MODEL_CLASS_LOOKUP: Dict[str, Type[OneAheadModel]] = {
    "SARIMA": SARIMAModel,
    "ARIMA": ARIMAModel,
    "RNN": RNNModel,
    "NAIVE": NaiveModel,
    "SARIMA_SVR": SARIMASVRModel,
}


class ModelClassLookupCallback:

    def __init__(self, model_name: str, m: int):
        if model_name not in _MODEL_CLASS_LOOKUP.keys():
            raise KeyError("Invalid model (or not implemented yet)")
        self.model_name = model_name
        self.m = m

    def __call__(self, **kwargs) -> OneAheadModel:
        return _MODEL_CLASS_LOOKUP[self.model_name](m=self.m, **kwargs)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return _MODEL_CLASS_LOOKUP[self.model_name].suggest_params(trial)
