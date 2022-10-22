import os
from typing import Dict, Type

import keras
import optuna
import pandas as pd
import pmdarima as pm
import tensorflow as tf
from keras import layers

from tg.models import OneAheadModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NaiveModel(OneAheadModel):

    tunable = True

    def __init__(self, constant: float = 0.0):
        super().__init__()
        self.constant = constant

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> None:
        self.last_value = y.values[-1]
        self._is_fitted = True

    def predict_one_ahead(self) -> float:
        if not self._is_fitted or self.last_value is None:
            raise ValueError("Model not fitted yet.")
        return self.last_value * (1 + self.constant)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict:
        return {"constant": trial.suggest_uniform("constant", -1, 1)}


class ARIMAModel(OneAheadModel):

    min_fit_size = 3

    def __init__(self):
        super().__init__()

    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> OneAheadModel:
        self.model = pm.auto_arima(y, seasonal=False)
        self._is_fitted = True
        return self

    def predict_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.model.predict(n_periods=1)[0]

    def predict_residuals(self) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.model.predict_in_sample() - self.model.y

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {}


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

    single_input = False
    tunable = True

    def __init__(self, hidden_units: int, epochs: int):

        super().__init__()
        self.hidden_units = hidden_units
        self.epochs = epochs

    def _create_model(self, shape: int) -> keras.Sequential:
        model = keras.Sequential()
        model.add(
            layers.SimpleRNN(
                units=self.hidden_units,
                activation="relu",
                input_shape=(shape, 1),
            ))
        model.add(layers.Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, y: pd.Series, X: pd.DataFrame) -> None:
        self.one_ahead_input = X.iloc[-1].shift(-1).fillna(
            y[-1]).values.reshape(1, -1)
        self.model = self._create_model(shape=X.shape[1])
        self.model.fit(X.values, y.values, epochs=self.epochs, verbose=0)

    def predict_one_ahead(self) -> float:
        return self.model.predict(self.one_ahead_input).reshape(-1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        hidden_units = trial.suggest_int("hidden_units", 5, 25, 5)
        epochs = trial.suggest_int("epochs", 300, 800, 100)

        return {"hidden_units": hidden_units, "epochs": epochs}


_MODEL_CLASS_LOOKUP: Dict[str, Type[OneAheadModel]] = {
    "NAIVE": NaiveModel,
    "ARIMA": ARIMAModel,
    "SARIMA": SARIMAModel,
    "RNN": RNNModel
}


class ModelClassLookupCallback:

    def __init__(self, model_name: str) -> None:
        if model_name not in _MODEL_CLASS_LOOKUP.keys():
            raise KeyError("Invalid model (or not implemented yet)")
        self.model_name = model_name
        self.single_input = _MODEL_CLASS_LOOKUP[model_name].single_input
        self.min_fit_size = _MODEL_CLASS_LOOKUP[model_name].min_fit_size
        self.tunable = _MODEL_CLASS_LOOKUP[model_name].tunable

    def __call__(self, **kwargs) -> OneAheadModel:
        return _MODEL_CLASS_LOOKUP[self.model_name](**kwargs)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return _MODEL_CLASS_LOOKUP[self.model_name].suggest_params(trial)
