from typing import Dict, Literal, Type

import keras
import numpy as np
import optuna
import pandas as pd
import pmdarima as pm
from keras import layers
from sklearn.svm import SVR

from tg.model_interfaces import HybridModel, OneAheadModel


class NaiveModel(OneAheadModel):

    tunable = True

    def __init__(self, constant: float = 0.0):
        super().__init__()
        self.constant = constant

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps=None) -> None:
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

    def __init__(self):
        super().__init__()

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps=None) -> OneAheadModel:
        self.model = pm.auto_arima(y, seasonal=False)
        self.y = y
        self._is_fitted = True
        return self

    def predict_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        return np.array(self.model.predict(n_periods=1))[0]

    def predict_residuals(self) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        return np.array(self.model.predict_in_sample() - self.y.values)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, int]:
        return {}


class SARIMAModel(OneAheadModel):

    def __init__(self):
        super().__init__()

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps=None) -> OneAheadModel:
        self.model = pm.auto_arima(y.values, seasonal=True, m=timesteps)
        self.y = y
        self._is_fitted = True
        return self

    def predict_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        return np.array(self.model.predict(n_periods=1))[0]

    def predict_residuals(self) -> pd.Series:
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        return np.array(self.model.predict_in_sample() - self.y.values)

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

    def fit(self, y: pd.Series, X: pd.DataFrame, timesteps=None) -> None:

        if len(X) - len(y) != 1:
            raise ValueError("X must be one timestep longer than y")

        self.one_ahead_input, X = X.iloc[-1].values.reshape(1, -1), X.iloc[:-1]
        self.model = self._create_model(shape=X.shape[1])
        self.model.fit(X.values, y.values, epochs=self.epochs, verbose=0)

    def predict_one_ahead(self) -> float:
        return self.model.predict(self.one_ahead_input).reshape(-1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        hidden_units = trial.suggest_int("hidden_units", 5, 25, 5)
        epochs = trial.suggest_int("epochs", 300, 800, 100)
        return {"hidden_units": hidden_units, "epochs": epochs}


class SVRModel(OneAheadModel):

    single_input = False
    tunable = True

    def __init__(self, C: float, epsilon: float,
                 kernel: Literal["linear", "poly", "rbf", "sigmoid"]):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel

    def fit(self, y: pd.Series, X: pd.DataFrame, timesteps=None) -> None:

        if len(X) - len(y) != 1:
            raise ValueError("X must be one timestep longer than y")

        self.one_ahead_input, X = X.iloc[-1].values.reshape(1, -1), X.iloc[:-1]
        self.model = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        self.model.fit(X, y)

    def predict_one_ahead(self) -> float:
        return self.model.predict(self.one_ahead_input).reshape(-1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        C = trial.suggest_loguniform("C", 1e-3, 1e3)
        epsilon = trial.suggest_loguniform("epsilon", 1e-3, 1e3)
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"])
        return {"C": C, "epsilon": epsilon, "kernel": kernel}


class ARIMARNNModel(HybridModel):

    tunable = True

    def __init__(self, hidden_units: int, epochs: int) -> None:
        super().__init__(ARIMAModel, RNNModel, method='residue')
        self.first_model = ARIMAModel()
        self.second_model = RNNModel(hidden_units=hidden_units, epochs=epochs)

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps: int = None) -> None:
        super().fit(y=y, X=X, timesteps=timesteps)

    def predict_one_ahead(self) -> float:
        return super().predict_one_ahead()

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            **ARIMAModel.suggest_params(trial),
            **RNNModel.suggest_params(trial)
        }


class SARIMASVRModel(HybridModel):

    tunable = True

    def __init__(self, C: float, epsilon: float,
                 kernel: Literal["linear", "poly", "rbf", "sigmoid"]) -> None:
        super().__init__(SARIMAModel, SVRModel, method='residue')
        self.first_model = SARIMAModel()
        self.second_model = SVRModel(C=C, epsilon=epsilon, kernel=kernel)

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps: int = None) -> None:
        super().fit(y=y, X=X, timesteps=timesteps)

    def predict_one_ahead(self) -> float:
        return super().predict_one_ahead()

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            **SARIMAModel.suggest_params(trial),
            **SVRModel.suggest_params(trial)
        }


_MODEL_CLASS_LOOKUP: Dict[str, Type[OneAheadModel]] = {
    'NAIVE': NaiveModel,
    'ARIMA': ARIMAModel,
    'SARIMA': SARIMAModel,
    'RNN': RNNModel,
    'SVR': SVRModel,
    'ARIMA_RNN': ARIMARNNModel,
    'SARIMA_SVR': SARIMASVRModel,
}


class ModelClassLookupCallback:

    def __init__(self, model_name: str) -> None:

        if model_name not in _MODEL_CLASS_LOOKUP.keys():
            raise KeyError("Invalid model (or not implemented yet)")

        self.model_name = model_name
        self.instance = _MODEL_CLASS_LOOKUP[model_name]

    def __call__(self, **kwargs) -> OneAheadModel:
        return self.instance(**kwargs)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return self.instance.suggest_params(trial)
