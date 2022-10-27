from typing import Dict, Literal, Type

import keras
import numpy as np
import optuna
import pandas as pd
import pmdarima as pm
from keras import layers
from skelm import ELMRegressor
from sklearn.svm import SVR
from sktime.forecasting.trend import STLForecaster

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


class ELMModel(OneAheadModel):

    single_input = False
    tunable = True

    def __init__(
        self,
        alpha: float,
        n_neurons: int,
        ufunc: Literal['tanh', 'sigm', 'relu', 'lin'] = 'tanh',
        include_original_features: bool = False,
        density: float = 1,
        pairwise_metric: Literal['euclidean', 'cityblock',
                                 'cosine'] = 'euclidean',
    ):
        super().__init__()
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.ufunc = ufunc
        self.include_original_features = include_original_features
        self.density = density
        self.pairwise_metric = pairwise_metric

    def fit(self, y: pd.Series, X: pd.DataFrame, timesteps=None) -> None:

        if len(X) - len(y) != 1:
            raise ValueError("X must be one timestep longer than y")

        self.one_ahead_input, X = X.iloc[-1].values.reshape(1, -1), X.iloc[:-1]
        self.model = ELMRegressor(
            alpha=self.alpha,
            n_neurons=self.n_neurons,
            ufunc=self.ufunc,
            include_original_features=self.include_original_features,
            density=self.density,
            pairwise_metric=self.pairwise_metric)
        self.model.fit(X, y)

    def predict_one_ahead(self) -> float:
        return self.model.predict(self.one_ahead_input).reshape(-1)[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        alpha = trial.suggest_loguniform("alpha", 1e-7, 1e7)
        n_neurons = trial.suggest_int("n_neurons", 5, 150, 5)
        ufunc = trial.suggest_categorical("ufunc",
                                          ['tanh', 'sigm', 'relu', 'lin'])
        include_original_features = trial.suggest_categorical(
            "include_original_features", [True, False])
        density = trial.suggest_uniform("density", 0.1, 1.0)
        pairwise_metric = trial.suggest_categorical(
            "pairwise_metric", ['euclidean', 'cityblock', 'cosine'])
        return {
            "alpha": alpha,
            "n_neurons": n_neurons,
            "ufunc": ufunc,
            "include_original_features": include_original_features,
            "density": density,
            "pairwise_metric": pairwise_metric
        }


class STLModel(OneAheadModel):

    tunable = True

    def __init__(self,
                 seasonal: int = 7,
                 seasonal_deg: int = 1,
                 trend_deg: int = 1,
                 low_pass_deg: int = 1,
                 seasonal_jump: int = 1,
                 trend_jump: int = 1,
                 low_pass_jump: int = 1,
                 robust: bool = False):
        super().__init__()
        self.seasonal = seasonal
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        self.robust = robust

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame = None,
            timesteps: int = None) -> None:

        if not timesteps:
            raise ValueError("timesteps must be provided")

        self.model = STLForecaster(sp=timesteps,
                                   seasonal=self.seasonal,
                                   seasonal_deg=self.seasonal_deg,
                                   trend_deg=self.trend_deg,
                                   low_pass_deg=self.low_pass_deg,
                                   seasonal_jump=self.seasonal_jump,
                                   trend_jump=self.trend_jump,
                                   low_pass_jump=self.low_pass_jump,
                                   robust=self.robust)
        self.model.fit(y.values)
        self._is_fitted = True

    def get_trend(self) -> pd.Series:
        return self.model.trend_

    def get_seasonal(self) -> pd.Series:
        return self.model.seasonal_

    def get_residuals(self) -> pd.Series:
        return self.model.resid_

    def predict_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
        return self.model.predict(fh=[1])[0]

    def predict_trend_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
        return self.model.forecaster_trend_.predict(fh=[1])[0]

    def predict_seasonal_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
        return self.model.forecaster_seasonal_.predict(fh=[1])[0]

    def predict_residuals_one_ahead(self) -> float:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
        return self.model.forecaster_resid_.predict(fh=[1])[0]

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        seasonal = trial.suggest_int("seasonal", 7, 99, 2)
        seasonal_deg = trial.suggest_int("seasonal_deg", 0, 1)
        trend_deg = trial.suggest_int("trend_deg", 0, 1)
        low_pass_deg = trial.suggest_int("low_pass_deg", 0, 1)
        seasonal_jump = trial.suggest_int("seasonal_jump", 1, 10)
        trend_jump = trial.suggest_int("trend_jump", 1, 10)
        low_pass_jump = trial.suggest_int("low_pass_jump", 1, 10)
        robust = trial.suggest_categorical("robust", [True, False])
        return {
            "seasonal": seasonal,
            "seasonal_deg": seasonal_deg,
            "trend_deg": trend_deg,
            "low_pass_deg": low_pass_deg,
            "seasonal_jump": seasonal_jump,
            "trend_jump": trend_jump,
            "low_pass_jump": low_pass_jump,
            "robust": robust
        }


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
    'ELM': ELMModel,
    'STL': STLModel,
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
