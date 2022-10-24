import logging
from functools import partial
from typing import Callable, List, Tuple, Type, Union
from uuid import uuid4

import mlflow
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from tg import get_data_path, get_root_path
from tg.metrics import generate_all_metrics
from tg.splitters import Splitter
from tg.models import ModelClassLookupCallback


class ModelInteractor:

    def __init__(self, model_name: str) -> None:

        self.model_name = model_name
        self.model_class = ModelClassLookupCallback(model_name)
        self._loaded = False

    def load(self,
             y: pd.Series,
             X: pd.DataFrame = None,
             dataset_name: str = None,
             timesteps: int = None,
             train_size: int = None) -> None:

        self.y = y
        self.X = X
        self.dataset_name = dataset_name
        self.timesteps = timesteps
        self.train_size = train_size
        self._loaded = True

    def split_trains_test(
        self,
        y: pd.Series,
        splitter_class: Type[Splitter],
        splitter_args: dict = {},
        X: pd.DataFrame = None
    ) -> Union[Tuple[List[pd.Series], pd.Series], Tuple[List[Tuple[
            pd.Series, pd.DataFrame]], Tuple[pd.Series, pd.DataFrame]]]:

        if not self._loaded:
            raise ValueError('You should load dataset first!')

        splitter = splitter_class(**splitter_args)
        slices = splitter.split(y)

        train_indexes_list = [s[0] for s in slices]
        test_indexes_list = [s[1] for s in slices]

        indexes = np.array([idx[0] for idx in test_indexes_list])

        y_trains = [y.iloc[idx] for idx in train_indexes_list]
        y_test = y.iloc[indexes]

        if not self.model_class.instance.single_input:
            X_trains = [
                X.iloc[np.append(np.array(idx),
                                 np.max(idx) + 1)]
                for idx in train_indexes_list
            ]
            trains = list(zip(y_trains, X_trains))
            X_test = X.iloc[np.append(np.array(indexes), np.max(indexes) + 1)]
            return trains, (y_test, X_test)
        return y_trains, y_test

    def fit_predict(
        self,
        trains: Union[List[pd.Series], List[Tuple[pd.Series, pd.DataFrame]]],
        test: Union[pd.Series, Tuple[pd.Series, pd.DataFrame]],
        parameters: dict,
    ) -> pd.Series:

        partial_predict = partial(self._fit_predict, parameters=parameters)

        preds = []
        for train in tqdm(trains, position=0, leave=True):
            if not self.model_class.instance.single_input:
                preds.append(partial_predict(y=train[0], X=train[1]))
            else:
                preds.append(partial_predict(y=train))

        if not self.model_class.instance.single_input:
            return pd.Series(preds, index=test[0].index, name=test[0].name)

        return pd.Series(preds, index=test.index, name=test.name)

    def evaluate(
            self, preds: pd.Series,
            test: Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]) -> None:

        if not self.model_class.instance.single_input:
            return generate_all_metrics(preds, test[0])
        return generate_all_metrics(preds, test)

    def execute_mlflow(self,
                       parameters: dict,
                       splitter_class: Type[Splitter],
                       splitter_args: dict = {}) -> dict:

        if not self._loaded:
            raise ValueError('You should load dataset first!')
        trains, test = self.split_trains_test(y=self.y,
                                              splitter_class=splitter_class,
                                              splitter_args=splitter_args,
                                              X=self.X)
        filepath = get_data_path(
            f"results/{self.dataset_name}_{self.model_name}_{uuid4().hex}.csv")

        mlflow.set_tracking_uri("file:///{}".format(get_root_path("mlruns")))
        with mlflow.start_run():

            preds = self.fit_predict(trains=trains,
                                     test=test,
                                     parameters=parameters)
            metrics = self.evaluate(preds=preds, test=test)

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_param("dataset_name", self.dataset_name)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("params", parameters)
            mlflow.log_param("n_train_points", self.train_size)
            preds.to_csv(filepath)
            mlflow.log_artifact(filepath)

        return metrics

    def tune_hyperparameters(self,
                             splitter_class: Type[Splitter],
                             splitter_args: dict = {},
                             n_trials: int = 5) -> dict:

        if not self._loaded:
            raise ValueError('You should load dataset first!')

        if not self.model_class.instance.tunable:
            logging.warning(
                f"Model {self.model_name} is not tunable. Returning default parameters."
            )
            return {}

        study_name = f"{self.dataset_name}/{self.model_name}/v1"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage="sqlite:///{}".format(get_root_path("optuna.db")),
            load_if_exists=True,
        )

        partial_objective = partial(
            self._objective,
            suggest_params=self.model_class.suggest_params,
            splitter_class=splitter_class,
            splitter_args=splitter_args,
        )
        study.optimize(partial_objective, n_trials=n_trials)
        return study.best_params

    def get_best_params(self) -> None:
        study_name = f"{self.dataset_name}/{self.model_name}/v1"
        study = optuna.create_study(study_name=study_name,
                                    direction="minimize",
                                    storage="sqlite:///{}".format(
                                        get_root_path("optuna.db")),
                                    load_if_exists=True)
        return study.best_params

    def _objective(self,
                   trial: optuna.Trial,
                   suggest_params: Callable[[optuna.Trial], dict],
                   splitter_class: Type[Splitter],
                   splitter_args: dict = {}) -> float:

        parameters = suggest_params(trial)
        trains, test = self.split_trains_test(y=self.y,
                                              splitter_class=splitter_class,
                                              splitter_args=splitter_args,
                                              X=self.X)
        preds = self.fit_predict(trains=trains,
                                 test=test,
                                 parameters=parameters)
        metrics = self.evaluate(preds=preds, test=test)

        trial.set_user_attr("metrics", metrics)
        return metrics['rmse']

    def _fit_predict(self,
                     parameters: dict,
                     y: pd.Series,
                     X: pd.DataFrame = None) -> float:
        model = self.model_class(**parameters)
        model.fit(y, X, self.timesteps)
        return model.predict_one_ahead()
