from functools import partial
from typing import Callable, Type
from uuid import uuid4

import mlflow
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from tg import get_data_path, get_root_path
from tg.datasets import DatasetFactoryLookupCallback
from tg.metrics import generate_all_metrics
from tg.splitters import Splitter
from tg.ts_models import ModelClassLookupCallback


class ModelInteractor:

    def __init__(self, model_name: str) -> None:

        self.model_name = model_name
        self.model_class = ModelClassLookupCallback(model_name)

    def load_time_series(self, dataset_name: str):

        self.dataset_name = dataset_name
        self.dataset = DatasetFactoryLookupCallback(dataset_name)()

    def split_trains_test(self,
                          splitter: Splitter,
                          splitter_args: dict = {},
                          X: pd.DataFrame = pd.DataFrame(),
                          tuning=False) -> None:

        if not hasattr(self, 'dataset'):
            raise ValueError('You should load dataset first!')

        self.splitter = splitter(**splitter_args)

        slices = self.splitter.split(self.dataset)
        if tuning:
            slices = self.splitter.split(
                self.dataset[:self.dataset.train_size])
        

        train_indexes_list = [s[0] for s in slices]
        test_indexes_list = [s[1] for s in slices]

        indexes = np.array([idx[0] for idx in test_indexes_list])

        y_trains = [self.dataset.loc[idx] for idx in train_indexes_list]
        y_test = self.dataset.loc[indexes]

        if not X.empty:
            X_trains = [X.loc[idx] for idx in train_indexes_list]
            X_test = X.loc[indexes]
            self.trains, self.test = (y_trains, X_trains), (y_test, X_test)
        else:
            self.trains, self.test = y_trains, y_test

    def fit_predict(
        self, parameters: dict, X: pd.DataFrame = pd.DataFrame()) -> None:

        if not hasattr(self, 'trains'):
            raise ValueError('You should split dataset first!')

        partial_predict = partial(self._fit_predict, parameters)

        preds = []
        for train in tqdm(self.trains):
            if not X.empty:
                preds.append(partial_predict(y=train[0], X=train[1]))
            else:
                preds.append(partial_predict(y=train))
        self.preds = pd.Series(preds,
                               index=self.test.index,
                               name=self.test.name)

    def evaluate(self) -> None:

        if not hasattr(self, 'preds'):
            raise ValueError('You should fit_predict model first!')

        if isinstance(self.test, tuple):
            self.metrics = generate_all_metrics(
                self.preds, self.test[0])
        else:
            self.metrics = generate_all_metrics(self.preds, self.test)

    def execute_mlflow(
        self, parameters: dict, X: pd.DataFrame = pd.DataFrame()) -> None:

        if not hasattr(self, 'dataset'):
            raise ValueError('You should load dataset first!')

        filepath = get_data_path(f"results/{self.dataset_name}_{self.model_name}_{uuid4().hex}.csv")
        with mlflow.start_run():
            self.fit_predict(parameters, X)
            self.evaluate()

            for metric_name, value in self.metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.log_param("dataset_name", self.dataset_name)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("params", parameters)
            mlflow.log_param("n_train_points", self.dataset.train_size)

            self.preds.to_csv(filepath)
            mlflow.log_artifact(filepath)

    def tune_hyperparameters(self,
                             splitter: Splitter,
                             splitter_args: dict = {},
                             n_trials: int = 5) -> None:

        if not hasattr(self, 'dataset'):
            raise ValueError('You should load dataset first!')

        study_name = f"{self.dataset_name}/{self.model_name}/v2"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage="sqlite:///{}".format(get_root_path("optuna.db")),
            load_if_exists=False,
        )

        partial_objective = partial(
            self._objective,
            suggest_params=self.model_class.suggest_params,
            splitter=splitter,
            splitter_args=splitter_args,
        )
        study.optimize(partial_objective, n_trials=n_trials)
        self.hyperparams_best_value = study.best_value
        self.hyperparams_best_trial = study.best_trial

    def get_best_params(self) -> None:
        study_name = f"{self.dataset_name}/{self.model_name}/v2"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage="sqlite:///{}".format(get_root_path("optuna.db")),
            load_if_exists=False
        )
        self.hyperparams_best_value = study.best_value
        self.hyperparams_best_trial = study.best_trial

    def _objective(self,
                   trial: optuna.Trial,
                   suggest_params: Callable[[optuna.Trial], dict],
                   splitter: Type[Splitter],
                   splitter_args: dict = {}) -> float:

        parameters = suggest_params(trial)
        self.split_trains_test(splitter=splitter,
                               splitter_args=splitter_args,
                               tuning=True)
        self.fit_predict(parameters)
        self.evaluate()

        trial.set_user_attr("metrics", self.metrics)
        return self.metrics['mape']

    def _fit_predict(self,
                     parameters: dict,
                     y: pd.Series,
                     X: pd.DataFrame = pd.DataFrame()) -> float:
        model = self.model_class(**parameters)
        model.fit(y, X)
        print(parameters, model.predict_one_ahead())
        return model.predict_one_ahead()
