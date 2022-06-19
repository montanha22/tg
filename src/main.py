from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Tuple, Type
from uuid import uuid4

import mlflow
import optuna
import pandas as pd

from src import metrics, models
from src.datasets import DATASET_FACTORY_LOOKUP
from src.splitters import AnchoredSplitter, Splitter
from src.ts_models import MODEL_CLASS_LOOKUP


def objective(
    trial: optuna.Trial,
    model_class: Type[models.OneAheadModel],
    suggest_params: Callable[[optuna.Trial], dict],
    train: pd.Series,
) -> float:

    parameters = suggest_params(trial)

    trains, test = split_trains_test(train, AnchoredSplitter(min_train_points=40))
    preds = fit_trains_and_predict_next(trains, test, model_class, parameters)

    trial.set_user_attr("metrics", metrics.generate_all_metrics(preds, test))

    return metrics.rmse(preds, test)


def split_trains_test(
    y: pd.Series,
    splitter: Splitter,
) -> Tuple[List[pd.Series], pd.Series]:

    slices = splitter.split(y)

    train_indexes_list = [s[0] for s in slices]
    test_indexes_list = [s[1] for s in slices]

    indexes = [idx[0] for idx in test_indexes_list]

    trains = [y.loc[idx] for idx in train_indexes_list]
    test = y.loc[indexes]

    return trains, test


def _predict(
    model_class: Type[models.OneAheadModel],
    parameters: dict,
    train: pd.Series,
) -> float:
    model = model_class(**parameters)
    model.fit(train)
    return model.predict_one_ahead()


def fit_trains_and_predict_next(
    trains: List[pd.Series],
    test: pd.Series,
    model_class: Type[models.OneAheadModel],
    parameters: dict,
    threads: bool = False,
) -> pd.Series:

    partial_predict = partial(_predict, model_class, parameters)

    preds = None

    if threads:
        with Pool() as pool:
            preds = pool.map(partial_predict, trains)
    else:
        preds = [partial_predict(train) for train in trains]

    return pd.Series(preds, index=test.index, name=test.name)


def tune_hyperparameters(
    study_name: str,
    model_class: Type[models.OneAheadModel],
    train: pd.Series,
    n_trials: int = 5,
) -> optuna.trial.Trial:

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )

    partial_objective = partial(
        objective,
        model_class=model_class,
        suggest_params=model_class.suggest_params,
        train=train,
    )
    study.optimize(partial_objective, n_trials=n_trials)

    return study.best_value, study.best_trial


def train_and_test_model():
    dataset_name = "AIR_PASSENGERS"
    # model_name = "RNN"
    # params = {"epochs": 700, "hidden_units": 25, "timesteps": 12}
    # model_name = "SARIMA"
    # params = {"m": 12}
    # model_name = "NAIVE"
    # params = {"constant": 0}
    model_name = "ARIMA"
    params = {}

    n_train_points = 113

    filepath = f"src/data/results/{dataset_name}_{model_name}_{uuid4().hex}.csv"

    with mlflow.start_run():
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("params", params)
        mlflow.log_param("n_train_points", n_train_points)

        dataset = DATASET_FACTORY_LOOKUP[dataset_name]()
        train = dataset

        model_class = MODEL_CLASS_LOOKUP[model_name]

        trains, test = split_trains_test(
            train, AnchoredSplitter(min_train_points=n_train_points)
        )
        preds = fit_trains_and_predict_next(trains, test, model_class, params)

        results = metrics.generate_all_metrics(preds, test)

        for metric_name, value in results.items():
            mlflow.log_metric(metric_name, value)

        preds.to_csv(filepath)
        mlflow.log_artifact(filepath)

    return metrics.rmse(preds, test)


def get_best_study_params() -> dict:
    dataset_name = "AIR_PASSENGERS"
    model_name = "RNN"

    study_name = f"{dataset_name}/{model_name}/v2"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    print(study.best_params)
    return study.best_params


def tune_hyperparameters_with_optuna():
    dataset_name = "AIR_PASSENGERS"
    model_name = "RNN"
    train_pct = 0.8

    study_name = f"{dataset_name}/{model_name}/v2"

    dataset = DATASET_FACTORY_LOOKUP[dataset_name]()
    train = dataset.iloc[: int(len(dataset) * train_pct)]

    best_value, best_trial = tune_hyperparameters(
        study_name=study_name,
        model_class=MODEL_CLASS_LOOKUP[model_name],
        train=train,
        n_trials=100,
    )

    print(best_value, best_trial.params)


def main():
    # tune_hyperparameters_with_optuna()
    # get_best_study_params()
    train_and_test_model()


if __name__ == "__main__":
    main()
