from tqdm.contrib.concurrent import process_map

from tg.datasets import DatasetFactoryLookupCallback
from tg.model_interactor import ModelInteractor
from tg.splitters import AnchoredSplitter

DATASET_NAME = 'NOISY_SINE30'
DATA_FACTORY = DatasetFactoryLookupCallback(dataset_name=DATASET_NAME)


def execute_mlflow(models: list):

    for model_name in models:

        y, X = DATA_FACTORY(model_name=model_name)

        mi = ModelInteractor(model_name=model_name)
        mi.load(y=y,
                X=X,
                dataset_name=DATA_FACTORY.dataset_name,
                timesteps=DATA_FACTORY.dataset.period,
                train_size=DATA_FACTORY.dataset.train_size)

        best_params = mi.get_best_params()

        mi.execute_mlflow(splitter_class=AnchoredSplitter,
                          splitter_args={
                              'min_train_points':
                              DATA_FACTORY.dataset.train_size
                          },
                          parameters=best_params)


def _tune(model_name: str):
    y, X = DATA_FACTORY(model_name=model_name)
    mi = ModelInteractor(model_name=model_name)
    mi.load(y=y,
            X=X,
            dataset_name=DATA_FACTORY.dataset_name,
            timesteps=DATA_FACTORY.dataset.period,
            train_size=DATA_FACTORY.dataset.train_size)
    mi.tune_hyperparameters(splitter_class=AnchoredSplitter,
                            splitter_args={
                                'min_train_points':
                                DATA_FACTORY.dataset.tuning_train_size
                            },
                            n_trials=1)


def parallel_hyperparameter_tuning(models: list):

    process_map(_tune, models, max_workers=8)


def main():

    models = [
        'NAIVE', 'ARIMA', 'SARIMA', 'RNN', 'SVR', 'ARIMA_RNN', 'SARIMA_SVR'
    ]

    parallel_hyperparameter_tuning(models=models)
    # execute_mlflow(models=models)


if __name__ == "__main__":
    main()
