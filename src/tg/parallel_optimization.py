from multiprocessing import Pool

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tg.datasets import DatasetFactoryLookupCallback
from tg.interactors import ModelInteractor
from tg.splitters import AnchoredSplitter


def _mlflow_run(model_name: str):
    dataset_name = 'AIR_PASSENGERS'
    data_factory = DatasetFactoryLookupCallback(dataset_name=dataset_name)

    y, X = data_factory(model_name=model_name)

    mi = ModelInteractor(model_name=model_name)
    mi.load(y=y,
            X=X,
            dataset_name=data_factory.dataset_name,
            timesteps=data_factory.dataset.period,
            train_size=data_factory.dataset.train_size)

    best_params = mi.tune_hyperparameters(
        splitter_class=AnchoredSplitter,
        splitter_args={
            'min_train_points': data_factory.dataset.tuning_train_size
        },
        n_trials=50)

    mi.execute_mlflow(
        splitter_class=AnchoredSplitter,
        splitter_args={'min_train_points': data_factory.dataset.train_size},
        parameters=best_params)


def main():

    process_map(_mlflow_run, [
        'NAIVE',
        'ARIMA',
        'SARIMA',
        'RNN',
        'SVR',
        'ARIMA_RNN',
        'SARIMA_SVR',
    ],
                max_workers=8)


if __name__ == "__main__":
    main()
