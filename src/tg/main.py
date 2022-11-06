import functools
from typing import Literal
import warnings

from tqdm.contrib.concurrent import process_map

from tg.datasets import DatasetFactoryLookupCallback
from tg.model_interactor import ModelInteractor
from tg.splitters import AnchoredSplitter

warnings.filterwarnings("ignore")


def _main(dataset_name: str,
          models: list,
          fn: Literal['tune', 'mlflow'] = 'tune'):

    data_factory = DatasetFactoryLookupCallback(dataset_name=dataset_name)

    if fn == 'tune':
        tune_hyperparameters(models=models, data_factory=data_factory)

    elif fn == 'mlflow':
        execute_mlflow(models=models, data_factory=data_factory)


def _execute_mlflow(model_name: str,
                    data_factory: DatasetFactoryLookupCallback):

    y, X = data_factory(model_name=model_name)
    mi = ModelInteractor(model_name=model_name)
    mi.load(y=y,
            X=X,
            dataset_name=data_factory.dataset_name,
            timesteps=data_factory.dataset().period,
            train_size=data_factory.dataset().train_size,
            stack_size=data_factory.dataset().stack_size)
    mi.execute_mlflow(
        splitter_class=AnchoredSplitter,
        splitter_args={'min_train_points': data_factory.dataset().train_size},
        parameters=mi.get_best_params())


def execute_mlflow(models: list, data_factory: DatasetFactoryLookupCallback):

    process_map(functools.partial(_execute_mlflow, data_factory=data_factory),
                models)


def _tune_hyperparameters(model_name: str,
                          data_factory: DatasetFactoryLookupCallback):

    y, X = data_factory(model_name=model_name)
    mi = ModelInteractor(model_name=model_name)
    mi.load(y=y,
            X=X,
            dataset_name=data_factory.dataset_name,
            timesteps=data_factory.dataset().period,
            train_size=data_factory.dataset().train_size,
            stack_size=data_factory.dataset().stack_size)
    mi.tune_hyperparameters(splitter_class=AnchoredSplitter,
                            splitter_args={
                                'min_train_points':
                                data_factory.dataset().tuning_train_size
                            },
                            n_trials=100)


def tune_hyperparameters(models: list,
                         data_factory: DatasetFactoryLookupCallback):
    process_map(
        functools.partial(_tune_hyperparameters, data_factory=data_factory),
        models)


def main():

    ALL_DATASETS = [
        "AIR_PASSENGERS", "PERFECT_SINE30", "NOISY_SINE30", "HOMICIDES",
        'RANDOM_WALK'
    ]
    ALL_MODELS = [
        'NAIVE', 'ARIMA', 'SARIMA', 'RNN', 'SVR', 'ELM', 'STL', 'ES', 'LSTM',
        'ARIMA_RNN', 'SARIMA_SVR', 'STL_ELM', 'ES_LSTM', 'ES_ELM'
    ]
    SINGLE_MODELS = [
        'NAIVE', 'ARIMA', 'SARIMA', 'RNN', 'SVR', 'ELM', 'STL', 'ES', 'LSTM'
    ]
    HYBRID_MODELS = ['ARIMA_RNN', 'SARIMA_SVR', 'STL_ELM', 'ES_LSTM', 'ES_ELM']
    TUNNABLE_MODELS = [
        'NAIVE', 'RNN', 'SVR', 'ELM', 'STL', 'ES', 'LSTM', 'ARIMA_RNN',
        'SARIMA_SVR', 'STL_ELM', 'ES_LSTM', 'ES_ELM'
    ]
    TUNNABLE_SINGLE_MODELS = [
        'NAIVE', 'RNN', 'SVR', 'ELM', 'STL', 'ES', 'LSTM'
    ]

    process_map(functools.partial(_main, models=['ES_ELM'], fn='tune'),
                ALL_DATASETS)

    process_map(functools.partial(_main, models=['ES_ELM'], fn='mlflow'),
                ["RANDOM_WALK"])


if __name__ == "__main__":
    main()
