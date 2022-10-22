import sys

from tg.interactors import DataInteractor, ModelInteractor
from tg.splitters import AnchoredSplitter


def main():

    dataset_name = "AIR_PASSENGERS"
    model_name = "ARIMA"

    di = DataInteractor(dataset_name=dataset_name)
    y, X = di.get_data(model_name=model_name)

    mi = ModelInteractor(model_name=model_name)
    mi.load(dataset_name=dataset_name, y=y, X=X)
    params = mi.tune_hyperparameters(
        splitter_class=AnchoredSplitter,
        splitter_args={'min_train_points': di.y.tuning_train_size},
        n_trials=5)
    mi.execute_mlflow(parameters=params,
                      splitter_class=AnchoredSplitter,
                      splitter_args={'min_train_points': di.y.train_size})


if __name__ == "__main__":
    main()
