from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Tuple, Type
from uuid import uuid4

import mlflow
import optuna
import pandas as pd
from tqdm import tqdm

from tg import metrics, models
from tg.datasets import DATASET_FACTORY_LOOKUP
from tg.splitters import AnchoredSplitter, Splitter
from tg.ts_models import ModelClassLookupCallback
from tg.interactors import ModelInteractor
from tg.splitters import AnchoredSplitter


def main():
    dataset_name = "AIR_PASSENGERS"

    # model_name = "SARIMA"
    # params = {"m": 12}

    model_name = "NAIVE"
    params = {"constant": 0.0}

    mi = ModelInteractor(model_name=model_name)
    mi.load(dataset_name=dataset_name)
    mi.tune_hyperparameters(
        splitter_class=AnchoredSplitter,
        splitter_args={'min_train_points': mi.y.tuning_train_size},
        n_trials=5,
    )
    # get_best_study_params()
    # train_and_test_model()


if __name__ == "__main__":
    main()
