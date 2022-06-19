from typing import Dict

import numpy as np
import pandas as pd


def mse(y_preds: pd.Series, y_true: pd.Series) -> float:
    se = (y_preds - y_true) ** 2
    return se.mean()


def rmse(y_preds: pd.Series, y_true: pd.Series) -> float:
    return np.sqrt(mse(y_preds, y_true))


def smape(y_preds: pd.Series, y_true: pd.Series) -> float:
    abs_diff = np.abs(y_preds - y_true)
    sum_of_abs = np.abs(y_preds) + np.abs(y_true)
    return 100 / len(y_preds) * np.sum(abs_diff / sum_of_abs)


def mae(y_preds: pd.Series, y_true: pd.Series) -> float:
    return np.abs(y_preds - y_true).mean()


def mape(y_preds: pd.Series, y_true: pd.Series) -> float:
    y_preds, y_true = np.array(y_preds), np.array(y_true)
    return np.mean(np.abs((y_preds - y_true) / y_true)) * 100


def generate_all_metrics(y_preds: pd.Series, y_true: pd.Series) -> Dict[str, float]:
    metrics = dict(
        rmse=rmse(y_preds, y_true),
        smape=smape(y_preds, y_true),
        mape=mape(y_preds, y_true),
        mae=mae(y_preds, y_true),
    )
    metrics = {k: float(v) for k, v in metrics.items()}
    return metrics
