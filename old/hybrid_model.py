import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import lagmat

from src.models import OneAheadModel


class HybridModel(OneAheadModel):

    def __init__(self,
                 first_model: OneAheadModel,
                 second_model: OneAheadModel,
                 first_model_params: dict = {},
                 second_model_params: dict = {}) -> None:
        self.first_model = first_model(**first_model_params)
        self.second_model = second_model(**second_model_params)

    def fit(self, y: pd.Series, m: int, method: str = 'residue') -> None:

        y_len = len(y)

        if method == 'residue':
            self.first_model.fit(y)
            first_model_pred = self.first_model.predict_one_ahead()
            first_model_errors = np.array(y - first_model_pred)[1:]
            last_first_model_errors = lagmat(first_model_errors, m, "both")
            lagged_y = lagmat(y[1:], m, "both")
            second_model_features = np.hstack(
                [last_first_model_errors, lagged_y])
            second_model_y_train = y.values[-(y_len - m) + 1:].reshape(-1)
            self.second_model.fit(second_model_features, second_model_y_train)
            last_first_model_errors = lagmat(first_model_errors, m - 1, "both",
                                             "in")
            lagged_y = lagmat(y, m - 1, "both", "in")
            self.second_model_input = np.hstack(
                [last_first_model_errors[-1], lagged_y[-1]]).reshape(1, -1)

    def predict_one_ahead(self) -> float:
        return self.second_model.predict_one_ahead(self.second_model_input)
