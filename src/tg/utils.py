import numpy as np


def stack_lags(x: np.ndarray, lags: int):
    return np.vstack([np.roll(x, -i) for i in range(lags + 1)]).T[:-lags]
