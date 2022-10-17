from typing import Dict, List, Protocol, Tuple, Type, Union

import pandas as pd


class Splitter(Protocol):

    def split(
            self, df: Union[pd.DataFrame,
                            pd.Series]) -> List[Tuple[pd.Index, pd.Index]]:
        ...


class AnchoredSplitter:

    def __init__(self,
                 min_train_points: int = 50,
                 ahead: int = 1,
                 step: int = 1):
        self.min_train_points = min_train_points
        self.ahead = ahead
        self.step = step

    def split(
            self, df: Union[pd.DataFrame,
                            pd.Series]) -> List[Tuple[pd.Index, pd.Index]]:
        indices = df.index
        slices = []
        for i in range(self.min_train_points, len(indices), self.step):
            slices.append((indices[:i], indices[i:i + self.ahead]))

        return slices


class WindowedSplitter:

    def __init__(self, window: int = 1, ahead: int = 1, step: int = 1):
        self.window = window
        self.ahead = ahead
        self.step = step

    def split(
            self, df: Union[pd.DataFrame,
                            pd.Series]) -> List[Tuple[pd.Index, pd.Index]]:
        indices = df.index
        slices = []
        for i in range(self.window, len(indices), self.step):
            slices.append(
                (indices[i - self.window:i], indices[i:i + self.ahead]))

        return slices


SPLITTERS_MAP: Dict[str, Type[Splitter]] = dict(
    anchored=AnchoredSplitter,
    windowed=WindowedSplitter,
)
