from typing import Protocol, Type

import pandas as pd

DEFAULT_MIN_TRAIN_POINTS = 50
DEFAULT_AHEAD = 1
DEFAULT_STEP = 1


class Splitter(Protocol):
    def split(self, df: pd.DataFrame | pd.Series) -> list[tuple[pd.Index, pd.Index]]:
        ...


class AnchoredSplitter:
    def __init__(self, min_train_points: int = 50, ahead: int = 1, step: int = 10):
        self.min_train_points = min_train_points
        self.ahead = ahead
        self.step = step

    def split(self, df: pd.DataFrame | pd.Series) -> list[tuple[pd.Index, pd.Index]]:
        indices = df.index
        slices = []
        for i in range(self.min_train_points, len(indices), self.step):
            slices.append((indices[:i], indices[i : i + self.ahead]))

        return slices


class WindowedSplitter:
    def __init__(self, window: int = 1, ahead: int = 1, step: int = 1):
        self.window = window
        self.ahead = ahead
        self.step = step

    def split(self, df: pd.DataFrame | pd.Series) -> list[tuple[pd.Index, pd.Index]]:
        indices = df.index
        slices = []
        for i in range(self.window, len(indices), self.step):
            slices.append((indices[i - self.window : i], indices[i : i + self.ahead]))

        return slices


SPLITTERS_MAP: dict[str, Type[Splitter]] = dict(
    anchored=AnchoredSplitter,
    windowed=WindowedSplitter,
)
