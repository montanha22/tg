"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from seasonal_time_series.pipelines import air_passengers_pipeline as app


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    app_pipeline = app.create_pipeline()
    return {
        "__default__": app_pipeline,  # pipeline([]),
        "app": app_pipeline,
    }
