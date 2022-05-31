"""
This is a boilerplate pipeline 'air_passengers_pipeline'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node
from . import nodes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                nodes.split_trains_test,
                inputs=["air_passengers", "params:splitter", "params:splitter_kwargs"],
                outputs=dict(trains="trains", test="test"),
                name="split_trains_test",
            ),
            node(
                nodes.fit_trains_and_predict_next,
                inputs=["trains", "test", "params:model", "params:model_kwargs"],
                outputs="preds",
                name="fit_trains_and_predict_next",
            ),
            node(
                nodes.generate_metrics,
                inputs=["preds", "test"],
                outputs=dict(rmse="rmse", smape="smape", mape="mape", mae="mae"),
                name="generate_metrics",
            ),
        ]
    )
