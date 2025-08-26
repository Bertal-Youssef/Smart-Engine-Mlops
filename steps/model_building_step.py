# steps/model_building_step.py
import logging
from typing import Annotated, Literal

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import ArtifactConfig, Model, step
from zenml.client import Client

from src.model_building import ModelBuilder, HistGBRStrategy, LinearRegressionStrategy

# Try to pick up an experiment tracker if one exists
try:
    _exp_tracker = Client().active_stack.experiment_tracker
    _exp_name = _exp_tracker.name if _exp_tracker else None
except Exception:
    _exp_name = None

model_meta = Model(
    name="engine_rul_predictor",
    version=None,
    license="Apache-2.0",
    description="RUL prediction model for turbofan engines (C-MAPSS).",
)

@step(enable_cache=False, experiment_tracker=_exp_name, model=model_meta)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm: Literal["hgb", "linreg"] = "hgb",
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Train a model pipeline for RUL.

    algorithm:
      - 'hgb'     -> HistGradientBoostingRegressor (recommended)
      - 'linreg'  -> LinearRegression (baseline)
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Build with selected strategy
    if algorithm == "hgb":
        strategy = HistGBRStrategy()
    else:
        strategy = LinearRegressionStrategy()

    builder = ModelBuilder(strategy=strategy)

    # Start an MLflow run if possible; no-op if no tracker is set
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog(log_models=True)
        logging.info(f"Training model with algorithm='{algorithm}'.")
        pipe = builder.build_model(X_train, y_train)
    finally:
        # End run if one was started here
        if mlflow.active_run():
            mlflow.end_run()

    return pipe
