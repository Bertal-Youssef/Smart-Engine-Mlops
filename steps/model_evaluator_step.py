# steps/model_evaluator_step.py
import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step

from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy

@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluate the trained pipeline on test data and return metrics + RMSE (for convenience).
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    # Because 'trained_model' is a Pipeline(preprocessor -> model), just call predict:
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())
    metrics = evaluator.evaluate(trained_model, X_test, y_test)

    rmse = metrics.get("RMSE", None)
    return metrics, rmse
