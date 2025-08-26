# src/model_building.py
import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """Build and train a model and return a fitted sklearn estimator/pipeline."""
        raise NotImplementedError


class HistGBRStrategy(ModelBuildingStrategy):
    """Strong default for tabular RUL: handles non-linearities and interactions well."""
    def __init__(
        self,
        learning_rate: float = 0.06,
        max_depth: int | None = 7,
        max_iter: int = 600,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        early_stopping: str | bool = "auto",
        validation_fraction: float = 0.1,
    ):
        self.params = dict(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            l2_regularization=l2_regularization,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
        )

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing HistGradientBoostingRegressor with identity preprocessor.")
        # Identity preprocessor with a .transform method so downstream code works uniformly
        preprocessor = FunctionTransformer(lambda X: X, validate=False)
        model = HistGradientBoostingRegressor(**self.params)

        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        logging.info("Training HistGradientBoostingRegressor.")
        pipe.fit(X_train, y_train)
        logging.info("Model training completed.")
        return pipe


class LinearRegressionStrategy(ModelBuildingStrategy):
    """Simple baseline; weaker than HGBR for non-linear sensor data, but useful for sanity checks."""
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        preprocessor = FunctionTransformer(lambda X: X, validate=False)
        model = LinearRegression()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        return pipe


class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)
