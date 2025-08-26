# steps/data_splitter_step.py

from typing import Tuple, Optional
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step


@step
def data_splitter_step(
    df: pd.DataFrame,
    target_column: str,
    selected_features: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split and make sure X contains ONLY the selected features (if provided).
    """
    if selected_features:
        cols = [c for c in selected_features if c in df.columns] + [target_column]
        df = df.loc[:, cols]

    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test
