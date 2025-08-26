# steps/feature_engineering_step.py

from typing import Optional
import pandas as pd
from zenml import step

# Use your existing strategies
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)


@step
def feature_engineering_step(
    df: pd.DataFrame,
    strategy: str = "standard_scaling",
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply FE on the given features, then RETURN ONLY those features + 'RUL'.
    This ensures the model is trained with exactly the selected features.
    """
    # Keep semantics simple and explicit
    feat_list: list[str] = features or []

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(feat_list))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(feat_list))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(feat_list))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(feat_list))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed = engineer.apply_feature_engineering(df)

    # Keep only the selected features + target (if present)
    keep = [c for c in feat_list if c in transformed.columns]
    if "RUL" in transformed.columns:
        keep.append("RUL")

    return transformed.loc[:, keep]
