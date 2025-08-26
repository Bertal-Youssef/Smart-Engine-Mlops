# steps/rul_labeling_step.py

import pandas as pd
from zenml import step

@step
def rul_labeling_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the Remaining Useful Life (RUL) target column to the TRAIN dataframe.

    RUL = (max cycle for engine_id) - (current cycle)

    Assumes columns:
        - 'engine_id' : int
        - 'cycle'     : int
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required = {"engine_id", "cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for RUL labeling: {sorted(missing)}")

    # compute RUL per engine without leakage
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).astype("int32")

    return df
