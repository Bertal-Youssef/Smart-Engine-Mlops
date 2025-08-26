# steps/data_ingestion_step.py

from pathlib import Path
import pandas as pd
from zenml import step
from src.ingest_data import DataIngestorFactory

@step
def data_ingestion_step(
    file_path: str,
    subset: str = "FD001",    # required for C-MAPSS
    part: str = "train",      # "train" | "test" | "rul"
) -> pd.DataFrame:
    """
    Ingest C-MAPSS data and return ONE DataFrame (train/test/rul) to match
    your house-price pattern where downstream steps expect a single DataFrame.
    """
    # Factory can take str or Path; it checks the path type to return ZIP/Dir ingestor
    ingestor = DataIngestorFactory.get_data_ingestor(file_path)

    # IMPORTANT: pass a Path to ingest() so .parent works
    ds = ingestor.ingest(Path(file_path), subset=subset, verbose=True)

    part = part.lower()
    if part == "train":
        return ds.train
    if part == "test":
        return ds.test
    if part == "rul":
        return ds.rul_truth

    raise ValueError("`part` must be one of: 'train', 'test', 'rul'")
