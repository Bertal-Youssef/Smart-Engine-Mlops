# src/ingest_data.py
"""
Ingestion for NASA Turbofan Engine Degradation (C-MAPSS) dataset
using the Factory design pattern.

This module provides:
  - DataIngestor (abstract)
  - CmapssZipDataIngestor (reads a .zip that contains train/test/RUL txt files)
  - CmapssDirectoryDataIngestor (reads the same files from an extracted folder)
  - DataIngestorFactory (chooses the right ingestor based on the input path)
  - CmapssDataset (a simple typed container for the three DataFrames)

Usage examples :
---------------------------------------
# 1) From a ZIP file sitting in data/raw/
python -c "from src.ingest_data import DataIngestorFactory; \
ds=DataIngestorFactory.get_data_ingestor('data/raw/archive.zip').ingest('data/raw/archive.zip', subset='FD001'); \
print(ds.train.shape, ds.test.shape, ds.rul_truth.shape)"

# 2) From an already-extracted directory (containing train_FD001.txt, test_FD001.txt, RUL_FD001.txt)
python -c "from src.ingest_data import DataIngestorFactory; \
ds=DataIngestorFactory.get_data_ingestor('data/raw/extracted').ingest('data/raw/extracted', subset='FD001'); \
print(ds.train.head())"
"""
from __future__ import annotations

import os
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------- Helper: Column Names ---------------------------
def cmapss_columns() -> list[str]:
    """
    The C-MAPSS text files are whitespace-separated and follow the common convention:
    id, cycle, 3 operational settings (setting_1..3), and 21 sensors (s1..s21).
    Some sensors may be all zeros in certain subsets, but we keep consistent names.
    """
    cols = ["engine_id", "cycle"]  # first two columns
    cols += [f"setting_{i}" for i in range(1, 4)]  # setting_1..3
    cols += [f"s{i}" for i in range(1, 22)]        # s1..s21
    return cols


# --------------------------- Typed Return Container -------------------------
@dataclass
class CmapssDataset:
    """A simple container holding the three core pieces for a given subset."""
    subset: str                 # e.g., "FD001"
    train: pd.DataFrame         # telemetry with labels computed later
    test: pd.DataFrame          # telemetry for testing
    rul_truth: pd.DataFrame     # ground-truth RUL at the last cycle per engine in test

    def summary(self) -> dict:
        """Quick metadata summary—handy in notebooks or logs."""
        return {
            "subset": self.subset,
            "train_shape": tuple(self.train.shape),
            "test_shape": tuple(self.test.shape),
            "rul_truth_shape": tuple(self.rul_truth.shape),
            "train_engines": int(self.train["engine_id"].nunique()),
            "test_engines": int(self.test["engine_id"].nunique()),
        }


# ------------------------------ Abstract Base ------------------------------
class DataIngestor(ABC):
    """
    Abstract base: all ingestors must implement .ingest(input_path, subset).
    This mirrors your pattern and lets us swap implementations easily.
    """

    @abstractmethod
    def ingest(self, input_path: str | Path, subset: str) -> CmapssDataset:
        """
        Ingests the data for a specific subset (FD001..FD004) and returns a CmapssDataset.

        Parameters
        ----------
        input_path : str | Path
            Path to the ZIP file (for zip ingestor) or to the directory with the txt files.
        subset : str
            One of "FD001", "FD002", "FD003", "FD004" (case-insensitive).

        Returns
        -------
        CmapssDataset
            A typed container holding train, test, and rul_truth DataFrames.
        """
        raise NotImplementedError


# ------------------------------ Concrete: ZIP ------------------------------
class CmapssZipDataIngestor(DataIngestor):
    """
    Reads a .zip containing train_FDxxx.txt / test_FDxxx.txt / RUL_FDxxx.txt.
    It extracts to a temporary subfolder next to the zip and loads just the subset you request.
    """

    def ingest(self, input_path: str | Path, subset: str) -> CmapssDataset:
        path = Path(input_path)
        if not path.exists() or not path.suffix.lower() == ".zip":
            raise ValueError("CmapssZipDataIngestor expects a path to a .zip file.")

        subset = subset.upper()  # normalize (FD001..FD004)
        # Build expected member names inside the zip (exactly like the dataset ships)
        train_name = f"train_{subset}.txt"
        test_name = f"test_{subset}.txt"
        rul_name = f"RUL_{subset}.txt"

        # Where we will extract (idempotent: if already exists, we still just read)
        extract_dir = path.parent / f"extracted_{subset}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Open the zip and extract only the three needed files into extract_dir
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            missing = [n for n in [train_name, test_name, rul_name] if n not in names]
            if missing:
                raise FileNotFoundError(
                    f"Subset {subset} files not found in zip: {missing}. "
                    "Make sure you have the official C-MAPSS archive."
                )
            for member in [train_name, test_name, rul_name]:
                zf.extract(member, path=extract_dir)

        # Now we can simply delegate to the directory reader to keep logic DRY.
        return CmapssDirectoryDataIngestor().ingest(extract_dir, subset)


# --------------------------- Concrete: Directory ---------------------------
class CmapssDirectoryDataIngestor(DataIngestor):
    """
    Reads train/test/RUL txt files from a directory.
    Expected filenames:
      train_FD001.txt, test_FD001.txt, RUL_FD001.txt  (or FD002..FD004)
    """

    def ingest(self, input_path: str | Path, subset: str) -> CmapssDataset:
        dir_path = Path(input_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError("CmapssDirectoryDataIngestor expects a directory path.")

        subset = subset.upper()

        # Build expected filenames
        train_fp = dir_path / f"train_{subset}.txt"
        test_fp = dir_path / f"test_{subset}.txt"
        rul_fp = dir_path / f"RUL_{subset}.txt"

        # Validate existence early with clear errors
        for fp in [train_fp, test_fp, rul_fp]:
            if not fp.exists():
                raise FileNotFoundError(
                    f"Expected file not found: {fp}. "
                    "Check subset name or extraction path."
                )

        # Read the whitespace-separated telemetry files into DataFrames with stable column names.
        # Many C-MAPSS distributions contain an extra trailing space—pandas handles this with delim_whitespace=True.
        colnames = cmapss_columns()
        train_df = pd.read_csv(train_fp, sep=r"\s+", header=None, names=colnames)
        test_df = pd.read_csv(test_fp, sep=r"\s+", header=None, names=colnames)

        # RUL files typically have a single integer per line: the true RUL at last cycle per test engine.
        # We'll store it as a single-column DataFrame with a conventional name.
        rul_truth = pd.read_csv(rul_fp, sep=r"\s+", header=None, names=["RUL"])

        # Return a typed container for downstream steps (EDA, labeling, features, etc.)
        return CmapssDataset(subset=subset, train=train_df, test=test_df, rul_truth=rul_truth)


# --------------------------------- Factory ---------------------------------
class DataIngestorFactory:
    """
    Chooses the appropriate ingestion strategy based on the input path:
      - If it's a .zip file  -> CmapssZipDataIngestor
      - If it's a directory  -> CmapssDirectoryDataIngestor
    """

    @staticmethod
    def get_data_ingestor(input_path: str | Path) -> DataIngestor:
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == ".zip":
            return CmapssZipDataIngestor()
        if path.is_dir():
            return CmapssDirectoryDataIngestor()
        raise ValueError(
            "Unsupported input for C-MAPSS ingestion. Provide either:\n"
            "  - a .zip file containing train/test/RUL txt files, or\n"
            "  - a directory where these files are already extracted."
        )


# ------------------------------- CLI Example --------------------------------
if __name__ == "__main__":
    # This small CLI allows quick manual testing (you can expand with argparse if you like).
    # Edit these two lines to match your local paths:
    example_input = "data/raw/archive.zip"  # or "data/raw/extracted"
    example_subset = "FD001"

    ingestor = DataIngestorFactory.get_data_ingestor(example_input)
    dataset = ingestor.ingest(example_input, subset=example_subset)

    print("✅ Loaded subset:", dataset.subset)
    print("   Train shape:", dataset.train.shape)
    print("   Test shape :", dataset.test.shape)
    print("   RUL shape  :", dataset.rul_truth.shape)
    print("   Summary    :", dataset.summary())
