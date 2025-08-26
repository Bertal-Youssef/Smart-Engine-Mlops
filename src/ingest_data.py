# src/ingest_data.py
from __future__ import annotations

import argparse
import sys
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


# --------------------------- Helper: Column Names ---------------------------
def cmapss_columns() -> list[str]:
    cols = ["engine_id", "cycle"]
    cols += [f"setting_{i}" for i in range(1, 4)]
    cols += [f"s{i}" for i in range(1, 22)]
    return cols


# --------------------------- Typed Return Container -------------------------
@dataclass
class CmapssDataset:
    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul_truth: pd.DataFrame

    def summary(self) -> dict:
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
    @abstractmethod
    def ingest(self, input_path: str | Path, subset: str, verbose: bool = False) -> CmapssDataset:
        """Ingest data from a zip or directory and return a CmapssDataset."""
        raise NotImplementedError


# ------------------------------ Concrete: ZIP ------------------------------
class CmapssZipDataIngestor(DataIngestor):
    """
    Reads a .zip that may contain files inside a subfolder (e.g., CMaps/train_FD001.txt).
    We match members by their filename (endswith), not by exact path.
    """

    def _find_members(self, zf: zipfile.ZipFile, subset: str) -> List[str]:
        subset = subset.upper()
        wanted = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
        names = zf.namelist()
        found = []
        for w in wanted:
            matches = [n for n in names if n.endswith("/" + w) or n.endswith("\\" + w) or n.endswith(w)]
            if not matches:
                raise FileNotFoundError(
                    f"Could not find {w} inside the zip. "
                    f"Zip contains: {names[:8]}{'...' if len(names)>8 else ''}"
                )
            # take the first match (usually CMaps/<file>)
            found.append(matches[0])
        return found

    def ingest(self, input_path: str | Path, subset: str, verbose: bool = False) -> CmapssDataset:
        # --- Robust to str or Path ---
        input_path = Path(input_path)
        subset = subset.upper()

        if verbose:
            print(f"🔎 ZIP ingestion from: {input_path}")

        extract_dir = input_path.parent / f"extracted_{subset}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(input_path, "r") as zf:
            members = self._find_members(zf, subset)
            if verbose:
                print("📦 Members to extract:", members)
            for m in members:
                zf.extract(m, path=extract_dir)

        # After extraction, files are under extract_dir/<maybe-subfolder>/filename.txt
        # Find them by filename regardless of subfolder depth.
        train_fp = next(extract_dir.rglob(f"train_{subset}.txt"))
        test_fp = next(extract_dir.rglob(f"test_{subset}.txt"))
        rul_fp = next(extract_dir.rglob(f"RUL_{subset}.txt"))

        if verbose:
            print(f"✅ Extracted to: {train_fp.parent}")

        # Delegate reading to the directory ingestor
        return CmapssDirectoryDataIngestor().ingest(train_fp.parent, subset=subset, verbose=verbose)


# --------------------------- Concrete: Directory ---------------------------
class CmapssDirectoryDataIngestor(DataIngestor):
    def ingest(self, input_path: str | Path, subset: str, verbose: bool = False) -> CmapssDataset:
        # --- Robust to str or Path ---
        input_path = Path(input_path)
        subset = subset.upper()

        if verbose:
            print(f"📂 Directory ingestion from: {input_path}")

        # Find by filename anywhere under the folder (handles extra nesting)
        train_fp = next(input_path.rglob(f"train_{subset}.txt"), None)
        test_fp = next(input_path.rglob(f"test_{subset}.txt"), None)
        rul_fp = next(input_path.rglob(f"RUL_{subset}.txt"), None)

        missing = [name for name, fp in [
            (f"train_{subset}.txt", train_fp),
            (f"test_{subset}.txt", test_fp),
            (f"RUL_{subset}.txt", rul_fp),
        ] if fp is None]

        if missing:
            raise FileNotFoundError(
                f"Missing expected files in {input_path}: {missing}\n"
                f"Tip: ensure your subset is correct (FD001..FD004)."
            )

        cols = cmapss_columns()
        train_df = pd.read_csv(train_fp, sep=r"\s+", header=None, names=cols)
        test_df = pd.read_csv(test_fp, sep=r"\s+", header=None, names=cols)
        rul_truth = pd.read_csv(rul_fp, sep=r"\s+", header=None, names=["RUL"])

        if verbose:
            print("✅ Loaded:")
            print("   train:", train_fp)
            print("   test :", test_fp)
            print("   RUL  :", rul_fp)

        return CmapssDataset(subset=subset, train=train_df, test=test_df, rul_truth=rul_truth)


# --------------------------------- Factory ---------------------------------
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(input_path: str | Path) -> DataIngestor:
        p = Path(input_path)
        if p.is_file() and p.suffix.lower() == ".zip":
            return CmapssZipDataIngestor()
        if p.is_dir():
            return CmapssDirectoryDataIngestor()
        raise ValueError("Provide either a .zip file path or a directory path.")


# ----------------------------------- CLI -----------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="C-MAPSS ingestion (Factory pattern)")
    ap.add_argument("--input", required=True, help="Path to .zip or directory")
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"],
                    help="Subset to load (default: FD001)")
    ap.add_argument("--verbose", action="store_true", help="Print progress")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    subset = args.subset.upper()

    ingestor = DataIngestorFactory.get_data_ingestor(input_path)
    dataset = ingestor.ingest(input_path, subset=subset, verbose=args.verbose)

    print("\n📊 Summary:", dataset.summary())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print a clean error on Windows/PowerShell
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
