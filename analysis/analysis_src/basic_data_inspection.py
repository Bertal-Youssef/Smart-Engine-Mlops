from abc import ABC, abstractmethod
import pandas as pd

# ===== Abstract Strategy =====
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """Run a specific inspection and print/return visuals."""
        raise NotImplementedError

# ===== Concrete Strategies =====
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        # Shows column dtypes + non-null counts; great first glance
        print("\n[Data Types & Non-Null Counts]")
        print(df.info())

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        # Numeric stats
        print("\n[Summary Stats — Numeric]")
        print(df.describe())

        # Categorical stats (check before describing)
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            print("\n[Summary Stats — Categorical]")
            print(df.describe(include=["O"]))
        else:
            print("\n[Summary Stats — Categorical]")
            print("No categorical columns found.")


# ===== Context =====
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy) -> None:
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame) -> None:
        self._strategy.inspect(df)
