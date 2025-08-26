from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#using template pattern
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        # High-level template: identify -> visualize
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None: ...
    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None: ...

class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        print("\n[Missing Values by Column]")
        missing = df.isnull().sum()
        print(missing[missing > 0])

    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        print("\n[Heatmap of Missing Values]")
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False)
        plt.title("Missing Values Heatmap")
        plt.show()
