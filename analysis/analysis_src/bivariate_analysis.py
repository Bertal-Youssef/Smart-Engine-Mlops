from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None: ...

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        # Useful for sensor vs. cycle or sensor vs. sensor
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[feature1], y=df[feature2], s=10, alpha=0.5)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1); plt.ylabel(feature2)
        plt.show()

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        # e.g., if you bin cycles into categories and compare sensor distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xticks(rotation=30)
        plt.show()

class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy
    def set_strategy(self, strategy: BivariateAnalysisStrategy) -> None:
        self._strategy = strategy
    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        self._strategy.analyze(df, feature1, feature2)
