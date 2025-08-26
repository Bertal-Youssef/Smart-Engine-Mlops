from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str) -> None: ...

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        # Histogram + KDE for sensor or cycle column
        plt.figure(figsize=(10, 5))
        sns.histplot(df[feature].dropna(), bins=40, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()

class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        # Kept for parity; often unused in C-MAPSS
        plt.figure(figsize=(10, 5))
        sns.countplot(x=feature, data=df)
        plt.title(f"Frequency of {feature}")
        plt.xticks(rotation=45)
        plt.show()

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy
    def set_strategy(self, strategy: UnivariateAnalysisStrategy) -> None:
        self._strategy = strategy
    def execute_analysis(self, df: pd.DataFrame, feature: str) -> None:
        self._strategy.analyze(df, feature)
