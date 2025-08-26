from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None: ...
    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame) -> None: ...

class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        # Correlations among sensors; choose numeric subset to avoid clutter
        plt.figure(figsize=(12, 10))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (numeric features)")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame) -> None:
        # For visibility, pass a small set of columns from the notebook
        sns.pairplot(df, corner=True, plot_kws=dict(s=10, alpha=0.4))
        plt.suptitle("Pair Plot (selected features)", y=1.02)
        plt.show()
