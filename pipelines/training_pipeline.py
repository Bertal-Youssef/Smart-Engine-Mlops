# pipelines/training_pipeline.py

from typing import List, Optional
from zenml import pipeline
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.rul_labeling_step import rul_labeling_step           # NEW
from steps.outlier_detection_step import outlier_detection_step  # uses your Z-score/IQR logic
from steps.data_splitter_step import data_splitter_step          # strategy-based splitter
from steps.model_building_step import model_building_step        # sklearn + mlflow
from steps.model_evaluator_step import model_evaluator_step      # metrics

@pipeline
def training_pipeline(
    file_path: str,
    subset: str = "FD001",
    part: str = "train",  # we train on the training partition
    missing_value_strategy: str = "mean",
    fe_strategy: str = "standard_scaling",
    features: Optional[List[str]] = None,
):
    # 1) Ingest a single DataFrame (train/test/rul). We use train here.
    df = data_ingestion_step(file_path=file_path, subset=subset, part=part)

    # 2) Handle missing values (kept for parity with the house project)
    df = handle_missing_values_step(df=df, strategy=missing_value_strategy)

    # 3) Label engineering: add the RUL target column
    #    RUL = max_cycle(engine) - cycle
    df = rul_labeling_step(df=df)

    # 4) Feature engineering (scale / transform only selected feature columns)
    df = feature_engineering_step(df=df, strategy=fe_strategy, features=features)

    # 5) Outlier detection and handling (remove/cap); validate column exists
    #    We mirror your house project API by naming the target column.
    df = outlier_detection_step(df=df, column_name="RUL")  # Z-score by default

    # 6) Split into train/test using your strategy-based splitter
    X_train, X_test, y_train, y_test = data_splitter_step(df, target_column="RUL")

    # 7) Train model (your sklearn + preprocessing pipeline logged to MLflow)
    model = model_building_step(X_train=X_train, y_train=y_train)

    # 8) Evaluate
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    # (Optionally: return artifacts if you want to consume them)
    return model
