# pipelines/deployment_pipeline.py
from typing import List, Optional
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor_http import predictor_http
from steps.data_ingestion_step import data_ingestion_step
from steps.rul_labeling_step import rul_labeling_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor_http import predictor_http

@pipeline
def continuous_deployment_pipeline(
    file_path: str = "data/raw/archive.zip",
    subset: str = "FD001",
    features: Optional[List[str]] = None,
    missing_value_strategy: str = "mean",
    fe_strategy: str = "standard_scaling",
    algorithm: str = "hgb",
    workers: int = 1,                 # safer on WSL
    deploy_decision: bool = True,
):
    df = data_ingestion_step(file_path=file_path, subset=subset, part="train")
    df = rul_labeling_step(df=df)     # ensures 'RUL'
    df = handle_missing_values_step(df=df, strategy=missing_value_strategy)
    df = outlier_detection_step(df=df, column_name="RUL")
    df = feature_engineering_step(df=df, strategy=fe_strategy, features=features)

    X_train, X_test, y_train, y_test = data_splitter_step(df=df, target_column="RUL")
    model = model_building_step(X_train=X_train, y_train=y_train, algorithm=algorithm)
    model_evaluator_step(trained_model=model, X_test=X_test, y_test=y_test)

    # You can add port=8000 to be explicit: port=8000
    mlflow_model_deployer_step(
        model=model,
        workers=workers,
        deploy_decision=deploy_decision,
    )




@pipeline(enable_cache=False)
def inference_pipeline(input_json: str):
    url = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )
    predictor_http(service_url=url, input_json=input_json)


