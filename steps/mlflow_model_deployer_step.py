# steps/mlflow_model_deployer_step.py
"""
Thin wrapper so you can import `steps.mlflow_model_deployer_step`
while using the official ZenML MLflow deploy step under the hood.
"""
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step as mlflow_model_deployer_step
