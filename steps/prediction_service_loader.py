from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, step_name: str) -> str:
    md = MLFlowModelDeployer.get_active_model_deployer()
    services = md.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        running=True,
    )
    if not services:
        raise RuntimeError(f"No MLflow service running for {pipeline_name}/{step_name}")
    return services[0].prediction_url or "http://127.0.0.1:8000"