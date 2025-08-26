# run_deployment.py

import json
import time
import requests
import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

# The deployed model currently expects ALL 26 C-MAPSS columns (per MLflow signature)
SIGNATURE_COLS_26 = [
    "engine_id", "cycle", "setting_1", "setting_2", "setting_3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9",
    "s10","s11","s12","s13","s14","s15","s16",
    "s17","s18","s19","s20","s21",
]

@click.command()
@click.option("--subset", default="FD001", show_default=True)
@click.option("--file-path", default="data/raw/archive.zip", show_default=True)
@click.option("--workers", default=1, show_default=True)
def main(subset: str, file_path: str, workers: int):
    # 1) Train & (re)deploy
    # NOTE: Even if you trained with a smaller feature list, your deployed model
    # is currently saved with a 26-column signature. We'll match that here.
    continuous_deployment_pipeline(
        file_path=file_path,
        subset=subset,
        features=["cycle","setting_1","setting_2","setting_3","s2","s3","s4","s7","s8","s9"],
        missing_value_strategy="mean",
        fe_strategy="standard_scaling",
        algorithm="hgb",
        workers=workers,     # 1 is safer on WSL/WSL2
        deploy_decision=True,
    )

    print(
        "Now run:\n"
        f"  mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "to inspect experiments."
    )

    # 2) Resolve the deployed service and wait on /ping
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )
    if not services:
        raise RuntimeError("No MLflow model server found. Check the deployment step logs.")
    service = services[-1]

    base = (service.prediction_url or "http://127.0.0.1:8000").rstrip("/")
    ping = f"{base}/ping"
    print(f"Waiting for model server at {base} ...")

    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if requests.head(ping, timeout=2).status_code == 200:
                print("[green]✓ Server is ready[/green]")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("Model server did not become healthy at /ping. See service.log.")

    sample = {
    "dataframe_records": [{
        "engine_id": 1, "cycle": 50.0, "setting_1": 0.5, "setting_2": -0.2, "setting_3": 0.1,
        "s1": 0.0, "s2": 550.0, "s3": 1350.0, "s4": 12.0, "s5": 0.0, "s6": 0.0,
        "s7": 3.2, "s8": 9005.0, "s9": 1.0, "s10": 0.0, "s11": 0.0, "s12": 0.0,
        "s13": 0.0, "s14": 0.0, "s15": 0.0, "s16": 0.0, "s17": 0, "s18": 0,
        "s19": 0.0, "s20": 0.0, "s21": 0.0
    }]
    }
    inference_pipeline(input_json=json.dumps(sample))

if __name__ == "__main__":
    main()
