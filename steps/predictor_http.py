from zenml import step
import time, json, requests

@step(enable_cache=False)
def predictor_http(service_url: str, input_json: str, wait_seconds: int = 60):
    base = service_url.rstrip("/")
    # Wait for /ping instead of ZenML daemon state (WSL-friendly)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            if requests.head(f"{base}/ping", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"Server not reachable at {base}/ping")

    payload = json.loads(input_json)
    r = requests.post(f"{base}/invocations",
                      headers={"Content-Type": "application/json"},
                      json=payload, timeout=30)
    r.raise_for_status()
    return r.json()
