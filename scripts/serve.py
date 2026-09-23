import os
from pathlib import Path
import yaml
from model_files import metadata, verify_model
from preflight import main as preflight


def main():
    preflight()
    versions, _ = metadata()
    cfg = yaml.safe_load(Path("/app/config.yml").read_text())
    model = cfg["model"]
    if model["cache_size"] < 2 * model["max_seq_len"] or model["max_batch_size"] != 2:
        raise RuntimeError("This bundle expects capacity for two complete context windows and max_batch_size=2.")
    if model["max_seq_len"] < 262144 or model["cache_size"] % 256:
        raise RuntimeError("Invalid long-context/cache configuration.")
    if model["model_name"] != versions["model"]["directory"]:
        raise RuntimeError("Unexpected model directory; update the deployment lock deliberately before changing models.")
    verify_model(Path(model["model_dir"]) / model["model_name"])
    # Downloading happens in the separate command, never implicitly at serving time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.execv("/opt/venv/bin/python", ["python", "/app/main.py"])


if __name__ == "__main__":
    main()
