import os
from pathlib import Path
import yaml
from model_files import metadata, verify_model
from preflight import main as preflight
from deployment import validate_limits


def main():
    preflight()
    versions, _ = metadata()
    cfg = yaml.safe_load(Path("/app/config.yml").read_text())
    model = cfg["model"]
    validate_limits(model)
    if model["model_name"] != versions["model"]["directory"]:
        raise RuntimeError("Unexpected model directory; update the deployment lock deliberately before changing models.")
    verify_model(Path(model["model_dir"]) / model["model_name"])
    # Downloading happens in the separate command, never implicitly at serving time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.execv("/opt/venv/bin/python", ["python", "/app/main.py"])


if __name__ == "__main__":
    main()
