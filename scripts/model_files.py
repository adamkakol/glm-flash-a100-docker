import hashlib
import json
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent


def metadata():
    return json.loads((DEPLOY / "versions.json").read_text()), json.loads((DEPLOY / "model-manifest.json").read_text())


def verify_model(directory, *, hashes=False, marker=True):
    versions, files = metadata()
    directory = Path(directory)
    if marker:
        stamp = json.loads((directory / "deployment-revision.json").read_text())
        if stamp["revision"] != versions["model"]["revision"]:
            raise RuntimeError("Model revision differs from the deployment lock.")
    for item in files:
        path = directory / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise RuntimeError(f"Missing/incomplete model file: {item['path']}. Run download_model.py.")
        if hashes and item.get("sha256"):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {item['path']}")
