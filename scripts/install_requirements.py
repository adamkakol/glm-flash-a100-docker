"""Reuse the pinned upstream CUDA-12 dependency set and verify the ExLlama wheel."""
import json
from pathlib import Path
import tomllib

project = tomllib.loads(Path("/app/pyproject.toml").read_text())["project"]
versions = json.loads(Path("/deploy/versions.json").read_text())
requirements = project["dependencies"] + project["optional-dependencies"]["cu12"]
wheel_name = versions["exllamav3"]["wheel"]
wheel_hash = versions["exllamav3"]["sha256"]
found = 0
for i, requirement in enumerate(requirements):
    if wheel_name in requirement:
        requirements[i] = requirement.replace(wheel_name, wheel_name + "#sha256=" + wheel_hash)
        found += 1
if found != 1:
    raise RuntimeError("Upstream wheel declaration does not match the deployment lock.")
Path("/tmp/requirements.in").write_text("\n".join(requirements) + "\n")
