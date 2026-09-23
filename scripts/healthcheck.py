"""Check both generation health and the configuration actually loaded."""
import json
from pathlib import Path
import urllib.request
import yaml
from deployment import validate_limits


def main():
    expected = yaml.safe_load(Path("/app/config.yml").read_text())["model"]
    validate_limits(expected)
    keys = json.loads(Path("/app/api_tokens.yml").read_text())
    key = keys["api_key"][0] if isinstance(keys["api_key"], list) else keys["api_key"]

    def get(path):
        request = urllib.request.Request("http://127.0.0.1:5000" + path,
                                         headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    if get("/health").get("status") != "healthy":
        raise RuntimeError("Inference engine reports unhealthy")
    model = get("/v1/model")
    if model["id"] != expected["model_name"]:
        raise RuntimeError("Unexpected loaded model")
    params = model["parameters"]
    for name in ("max_seq_len", "cache_size", "max_batch_size", "chunk_size", "cache_mode"):
        if params.get(name) != expected[name]:
            raise RuntimeError(f"Loaded {name} differs from config.yml")


if __name__ == "__main__":
    main()
