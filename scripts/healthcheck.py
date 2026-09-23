import json
from pathlib import Path
import urllib.request

keys = json.loads(Path("/app/api_tokens.yml").read_text())
key = keys["api_key"][0] if isinstance(keys["api_key"], list) else keys["api_key"]
request = urllib.request.Request("http://127.0.0.1:5000/v1/model", headers={"Authorization": "Bearer " + key})
with urllib.request.urlopen(request, timeout=5) as response:
    model = json.load(response)
params = model["parameters"]
if (model["id"] != "GLM-5.3-Flash-exl3-4.05bpw"
        or params["max_seq_len"] < 262144
        or params["cache_size"] < 2 * params["max_seq_len"]
        or params["max_batch_size"] != 2):
    raise SystemExit("Unexpected loaded model/context/batch configuration")
