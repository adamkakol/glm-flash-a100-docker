"""Parse NVIDIA topology without assuming host indices equal CUDA ordinals."""
import csv
import io
import re


def nvlink_pairs(text):
    lines = text.splitlines()
    header = next((line for line in lines if re.match(r'\s*GPU\d+\s+GPU\d+', line)), '')
    labels = re.findall(r'\bGPU\d+\b', header)
    directed = set()
    for line in lines:
        cells = line.split()
        if not cells or cells[0] not in labels:
            continue
        src = int(cells[0][3:])
        for label, link in zip(labels, cells[1:]):
            if re.fullmatch(r'NV[1-9]\d*', link):
                dst = int(label[3:])
                if src != dst:
                    directed.add((src, dst))
    return sorted((a, b) for a, b in directed if a < b and (b, a) in directed)


def require_nvlink_pair(topology, inventory, expected_uuids):
    """Inventory is nvidia-smi --query-gpu=index,uuid --format=csv,noheader."""
    by_uuid = {uuid.strip(): int(index.strip()) for index, uuid in csv.reader(io.StringIO(inventory)) if uuid.strip()}
    if len(expected_uuids) != 2:
        raise ValueError('Exactly two expected NVLink GPU UUIDs are required')
    try:
        pair = tuple(sorted(by_uuid[uuid] for uuid in expected_uuids))
    except KeyError as exc:
        raise RuntimeError('Selected NVLink GPU is missing from the container inventory') from exc
    if pair not in nvlink_pairs(topology):
        raise RuntimeError('Selected GPUs no longer report a bidirectional NVLink connection; recheck the bridge and configuration')
    return pair
