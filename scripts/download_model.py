"""Download only the pinned EXL3 revision; interrupted downloads can be resumed."""
import argparse
import json
from pathlib import Path
import shutil
from model_files import metadata, verify_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hashes", action="store_true", help="Also read all large files and check upstream SHA256 values.")
    args = parser.parse_args()
    versions, files = metadata()
    spec = versions["model"]
    target = Path("/models") / spec["directory"]
    target.mkdir(parents=True, exist_ok=True)
    needed = sum(item["size"] for item in files if not (target / item["path"]).is_file() or (target / item["path"]).stat().st_size != item["size"])
    if shutil.disk_usage(target).free < needed + 5 * 2**30:
        raise RuntimeError(f"Need approximately {needed / 2**30:.1f} GiB free plus 5 GiB on the model volume. Incomplete downloads may already consume space.")
    from huggingface_hub import snapshot_download
    print(f"Downloading {spec['repo_id']} at immutable revision {spec['revision']}", flush=True)
    snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=str(target), max_workers=4,
                      allow_patterns=[item["path"] for item in files])
    verify_model(target, hashes=args.verify_hashes, marker=False)
    (target / "deployment-revision.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("Download and file checks complete.")


if __name__ == "__main__":
    main()
