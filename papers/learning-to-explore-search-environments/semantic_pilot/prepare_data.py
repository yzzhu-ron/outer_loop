"""Fetch pinned public inputs, or verify existing files without network access.

The preceding pilot's downloader remains responsible for SciFact, FiQA, and
MiniLM. This script adds the checksum-pinned NFCorpus archive and the immutable
Qwen MLX snapshot. Dataset archives are never extracted.

Run with the pilot virtual environment. ``--verify-only`` neither downloads nor
imports model libraries and is safe while generation is using the model files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / "pilot"
NFCORPUS_SHA256 = "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b"
NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
QWEN_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
QWEN_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, expected: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive: {path}; run prepare_data.py without --verify-only")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Archive checksum mismatch for {path}: expected {expected}, got {actual}; inspect rather than updating the pin automatically")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def ensure_nfcorpus(path: Path, *, verify_only: bool = False) -> dict:
    if path.exists() or verify_only:
        return verify_archive(path, NFCORPUS_SHA256)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".download", delete=False) as output:
            temporary_path = Path(output.name)
            with urllib.request.urlopen(NFCORPUS_URL, timeout=120) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
        verify_archive(temporary_path, NFCORPUS_SHA256)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return verify_archive(path, NFCORPUS_SHA256)


def snapshot_path(cache_dir: Path, repo: str, revision: str) -> Path:
    return cache_dir / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def verify_snapshot(path: Path, repo: str, revision: str) -> dict:
    if path.name != revision or path.parent.name != "snapshots":
        raise ValueError(f"Expected immutable snapshot directory ending in snapshots/{revision}")
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if repo == QWEN_REPO:
        required.append("chat_template.jinja")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = set(index.get("weight_map", {}).values())
        if not shards:
            raise ValueError(f"No model weight shards listed in {index_path}")
        if not all(isinstance(shard, str) and Path(shard).name == shard for shard in shards):
            raise ValueError(f"Unexpected shard path in {index_path}")
        required.extend(sorted(shards))
    else:
        required.append("model.safetensors")
    if repo == ENCODER_REPO:
        required.extend(["modules.json", "sentence_bert_config.json", "1_Pooling/config.json"])
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete pinned snapshot {path}: {missing}; run prepare_data.py without --verify-only")
    return {
        "repo": repo,
        "revision": revision,
        "path": str(path),
        "required_files_present": sorted(set(required)),
        "verification": "Immutable revision path and required files; archive checksums are separately verified. Local model weights are not rehashed.",
    }


def _legacy_downloader():
    spec = importlib.util.spec_from_file_location("preceding_pilot_prepare_data", PILOT / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(*, verify_only: bool = False) -> dict:
    legacy = _legacy_downloader()
    if not verify_only:
        # A subprocess gives the unchanged legacy downloader its own HF cache
        # configuration, regardless of huggingface_hub imports in this process.
        environment = os.environ.copy()
        environment["HF_HOME"] = str(PILOT / "data" / "hf")
        subprocess.run([sys.executable, str(PILOT / "prepare_data.py")], env=environment, check=True)
    archives = {name: verify_archive(PILOT / "data" / f"{name}.zip", expected)
                for name, expected in legacy.ARCHIVES.items()}
    archives["nfcorpus"] = ensure_nfcorpus(ROOT / "data" / "nfcorpus.zip", verify_only=verify_only)
    encoder_path = snapshot_path(PILOT / "data" / "hf" / "hub", ENCODER_REPO, ENCODER_REVISION)
    qwen_cache = ROOT / "data" / "hf" / "hub"
    qwen_path = snapshot_path(qwen_cache, QWEN_REPO, QWEN_REVISION)
    if not verify_only:
        from huggingface_hub import snapshot_download
        downloaded = Path(snapshot_download(
            QWEN_REPO,
            revision=QWEN_REVISION,
            cache_dir=str(qwen_cache),
            allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
        ))
        if downloaded.resolve() != qwen_path.resolve():
            raise ValueError(f"Unexpected snapshot location: {downloaded}")
    return {
        "archives": archives,
        "models": {
            "encoder": verify_snapshot(encoder_path, ENCODER_REPO, ENCODER_REVISION),
            "generator": verify_snapshot(qwen_path, QWEN_REPO, QWEN_REVISION),
        },
        "verify_only": verify_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check local pinned inputs without network calls or downloads")
    args = parser.parse_args()
    print(json.dumps(prepare(verify_only=args.verify_only), indent=2))


if __name__ == "__main__":
    main()
