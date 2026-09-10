"""Audit a copied replay without raw caches, preserving the archived lock and hashes.

The new decision lock has a new UTC creation timestamp. Compare its complete
content after removing exactly that field; evaluate the unchanged copied original
lock so its original test-label hash binding remains valid. Never rewrite either
lock or rebind the frozen test labels to a new decision file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
ARTIFACTS = ("freeze.v1.json", "inputs.v1.json", "decisions.v1.json", "test_labels.v1.json", "results.v1.json")
TIMESTAMP = "prediction_lock_created_at_utc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(destination=None):
    if destination is None:
        destination = Path(tempfile.mkdtemp(prefix="intervention-transfer-portable-"))
    else:
        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((ROOT / "freeze.v1.json").read_text())
    before = {str(path): sha(path) for path in [*(PAPER / rel for rel in freeze["files"]), *(ROOT / name for name in ARTIFACTS)]}
    for rel, expected in freeze["files"].items():
        source = PAPER / rel
        if sha(source) != expected:
            raise ValueError("Frozen implementation hash mismatch: " + rel)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    copied = destination / "full_paper" / "intervention_transfer"
    for name in ARTIFACTS:
        shutil.copy2(ROOT / name, copied / name)
    raw_cache = destination / "semantic_pilot" / "cache"
    if raw_cache.exists():
        raise AssertionError("The portable fixture must contain no raw cache directory")
    for phase, output in (("decide", "replayed_decisions.json"), ("evaluate", "replayed_results.json")):
        command = [sys.executable, str(copied / "experiment.py"), phase, "--output", str(copied / output)]
        subprocess.run(command, cwd=copied, check=True, capture_output=True, text=True)
    archived = json.loads((copied / "decisions.v1.json").read_text())
    regenerated = json.loads((copied / "replayed_decisions.json").read_text())
    old_time = archived.pop(TIMESTAMP)
    new_time = regenerated.pop(TIMESTAMP)
    if archived != regenerated:
        raise AssertionError("Regenerated decisions differ beyond their declared lock timestamp")
    if (copied / "results.v1.json").read_bytes() != (copied / "replayed_results.json").read_bytes():
        raise AssertionError("Portable evaluation is not byte-identical to the archived results")
    if raw_cache.exists():
        raise AssertionError("Replay created an unexpected raw cache directory")
    if any(sha(path) != expected for path, expected in before.items()):
        raise AssertionError("Replay changed an original frozen artifact")
    report = {"status": "passed", "copy_directory": str(destination), "raw_cache_directory_exists": False,
        "decision_comparison": "Exact JSON equality after excluding only prediction_lock_created_at_utc",
        "archived_lock_created_at_utc": old_time, "replayed_lock_created_at_utc": new_time,
        "evaluation_decisions": "Unchanged copied original lock; original test-label decision hash binding preserved",
        "results_byte_identical": True, "original_artifacts_unchanged": True,
        "archived_decision_sha256": sha(copied / "decisions.v1.json"),
        "replayed_decision_sha256": sha(copied / "replayed_decisions.json"),
        "results_sha256": sha(copied / "results.v1.json")}
    with (destination / "replay_report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, help="New copy directory; defaults to a new temporary directory")
    args = parser.parse_args()
    print(json.dumps(replay(args.destination), indent=2, sort_keys=True))
