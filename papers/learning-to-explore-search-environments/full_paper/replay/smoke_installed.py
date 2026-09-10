"""Verify an installed wheel and portable replay from outside the checkout.

Run after installing SearchProbe into a fresh environment:
  python smoke_installed.py --python /path/to/venv/bin/python
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent


def smoke(python, evidence_dir):
    python = Path(python).absolute()
    executable = shutil.which("searchprobe-demo", path=str(python.parent))
    if executable is None:
        raise ValueError("The selected Python environment has no searchprobe-demo entry point; install the new wheel first")
    manifest = json.loads((HERE / "evidence_manifest.json").read_text())
    environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    with tempfile.TemporaryDirectory(prefix="searchprobe-outside-checkout-") as directory:
        cwd = Path(directory)
        for name in ("replay.py", "evidence_manifest.json"):
            shutil.copyfile(HERE / name, cwd / name)
        for name in manifest["files"]:
            destination = cwd / "evidence" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(evidence_dir) / name, destination)

        def run(command):
            result = subprocess.run([str(part) for part in command], cwd=cwd, env=environment,
                                    text=True, capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            return result

        module = run([python, "-c", "import importlib.util, json, pathlib, searchprobe, sys; print(json.dumps({'module': str(pathlib.Path(searchprobe.__file__).resolve()), 'prefix': sys.prefix, 'research_dependencies_available': [name for name in ('numpy','sklearn','torch','mlx') if importlib.util.find_spec(name) is not None]}))"])
        location = json.loads(module.stdout)
        if not Path(location["module"]).is_relative_to(Path(location["prefix"]).resolve()):
            raise ValueError("SearchProbe was imported outside the selected installed environment")
        if location["research_dependencies_available"]:
            raise ValueError("Use a fresh wheel environment without the research dependencies for this smoke test")
        run([executable, "--output-dir", "editable examples"])
        demo = json.loads((cwd / "editable examples/demo.json").read_text())
        for item in demo["examples"]:
            report = cwd / f"{item['name']}.rerun.json"
            run([python, "-m", "searchprobe", item["command"][1], cwd / "editable examples" / item["input"], "--output", report])
            if json.loads(report.read_text()) != json.loads((cwd / "editable examples" / item["report"]).read_text()):
                raise ValueError("Installed original command differs from bundled demo report")
        run([python, cwd / "replay.py", "--evidence-dir", cwd / "evidence", "--output", cwd / "replay.json"])
        replayed = json.loads((cwd / "replay.json").read_text())
        assert replayed["counts"]["main_run_rows_recomputed"] == 450
        assert replayed["counts"]["followup_run_rows_recomputed"] == 360
        assert sum(item["certified_unchanged"] for item in replayed["certificates"]) == 730
        return {"status": "passed", "installed_module": location["module"], "synthetic_examples": len(demo["examples"]),
                "replay_counts": replayed["counts"], "checkout_on_pythonpath": False,
                "research_dependencies_available": location["research_dependencies_available"],
                "evidence_source": "Only the 34 committed manifest files copied into a temporary directory; no ignored caches or network."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--evidence-dir", type=Path, default=HERE.parents[1] / "semantic_pilot" / "results")
    args = parser.parse_args()
    print(json.dumps(smoke(args.python, args.evidence_dir), indent=2))


if __name__ == "__main__":
    main()
