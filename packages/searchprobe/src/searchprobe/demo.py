"""Run bundled synthetic examples without a repository checkout or network."""

from __future__ import annotations

import argparse
from importlib.resources import as_file, files
import json
from pathlib import Path
import shlex

from .audit import audit_jsonl
from .decisions import audit_decision_json
from .responsiveness import audit_response_json


EXAMPLES = {
    "probes": ("synthetic.jsonl", "audit", audit_jsonl),
    "decision": ("source_model_synthetic.json", "decision-audit", audit_decision_json),
    "response": ("response_model.json", "response-audit", audit_response_json),
}


def run_demo(output_dir: str | Path, example: str = "all") -> dict:
    """Save installed example inputs and computed reports; refuse overwrites.

    Everything is synthetic, including outcomes, scores, bounds, and costs.
    No search, data download, model inference, or empirical replay occurs.
    """
    if example not in (*EXAMPLES, "all"):
        raise ValueError(f"Unknown example {example!r}; choose probes, decision, response, or all")
    output = Path(output_dir).resolve()
    prepared = []
    for name in EXAMPLES if example == "all" else (example,):
        filename, command, audit = EXAMPLES[name]
        resource = files("searchprobe").joinpath("example_data", filename)
        with as_file(resource) as path:
            report = audit(path)
        prepared.append((name, filename, command, resource.read_bytes(), report))
    summary = {
        "schema_version": 1, "synthetic": True,
        "scope": "Bundled demonstrations only. No retrieval, model calls, measured research outcomes, or target guarantees.",
        "examples": [{"name": name, "input": filename, "report": f"{name}.report.json",
                      "command": ["searchprobe", command, filename, "--output", f"{name}.report.json"]}
                     for name, filename, command, _, _ in prepared],
    }
    destinations = [output / "demo.json", output / "README.md"]
    destinations.extend(output / filename for _, filename, _, _, _ in prepared)
    destinations.extend(output / f"{name}.report.json" for name, _, _, _, _ in prepared)
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise ValueError("Demo files already exist; choose a new --output-dir. Existing files are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    for name, filename, _, content, report in prepared:
        (output / filename).write_bytes(content)
        (output / f"{name}.report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "demo.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    instructions = ["# SearchProbe synthetic examples", "", summary["scope"], "",
                    "The input files and reports are yours to inspect and edit. From this directory, rerun a command after editing its input:", "", "```sh"]
    instructions.extend(shlex.join(item["command"]) for item in summary["examples"])
    instructions.extend(["```", "", "Expected unchanged inputs: the probe audit finds 5 valid pairs and 1 exact query collision; the source decision radius is 1/3; the response example certifies 1/3 of rows unchanged with an absolute mean utility-change bound of 2/3.", "",
                         "A warning in the probe example is intentional. A source decision certificate assumes the supplied utility model; a response certificate assumes the supplied bounds. Neither establishes performance on future tasks.", ""])
    (output / "README.md").write_text("\n".join(instructions), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="searchprobe-demo", description=__doc__)
    parser.add_argument("example", choices=(*EXAMPLES, "all"), nargs="?", default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("searchprobe-demo"), help="Directory for input files and reports (default: ./searchprobe-demo); files are never overwritten")
    args = parser.parse_args(argv)
    try:
        summary = run_demo(args.output_dir, args.example)
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(f"Created {len(summary['examples'])} synthetic examples in {args.output_dir.resolve()}")
    print("No network, retrieval, or model calls. Open README.md in that directory to inspect and rerun them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
