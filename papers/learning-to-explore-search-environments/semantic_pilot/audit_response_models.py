"""Export the frozen B32 routers to SearchProbe's conditional score-box audit.

Only saved model parameters, test query IDs/features, and saved sensitivity
bounds enter the calculation. Outcome-containing files are hashed for source
integrity; no target utility, relevance label, or probe observation is used.
The exported decimals define an exact rational box, not a formally verified
enclosure of the entire floating-point training/inference implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

import numpy as np

from audit_router_sensitivity import base_predictions, change_certificate, coefficient_matrix
from learning import ACTIONS, SHRINKAGE


ROOT = Path(__file__).resolve().parent
PACKAGE_SOURCE = ROOT.parents[2] / "packages" / "searchprobe" / "src"
sys.path.insert(0, str(PACKAGE_SOURCE))
from searchprobe.responsiveness import audit_response_model


PAIR_BUDGET = 32
NUMERICAL_SCOPE = (
    "Base predictions are reconstructed with NumPy floating-point arithmetic and exported using Python's "
    "shortest round-trip decimal strings. Bounds are the saved B32 floating-point bounds, exported the same way. "
    "SearchProbe computes exactly for these supplied decimals. This does not establish that the decimals "
    "enclose all roundoff in base reconstruction, profile accumulation, linear corrections, or argmax in "
    "the real router; it is not a formal whole-router arithmetic certificate."
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number {value} in {path}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def _json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def response_model(record, metadata, saved_details):
    """Build one score-box model without accessing target or probe outcomes.

    The caller verifies file integrity first. This function verifies the saved
    B32 certificate by reconstruction, then compares the old conservative
    floating-point certificate with the package's exact supplied-decimal box.
    """
    environment = record["name"]
    tasks = record["tasks"]["test"]
    ids = list(tasks["ids"])
    predictions = base_predictions(metadata, tasks["features"])
    coefficients = coefficient_matrix(metadata)
    shrinkage = float(metadata["profile_shrinkage"])
    if shrinkage != SHRINKAGE or saved_details["profile_shrinkage"] != shrinkage:
        raise ValueError(f"{environment}: profile shrinkage differs from the frozen implementation")
    if saved_details["coefficients"] != coefficients.tolist():
        raise ValueError(f"{environment}: saved correction coefficients differ from the fitted model")
    saved = saved_details["queries"][str(PAIR_BUDGET)]
    if ids != saved["query_ids"] or len(ids) != len(predictions):
        raise ValueError(f"{environment}: test query IDs/order differ from the saved certificate")
    reconstructed = change_certificate(predictions, coefficients, PAIR_BUDGET, shrinkage=shrinkage)
    for key, value in reconstructed.items():
        if saved.get(key) != value:
            raise ValueError(f"{environment}: saved B32 certificate differs on {key}")
    bounds = saved["action_correction_abs_bounds"]
    model = {
        "schema_version": 1, "kind": "score_box_model", "model_id": f"{environment}-frozen-profile-B{PAIR_BUDGET}",
        "actions": list(ACTIONS), "row_ids": ids,
        "base_scores": [[str(float(value)) for value in row] for row in predictions],
        "correction_bounds": [str(float(value)) for value in bounds],
        "provenance": {
            "score_source": f"results/models.json, fold {record['corpus_name']}::{record['backend']}; base_predictions on cache/{environment}_outcomes.json test features. Source-trained frozen parameters; no target utilities used.",
            "bound_source": f"results/router_sensitivity_queries.json, {environment}/queries/{PAIR_BUDGET}/action_correction_abs_bounds; verified against the saved coefficients and change_certificate implementation.",
            "row_population": f"The {len(ids)} retained frozen test-query rows for {environment}, in their saved order. Backends share queries; rows across environments are not independent replications.",
            "synthetic": False,
        },
        "assumptions": [
            "At most 32 paired probes; every reciprocal-rank difference lies in [-1,1]; profile pseudocount is 4 and correction intercepts are zero.",
            "In the mathematical linear profile model each of four features has absolute value at most 32/(32+4); action bound is coefficient L1 norm times that value. The original action has zero correction.",
            "Use the supplied symmetric independent action box. Shared counts, coupled features, and reachable histories may make its extremes unattainable in the real router.",
            "Use the highest score with lowest action index winning exact ties, in the supplied action order.",
            NUMERICAL_SCOPE,
        ],
    }
    report = audit_response_model(model)
    original_actions = saved["baseline_actions"]
    new_actions = [row["original_action_index"] for row in report["rows"]]
    if new_actions != original_actions:
        raise ValueError(f"{environment}: exported decimal argmax differs from the saved baseline")
    certified = [row["certified_unchanged"] for row in report["rows"]]
    original_certified = saved["certified_unchangeable"]
    if any(old and not new for old, new in zip(original_certified, certified)):
        raise ValueError(f"{environment}: supplied-decimal audit contradicts an original unchanged certificate")
    return model, report, {
        "environment": environment, "pair_budget": PAIR_BUDGET, "n_queries": len(ids),
        "certified_unchanged": report["counts"]["certified_unchanged"],
        "potentially_changed": report["counts"]["potentially_changed"],
        "unchanged_fraction": report["unchanged_fraction"]["value"],
        "mean_utility_change_abs_upper_bound": report["mean_utility_change_bound"]["absolute_upper_bound"]["value"],
        "original_unchanged_fraction": saved["certified_unchangeable_fraction"],
        "original_certificate_matches": certified == original_certified,
        "baseline_actions_match": new_actions == original_actions,
    }


def _verify_sources(results, cache):
    """Verify the upstream hash chain before extracting any model inputs."""
    sensitivity_path = results / "router_sensitivity_provenance.json"
    sensitivity = _read_json(sensitivity_path)
    inputs = {"results/router_sensitivity_provenance.json": sensitivity_path}
    required = {f"results/{name}" for name in ("models.json", "analysis_metadata.json", "paired_outcomes.json")}
    if not required.issubset(sensitivity["input_sha256"]):
        raise ValueError("Saved sensitivity provenance is missing required input hashes")
    for name, expected in sensitivity["input_sha256"].items():
        parts = Path(name).parts
        if len(parts) != 2 or parts[0] not in ("results", "cache") or parts[1] in (".", ".."):
            raise ValueError(f"Unsupported sensitivity input path: {name}")
        path = (results if parts[0] == "results" else cache) / parts[1]
        if _sha256(path) != expected:
            raise ValueError(f"Sensitivity input hash mismatch: {name}")
        inputs[name] = path
    if "router_sensitivity_queries.json" not in sensitivity["output_sha256"]:
        raise ValueError("Saved sensitivity provenance is missing query certificate hash")
    for name, expected in sensitivity["output_sha256"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Unsupported sensitivity output path: {name}")
        path = results / name
        if _sha256(path) != expected:
            raise ValueError(f"Sensitivity output hash mismatch: {name}")
        inputs[f"results/{name}"] = path
    analysis = _read_json(results / "analysis_metadata.json")
    sources = {"learning.py": ROOT / "learning.py", "protocol.json": ROOT / "protocol.json",
               "audit_router_sensitivity.py": ROOT / "audit_router_sensitivity.py"}
    if _sha256(sources["learning.py"]) != analysis["learning_sha256"] or sensitivity["learning_sha256"] != analysis["learning_sha256"]:
        raise ValueError("Frozen learning source hash differs from the saved analysis")
    if _sha256(sources["protocol.json"]) != analysis["protocol_sha256"]:
        raise ValueError("Frozen protocol hash differs from the saved analysis")
    if _read_json(sources["protocol.json"]) != analysis["protocol"]:
        raise ValueError("Saved analysis protocol differs from the frozen protocol contents")
    if _sha256(sources["audit_router_sensitivity.py"]) != sensitivity["code_sha256"]:
        raise ValueError("Sensitivity implementation differs from its saved source hash")
    inputs.update({f"semantic_pilot/{name}": path for name, path in sources.items()})
    return analysis, sensitivity, inputs


def export_response_models(results_dir, cache_dir, *, python_executable=sys.executable):
    results, cache = Path(results_dir), Path(cache_dir)
    analysis, sensitivity, input_paths = _verify_sources(results, cache)
    models = _read_json(results / "models.json")
    details = _read_json(results / "router_sensitivity_queries.json")
    if not details or set(details) != set(analysis["input_record_sha256"]):
        raise ValueError("Sensitivity environments differ from the frozen analysis population")
    prepared, seen_folds = [], set()
    for environment in sorted(details):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", environment):
            raise ValueError("Environment IDs must be safe simple file stems")
        key = f"cache/{environment}_outcomes.json"
        if key not in input_paths:
            raise ValueError(f"Sensitivity provenance is missing the cache hash for {environment}")
        record = _read_json(input_paths[key])
        if _canonical_hash(record) != analysis["input_record_sha256"][environment]:
            raise ValueError(f"{environment}: canonical cache hash differs from the model fitting input")
        if record["name"] != environment or f"{record['corpus_name']}_{record['backend']}" != environment:
            raise ValueError(f"{environment}: cache identity differs from the saved environment")
        fold = f"{record['corpus_name']}::{record['backend']}"
        metadata = models[fold]
        seen_folds.add(fold)
        if metadata["analysis_learning_sha256"] != analysis["learning_sha256"] or metadata["analysis_protocol_sha256"] != analysis["protocol_sha256"]:
            raise ValueError(f"{environment}: fitted model source/protocol hashes differ from the saved analysis")
        if metadata["target_corpus"] != record["corpus_name"] or metadata["backend"] != record["backend"] or record["corpus_name"] in metadata["source_corpora"]:
            raise ValueError(f"{environment}: fitted model source/target identity is inconsistent")
        model, report, row = response_model(record, metadata, details[environment])
        prepared.append((environment, model, report, row))
    if seen_folds != set(models):
        raise ValueError("Saved fitted-model folds differ from the audited environments")

    # Compute all reports and run the real CLI in temporary storage before
    # touching derived outputs. Frozen source files are never changed.
    environment_vars = os.environ.copy()
    environment_vars["PYTHONPATH"] = str(PACKAGE_SOURCE) + (os.pathsep + environment_vars["PYTHONPATH"] if environment_vars.get("PYTHONPATH") else "")
    with tempfile.TemporaryDirectory(prefix="searchprobe-response-") as directory:
        temporary = Path(directory)
        for environment, model, report, row in prepared:
            model_path, report_path = temporary / f"{environment}.model.json", temporary / f"{environment}.audit.json"
            model_path.write_text(_json_text(model), encoding="utf-8")
            completed = subprocess.run([str(python_executable), "-m", "searchprobe", "response-audit", str(model_path), "--output", str(report_path)],
                                       env=environment_vars, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"SearchProbe CLI failed for {environment}: {completed.stderr.strip()}")
            if _read_json(report_path) != report:
                raise RuntimeError(f"SearchProbe CLI/API reports differ for {environment}")
            row.update(cli_matches_api=True, model_path=f"response_models/{environment}.model.json",
                       audit_path=f"response_models/{environment}.audit.json")

    # Fail closed if an output symlink or hard link would replace any source.
    output_dir = results / "response_models"
    destinations = [results / "response_audits.csv", output_dir / "provenance.json"]
    destinations.extend(results / row[key] for _, _, _, row in prepared for key in ("model_path", "audit_path"))
    for destination in destinations:
        if not destination.resolve().is_relative_to(results.resolve()):
            raise ValueError("A derived response output resolves outside the results directory")
        for source in input_paths.values():
            if destination.resolve() == source.resolve() or (destination.exists() and destination.samefile(source)):
                raise ValueError("A derived response output aliases a protected source file")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, files = [], []
    for environment, model, report, row in prepared:
        for path, value in ((row["model_path"], model), (row["audit_path"], report)):
            (results / path).write_text(_json_text(value), encoding="utf-8")
        rows.append(row)
        files.append({**row, "model_sha256": _sha256(results / row["model_path"]),
                      "audit_sha256": _sha256(results / row["audit_path"])})
    with (results / "response_audits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    package_code = {name: _sha256(PACKAGE_SOURCE / "searchprobe" / name) for name in ("responsiveness.py", "decisions.py", "cli.py", "__main__.py", "__init__.py")}
    provenance = {
        "schema_version": 1, "pair_budget": PAIR_BUDGET,
        "scope": "Reproducibility adapter from saved frozen-router scores and B32 bounds to SearchProbe's exact conditional supplied-decimal model. No refitting or new retrieval.",
        "code_sha256": _sha256(Path(__file__)), "searchprobe_code_sha256": package_code,
        "input_sha256": {name: _sha256(path) for name, path in sorted(input_paths.items())},
        "output_sha256": {str(path.relative_to(results)): _sha256(path) for path in destinations if path.name != "provenance.json"},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "uses_target_task_utility_labels": False, "uses_probe_outcomes": False,
        "uses_future_query_features_for_frozen_evaluation": True,
        "input_access": "Outcome-containing cache records are parsed and canonically hashed solely for source integrity. Only environment/corpus/backend identity and tasks.test.ids/features enter calculations. Paired outcomes and optional full-pool outputs are byte-hashed only; their numeric contents are not inspected.",
        "original_sensitivity_used_unselected_probe_outcomes": sensitivity["uses_unselected_probe_outcomes"],
        "original_certificate_check": "Every saved B32 certificate field is reconstructed and compared. Package baseline actions must agree, and every original certified row must remain certified. The original implementation's strict margin/slack rule may certify fewer tie-boundary rows than SearchProbe's exact tie rule.",
        "floating_point": NUMERICAL_SCOPE,
        "independent_box": "Action and row corrections are allowed to vary independently in the exported box. Real shared-budget profiles can occupy a smaller set, so potential responsiveness need not be attainable.",
        "inference_limit": "Conditional statement for these retained rows and supplied scores/bounds; no target-regret or generalization guarantee. The two backends share query populations and are not independent adaptation replicates.",
        "counts": {"models": len(rows), "rows": sum(row["n_queries"] for row in rows),
                   "certified_unchanged": sum(row["certified_unchanged"] for row in rows),
                   "potentially_changed": sum(row["potentially_changed"] for row in rows),
                   "cli_api_matches": sum(row["cli_matches_api"] for row in rows)},
        "files": files,
    }
    (output_dir / "provenance.json").write_text(_json_text(provenance), encoding="utf-8")
    for row in rows:
        print(f"{row['environment']}: {row['certified_unchanged']}/{row['n_queries']} unchanged; B{PAIR_BUDGET}; original certificate match={row['original_certificate_matches']}; CLI/API match={row['cli_matches_api']}")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    args = parser.parse_args(argv)
    export_response_models(args.results_dir, args.cache_dir)


if __name__ == "__main__":
    main()
