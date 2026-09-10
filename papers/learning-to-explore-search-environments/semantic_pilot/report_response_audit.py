"""Read-only checks connecting report claims to complete response-audit evidence."""

import csv
import hashlib
import json
from pathlib import Path

from audit_response_models import PAIR_BUDGET, _canonical_hash, _read_json, response_model
from searchprobe import responsiveness


# The frozen exporter records these response-related modules. This is not a
# claim to cover every transitive import (e.g. the separate paired-log audit).
PACKAGE_FILES = {"responsiveness.py", "decisions.py", "cli.py", "__main__.py", "__init__.py"}
CSV_FIELDS = (
    "environment", "pair_budget", "n_queries", "certified_unchanged", "potentially_changed",
    "unchanged_fraction", "mean_utility_change_abs_upper_bound", "original_unchanged_fraction",
    "original_certificate_matches", "baseline_actions_match", "cli_matches_api", "model_path", "audit_path",
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError("Response evidence: " + message)


def _same(left, right):
    # JSON distinguishes booleans from counts and preserves declared types.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate_response_evidence(root, results):
    """Return six validated CSV rows; never write files or inspect utilities.

    Caches are parsed/canonically hashed for provenance. Only their identities
    and test query IDs/features enter reconstruction; paired utility files and
    full-pool diagnostics are byte-hashed without parsing their contents.
    """
    try:
        return _validate_response_evidence(Path(root), Path(results))
    except (KeyError, TypeError, OSError, IndexError) as exc:
        raise ValueError(f"Response evidence is missing or malformed: {exc}") from exc


def _validate_response_evidence(root, results):
    root, results = Path(root), Path(results)
    protocol = _read_json(root / "protocol.json")
    names = {f"{corpus}_{backend}": (corpus, backend) for corpus in protocol["datasets"] for backend in ("bm25", "dense")}
    _require(len(names) == 6 and PAIR_BUDGET in protocol["probes"]["pair_budgets"], "main protocol must define six environments and B32")
    provenance = _read_json(results / "response_models" / "provenance.json")
    sensitivity = _read_json(results / "router_sensitivity_provenance.json")
    _require(type(provenance.get("schema_version")) is int and provenance["schema_version"] == 1
             and type(provenance.get("pair_budget")) is int and provenance["pair_budget"] == PAIR_BUDGET, "unsupported provenance schema/budget")
    for key, expected in (("uses_target_task_utility_labels", False), ("uses_probe_outcomes", False),
                          ("uses_future_query_features_for_frozen_evaluation", True)):
        _require(provenance.get(key) is expected, f"incorrect scope flag {key}")
    _require(type(sensitivity.get("uses_unselected_probe_outcomes")) is bool, "missing original sensitivity scope")
    _require(provenance.get("original_sensitivity_used_unselected_probe_outcomes") is sensitivity["uses_unselected_probe_outcomes"], "original sensitivity scope mismatch")
    _require(isinstance(provenance.get("floating_point"), str) and "not a formal whole-router" in provenance["floating_point"], "missing numerical scope limitation")

    original_inputs = {"results/models.json", "results/paired_outcomes.json", "results/analysis_metadata.json",
                       *{f"cache/{name}_outcomes.json" for name in names}}
    original_outputs = {"router_sensitivity.csv", "router_sensitivity_queries.json"}
    if sensitivity["uses_unselected_probe_outcomes"]:
        original_outputs.add("full_pool_sensitivity.csv")
    _require(set(sensitivity["input_sha256"]) == original_inputs and set(sensitivity["output_sha256"]) == original_outputs,
             "incomplete original sensitivity provenance")
    expected_inputs = original_inputs | {f"results/{name}" for name in original_outputs} | {
        "results/router_sensitivity_provenance.json", "semantic_pilot/learning.py",
        "semantic_pilot/protocol.json", "semantic_pilot/audit_router_sensitivity.py"}
    expected_outputs = {"response_audits.csv", *{f"response_models/{name}.{kind}.json" for name in names for kind in ("model", "audit")}}
    _require(set(provenance["input_sha256"]) == expected_inputs, "incomplete input hashes")
    _require(set(provenance["output_sha256"]) == expected_outputs, "incomplete output hashes")
    _require(set(provenance["searchprobe_code_sha256"]) == PACKAGE_FILES, "incomplete package-code hashes")
    checks = [(root / "audit_response_models.py", provenance["code_sha256"])]
    for name, expected in provenance["input_sha256"].items():
        prefix, suffix = name.split("/", 1)
        path = (results if prefix == "results" else root / "cache" if prefix == "cache" else root) / suffix
        checks.append((path, expected))
    checks.extend((results / name, expected) for name, expected in provenance["output_sha256"].items())
    package = Path(responsiveness.__file__).resolve().parent
    checks.extend((package / name, expected) for name, expected in provenance["searchprobe_code_sha256"].items())
    for path, expected in checks:
        _require(_hash(path) == expected, "hash drift: " + path.name)
    _require(all(provenance["input_sha256"][name] == value for name, value in sensitivity["input_sha256"].items())
             and all(provenance["input_sha256"]["results/" + name] == value for name, value in sensitivity["output_sha256"].items()),
             "upstream hash chain mismatch")
    _require(provenance["input_sha256"]["semantic_pilot/audit_router_sensitivity.py"] == sensitivity["code_sha256"], "original sensitivity code hash mismatch")

    analysis = _read_json(results / "analysis_metadata.json")
    _require(set(analysis["input_record_sha256"]) == set(names), "analysis environment population mismatch")
    _require(analysis["learning_sha256"] == sensitivity["learning_sha256"] == provenance["input_sha256"]["semantic_pilot/learning.py"]
             and analysis["protocol_sha256"] == provenance["input_sha256"]["semantic_pilot/protocol.json"]
             and _same(analysis["protocol"], protocol), "frozen analysis source/protocol mismatch")
    details, fitted = _read_json(results / "router_sensitivity_queries.json"), _read_json(results / "models.json")
    _require(set(details) == set(names) and set(fitted) == {f"{corpus}::{backend}" for corpus, backend in names.values()}, "model/certificate environment grid mismatch")
    files = provenance["files"]
    _require(len(files) == 6 and {item["environment"] for item in files} == set(names), "partial or duplicate provenance file grid")
    with (results / "response_audits.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == CSV_FIELDS, "unexpected summary CSV columns")
        rows = list(reader)
    _require(len(rows) == 6 and {row["environment"] for row in rows} == set(names), "partial or duplicate CSV environment grid")
    with (results / "router_sensitivity.csv").open(newline="", encoding="utf-8") as handle:
        original_rows = [row for row in csv.DictReader(handle) if row["pair_budget"] == str(PAIR_BUDGET)]
    _require(len(original_rows) == 6 and {row["environment"] for row in original_rows} == set(names), "original B32 CSV environment grid mismatch")
    total, unchanged = 0, 0
    for row in rows:
        name = row["environment"]
        corpus, backend = names[name]
        record = _read_json(root / "cache" / f"{name}_outcomes.json")
        _require(_canonical_hash(record) == analysis["input_record_sha256"][name], "canonical cache drift: " + name)
        _require(record["name"] == name and record["corpus_name"] == corpus and record["backend"] == backend, "cache identity mismatch: " + name)
        metadata = fitted[f"{corpus}::{backend}"]
        sources, expected_sources = metadata["source_corpora"], set(protocol["datasets"]) - {corpus}
        _require(metadata["target_corpus"] == corpus and metadata["backend"] == backend
                 and type(sources) is list and len(sources) == len(expected_sources) and set(sources) == expected_sources
                 and metadata["analysis_learning_sha256"] == analysis["learning_sha256"]
                 and metadata["analysis_protocol_sha256"] == analysis["protocol_sha256"], "fitted model provenance mismatch: " + name)
        expected_model, expected_report, expected_row = response_model(record, metadata, details[name])
        count = protocol["query_selection"]["test_counts"][corpus]
        _require(type(count) is int and expected_row["n_queries"] == count, "protocol query cohort mismatch: " + name)
        _require(expected_model["actions"] == protocol["generation"]["action_order"], "protocol action order mismatch: " + name)
        model_path, audit_path = f"response_models/{name}.model.json", f"response_models/{name}.audit.json"
        model, audit = _read_json(results / model_path), _read_json(results / audit_path)
        _require(_same(model, expected_model), "exported model differs from source reconstruction: " + name)
        _require(_same(audit, expected_report) and _same(audit, responsiveness.audit_response_json(results / model_path)), "saved audit differs from actual model/API: " + name)
        original = details[name]["queries"][str(PAIR_BUDGET)]
        _require(type(original["n_queries"]) is int and original["n_queries"] == count and original["query_ids"] == model["row_ids"]
                 and _same(original["certified_unchangeable"], [item["certified_unchanged"] for item in audit["rows"]]),
                 "original B32 cohort/certificate mismatch: " + name)
        original_summary = {"environment": name, "pair_budget": PAIR_BUDGET, "n_queries": count,
                            **{key: original[key] for key in ("profile_feature_abs_bound", "certified_unchangeable_fraction",
                               "changeable_query_fraction_upper_bound", "mean_bounded_utility_change_abs_upper_bound")},
                            **{f"{action}_correction_abs_bound": original["action_correction_abs_bounds"][i]
                               for i, action in enumerate(model["actions"])}}
        _require(next(item for item in original_rows if item["environment"] == name)
                 == {key: str(value) for key, value in original_summary.items()}, "original B32 CSV differs from reconstructed certificate: " + name)
        _require(expected_row["original_certificate_matches"] is True and expected_row["baseline_actions_match"] is True,
                 "original certificate/baseline flags are false: " + name)
        expected_row.update(cli_matches_api=True, model_path=model_path, audit_path=audit_path)
        _require(row == {key: str(expected_row[key]) for key in CSV_FIELDS}, "summary counts/flags differ from evidence: " + name)
        file_record = next(item for item in files if item["environment"] == name)
        expected_file = {**expected_row, "model_sha256": provenance["output_sha256"][model_path],
                         "audit_sha256": provenance["output_sha256"][audit_path]}
        _require(_same(file_record, expected_file), "per-file provenance differs from evidence: " + name)
        total += count
        unchanged += expected_row["certified_unchanged"]
    _require(_same(provenance["counts"], {"models": 6, "rows": total, "certified_unchanged": unchanged,
                                        "potentially_changed": total - unchanged, "cli_api_matches": 6}), "aggregate counts mismatch")
    return sorted(rows, key=lambda row: row["environment"])
