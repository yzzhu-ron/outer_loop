"""Offline synthetic checks that stale response evidence cannot enter a report."""

from contextlib import redirect_stdout
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_response_models as adapter
import audit_router_sensitivity as original
import report_response_audit as validator


SOURCE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, *, rows_override=None):
    results, cache = root / "results", root / "cache"
    results.mkdir()
    cache.mkdir()
    for name in ("protocol.json", "learning.py", "audit_router_sensitivity.py", "audit_response_models.py"):
        shutil.copyfile(SOURCE / name, root / name)
    protocol = json.loads((root / "protocol.json").read_text())
    models, paired, hashes = {}, {}, {}
    for corpus in protocol["datasets"]:
        count = protocol["query_selection"]["test_counts"][corpus] if rows_override is None else rows_override
        for backend in ("bm25", "dense"):
            name = f"{corpus}_{backend}"
            ids = [f"synthetic-{corpus}-{i}" for i in range(count)]
            record = {"name": name, "corpus_name": corpus, "backend": backend,
                      "tasks": {"test": {"ids": ids, "features": [[0, 1]] * count, "buckets": [0] * count,
                                         "scores": "FORBIDDEN_TARGET_UTILITY", "recalls": "FORBIDDEN_TARGET_RECALL"}},
                      "pools": "FORBIDDEN_PROBE_OBSERVATIONS"}
            write_json(cache / f"{name}_outcomes.json", record)
            hashes[name] = adapter._canonical_hash(record)
            paired[name] = {"query_ids": ids, "source_router_actions": [0] * count}
            models[f"{corpus}::{backend}"] = {
                "calibrator_intercept": False, "profile_shrinkage": 4.0,
                "calibrator_coefficients": {action: [0, 0, 0, 0] for action in original.ACTIONS[1:]},
                "base_router_parameters": {"type": "fixed", "scores": [0.8, 0.4, 0.2, 0.1, 0]},
                "target_corpus": corpus, "backend": backend,
                "source_corpora": sorted(set(protocol["datasets"]) - {corpus}),
                "analysis_learning_sha256": digest(root / "learning.py"),
                "analysis_protocol_sha256": digest(root / "protocol.json"),
            }
    write_json(results / "models.json", models)
    write_json(results / "paired_outcomes.json", paired)
    write_json(results / "analysis_metadata.json", {
        "learning_sha256": digest(root / "learning.py"), "protocol_sha256": digest(root / "protocol.json"),
        "protocol": protocol, "input_record_sha256": hashes,
    })
    # Only temporary synthetic evidence is generated. The validator itself
    # must neither regenerate an output nor call the command-line tool.
    with redirect_stdout(StringIO()), patch.object(original, "ROOT", root):
        original.main(["--results-dir", str(results), "--cache-dir", str(cache)])
    with redirect_stdout(StringIO()), patch.object(adapter, "ROOT", root):
        adapter.export_response_models(results, cache)
    return results


def update_output_hash(results, path):
    target = results / "response_models" / "provenance.json"
    provenance = json.loads(target.read_text())
    relative = str(path.relative_to(results))
    provenance["output_sha256"][relative] = digest(path)
    for item in provenance["files"]:
        for kind in ("model", "audit"):
            if item[f"{kind}_path"] == relative:
                item[f"{kind}_sha256"] = digest(path)
    write_json(target, provenance)


def mutate_csv(results, update):
    path = results / "response_audits.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    update(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    update_output_hash(results, path)


class ResponseReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = fixture(self.root)
        self.provenance_path = self.results / "response_models" / "provenance.json"

    def validate(self):
        return validator.validate_response_evidence(self.root, self.results)

    def test_complete_six_environment_evidence_validates_without_writes_or_cli(self):
        before = {path: digest(path) for path in self.root.rglob("*") if path.is_file()}
        with patch.object(adapter.subprocess, "run", side_effect=AssertionError("Validator called CLI")), \
                patch.object(Path, "write_text", side_effect=AssertionError("Validator wrote a file")):
            rows = self.validate()
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(int(row["n_queries"]) for row in rows), 730)
        self.assertTrue(all(row["unchanged_fraction"] == "1.0" for row in rows))
        self.assertEqual({path: digest(path) for path in self.root.rglob("*") if path.is_file()}, before)

    def test_missing_input_output_or_package_hashes_fail_closed(self):
        pristine = self.provenance_path.read_text()
        for field in ("input_sha256", "output_sha256", "searchprobe_code_sha256"):
            value = json.loads(pristine)
            value[field].pop(next(iter(value[field])))
            write_json(self.provenance_path, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "incomplete"):
                self.validate()

    def test_missing_provenance_field_has_a_clear_validation_error(self):
        provenance = json.loads(self.provenance_path.read_text())
        del provenance["input_sha256"]
        write_json(self.provenance_path, provenance)
        with self.assertRaisesRegex(ValueError, "missing or malformed"):
            self.validate()

    def test_package_or_adapter_code_drift_is_rejected(self):
        pristine = self.provenance_path.read_text()
        for field in ("package", "adapter"):
            value = json.loads(pristine)
            if field == "package":
                value["searchprobe_code_sha256"]["responsiveness.py"] = "0" * 64
            else:
                value["code_sha256"] = "0" * 64
            write_json(self.provenance_path, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "hash drift"):
                self.validate()

    def test_changed_cache_bytes_are_rejected(self):
        path = self.root / "cache" / "fiqa_bm25_outcomes.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash drift"):
            self.validate()

    def test_modified_audit_is_recomputed_even_when_its_hash_is_updated(self):
        path = self.results / "response_models" / "fiqa_bm25.audit.json"
        report = json.loads(path.read_text())
        report["counts"]["certified_unchanged"] -= 1
        write_json(path, report)
        update_output_hash(self.results, path)
        with self.assertRaisesRegex(ValueError, "actual model/API"):
            self.validate()

    def test_internally_consistent_model_and_audit_must_match_fitted_inputs(self):
        model_path = self.results / "response_models" / "fiqa_bm25.model.json"
        audit_path = self.results / "response_models" / "fiqa_bm25.audit.json"
        model = json.loads(model_path.read_text())
        model["base_scores"][0][0] = "0.79"
        write_json(model_path, model)
        write_json(audit_path, validator.responsiveness.audit_response_json(model_path))
        update_output_hash(self.results, model_path)
        update_output_hash(self.results, audit_path)
        with self.assertRaisesRegex(ValueError, "source reconstruction"):
            self.validate()

    def test_duplicate_csv_environments_cannot_hide_a_missing_environment(self):
        mutate_csv(self.results, lambda rows: rows.__setitem__(-1, dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "CSV environment grid"):
            self.validate()

    def test_summary_counts_and_comparison_flags_are_checked_against_evidence(self):
        path = self.results / "response_audits.csv"
        pristine = path.read_text()
        for key, value in (("n_queries", "999"), ("certified_unchanged", "0"), ("cli_matches_api", "False")):
            path.write_text(pristine)
            mutate_csv(self.results, lambda rows: rows[0].__setitem__(key, value))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "summary counts/flags"):
                self.validate()

    def test_per_file_metadata_and_aggregate_counts_are_verified(self):
        pristine = self.provenance_path.read_text()
        for field in ("files", "counts"):
            provenance = json.loads(pristine)
            if field == "files":
                provenance["files"][0]["cli_matches_api"] = 1
            else:
                provenance["counts"]["rows"] += 1
            write_json(self.provenance_path, provenance)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "per-file provenance|aggregate counts"):
                self.validate()

    def test_original_b32_certificate_is_reconstructed_after_hash_updates(self):
        path = self.results / "router_sensitivity_queries.json"
        details = json.loads(path.read_text())
        details["fiqa_bm25"]["queries"]["32"]["certified_unchangeable"][0] = False
        write_json(path, details)
        upstream_path = self.results / "router_sensitivity_provenance.json"
        upstream = json.loads(upstream_path.read_text())
        upstream["output_sha256"][path.name] = digest(path)
        write_json(upstream_path, upstream)
        provenance = json.loads(self.provenance_path.read_text())
        provenance["input_sha256"]["results/" + path.name] = digest(path)
        provenance["input_sha256"]["results/" + upstream_path.name] = digest(upstream_path)
        write_json(self.provenance_path, provenance)
        with self.assertRaisesRegex(ValueError, "saved B32 certificate differs"):
            self.validate()

    def test_original_b32_display_csv_must_match_the_reconstructed_certificate(self):
        path = self.results / "router_sensitivity.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        row = next(row for row in rows if row["environment"] == "fiqa_bm25" and row["pair_budget"] == "32")
        row["certified_unchangeable_fraction"] = "0.0"
        row["changeable_query_fraction_upper_bound"] = "1.0"
        row["mean_bounded_utility_change_abs_upper_bound"] = "1.0"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        upstream_path = self.results / "router_sensitivity_provenance.json"
        upstream = json.loads(upstream_path.read_text())
        upstream["output_sha256"][path.name] = digest(path)
        write_json(upstream_path, upstream)
        provenance = json.loads(self.provenance_path.read_text())
        provenance["input_sha256"]["results/" + path.name] = digest(path)
        provenance["input_sha256"]["results/" + upstream_path.name] = digest(upstream_path)
        write_json(self.provenance_path, provenance)
        with self.assertRaisesRegex(ValueError, "original B32 CSV differs"):
            self.validate()

    def test_fitted_source_set_must_include_exactly_the_other_two_corpora(self):
        path = self.results / "models.json"
        pristine = path.read_text()
        for sources in ([], ["scifact"], ["scifact", "scifact"], ["scifact", "unrelated"]):
            models = json.loads(pristine)
            models["fiqa::bm25"]["source_corpora"] = sources
            write_json(path, models)
            upstream_path = self.results / "router_sensitivity_provenance.json"
            upstream = json.loads(upstream_path.read_text())
            upstream["input_sha256"]["results/models.json"] = digest(path)
            write_json(upstream_path, upstream)
            provenance = json.loads(self.provenance_path.read_text())
            provenance["input_sha256"]["results/models.json"] = digest(path)
            provenance["input_sha256"]["results/router_sensitivity_provenance.json"] = digest(upstream_path)
            write_json(self.provenance_path, provenance)
            with self.subTest(sources=sources), self.assertRaisesRegex(ValueError, "fitted model provenance"):
                self.validate()

    def test_protocol_query_counts_are_required_even_for_consistent_small_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = fixture(root, rows_override=1)
            with self.assertRaisesRegex(ValueError, "protocol query cohort"):
                validator.validate_response_evidence(root, results)

    def test_calculation_does_not_access_target_utilities_or_probe_outcomes(self):
        read_json = validator._read_json

        class Guard(dict):
            def __getitem__(self, key):
                if key in ("scores", "recalls", "qrels", "pools", "buckets", "train"):
                    raise AssertionError(f"Accessed forbidden input {key}")
                return super().__getitem__(key)

        def guarded(path):
            value = read_json(path)
            if Path(path).parent.name == "cache":
                value["tasks"]["test"] = Guard(value["tasks"]["test"])
                value["tasks"] = Guard(value["tasks"])
                value = Guard(value)
            if Path(path).name == "paired_outcomes.json":
                raise AssertionError("Validator parsed paired utility outcomes")
            return value

        with patch.object(validator, "_read_json", side_effect=guarded):
            self.assertEqual(len(self.validate()), 6)


if __name__ == "__main__":
    unittest.main()
