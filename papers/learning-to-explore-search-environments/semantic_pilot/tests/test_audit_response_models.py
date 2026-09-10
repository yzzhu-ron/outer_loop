"""Provenance, source-boundary, and CLI tests for the response-model adapter."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_response_models as adapter
from audit_router_sensitivity import audit_environment, main as original_audit_main
from learning import ACTIONS


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def model_metadata(scores=None, coefficients=None):
    coefficients = np.zeros((5, 4)) if coefficients is None else coefficients
    return {
        "calibrator_intercept": False, "profile_shrinkage": 4.0,
        "calibrator_coefficients": {action: list(coefficients[i]) for i, action in enumerate(ACTIONS[1:], 1)},
        "base_router_parameters": {"type": "fixed", "scores": [0.8, 0.4, 0.2, 0.1, 0] if scores is None else scores},
        "target_corpus": "fixture", "backend": "bm25", "source_corpora": ["source-only"],
        "analysis_learning_sha256": digest(ROOT / "learning.py"),
        "analysis_protocol_sha256": digest(ROOT / "protocol.json"),
    }


def record():
    return {"name": "fixture_bm25", "corpus_name": "fixture", "backend": "bm25",
            "tasks": {"test": {"ids": ["q1", "q2", "q3"], "features": [[0, 1], [1, 0], [0, 0]],
                               "buckets": [0, 1, 2], "scores": "TARGET_UTILITY_MUST_NOT_BE_USED", "recalls": "forbidden"}},
            "pools": "UNSELECTED_PROBES_MUST_NOT_BE_USED"}


def fixture(root):
    results, cache = root / "results", root / "cache"
    results.mkdir()
    cache.mkdir()
    source, metadata = record(), model_metadata()
    write_json(cache / "fixture_bm25_outcomes.json", source)
    write_json(results / "models.json", {"fixture::bm25": metadata})
    write_json(results / "paired_outcomes.json", {"fixture_bm25": {"query_ids": source["tasks"]["test"]["ids"], "source_router_actions": [0, 0, 0]}})
    write_json(results / "analysis_metadata.json", {
        "learning_sha256": metadata["analysis_learning_sha256"], "protocol_sha256": metadata["analysis_protocol_sha256"],
        "protocol": json.loads((ROOT / "protocol.json").read_text()),
        "input_record_sha256": {"fixture_bm25": adapter._canonical_hash(source)},
    })
    with redirect_stdout(StringIO()):
        original_audit_main(["--results-dir", str(results), "--cache-dir", str(cache)])
    return results, cache


def refresh_upstream_hash(results, key, path, *, output=False):
    target = results / "router_sensitivity_provenance.json"
    provenance = json.loads(target.read_text())
    provenance["output_sha256" if output else "input_sha256"][key] = digest(path)
    write_json(target, provenance)


class ResponseAdapterTests(unittest.TestCase):
    def test_real_cli_roundtrip_verifies_inputs_and_preserves_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache = fixture(Path(directory))
            source_hashes = {path: digest(path) for folder in (results, cache) for path in folder.glob("*")}
            with redirect_stdout(StringIO()):
                provenance = adapter.export_response_models(results, cache)
            self.assertEqual(provenance["counts"], {"models": 1, "rows": 3, "certified_unchanged": 3, "potentially_changed": 0, "cli_api_matches": 1})
            self.assertFalse(provenance["uses_target_task_utility_labels"])
            self.assertFalse(provenance["uses_probe_outcomes"])
            self.assertIn("not a formal whole-router", provenance["floating_point"])
            self.assertTrue(provenance["files"][0]["original_certificate_matches"])
            self.assertTrue(provenance["files"][0]["baseline_actions_match"])
            for path, expected in source_hashes.items():
                self.assertEqual(digest(path), expected)
            for name, expected in provenance["output_sha256"].items():
                self.assertEqual(digest(results / name), expected)
            with (results / "response_audits.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["certified_unchanged"], "3")
            self.assertEqual(rows[0]["mean_utility_change_abs_upper_bound"], "0.0")
            model = json.loads((results / rows[0]["model_path"]).read_text())
            self.assertFalse(model["provenance"]["synthetic"])
            self.assertEqual(model["row_ids"], ["q1", "q2", "q3"])
            self.assertIsInstance(model["base_scores"][0][0], str)
            for file in (results / "response_models").glob("*.json"):
                self.assertNotIn("TARGET_UTILITY_MUST_NOT_BE_USED", file.read_text())

    def test_calculation_never_accesses_labels_or_probe_observations(self):
        class Guard(dict):
            def __getitem__(self, key):
                if key in ("scores", "recalls", "qrels", "pools", "buckets", "train"):
                    raise AssertionError(f"Accessed forbidden input {key}")
                return super().__getitem__(key)

        source = record()
        metadata = model_metadata()
        saved = audit_environment(source, metadata)
        source["tasks"]["test"] = Guard(source["tasks"]["test"])
        source["tasks"] = Guard(source["tasks"])
        _, report, row = adapter.response_model(Guard(source), metadata, saved)
        self.assertEqual(report["counts"]["certified_unchanged"], 3)
        self.assertTrue(row["original_certificate_matches"])

    def test_old_conservative_tie_rule_can_be_less_tight_than_exact_box(self):
        coefficients = np.zeros((5, 4))
        # 32/(32+4) * 0.125 is rounded upward; use that saved bound as
        # the exact base margin, so the earlier action survives the tie.
        coefficients[1, 0] = 0.125
        bound = float(np.nextafter(0.125 * (32 / 36), np.inf))
        metadata = model_metadata([bound, 0, -1, -1, -1], coefficients)
        source = record()
        saved = audit_environment(source, metadata)
        self.assertEqual(saved["queries"]["32"]["certified_unchangeable"], [False] * 3)
        _, report, row = adapter.response_model(source, metadata, saved)
        self.assertEqual(report["counts"]["certified_unchanged"], 3)
        self.assertFalse(row["original_certificate_matches"])

    def test_changed_source_bytes_are_rejected_before_any_export(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache = fixture(Path(directory))
            path = cache / "fixture_bm25_outcomes.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "input hash mismatch"):
                adapter.export_response_models(results, cache)
            self.assertFalse((results / "response_models").exists())
            self.assertFalse((results / "response_audits.csv").exists())

    def test_changed_canonical_record_cannot_be_hidden_by_new_raw_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache = fixture(Path(directory))
            path = cache / "fixture_bm25_outcomes.json"
            source = json.loads(path.read_text())
            source["tasks"]["test"]["features"][0][0] = 5
            write_json(path, source)
            refresh_upstream_hash(results, "cache/fixture_bm25_outcomes.json", path)
            with self.assertRaisesRegex(ValueError, "canonical cache hash"):
                adapter.export_response_models(results, cache)

    def test_saved_certificate_must_reconstruct_even_if_output_hash_is_updated(self):
        for field, value in (("action_correction_abs_bounds", [0, 0.5, 0, 0, 0]),
                             ("baseline_actions", [1, 1, 1]), ("certified_unchangeable", [False] * 3)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                results, cache = fixture(Path(directory))
                path = results / "router_sensitivity_queries.json"
                details = json.loads(path.read_text())
                details["fixture_bm25"]["queries"]["32"][field] = value
                write_json(path, details)
                refresh_upstream_hash(results, path.name, path, output=True)
                with self.assertRaisesRegex(ValueError, f"saved B32 certificate differs on {field}"):
                    adapter.export_response_models(results, cache)

    def test_frozen_code_and_protocol_provenance_are_checked(self):
        for field in ("learning_sha256", "protocol_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                results, cache = fixture(Path(directory))
                path = results / "analysis_metadata.json"
                analysis = json.loads(path.read_text())
                analysis[field] = "0" * 64
                write_json(path, analysis)
                refresh_upstream_hash(results, "results/analysis_metadata.json", path)
                with self.assertRaisesRegex(ValueError, "Frozen .* hash"):
                    adapter.export_response_models(results, cache)
        with tempfile.TemporaryDirectory() as directory:
            results, cache = fixture(Path(directory))
            path = results / "router_sensitivity_provenance.json"
            provenance = json.loads(path.read_text())
            provenance["code_sha256"] = "0" * 64
            write_json(path, provenance)
            with self.assertRaisesRegex(ValueError, "Sensitivity implementation"):
                adapter.export_response_models(results, cache)

    def test_missing_or_changed_environment_population_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache = fixture(Path(directory))
            path = results / "router_sensitivity_queries.json"
            write_json(path, {})
            refresh_upstream_hash(results, path.name, path, output=True)
            with self.assertRaisesRegex(ValueError, "environments differ"):
                adapter.export_response_models(results, cache)

    def test_fitted_model_source_target_identity_and_hashes_are_checked(self):
        for update in ({"analysis_protocol_sha256": "bad"}, {"target_corpus": "other"},
                       {"source_corpora": ["fixture"]}):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as directory:
                results, cache = fixture(Path(directory))
                path = results / "models.json"
                models = json.loads(path.read_text())
                models["fixture::bm25"].update(update)
                write_json(path, models)
                refresh_upstream_hash(results, "results/models.json", path)
                with self.assertRaisesRegex(ValueError, "fitted model"):
                    adapter.export_response_models(results, cache)

    def test_query_order_and_shrinkage_drift_are_rejected(self):
        source, metadata = record(), model_metadata()
        saved = audit_environment(source, metadata)
        reordered = deepcopy(source)
        reordered["tasks"]["test"]["ids"].reverse()
        with self.assertRaisesRegex(ValueError, "query IDs/order"):
            adapter.response_model(reordered, metadata, saved)
        metadata["profile_shrinkage"] = 8
        with self.assertRaisesRegex(ValueError, "shrinkage"):
            adapter.response_model(source, metadata, saved)

    def test_output_symlink_cannot_overwrite_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            results, cache = fixture(Path(directory))
            source = results / "models.json"
            expected = digest(source)
            (results / "response_audits.csv").symlink_to(source)
            with self.assertRaisesRegex(ValueError, "protected source file"):
                adapter.export_response_models(results, cache)
            self.assertEqual(digest(source), expected)
            self.assertFalse((results / "response_models").exists())

    def test_strict_json_and_untrusted_provenance_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for body in ('{"key":1,"key":2}', '{"value":NaN}'):
                path.write_text(body)
                with self.assertRaises(ValueError):
                    adapter._read_json(path)
            results, cache = fixture(Path(directory))
            path = results / "router_sensitivity_provenance.json"
            provenance = json.loads(path.read_text())
            provenance["input_sha256"]["results/../../outside.json"] = "anything"
            write_json(path, provenance)
            with self.assertRaisesRegex(ValueError, "Unsupported sensitivity input path"):
                adapter.export_response_models(results, cache)


if __name__ == "__main__":
    unittest.main()
