"""Mathematical and information-boundary tests for the frozen-router audit."""

import contextlib
import csv
import hashlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_router_sensitivity import audit_environment, base_predictions, change_certificate, coefficient_matrix, main
from learning import ACTIONS, Profile


def metadata(coefficients=None, scores=None):
    coefficients = np.zeros((5, 4)) if coefficients is None else np.asarray(coefficients)
    return {"calibrator_intercept": False, "profile_shrinkage": 4.0,
            "calibrator_coefficients": {action: coefficients[i].tolist() for i, action in enumerate(ACTIONS[1:], 1)},
            "base_router_parameters": {"type": "fixed", "scores": [0.5, 0.4, 0.2, 0.1, 0.0] if scores is None else scores}}


def candidate(identifier, action=1, family="direct_question", bucket=0):
    return {"id": identifier, "action": action, "family": family, "bucket": bucket, "features": [0.0] * 8, "cost": 2}


class CertificateTests(unittest.TestCase):
    def test_zero_budget_and_zero_coefficients_preserve_tied_actions(self):
        predictions = np.zeros((3, 5))
        coefficients = np.ones((5, 4))
        coefficients[0] = 0
        self.assertEqual(change_certificate(predictions, coefficients, 0)["certified_unchangeable"], [True] * 3)
        self.assertEqual(change_certificate(predictions, np.zeros((5, 4)), 32)["certified_unchangeable"], [True] * 3)

    def test_absolute_bound_matches_coefficient_l1_formula(self):
        coefficients = np.zeros((5, 4))
        coefficients[1] = [1, -2, 3, -4]
        result = change_certificate(np.array([[1, 0, 0, 0, 0]]), coefficients, 4)
        self.assertEqual(result["profile_feature_abs_bound"], 0.5)
        self.assertAlmostEqual(result["action_correction_abs_bounds"][1], 5)
        self.assertEqual(result["action_correction_abs_bounds"][0], 0)

    def test_winner_can_move_down_even_when_challenger_is_original(self):
        coefficients = np.zeros((5, 4))
        coefficients[1, 0] = 1
        base = np.array([[0.5, 0.6, 0.0, 0.0, 0.0]])
        result = change_certificate(base, coefficients, 1)
        self.assertEqual(result["certified_unchangeable"], [False])
        profile = Profile()
        profile.update(candidate("negative"), {"delta": -1})
        corrected = base.copy()
        corrected[:, 1] += profile.routing_features(1, [0]) @ coefficients[1]
        self.assertEqual(corrected.argmax(axis=1).tolist(), [0])

    def test_large_margin_certificate_and_budget_monotonicity(self):
        coefficients = np.full((5, 4), 0.02)
        coefficients[0] = 0
        base = np.array([[1, 0, 0, 0, 0], [0.1, 0.08, 0, 0, 0]])
        previous_bound, previous_uncertified = 0, 0
        for budget in (0, 1, 4, 8, 16, 32):
            result = change_certificate(base, coefficients, budget)
            self.assertTrue(result["certified_unchangeable"][0])
            self.assertGreaterEqual(result["action_correction_abs_bounds"][1], previous_bound)
            self.assertGreaterEqual(result["changeable_query_fraction_upper_bound"], previous_uncertified)
            previous_bound = result["action_correction_abs_bounds"][1]
            previous_uncertified = result["changeable_query_fraction_upper_bound"]

    def test_all_small_extreme_histories_respect_the_bound_and_certificate(self):
        coefficients = np.array([[0, 0, 0, 0], [0.2, -0.1, 0.1, 0.3], [-0.1, 0.1, 0.1, -0.2], [0.03, 0, 0, 0], [0, 0.02, 0, 0]])
        base = np.array([[0.8, 0.1, 0, 0, 0], [0.2, 0.19, 0.18, 0, 0]])
        options = [(candidate(f"{a}-{f}-{b}-{d}", a, f, b), d)
                   for a in (1, 2) for f in ("direct_question", "indirect_question") for b in (0, 1) for d in (-1, 1)]
        result = change_certificate(base, coefficients, 2)
        bounds = np.asarray(result["action_correction_abs_bounds"])
        certified = np.asarray(result["certified_unchangeable"])
        for first, second in itertools.product(options, repeat=2):
            profile = Profile()
            for c, delta in (first, second):
                profile.update(c, {"delta": delta})
            correction = np.zeros_like(base)
            for action in range(1, 5):
                correction[:, action] = profile.routing_features(action, [0, 1]) @ coefficients[action]
            self.assertTrue(np.all(np.abs(correction) <= bounds[None, :] + 1e-14))
            changed = (base + correction).argmax(axis=1) != base.argmax(axis=1)
            self.assertFalse(np.any(changed & certified))


class EvaluationBoundaryTests(unittest.TestCase):
    def test_saved_scaler_and_coefficients_reconstruct_base_predictions(self):
        model = metadata()
        model["base_router_parameters"] = {"type": "standard_scaler_ridge", "scaler_mean": [1, 2], "scaler_scale": [2, 4],
            "coefficients": [[1, 0], [0, 1], [1, 1], [0, 0], [-1, 0]], "intercept": [0, 0, 0, 0.5, 0]}
        np.testing.assert_allclose(base_predictions(model, [[3, 6]]), [[1, 1, 2, 0.5, -1]])

    def test_task_labels_are_never_accessed(self):
        class LabelGuard(dict):
            def __getitem__(self, key):
                if key in ("scores", "recalls", "rankings", "qrels"):
                    raise AssertionError("Sensitivity diagnostic accessed target outcomes")
                return super().__getitem__(key)

        tasks = LabelGuard(ids=["q"], features=[[0, 1]], buckets=[0], scores="forbidden", recalls="forbidden")
        result = audit_environment({"name": "fixture", "tasks": {"test": tasks}, "pools": {}}, metadata(), expected_baseline=[0])
        self.assertEqual(len(result["bounds"]), 5)
        self.assertTrue(all(row["certified_unchangeable_fraction"] == 1 for row in result["bounds"]))

    def test_full_pool_cancellation_is_not_an_oracle_ceiling(self):
        coefficients = np.zeros((5, 4))
        coefficients[1, 0] = 1
        model = metadata(coefficients)
        record = {"name": "fixture", "tasks": {"test": {"ids": ["q"], "features": [[0, 1]], "buckets": [0]}},
                  "pools": {"11": {"candidates": [candidate("positive"), candidate("negative")],
                                   "observations": {"positive": {"delta": 1}, "negative": {"delta": -1}}}}}
        result = audit_environment(record, model, include_full_pool=True)
        self.assertEqual(result["full_pool"][0]["observed_full_pool_changed_action_fraction"], 0)
        self.assertEqual(result["full_pool"][0]["certified_unchangeable_fraction_at_full_pair_count"], 0)
        profile = Profile()
        profile.update(candidate("positive"), {"delta": 1})
        base = base_predictions(model, [[0, 1]])
        base[:, 1] += profile.routing_features(1, [0]) @ coefficients[1]
        self.assertEqual(base.argmax(axis=1).tolist(), [1])

    def test_unknown_intercept_nonzero_original_or_profile_drift_is_rejected(self):
        model = metadata()
        model["calibrator_intercept"] = True
        with self.assertRaises(ValueError):
            coefficient_matrix(model)
        model = metadata()
        model["calibrator_coefficients"]["original"] = [1, 0, 0, 0]
        with self.assertRaises(ValueError):
            coefficient_matrix(model)
        model = metadata()
        model["profile_shrinkage"] = 8
        with self.assertRaises(ValueError):
            audit_environment({"name": "fixture", "tasks": {"test": {"ids": ["q"], "features": [[0]], "buckets": [0]}}}, model)

    def test_command_writes_certificates_without_numeric_target_labels(self):
        model = metadata()
        record = {"name": "fixture_bm25", "corpus_name": "fixture", "backend": "bm25",
                  "tasks": {"test": {"ids": ["q"], "features": [[0, 1]], "buckets": [0], "scores": "not available", "recalls": "not available"}},
                  "pools": {"11": {"candidates": [candidate("p")], "observations": {"p": {"delta": 1}}}}}
        learning_hash = hashlib.sha256((Path(__file__).resolve().parents[1] / "learning.py").read_bytes()).hexdigest()
        model.update(analysis_learning_sha256=learning_hash, analysis_protocol_sha256="fixture-protocol")
        record_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, cache = root / "results", root / "cache"
            result.mkdir()
            cache.mkdir()
            (result / "models.json").write_text(json.dumps({"fixture::bm25": model}))
            (result / "paired_outcomes.json").write_text(json.dumps({"fixture_bm25": {"query_ids": ["q"], "source_router_actions": [0]}}))
            (result / "analysis_metadata.json").write_text(json.dumps({"learning_sha256": learning_hash, "protocol_sha256": "fixture-protocol", "input_record_sha256": {"fixture_bm25": record_hash}}))
            (cache / "fixture_bm25_outcomes.json").write_text(json.dumps(record))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--results-dir", str(result), "--cache-dir", str(cache), "--include-full-pool"])
            with (result / "router_sensitivity.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(float(row["certified_unchangeable_fraction"]) == 1 for row in rows))
            provenance = json.loads((result / "router_sensitivity_provenance.json").read_text())
            self.assertFalse(provenance["uses_target_task_utility_labels"])
            self.assertTrue(provenance["uses_unselected_probe_outcomes"])
            self.assertTrue((result / "full_pool_sensitivity.csv").exists())
            for name, checksum in provenance['output_sha256'].items():
                self.assertEqual(hashlib.sha256((result / name).read_bytes()).hexdigest(), checksum)
            self.assertEqual(set(provenance['output_sha256']), {'router_sensitivity.csv', 'router_sensitivity_queries.json', 'full_pool_sensitivity.csv'})
            self.assertEqual(set(provenance['input_sha256']), {'results/models.json', 'results/paired_outcomes.json', 'results/analysis_metadata.json', 'cache/fixture_bm25_outcomes.json'})
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--results-dir", str(result), "--cache-dir", str(cache)])
            self.assertFalse((result / 'full_pool_sensitivity.csv').exists())
            provenance = json.loads((result / 'router_sensitivity_provenance.json').read_text())
            self.assertNotIn('full_pool_sensitivity.csv', provenance['output_sha256'])
            self.assertFalse(provenance['uses_unselected_probe_outcomes'])


if __name__ == "__main__":
    unittest.main()
