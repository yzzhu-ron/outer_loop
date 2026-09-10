"""Tests of information boundaries, budget accounting, and source-only fitting."""

import copy
import csv
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning import Acquisition, Profile, _selected_task_costs, analyze_environments, run_onboarding


def candidate(index, *, cost=2):
    return {"id": f"p{index}", "action": index % 4 + 1, "family": ("direct_question", "indirect_question")[(index // 4) % 2], "bucket": (index // 8) % 4, "features": [float(index % 3)] * 8, "cost": cost}


def observation(delta=0.5):
    return {"delta": delta, "rr_original": 0.25, "rr_alternative": 0.25 + delta, "overlap": 0.0}


def synthetic_records():
    records = []
    for family_index, corpus in enumerate(("alpha", "beta", "gamma")):
        for backend in ("bm25", "dense"):
            rng = np.random.default_rng(100 + family_index)
            tasks = {}
            for split, n in (("train", 12), ("calibration", 6), ("utility", 5), ("test", 7)):
                features = rng.normal(size=(n, 10))
                scores = np.clip(0.4 + features[:, :5] * 0.1 + np.arange(5) * (0.01 if family_index % 2 else -0.01), 0, 1)
                costs = [[{"search_calls": 2 if a == 4 else 1, "llm_calls": int(a >= 2), "input_tokens": 20 * int(a >= 2), "output_tokens": 10 * int(a >= 2)} for a in range(5)] for _ in range(n)]
                tasks[split] = {"ids": [f"{corpus}/{split}/{i}" for i in range(n)], "features": features.tolist(), "buckets": [i % 4 for i in range(n)], "scores": scores.tolist(), "recalls": np.clip(scores + 0.2, 0, 1).tolist(), "action_costs": costs}
            candidates = [candidate(i, cost=3 if i % 4 == 3 else 2) for i in range(12)]
            pool = {"candidates": candidates, "observations": {c["id"]: observation((1 if family_index % 2 else -1) * (0.1 + int(c["action"]) / 10)) for c in candidates}, "setup_costs": {"sample_calls": 16, "search_calls": 7, "llm_calls": 12, "input_tokens": 240, "output_tokens": 120, "seconds": 60.0}}
            records.append({"name": f"{corpus}_{backend}", "corpus_name": corpus, "backend": backend, "tasks": tasks, "pools": {"11": pool}})
    return records


class BoundaryTests(unittest.TestCase):
    def test_zero_budget_never_requests_an_observation(self):
        def forbidden(_):
            self.fail("Zero-budget onboarding accessed an observation")
        for method in ("source_router", "random", "fixed", "information_gain", "learned_value"):
            result = run_onboarding([candidate(0)], forbidden, method=method, budget=0)
            self.assertEqual(result["selected_ids"], [])
            self.assertEqual(result["search_calls"], 0)
            self.assertEqual(result["profile"].count.sum(), 0)

    def test_callback_sees_only_selected_candidates_and_no_future_tasks(self):
        calls = []
        candidates = [candidate(i) for i in range(8)]
        def observe(candidate_id):
            calls.append(candidate_id)
            return observation()
        result = run_onboarding(candidates, observe, method="fixed", budget=2)
        self.assertEqual(calls, result["selected_ids"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(set(calls)), 2)
        self.assertNotIn("tasks", inspect.signature(run_onboarding).parameters)
        self.assertNotIn("observations", inspect.signature(run_onboarding).parameters)

    def test_pre_observation_outcome_leak_is_rejected(self):
        bad = candidate(0)
        bad["delta"] = 0.7
        with self.assertRaisesRegex(ValueError, "Outcome fields"):
            run_onboarding([bad], lambda _: observation(), method="learned_value", budget=1)

    def test_probe_count_and_search_cost_caps_are_both_enforced(self):
        candidates = [candidate(0, cost=3), candidate(1, cost=2), candidate(2, cost=2)]
        calls = []
        result = run_onboarding(candidates, lambda x: calls.append(x) or observation(), method="fixed", budget=5, max_search_calls=4)
        self.assertLessEqual(result["search_calls"], 4)
        self.assertEqual(calls, result["selected_ids"])
        affordable = run_onboarding(candidates, lambda _: observation(), method="fixed", budget=5, max_search_calls=2)
        self.assertEqual(affordable["selected_ids"], ["p1"])
        self.assertEqual(affordable["search_calls"], 2)
        self.assertEqual(affordable["costs"]["search_calls"], 2)

    def test_same_seed_preserves_nested_budget_prefixes(self):
        candidates = [candidate(i) for i in range(20)]
        for method in ("random", "fixed", "information_gain", "learned_value"):
            short = run_onboarding(candidates, lambda _: observation(), method=method, budget=4, seed=74)
            long = run_onboarding(candidates, lambda _: observation(), method=method, budget=8, seed=74)
            self.assertEqual(short["selected_ids"], long["selected_ids"][:4])
            self.assertEqual(short["profile"].as_dict(), long["snapshots"][4].as_dict())

    def test_profile_contains_only_numeric_sufficient_statistics(self):
        profile = Profile()
        c = candidate(0)
        c.update(action=3, bucket=2)
        profile.update(c, {**observation(0.6), "document_text": "sensitive example text"})
        self.assertEqual(set(profile.as_dict()), {"count", "sum", "sumsq"})
        self.assertNotIn("sensitive", json.dumps(profile.as_dict()))
        self.assertEqual(profile.count.shape, (2, 4, 4))
        np.testing.assert_allclose(profile.routing_features(3, [2, 0]), [[0.12, 0.12, 0, 0], [0.12, 0, 0, 0]])
        snapshot = profile.copy()
        profile.update(c, observation(-0.2))
        self.assertEqual(snapshot.count.sum(), 1)

    def test_unselected_observations_cannot_change_acquisition(self):
        candidates = [candidate(i) for i in range(6)]
        selected = []
        def observe(candidate_id):
            selected.append(candidate_id)
            # Any accidental acquisition lookahead requests more than the budget.
            self.assertLessEqual(len(selected), 1)
            return observation(0.25)
        result = run_onboarding(candidates, observe, method="information_gain", budget=1, seed=22)
        self.assertEqual(result["selected_ids"], selected)
        self.assertEqual(result["profile"].count.sum(), 1)

    def test_information_gain_reduces_with_repeated_cell_measurements(self):
        acquisition, profile, c = Acquisition(), Profile(), candidate(0)
        before = acquisition.score(c, profile, "information_gain")
        profile.update(c, observation())
        self.assertLess(acquisition.score(c, profile, "information_gain"), before)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = synthetic_records()
        cls.directory = tempfile.TemporaryDirectory()
        cls.result = analyze_environments(cls.records, cls.directory.name, model_cache_dir=Path(cls.directory.name) / "fitted_models")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_cross_family_model_metadata_excludes_target(self):
        metadata = json.loads((Path(self.directory.name) / "models.json").read_text())
        for model in metadata.values():
            self.assertNotIn(model["target_corpus"], model["source_corpora"])
            self.assertEqual(len(model["source_corpora"]), 2)
            self.assertIn(model["selected"], ("fixed", "ridge_10", "ridge_100"))

    def test_setup_and_task_costs_are_accounted_separately(self):
        for row in self.result["adaptation"]:
            if row["budget"] == 0 or row["method"] == "source_router":
                self.assertEqual(row["onboarding_search_calls"], 0)
                self.assertEqual(row["onboarding_llm_calls"], 0)
                self.assertEqual(row["onboarding_sample_calls"], 0)
                self.assertEqual(row["measured_onboarding_generation_seconds"], 0)
                self.assertAlmostEqual(row["delta_vs_source_router"], 0)
                self.assertEqual(row["changed_action_fraction"], 0)
            else:
                self.assertEqual(row["onboarding_llm_calls"], 12)
                self.assertGreaterEqual(row["onboarding_search_calls"], 7 + 2 * row["selected_probe_pairs"])
                self.assertEqual(row["onboarding_input_tokens"], 240)
                self.assertEqual(row["onboarding_sample_calls"], 16)
                self.assertEqual(row["measured_onboarding_generation_seconds"], 60)
            self.assertGreaterEqual(row["mean_task_search_calls"], 1)
            self.assertLessEqual(row["mean_task_search_calls"], 2)

    def test_test_labels_and_queries_cannot_change_probe_selection(self):
        changed = copy.deepcopy(self.records)
        for record in changed:
            tasks = record["tasks"]["test"]
            tasks["scores"] = (1 - np.asarray(tasks["scores"])).tolist()
            tasks["features"] = (np.asarray(tasks["features"]) * -7).tolist()
            tasks["buckets"] = [(b + 1) % 4 for b in tasks["buckets"]]
        with tempfile.TemporaryDirectory() as other:
            analyze_environments(changed, other, model_cache_dir=Path(other) / "fitted_models")
            before = json.loads((Path(self.directory.name) / "profiles.json").read_text())
            after = json.loads((Path(other) / "profiles.json").read_text())
            self.assertEqual(before, after)
            before_models = json.loads((Path(self.directory.name) / "models.json").read_text())
            after_models = json.loads((Path(other) / "models.json").read_text())
            for name in before_models:
                # Model bytes must match; only the local output directory differs.
                before_models[name]["serialized_model"].pop("path")
                after_models[name]["serialized_model"].pop("path")
            self.assertEqual(before_models, after_models)

    def test_saved_scores_reproduce_reported_quality_and_macro(self):
        paired = json.loads((Path(self.directory.name) / "paired_outcomes.json").read_text())
        for row in self.result["adaptation"]:
            key = f"{row['seed']}/{row['method']}/{row['budget']}"
            self.assertAlmostEqual(row["ndcg"], float(np.mean(paired[row["environment"]]["runs"][key]["ndcg"])))
            self.assertAlmostEqual(row["recall_at_10"], float(np.mean(paired[row["environment"]]["runs"][key]["recall_at_10"])))
        for row in self.result["family_macro"]:
            families = json.loads(row["family_ndcg"])
            self.assertEqual(row["n_families"], 3)
            self.assertAlmostEqual(row["equal_family_ndcg"], float(np.mean(list(families.values()))))
            family_recalls = json.loads(row["family_recall_at_10"])
            self.assertAlmostEqual(row["equal_family_recall_at_10"], float(np.mean(list(family_recalls.values()))))

    def test_sample_costs_amortized_setup_time_and_serving_defaults(self):
        with (Path(self.directory.name) / "cost_curves.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertNotIn("total_generation_seconds", row)
            self.assertAlmostEqual(float(row["amortized_setup_generation_seconds_per_future_task"]) * int(row["future_tasks"]), float(row["measured_onboarding_generation_seconds"]))
            expected_samples = 0 if int(row["budget"]) == 0 or row["method"] == "source_router" else 16
            self.assertEqual(float(row["total_sample_calls"]), expected_samples)
        missing = _selected_task_costs({"action_costs": [[{"search_calls": 1}] * 5]}, np.array([0]))
        self.assertEqual(missing, {"search_calls": 1, "sample_calls": 0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0})
        missing_arrays = _selected_task_costs({"action_costs": {"search_calls": [[1] * 5]}}, np.array([0]))
        self.assertEqual(missing, missing_arrays)

    def test_original_and_source_fixed_cost_baselines_are_saved(self):
        for row in self.result["headroom"]:
            self.assertEqual(row["original_mean_task_search_calls"], 1)
            self.assertEqual(row["original_mean_task_llm_calls"], 0)
            self.assertEqual(row["original_mean_task_sample_calls"], 0)
            self.assertIn("source_fixed_mean_task_input_tokens", row)
            self.assertIn("source_fixed_recall_at_10", row)

    def test_serialized_models_and_analysis_provenance_reproduce_predictions(self):
        root = Path(self.directory.name)
        metadata = json.loads((root / "models.json").read_text())
        provenance = json.loads((root / "analysis_metadata.json").read_text())
        self.assertEqual(provenance["learning_sha256"], hashlib.sha256((Path(__file__).resolve().parents[1] / "learning.py").read_bytes()).hexdigest())
        self.assertIn("protocol", provenance)
        self.assertEqual(len(provenance["input_record_sha256"]), 6)
        paired = json.loads((root / "paired_outcomes.json").read_text())
        for record in self.records:
            fold = f"{record['corpus_name']}::{record['backend']}"
            model = metadata[fold]
            serialized = model["serialized_model"]
            self.assertEqual(hashlib.sha256(Path(serialized["path"]).read_bytes()).hexdigest(), serialized["sha256"])
            bundle = joblib.load(serialized["path"])
            predictions = bundle["base_router"].predict(record["tasks"]["test"])
            self.assertEqual(predictions.argmax(axis=1).tolist(), paired[record["name"]]["source_router_actions"])
            self.assertEqual(bundle["learning_sha256"], provenance["learning_sha256"])


if __name__ == "__main__":
    unittest.main()
