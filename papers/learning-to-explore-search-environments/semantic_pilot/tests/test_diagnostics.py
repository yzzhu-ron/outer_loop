"""Independent checks of paired uncertainty, fusion accounting, and cohorts."""

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_diagnostics as diagnostics
import run_semantic as runner


def fixture():
    paired, records = {}, {}
    prepared = {"contract": {"fixture": "fixed"}, "datasets": {}}
    for corpus in ("alpha", "beta", "gamma"):
        name = f"{corpus}_bm25"
        ids = {part: [f"{corpus}-{part}-{i}" for i in range(2)] for part in ("train", "calibration", "utility", "test")}
        qrels = {part: {qid: {f"relevant-{i}": 1} for i, qid in enumerate(values)} for part, values in ids.items()}
        rankings = [
            [["z0"], ["z0"], ["relevant-0"], ["z0"], ["relevant-0"]],
            [["relevant-1"], ["z1"], ["z1"], ["z1"], []],
        ]
        scores = [[diagnostics.ndcg_at_k(rank, qrels["test"][qid]) for rank in ranks] for qid, ranks in zip(ids["test"], rankings)]
        recalls = [[diagnostics.recall_at_k(rank, qrels["test"][qid]) for rank in ranks] for qid, ranks in zip(ids["test"], rankings)]
        costs = [[{"search_calls": 1 if a < 4 else 2 if i == 0 else 0,
                   "llm_calls": int(a >= 2), "input_tokens": 80 * int(a >= 2), "output_tokens": 20 * int(a >= 2)}
                  for a in range(5)] for i in range(2)]
        tasks = {part: {"ids": values, "scores": [[0, 0, 0, 0, 1]] * 2} for part, values in ids.items() if part != "test"}
        tasks["test"] = {"ids": ids["test"], "scores": scores, "recalls": recalls, "rankings": rankings, "action_costs": costs, "buckets": [0, 1]}
        candidate = {"id": "shared-probe", "family": "direct_question", "action": 1, "bucket": 0, "features": [0] * 8, "cost": 2}
        pools = {str(seed): {"candidates": [candidate.copy()], "observations": {"shared-probe": {"delta": -0.25, "rr_original": 0.5, "rr_alternative": 0.25}}} for seed in (11, 23, 47)}
        records[name] = {"name": name, "corpus_name": corpus, "backend": "bm25", "contract": prepared["contract"].copy(), "tasks": tasks, "pools": pools}
        runs = {f"{seed}/{method}/{budget}": {"actions": [0, 0], "ndcg": [row[0] for row in scores], "recall_at_10": [row[0] for row in recalls]}
                for seed in pools for method in (*diagnostics.METHODS, "source_router") for budget in diagnostics.BUDGETS}
        paired[name] = {"query_ids": list(ids["test"]), "action_names": list(diagnostics.ACTIONS), "action_scores": copy.deepcopy(scores), "action_recalls_at_10": copy.deepcopy(recalls), "source_router_actions": [0, 0], "runs": runs}
        prepared["datasets"][corpus] = {"ids": ids, "qrels": qrels}
    return paired, records, prepared


class BootstrapTests(unittest.TestCase):
    def test_constant_paired_gain_and_zero_difference_have_exact_intervals(self):
        for value in (0.0, 0.25, -0.25):
            result = diagnostics.interval(np.full((3, 7), value), np.random.default_rng(20))
            self.assertEqual(result, {"mean": value, "lo": value, "hi": value})

    def test_seed_uncertainty_is_retained_with_one_shared_query(self):
        result = diagnostics.interval(np.array([[0.0], [0.5], [1.0]]), np.random.default_rng(19))
        self.assertEqual(result["mean"], 0.5)
        self.assertGreater(result["hi"] - result["lo"], 0.5)

    def test_query_indices_are_shared_across_resampled_profiles(self):
        class FixedDraws:
            def __init__(self):
                self.calls = []

            def integers(self, high, size):
                self.calls.append((high, size))
                return np.array([[0, 0], [1, 1]]) if len(self.calls) == 1 else np.array([[0, 0, 0], [2, 2, 2]])

        rng = FixedDraws()
        result = diagnostics.interval(np.array([[0, 2, 4], [10, 12, 14]]), rng, draws=2)
        self.assertEqual(rng.calls, [(2, (2, 2)), (3, (2, 3))])
        self.assertEqual(result["mean"], 7)
        self.assertAlmostEqual(result["lo"], 0.35)
        self.assertAlmostEqual(result["hi"], 13.65)


class CohortTests(unittest.TestCase):
    def test_valid_cohort(self):
        diagnostics.validate_inputs(*fixture())

    def test_same_length_query_reordering_fails_closed(self):
        paired, records, prepared = fixture()
        paired["alpha_bm25"]["query_ids"] = list(reversed(paired["alpha_bm25"]["query_ids"]))
        with self.assertRaisesRegex(ValueError, "ID/order"):
            diagnostics.validate_inputs(paired, records, prepared)

    def test_stale_scores_contract_or_labels_fail_closed(self):
        for corruption, message in (("scores", "scores differ"), ("contract", "contract mismatch"), ("qrels", "relevance cohort")):
            paired, records, prepared = copy.deepcopy(fixture())
            if corruption == "scores":
                records["alpha_bm25"]["tasks"]["test"]["scores"][0][0] = 0.4
            elif corruption == "contract":
                records["alpha_bm25"]["contract"] = {"fixture": "stale"}
            else:
                prepared["datasets"]["alpha"]["qrels"]["test"]["unmatched-query"] = {"doc": 1}
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, message):
                diagnostics.validate_inputs(paired, records, prepared)

    def test_wrong_actions_or_outcomes_fail_closed(self):
        for corruption, message in (("actions", "invalid selected actions"), ("ndcg", "outcomes disagree"), ("recall_at_10", "recalls disagree")):
            paired, records, prepared = copy.deepcopy(fixture())
            key = next(iter(paired["alpha_bm25"]["runs"]))
            if corruption == "actions":
                paired["alpha_bm25"]["runs"][key]["actions"] = [0, 5]
            else:
                paired["alpha_bm25"]["runs"][key][corruption] = [0.25, 0.25]
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, message):
                diagnostics.validate_inputs(paired, records, prepared)

    def test_stale_action_recalls_fail_closed(self):
        paired, records, prepared = fixture()
        paired['alpha_bm25']['action_recalls_at_10'][0][0] = 0.4
        with self.assertRaisesRegex(ValueError, 'action recalls differ'):
            diagnostics.validate_inputs(paired, records, prepared)

    def test_stale_or_invalid_standalone_baseline_fails_closed(self):
        for actions, message in (([-1, 0], "invalid standalone baseline"), ([2, 0], "baseline actions differ")):
            paired, records, prepared = fixture()
            paired["alpha_bm25"]["source_router_actions"] = actions
            with self.subTest(actions=actions), self.assertRaisesRegex(ValueError, message):
                diagnostics.validate_inputs(paired, records, prepared)

    def test_reordered_source_training_cohort_fails_closed(self):
        paired, records, prepared = fixture()
        original = records["alpha_bm25"]["tasks"]["train"]["ids"]
        records["alpha_bm25"]["tasks"]["train"]["ids"] = list(reversed(original))
        with self.assertRaisesRegex(ValueError, "train query ID/order"):
            diagnostics.validate_inputs(paired, records, prepared)


class FusionTests(unittest.TestCase):
    def test_headroom_figure_does_not_clip_fusion_above_the_action_menu_oracle(self):
        from matplotlib.figure import Figure

        intervals = [{'environment': 'synthetic_bm25', 'method': method, 'budget': 0, 'comparator': 'source_router',
                      'mean': 0, 'lo': 0, 'hi': 0} for method in diagnostics.METHODS]
        headroom = [{'environment': 'synthetic_bm25', 'comparator': 'target_fixed_oracle', 'mean': 0.1}]
        actions = [{'environment': 'synthetic_bm25', 'action': name, 'bucket': -1, 'mean_ndcg': 0.1} for name in diagnostics.ACTIONS]
        fusion = [{'environment': 'synthetic_bm25', 'ndcg': 0.9}]
        limits = []
        def capture(figure, path, **_):
            if Path(path).stem == 'headroom':
                limits.append(figure.axes[0].get_ylim())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            with patch.object(diagnostics, 'ROOT', root), patch.object(Figure, 'savefig', capture):
                diagnostics.plot_figures(intervals, headroom, actions, [], fusion)
        self.assertEqual(len(limits), 2)
        self.assertTrue(all(low <= 0.9 < high for low, high in limits))

    def test_rrf_rank_and_cutoff(self):
        self.assertEqual(diagnostics.rrf([["a", "b"], ["b", "c"]], (0, 1)), ["b", "a", "c"])
        ranking = diagnostics.rrf([[f"a{i}" for i in range(10)], [f"b{i}" for i in range(10)]], (0, 1))
        self.assertEqual(len(ranking), 10)
        self.assertEqual(len(set(ranking)), 10)

    def test_decomposition_comparator_is_nested_fusion_as_declared(self):
        decomposed = diagnostics.rrf([["b"], ["b"]], (0, 1))
        nested = diagnostics.rrf([["a"], decomposed], (0, 1))
        flattened = diagnostics.rrf([["a"], ["b"], ["b"]], (0, 1, 2))
        self.assertEqual(nested, ["a", "b"])
        self.assertEqual(flattened, ["b", "a"])

    def test_valid_and_invalid_generated_actions_keep_correct_logical_costs(self):
        class Search:
            def batch(self, queries):
                return [[f"doc-{i}"] for i, _ in enumerate(queries)]

        valid = {"valid": True, "semantic_query": "semantic words", "hypothetical_document": "synthetic passage", "subqueries": ["first part", "second part"], "usage": {"prompt_tokens": 80, "output_tokens": 20}}
        ranks, costs = runner.search_actions("original query", valid, Search())
        self.assertEqual(costs[0]["search_calls"] + costs[4]["search_calls"], 3)
        self.assertEqual(costs[0]["llm_calls"] + costs[4]["llm_calls"], 1)
        self.assertEqual(costs[4]["input_tokens"], 80)
        invalid = {"valid": False, "usage": valid["usage"]}
        ranks, costs = runner.search_actions("original query", invalid, Search())
        self.assertEqual(ranks[4], [])
        self.assertEqual(costs[0]["search_calls"] + costs[4]["search_calls"], 1)
        self.assertEqual(costs[0]["llm_calls"] + costs[4]["llm_calls"], 1)
        self.assertEqual(costs[4]["output_tokens"], 20)

    def test_source_selected_fusion_costs_deduplication_and_input_hashes(self):
        paired, records, prepared = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "cache").mkdir()
            (root / "results" / "paired_outcomes.json").write_text(json.dumps(paired))
            metadata = {"input_record_sha256": {name: hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for name, record in records.items()}}
            (root / "results" / "analysis_metadata.json").write_text(json.dumps(metadata))
            (root / "cache" / "prepared.json").write_text(json.dumps(prepared))
            (root / "analysis_protocol.json").write_text(json.dumps({"test_fixture": True}))
            for name, record in records.items():
                (root / "cache" / f"{name}_outcomes.json").write_text(json.dumps(record))
            def fixture_figures(*_):
                figures = root / 'results' / 'figures'
                figures.mkdir(exist_ok=True)
                for name in ('headroom', 'adaptation', 'alignment'):
                    (figures / f'{name}.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            with patch.object(diagnostics, "ROOT", root), patch.object(diagnostics, "plot_figures", side_effect=fixture_figures):
                diagnostics.main()
            with (root / "results" / "fusion_baseline.csv").open() as handle:
                fusion = list(csv.DictReader(handle))
            self.assertEqual(len(fusion), 3)
            for row in fusion:
                self.assertEqual(row["alternative"], "decomposed")
                self.assertEqual(float(row["mean_task_search_calls"]), 2)
                self.assertEqual(float(row["mean_task_llm_calls"]), 1)
                self.assertEqual(float(row["mean_task_input_tokens"]), 80)
            with (root / "results" / "probe_contrasts.csv").open() as handle:
                probes = list(csv.DictReader(handle))
            self.assertTrue(all(int(row["n_unique_pairs"]) == 1 for row in probes))
            provenance = json.loads((root / "results" / "diagnostics_provenance.json").read_text())
            for name in records:
                self.assertIn(f"cache/{name}_outcomes.json", provenance["input_sha256"])
            for name in ('headroom', 'adaptation', 'alignment'):
                output_name = f'figures/{name}.svg'
                self.assertEqual(provenance['output_sha256'][output_name], hashlib.sha256((root / 'results' / output_name).read_bytes()).hexdigest())
            stale = copy.deepcopy(records["alpha_bm25"])
            stale["tasks"]["test"]["action_costs"][0][0]["search_calls"] = 9
            (root / "cache" / "alpha_bm25_outcomes.json").write_text(json.dumps(stale))
            with patch.object(diagnostics, "ROOT", root), patch.object(diagnostics, "plot_figures"), self.assertRaisesRegex(ValueError, "differs from the fitted analysis input"):
                diagnostics.main()


if __name__ == "__main__":
    unittest.main()
