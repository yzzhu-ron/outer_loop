"""Synthetic-only checks for the registered intervention-transfer protocol.

No test imports or reads replay evidence, cached rankings, or corpus outcomes.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import experiment as exp
import export_evidence as exporter


def confounded_sources():
    """Opposite within-family and between-family associations, analytically known."""
    sources = []
    z = np.linspace(-.1, .1, 7)
    for c in range(2):
        train = np.full((7, 4, 12), .1)
        train[:, :, 1] += (.2 + .4 * c - z)[:, None]
        x = np.zeros((3, 7, exp.P))
        x[:, :, 1] = .4 * c + z
        sources.append({"train_scores": train, "calibration_scores": train.copy(),
                        "fit_profiles": x, "calibration_profiles": x[:2].copy()})
    return sources


def shuffle_sources():
    sources = []
    z = np.linspace(-.2, .2, 7)
    for c in range(2):
        train = np.full((7, 4, 12), .5)
        cal = np.full((7, 4, 12), .5)
        train[:, :, 1] += (z + .08 * c)[:, None]
        cal[:, :, 1] += (z + .02 * c)[:, None]
        x = np.zeros((3, 7, exp.P)); x[:, :, 1] = z + .1 * c
        cal_x = x[:2].copy(); cal_x[0, :, 1] -= .02; cal_x[1, :, 1] += .02
        sources.append({"train_scores": train, "calibration_scores": cal,
                        "fit_profiles": x, "calibration_profiles": cal_x})
    return sources


def probe_row(document, j=3):
    rr = [.5] + [.25] * (exp.P - 1)
    rr[1] = .5 + .08 * (j - 3)
    rr[2] = .5 - .08 * (j - 3)
    return {"rr": rr, "action_rankings": [[document] for _ in range(5)],
            "cost": dict(zip(exp.COST_KEYS, (6, 12, 2, 100, 20, 1)))}


def synthetic_evidence():
    inputs = {"stage": "inputs_without_test_utilities", "corpora": {}}
    labels = {"stage": "test_utilities_after_decision_lock", "corpora": {}}
    for c, family in enumerate(("a", "b", "c")):
        docs = [f"{family}-d{i}" for i in range(64)]
        panels = {"fit": [docs[i:i + 8] for i in (0, 8, 16)],
                  "calibration": [docs[i:i + 8] for i in (24, 32)],
                  "target": {str(seed): docs[i:i + 8] for seed, i in zip((11, 23, 47), (40, 48, 56))}}
        costs = exporter.policy_costs([["q"], ["k"], ["s"], ["h"], ["d1", "d2"]],
                                      {"valid": True, "usage": {"prompt_tokens": 7, "output_tokens": 3}})
        record = {"panels": panels, "worlds": {},
                  "test_meta": {"ids": [f"q{i}" for i in range(3)], "policy_costs": [deepcopy(costs) for _ in range(3)]}}
        target = {"worlds": {}}
        for j in range(7):
            scores = np.full((3, 12), .5)
            scores[:, 1] += .01 * c + .025 * (j - 3)
            scores[:, 2] += .01 * c - .025 * (j - 3)
            record["worlds"][str(j)] = {
                "train": {"scores": scores.tolist()},
                "calibration": {"scores": scores[:2].tolist()},
                "probes": {doc: probe_row(doc, j) for doc in docs}}
            target["worlds"][str(j)] = {"ids": list(record["test_meta"]["ids"]),
                                       "scores": scores.tolist(), "recalls": (scores * .8).tolist()}
        inputs["corpora"][family] = record
        labels["corpora"][family] = target
    return inputs, labels


class BridgeTests(unittest.TestCase):
    def test_within_centering_removes_corpus_confounding(self):
        sources = confounded_sources()
        within = exp.fit_bridge(sources)
        pooled = exp.fit_bridge(sources, mode="pooled")
        self.assertAlmostEqual(within["beta"][1], -1 / 1.1)
        self.assertGreater(pooled["beta"][1], 0)
        self.assertAlmostEqual(within["mu_D"][1], .4)
        self.assertAlmostEqual(within["mu_X"][1], .2)
        self.assertEqual(within["beta"][0], 0)

    def test_deployment_uses_source_anchor_without_clipping(self):
        model = exp.fit_bridge(confounded_sources()) | {"eta": 1.0}
        snapshot = deepcopy(model)
        np.testing.assert_allclose(exp.predict(model, model["mu_X"]), model["mu_D"])
        target_x = np.array(model["mu_X"]); target_x[1] = -1; target_x[0] = 999
        pred = exp.predict(model, target_x)
        self.assertAlmostEqual(pred[1], .4 + (-1 / 1.1) * (-1 - .2))
        self.assertGreater(pred[1], 1)
        self.assertEqual(pred[0], 0)
        self.assertEqual(model, snapshot)
        self.assertEqual(target_x[0], 999)
        np.testing.assert_allclose(exp.predict(model | {"eta": 0}), model["mu_D"])
        with self.assertRaisesRegex(ValueError, "requires observed"):
            exp.predict(model)

    def test_no_within_variation_is_degenerate_even_with_corpus_offsets(self):
        sources = confounded_sources()
        for c, source in enumerate(sources):
            source["fit_profiles"][:, :, 1] = .4 * c
        model = exp.fit_bridge(sources)
        self.assertLess(model["variance"][1], 1e-20)
        self.assertTrue(model["degenerate"][1])
        self.assertEqual(model["beta"][1], 0)
        self.assertTrue(np.isfinite(exp.predict(model | {"eta": 1}, np.zeros(exp.P))).all())

    def test_calibration_selects_eta_without_refitting(self):
        sources = confounded_sources()
        before = exp.fit_bridge(sources)
        for source in sources:
            source["calibration_scores"][:] = .37
            source["calibration_profiles"][:] = .91
        after = exp.fit_bridge(sources)
        for key in ("mu_D", "mu_X", "beta", "variance", "lambda_source_D"):
            np.testing.assert_allclose(before[key], after[key], atol=0, rtol=0)
        self.assertEqual(after["eta"], 0)
        np.testing.assert_allclose([row["mean_ndcg"] for row in after["calibration"]], [.37] * 3)

    def test_shuffle_jointly_permutes_train_and_calibration_then_recalibrates(self):
        sources = shuffle_sources(); untouched = deepcopy(sources)
        seed = 20260911
        shuffled = exp.fit_bridge(sources, shuffle_seed=seed)
        rng = np.random.default_rng(seed); manually = deepcopy(sources); permutations = []
        for source in manually:
            order = rng.permutation(7); permutations.append(order.tolist())
            for field in ("train_scores", "calibration_scores"):
                source[field] = source[field][order].copy()
        expected = exp.fit_bridge(manually)
        self.assertEqual(shuffled["source_permutations"], permutations)
        self.assertEqual(shuffled["shuffle_seed"], seed)
        for key in ("mu_D", "mu_X", "beta", "variance", "lambda_source_D", "calibration", "eta"):
            self.assertEqual(shuffled[key], expected[key], key)
        original = exp.fit_bridge(sources)
        np.testing.assert_allclose(original["mu_D"], shuffled["mu_D"], atol=1e-15)
        np.testing.assert_allclose(original["mu_X"], shuffled["mu_X"], atol=0)
        train_only = deepcopy(manually)
        for source, unpermuted in zip(train_only, sources):
            source["calibration_scores"] = unpermuted["calibration_scores"]
        wrong = exp.fit_bridge(train_only)
        self.assertGreater(abs(shuffled["calibration"][2]["mean_ndcg"] - wrong["calibration"][2]["mean_ndcg"]), .001)
        for source, saved in zip(sources, untouched):
            for field in source:
                np.testing.assert_array_equal(source[field], saved[field])

    def test_reference_policy_is_excluded_from_fitting(self):
        sources = confounded_sources(); before = exp.fit_bridge(sources)
        for source in sources:
            source["train_scores"][:, :, -1] = 999
            source["calibration_scores"][:, :, -1] = -999
        self.assertEqual(before, exp.fit_bridge(sources))


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.model = {"mu_D": [0.] * exp.P, "mu_X": [0.] * exp.P,
                      "beta": [1.] * exp.P, "eta": 1.0}
        self.documents = [f"d{i}" for i in range(8)]

    def test_registered_menu_and_ties(self):
        self.assertEqual(len(set(exp.POLICIES)), 11)
        calls = lambda p: sum(2 if a == 4 else 1 for a in p)
        self.assertTrue(all(calls(p) <= 2 for p in exp.POLICIES))
        self.assertEqual(calls(exp.ALL_POLICIES[-1]), 6)
        self.assertEqual(exp.choose([0.] * exp.P), 1)
        for tied, expected in (((2, 3), 3), ((4, 5), 5), ((2, 5), 2)):
            prediction = np.zeros(exp.P); prediction[list(tied)] = 1
            self.assertEqual(exp.choose(prediction), expected)
        for prediction in ([0.] * 10, [np.nan] * exp.P, [np.inf] * exp.P):
            with self.assertRaises(ValueError):
                exp.choose(prediction)

    def test_exact_eight_callbacks_and_actual_costs_with_trace_whitelist(self):
        calls = []
        def observe(doc):
            calls.append(doc)
            return probe_row(doc, 5) | {"target_labels": object(), "hidden_lambda": object(), "unselected": object()}
        run = exp.decide_target(self.model, self.documents, observe)
        self.assertEqual(calls, self.documents)
        self.assertEqual(run["onboarding_cost"], dict(zip(exp.COST_KEYS, (48, 96, 16, 800, 160, 8))))
        self.assertEqual(len(run["observations"]), 8)
        for row in run["observations"]:
            self.assertEqual(set(row), {"document_id", "rr", "cost", "action_rankings"})
        self.assertEqual(run["profile"][0], 0)
        self.assertAlmostEqual(run["profile"][1], .16)
        json.dumps(run, allow_nan=False)

    def test_eta_zero_skips_onboarding_entirely(self):
        observe = Mock(side_effect=AssertionError("No target access allowed"))
        run = exp.decide_target(self.model | {"eta": 0}, self.documents, observe)
        observe.assert_not_called()
        self.assertIsNone(run["profile"])
        self.assertEqual(run["observations"], [])
        self.assertEqual(run["onboarding_cost"], exp.zero_cost())

    def test_invalid_panels_and_profiles_are_rejected(self):
        for documents in (self.documents[:-1], ["d"] * 8):
            observe = Mock()
            with self.assertRaises(ValueError):
                exp.decide_target(self.model, documents, observe)
            observe.assert_not_called()
        rows = [probe_row(doc) for doc in self.documents]
        with self.assertRaises(ValueError):
            exp.profile(rows[:-1])
        for value in (-.1, 1.1, float("nan")):
            bad = deepcopy(rows); bad[0]["rr"][1] = value
            with self.assertRaises(ValueError):
                exp.profile(bad)
        bad = deepcopy(rows); bad[0]["rr"].pop()
        with self.assertRaises(ValueError):
            exp.profile(bad)

    def test_unavailable_probe_remains_in_eight_document_denominator(self):
        rows = [probe_row(doc, 5) for doc in self.documents]
        rows[0]["rr"] = [0.] * exp.P
        self.assertAlmostEqual(exp.profile(rows)[1], 7 * .16 / 8)


class CostAndExportTests(unittest.TestCase):
    def test_generation_attempts_charge_invalid_rows_and_no_absent_attempt(self):
        self.assertEqual(exporter.generation_cost(None), exp.zero_cost())
        expected = exp.zero_cost() | {"llm_calls": 1, "input_tokens": 17, "output_tokens": 4}
        self.assertEqual(exporter.generation_cost({"valid": False, "usage": {"prompt_tokens": 17, "output_tokens": 4}}), expected)

    def test_shared_generation_bundle_and_actual_action_lengths(self):
        generated = {"valid": True, "usage": {"prompt_tokens": 11, "output_tokens": 5}}
        costs = exporter.policy_costs([["q"], ["k"], ["s"], ["h"], ["d1", "d2"]], generated)
        self.assertEqual(len(costs), 12)
        self.assertEqual(costs[-1]["logical_search_calls"], 6)
        self.assertEqual(costs[-1]["underlying_search_calls"], 12)
        self.assertEqual(costs[-1]["llm_calls"], 1)
        self.assertEqual(costs[10]["llm_calls"], 1)
        self.assertEqual(costs[5]["llm_calls"], 0)
        invalid = exporter.policy_costs([["q"], ["k"], [], [], []], generated | {"valid": False})
        self.assertEqual(invalid[3]["logical_search_calls"], 0)
        self.assertEqual(invalid[3]["llm_calls"], 1)
        self.assertEqual(invalid[-1]["logical_search_calls"], 2)

    def test_failed_question_charges_only_existing_attempt_and_sample(self):
        data = {"probe_questions": {"d": {"valid": False, "usage": {"prompt_tokens": 10, "output_tokens": 2}}},
                "probe_rewrites": {}}
        backend = Mock()
        row = exporter.document_observation(data, "d", backend)
        self.assertEqual(row["status"], "unavailable_question")
        self.assertEqual(row["rr"], [0.] * exp.P)
        self.assertEqual(row["action_rankings"], [[] for _ in range(5)])
        self.assertEqual(row["cost"], exp.zero_cost() | {"llm_calls": 1, "input_tokens": 10, "output_tokens": 2, "sample_calls": 1})
        backend.search_group.assert_not_called()

    def test_valid_question_invalid_rewrite_keeps_original_and_keyword_costs(self):
        question = {"valid": True, "indirect_question": "What helps growing plants?", "usage": {"prompt_tokens": 10, "output_tokens": 2}}
        rewrite = {"valid": False, "usage": {"prompt_tokens": 7, "output_tokens": 1}}
        data = {"probe_questions": {"d": question}, "probe_rewrites": {"d:indirect_question": rewrite}}
        groups = exporter.action_queries(question["indirect_question"], rewrite)
        ranks = {q: ["d"] for group in groups for q in group}
        backend = exporter.FrozenHybridBackend(ranks, ranks, .5)
        row = exporter.document_observation(data, "d", backend)
        self.assertEqual(row["status"], "unavailable_generated_actions")
        self.assertEqual(row["rr"][:5], [1., 1., 0., 0., 0.])
        self.assertEqual(row["cost"], dict(zip(exp.COST_KEYS, (2, 4, 2, 17, 3, 1))))
        del data["probe_rewrites"]["d:indirect_question"]
        with self.assertRaisesRegex(ValueError, "lacks its frozen rewrite attempt"):
            exporter.document_observation(data, "d", backend)

    def test_input_export_never_indexes_test_qrels(self):
        class SourceOnlyQrels(dict):
            def __getitem__(self, key):
                if key == "test":
                    raise AssertionError("Input exporter accessed sealed target qrels")
                return super().__getitem__(key)
        docs = [f"d{i:02}" for i in range(48)]
        failed = {"valid": False, "usage": {"prompt_tokens": 1, "output_tokens": 1}}
        data = {"pools": {str(seed): docs[i:i + 16] for seed, i in zip((11, 23, 47), (0, 16, 32))},
                "ids": {part: [part] for part in ("train", "calibration", "test")},
                "queries": {part: "growing plants" for part in ("train", "calibration", "test")},
                "task_rewrites": {part: failed for part in ("train", "calibration", "test")},
                "probe_questions": {doc: failed for doc in docs}, "probe_rewrites": {},
                "qrels": SourceOnlyQrels(train={"train": {"d00": 1}}, calibration={"calibration": {"d00": 1}})}
        ranks = {q: ["d00"] for group in exporter.action_queries("growing plants", failed) for q in group}
        with patch.object(exporter, "load_rank_maps", return_value={"bm25": ranks, "dense": ranks}):
            result = exporter.export_inputs({"datasets": {"synthetic": data}})
        self.assertEqual(result["stage"], "inputs_without_test_utilities")
        record = result["corpora"]["synthetic"]
        self.assertEqual(set(record["test_meta"]), {"ids", "policy_costs"})
        for world in record["worlds"].values():
            self.assertEqual(set(world), {"train", "calibration", "probes"})
        selected = [doc for kind in ("fit", "calibration") for panel in record["panels"][kind] for doc in panel]
        self.assertEqual(len(selected), 40)
        self.assertEqual(len(set(selected)), 40)
        self.assertEqual(record["panels"]["target"]["11"], docs[:8])

    def test_test_export_rejects_unlocked_decisions_before_reading_caches(self):
        with patch.object(exporter, "load_rank_maps", side_effect=AssertionError("Unexpected cache access")) as reader:
            with self.assertRaisesRegex(ValueError, "before decisions are locked"):
                exporter.export_test_labels({}, {"stage": "unlocked"}, {})
            reader.assert_not_called()

    def test_artifact_writer_cannot_overwrite_locked_file(self):
        with TemporaryDirectory(prefix="synthetic-intervention-") as folder:
            output = Path(folder) / "decisions.json"
            exp.write_json(output, {"locked": True})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                exp.write_json(output, {"locked": False})
            self.assertEqual(output.read_bytes(), original)


class EvaluationTests(unittest.TestCase):
    def test_bootstrap_reuses_queries_across_lambdas_and_panels(self):
        values = np.zeros((7, 3, 2))
        values[0] = [1, -1]; values[1] = [-1, 1]
        self.assertEqual(exp.paired_macro_interval({"family": values}, draws=100), {"mean": 0., "lo": 0., "hi": 0.})

    def test_bootstrap_weights_families_equally_with_unequal_query_counts(self):
        values = {"small": np.ones((7, 3, 2)), "large": -np.ones((7, 3, 19))}
        self.assertEqual(exp.paired_macro_interval(values, draws=100), {"mean": 0., "lo": 0., "hi": 0.})

    def test_bootstrap_is_reproducible(self):
        values = {"family": np.broadcast_to(np.arange(9), (7, 3, 9))}
        first = exp.paired_macro_interval(values, draws=100, seed=123)
        self.assertEqual(first, exp.paired_macro_interval(values, draws=100, seed=123))
        self.assertLess(first["lo"], first["hi"])

    def test_stage_checks_precede_payload_access(self):
        with self.assertRaisesRegex(ValueError, "label-separated"):
            exp.make_decisions({"stage": "test_utilities_after_decision_lock"})
        with self.assertRaisesRegex(ValueError, "locked decisions"):
            exp.evaluate({"stage": "unlocked"}, {}, {})

    def test_synthetic_end_to_end_label_isolation_and_lock_completeness(self):
        inputs, labels = synthetic_evidence()
        decisions = exp.make_decisions(inputs, shuffle_seeds=[])
        exp.validate_decision_lock(decisions, inputs)
        snapshot = json.dumps(decisions, sort_keys=True); input_snapshot = json.dumps(inputs, sort_keys=True)
        result = exp.evaluate(decisions, labels, inputs)
        self.assertEqual(set(result["families"]), {"a", "b", "c"})
        self.assertEqual(len(result["fixed_policy_frontier"]["a"]), 12)
        self.assertEqual(len(result["response_transport"]["a"]["within"]["centered_mse_by_policy"]), exp.P)
        self.assertEqual(len(result["intervals"]["by_family"]), 3)
        self.assertEqual(result["costs"]["a"]["source_fixed"]["onboarding_per_environment_panel"], exp.zero_cost())
        self.assertEqual(result["costs"]["a"]["within_unshrunk"]["onboarding_per_environment_panel"]["sample_calls"], 8)
        self.assertEqual(result["costs"]["a"]["rrf_all"]["serving_per_query"]["llm_calls"], 1)
        self.assertAlmostEqual(result["costs"]["a"]["within_unshrunk"]["amortized_per_query"]["100"]["sample_calls"], .08)
        modified = deepcopy(labels)
        for target in modified["corpora"].values():
            for world in target["worlds"].values():
                world["scores"] = np.zeros((3, 12)).tolist()
        changed = exp.evaluate(decisions, modified, inputs)
        self.assertNotEqual(result["macro"], changed["macro"])
        self.assertEqual(json.dumps(decisions, sort_keys=True), snapshot)
        self.assertEqual(json.dumps(inputs, sort_keys=True), input_snapshot)
        self.assertEqual(decisions, exp.make_decisions(inputs, shuffle_seeds=[]))
        broken = deepcopy(decisions)
        del broken["folds"]["a"]["runs"]["0"]["11"]["within"]
        with self.assertRaisesRegex(ValueError, "Missing registered method"):
            exp.validate_decision_lock(broken, inputs)
        broken = deepcopy(decisions)
        run = broken["folds"]["a"]["runs"]["0"]["11"]["within"]
        run["policy"] = (run["policy"] + 1) % exp.P
        with self.assertRaisesRegex(ValueError, "differs from its predictions"):
            exp.validate_decision_lock(broken, inputs)

    def test_heldout_family_training_utilities_cannot_change_its_own_decisions(self):
        inputs, _ = synthetic_evidence()
        before = exp.make_decisions(inputs, shuffle_seeds=[])["folds"]["a"]
        changed = deepcopy(inputs)
        for world in changed["corpora"]["a"]["worlds"].values():
            world["train"]["scores"] = np.full((3, 12), .99).tolist()
            world["calibration"]["scores"] = np.full((2, 12), .01).tolist()
        after = exp.make_decisions(changed, shuffle_seeds=[])["folds"]["a"]
        self.assertEqual(before, after)

    def test_fixed_train_cal_uses_query_weighted_union_and_no_onboarding(self):
        inputs, _ = synthetic_evidence()
        for record in inputs["corpora"].values():
            for world in record["worlds"].values():
                train = np.full((3, 12), .2); train[:, 1] = .8
                cal = np.full((2, 12), .2); cal[:, 1] = 0; cal[:, 2] = 1
                world["train"]["scores"] = train.tolist()
                world["calibration"]["scores"] = cal.tolist()
        decisions = exp.make_decisions(inputs, shuffle_seeds=[])
        for fold in decisions["folds"].values():
            model = fold["models"]["source_fixed_train_cal"]
            self.assertAlmostEqual(model["mu_D"][1], .28)
            self.assertAlmostEqual(model["mu_D"][2], .32)
            self.assertEqual(model["eta"], 0)
            for panels in fold["runs"].values():
                for runs in panels.values():
                    self.assertEqual(runs["source_fixed"]["policy"], 1)
                    self.assertEqual(runs["source_fixed_train_cal"]["policy"], 2)
                    self.assertEqual(runs["source_fixed_train_cal"]["onboarding_cost"], exp.zero_cost())

    def test_gate_requires_improvement_and_family_protection_against_both_baselines(self):
        inputs, labels = synthetic_evidence()
        decisions = exp.make_decisions(inputs, shuffle_seeds=[])
        for fold in decisions["folds"].values():
            for panels in fold["runs"].values():
                for runs in panels.values():
                    for method, policy in (("within", 1), ("source_fixed", 0), ("source_fixed_train_cal", 2)):
                        prediction = np.full(exp.P, -.1); prediction[0] = 0
                        if policy:
                            prediction[policy] = .1
                        runs[method]["prediction"] = prediction.tolist()
                        runs[method]["policy"] = policy
        cases = (
            ((.01, .01, .01), (.006, .006, .006), True),
            ((.01, .01, .01), (0., 0., 0.), False),
            ((.004, .004, .004), (.01, .01, .01), False),
            ((.03, 0., 0.), (.01, .01, .01), False),
            ((.05, .05, -.006), (.01, .01, .01), False),
            ((.01, .01, .01), (.05, .05, -.006), False),
        )
        for first_gains, second_gains, expected in cases:
            with self.subTest(first=first_gains, second=second_gains):
                target_labels = deepcopy(labels)
                for family, first, second in zip(("a", "b", "c"), first_gains, second_gains):
                    scores = np.full((3, 12), .5)
                    scores[:, 1] = .5 + first
                    scores[:, 2] = .5 + first - second
                    for world in target_labels["corpora"][family]["worlds"].values():
                        world["scores"] = scores.tolist()
                result = exp.evaluate(decisions, target_labels, inputs)
                self.assertEqual(result["advancement_gate"]["passed"], expected)
                self.assertEqual(set(result["advancement_gate"]["required_against_both"]), {"source_fixed", "source_fixed_train_cal"})

    def test_mean_profile_aliases_distinguish_full_responses(self):
        inputs, labels = synthetic_evidence()
        decisions = exp.make_decisions(inputs, shuffle_seeds=[])
        fold = decisions["folds"]["a"]
        for j in range(7):
            run = fold["runs"][str(j)]["11"]["within_unshrunk"]
            run["profile"] = [0.] * exp.P
        utility = np.zeros((7, exp.P)); utility[:3, 1] = 1; utility[3:, 2] = 1
        audit = exp.alias_audit(fold, ["11"], utility)
        group = audit["panels"]["11"][0]
        self.assertEqual(group["lambda_indices"], list(range(7)))
        self.assertTrue(group["no_common_optimal_policy"])
        self.assertEqual(group["distinct_document_rr_states"], 7)
        self.assertEqual(group["distinct_action_ranklist_states"], 1)
        self.assertGreater(group["adaptation_headroom"], 0)

    def test_code_constants_match_final_registration(self):
        registration = json.loads((Path(__file__).parent / "protocol.v1.json").read_text())
        self.assertEqual(registration["lambda_grid"], list(exp.LAMBDAS))
        self.assertEqual(registration["policy_menu"]["ordered_policies"], [[exp.ACTIONS[a] for a in policy] for policy in exp.POLICIES])
        gate = registration["evaluation"]["advancement_gate"]
        self.assertEqual(gate["macro_gain_vs_source_fixed_at_least"], .005)
        self.assertEqual(gate["positive_families_at_least"], 2)
        self.assertEqual(gate["maximum_loss_in_any_family"], .005)


if __name__ == "__main__":
    unittest.main()
