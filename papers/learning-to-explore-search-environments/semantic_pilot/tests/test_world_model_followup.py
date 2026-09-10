"""Synthetic checks of the separate source-world responsiveness intervention."""
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import world_model_followup as world


def candidate(identifier, action=1, bucket=2, cost=2):
    return {'id': identifier, 'family': 'direct_question', 'action': action, 'bucket': bucket,
            'cost': cost, 'features': [0] * 8}


def decision_fixture():
    utilities = np.zeros((2, 4, 5))
    utilities[0, 0, 0] = utilities[1, 0, 1] = 1
    utilities[:, 1, 4] = 1
    means = np.zeros((2, 2, 4, 4))
    means[:, 0, 0, 2] = [-0.04, 0.04]
    means[:, 0, 1, 2] = [-0.0575, 0.0575]
    return world.WorldModel(utilities, [0.5, 0.5, 0, 0], means, np.full((2, 4, 4), 0.01), ('one', 'two'))


def records_fixture():
    records, frozen = [], {}
    for corpus_index, corpus in enumerate(('alpha', 'beta', 'gamma')):
        for backend in ('bm25', 'dense'):
            tasks = {}
            for split, n in (('train', 8), ('calibration', 2), ('utility', 2), ('test', 3)):
                scores = [[0.1 * (1 + (a + corpus_index + q) % 5) for a in range(5)] for q in range(n)]
                costs = [[{'search_calls': 2 if a == 4 else 1, 'llm_calls': int(a >= 2),
                           'input_tokens': 80 * int(a >= 2), 'output_tokens': 20 * int(a >= 2)} for a in range(5)] for _ in range(n)]
                tasks[split] = {'ids': [f'{corpus}-{split}-{i}' for i in range(n)], 'buckets': [i % 4 for i in range(n)],
                    'features': [[99] * 8 for _ in range(n)], 'scores': scores,
                    'recalls': [[min(1, value + 0.2) for value in row] for row in scores], 'action_costs': costs}
            candidates = [candidate(f'p{i}', i % 4 + 1, 2 + i % 2, 3 if i % 4 == 3 else 2) for i in range(6)]
            observations = {c['id']: {'delta': (corpus_index - 1) * (0.1 + 0.05 * c['action'])} for c in candidates}
            pool = {'candidates': candidates, 'observations': observations,
                    'setup_costs': {'sample_calls': 16, 'llm_calls': 8, 'input_tokens': 80, 'output_tokens': 20, 'seconds': 3}}
            name = corpus + '_' + backend
            records.append({'name': name, 'corpus_name': corpus, 'backend': backend, 'tasks': tasks,
                            'pools': {str(seed): copy.deepcopy(pool) for seed in world.SEEDS}})
            frozen[name] = {'query_ids': list(tasks['test']['ids']), 'action_scores': copy.deepcopy(tasks['test']['scores']),
                'action_recalls_at_10': copy.deepcopy(tasks['test']['recalls']), 'source_router_actions': [0] * 3}
    return records, frozen


class GaussianDecisionTests(unittest.TestCase):
    def test_more_expensive_information_can_lose_on_decision_value_per_search(self):
        model = decision_fixture()
        a, b = candidate('cheap'), candidate('expensive', action=2, cost=3)
        prior = np.log([0.5, 0.5])
        self.assertLess(model.gain(a, prior, 'information_gain') / 2, model.gain(b, prior, 'information_gain') / 3)
        self.assertGreater(model.gain(a, prior, 'decision_value') / 2, model.gain(b, prior, 'decision_value') / 3)
        # Exact binary Gaussian Bayes accuracy: Phi(d/2), with this bucket's
        # source task weight 1/2. GH12 approximates the nonsmooth utility max.
        exact = 0.5 * (0.5 * (1 + math.erf(0.4 / math.sqrt(2))) - 0.5)
        self.assertAlmostEqual(model.gain(a, prior, 'decision_value'), exact, delta=0.005)

    def test_common_optimum_has_zero_decision_value_despite_information(self):
        model = decision_fixture()
        model.utilities[:] = 0
        model.utilities[0, :, 4], model.utilities[1, :, 4] = 0.6, 1.0
        model.probe_means[:, 0, 0, 2] = [-0.2, 0.2]
        prior = np.log([0.8, 0.2])
        self.assertGreater(model.gain(candidate('p'), prior, 'information_gain'), 0.1)
        self.assertEqual(model.gain(candidate('p'), prior, 'decision_value'), 0)

    def test_identical_likelihoods_and_extreme_log_odds_remain_finite(self):
        original = np.log([0.7, 0.3])
        np.testing.assert_allclose(world.posterior_after(original, [0.1, 0.1], 0.01, 0.9), original, atol=1e-14)
        result = world.posterior_after([-10000, 0], [-0.5, 0.5], 0.01, -1)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(np.exp(result).sum(), 1)
        model = decision_fixture()
        model.probe_means[:] = 0
        self.assertAlmostEqual(model.gain(candidate('p'), original, 'information_gain'), 0, places=14)
        self.assertEqual(model.gain(candidate('p'), original, 'decision_value'), 0)

    def test_world_label_permutation_does_not_change_decisions_or_values(self):
        model = decision_fixture()
        swapped = world.WorldModel(model.utilities[::-1], model.task_mix, model.probe_means[::-1], model.probe_variance, ('two', 'one'))
        prior = np.log([0.8, 0.2])
        np.testing.assert_array_equal(model.route([0, 1, 2, 3], prior), swapped.route([0, 1, 2, 3], prior[::-1]))
        for method in ('information_gain', 'decision_value'):
            self.assertAlmostEqual(model.gain(candidate('p'), prior, method), swapped.gain(candidate('p'), prior[::-1], method), places=14)

    def test_world_posterior_can_change_actions_in_a_bucket_without_probes(self):
        model = decision_fixture()
        self.assertEqual(model.route([0], np.log([0.8, 0.2])).tolist(), [0])
        self.assertEqual(model.route([0], np.log([0.2, 0.8])).tolist(), [1])
        self.assertEqual(model.responsive_actions()[0], [0, 1])
        self.assertEqual(model.responsive_actions()[1], [4])


class SourceBoundaryTests(unittest.TestCase):
    def test_only_source_train_and_source_probes_are_read(self):
        records, _ = records_fixture()
        expected = world.fit_for_target(records, 'gamma', 'bm25').as_dict()
        class TargetTasks(dict):
            def __getitem__(self, key):
                raise AssertionError('Held-out target tasks were accessed in fitting')
        class SourcePartitions(dict):
            def __getitem__(self, key):
                if key != 'train':
                    raise AssertionError('Non-training source labels were accessed')
                return super().__getitem__(key)
        class SourceTrain(dict):
            def __getitem__(self, key):
                if key == 'features':
                    raise AssertionError('Task features were accessed in source fitting')
                return super().__getitem__(key)
        for record in records:
            if record['corpus_name'] == 'gamma':
                record['tasks'] = TargetTasks(record['tasks'])
            else:
                record['tasks'] = SourcePartitions(train=SourceTrain(record['tasks']['train']))
        self.assertEqual(world.fit_for_target(records, 'gamma', 'bm25').as_dict(), expected)

    def test_exact_shrinkage_deduplication_and_pooled_within_source_variance(self):
        records, _ = records_fixture()
        sources = [r for r in records if r['backend'] == 'bm25' and r['corpus_name'] != 'gamma']
        for index, record in enumerate(sources):
            record['tasks']['train'].update(ids=['q0', 'q1'], buckets=[0, 1], scores=[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
            candidates = [candidate('a'), candidate('b'), candidate('c', bucket=3)]
            values = [0, 1, -0.5] if index == 0 else [0.5, 0.5, 0.5]
            pool = {'candidates': candidates, 'observations': {c['id']: {'delta': value} for c, value in zip(candidates, values)}}
            record['pools'] = {str(seed): copy.deepcopy(pool) for seed in world.SEEDS}
        model = world.fit_world_model(sources)
        self.assertAlmostEqual(model.utilities[0, 0, 0], 5 / 9)
        self.assertAlmostEqual(model.utilities[0, 1, 0], 4 / 9)
        self.assertAlmostEqual(model.utilities[0, 2, 0], 0.5)
        self.assertAlmostEqual(model.probe_means[0, 0, 0, 2], 5 / 18)
        self.assertAlmostEqual(model.probe_means[0, 0, 0, 0], 1 / 6)
        self.assertAlmostEqual(model.fitting['source_probe_family_action_fallback_means'][0][0][0], 1 / 6)
        self.assertAlmostEqual(model.probe_variance[0, 0, 2], 0.25)
        self.assertEqual(model.fitting['source_probe_cell_counts'][0][0][0][2], 2)
        self.assertEqual(model.fitting['within_source_variance_df'][0][0][2], 2)
        self.assertTrue(np.all(np.asarray(model.fitting['empty_probe_cells_using_family_action_fallback'])[:, :, :, :2]))

    def test_target_task_mutation_cannot_change_onboarding(self):
        records, _ = records_fixture()
        model = world.fit_for_target(records, 'gamma', 'bm25')
        target = next(r for r in records if r['name'] == 'gamma_bm25')
        pool = target['pools']['11']
        def run(fitted):
            return world.run_onboarding(fitted, pool['candidates'], lambda identifier: pool['observations'][identifier],
                                        method='decision_value', budget=4, seed=23)
        expected = run(model)
        target['tasks'] = {'train': {'scores': 'forbidden'}, 'test': {'features': 'changed', 'scores': 'changed', 'qrels': 'changed'}}
        actual = run(world.fit_for_target(records, 'gamma', 'bm25'))
        self.assertEqual(actual['selected_ids'], expected['selected_ids'])
        self.assertEqual(actual['trace'], expected['trace'])
        self.assertNotIn('tasks', inspect.signature(world.run_onboarding).parameters)


class OnboardingAndEvaluationTests(unittest.TestCase):
    def test_zero_budget_selected_callback_and_nested_prefixes(self):
        model = decision_fixture()
        candidates = [candidate(str(i), i % 4 + 1) for i in range(6)]
        for method in world.METHODS:
            def forbidden(_):
                self.fail('Zero budget observed a probe')
            empty = world.run_onboarding(model, candidates, forbidden, method=method, budget=0)
            self.assertEqual(empty['selected_ids'], [])
            calls = []
            short = world.run_onboarding(model, candidates, lambda identifier: calls.append(identifier) or {'delta': 0.2}, method=method, budget=2, seed=19)
            longer = world.run_onboarding(model, candidates, lambda _: {'delta': 0.2}, method=method, budget=4, seed=19)
            self.assertEqual(calls, short['selected_ids'])
            self.assertEqual(short['selected_ids'], longer['selected_ids'][:2])
            self.assertEqual(short['trace'], longer['trace'][:2])
            np.testing.assert_array_equal(short['snapshots'][2], longer['snapshots'][2])

    def test_preobserved_fields_duplicate_ids_and_invalid_deltas_are_rejected(self):
        model = decision_fixture()
        invalid = candidate('p')
        invalid['delta'] = 1
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            world.run_onboarding(model, [invalid], lambda _: {'delta': 1}, method='fixed', budget=1)
        with self.assertRaisesRegex(ValueError, 'unique'):
            world.run_onboarding(model, [candidate('p'), candidate('p')], lambda _: {'delta': 1}, method='fixed', budget=1)
        with self.assertRaisesRegex(ValueError, 'Selected RR delta'):
            world.run_onboarding(model, [candidate('p')], lambda _: {'delta': 2}, method='fixed', budget=1)

    def test_full_synthetic_grid_costs_baselines_and_provenance(self):
        records, original = records_fixture()
        records[0]['pools']['11']['candidates'] = []
        records[0]['pools']['11']['observations'] = {}
        before = copy.deepcopy((records, original))
        protocol = json.loads(world.PROTOCOL_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'followup'
            result = world.evaluate_records(records, original, output, protocol=protocol)
            self.assertEqual(len(result['adaptation']), 6 * 3 * 4 * 5)
            self.assertEqual(len(result['paired_intervals']), 6 * 4 * 5 * 3)
            for row in result['adaptation']:
                saved = result['paired_outcomes'][row['environment']]['runs'][f"{row['seed']}/{row['method']}/{row['budget']}"]
                self.assertAlmostEqual(row['ndcg'], np.mean(saved['ndcg']))
                self.assertAlmostEqual(row['recall_at_10'], np.mean(saved['recall_at_10']))
                self.assertAlmostEqual(row['posterior_world_0'] + row['posterior_world_1'], 1)
                if row['budget'] == 0:
                    self.assertEqual(row['changed_action_fraction_vs_zero_budget'], 0)
                    self.assertEqual(row['delta_vs_zero_budget'], 0)
                    self.assertTrue(all(row['onboarding_' + key] == 0 for key in world.COST_KEYS))
                    self.assertEqual(saved['selected_probe_ids'], [])
                else:
                    self.assertEqual(row['onboarding_sample_calls'], 16)
                    self.assertEqual(row['onboarding_llm_calls'], 8)
                    self.assertEqual(row['onboarding_output_tokens'], 20)
                    self.assertEqual(row['measured_onboarding_generation_seconds'], 3)
                    record = next(record for record in records if record['name'] == row['environment'])
                    by_id = {c['id']: c['cost'] for c in record['pools'][row['seed']]['candidates']}
                    self.assertEqual(row['onboarding_search_calls'], sum(by_id[identifier] for identifier in saved['selected_probe_ids']))
                    if row['environment'] == records[0]['name'] and row['seed'] == '11':
                        self.assertEqual(row['selected_probe_pairs'], 0)
                        self.assertEqual(row['changed_action_fraction_vs_zero_budget'], 0)
            for row in result['paired_intervals']:
                if row['method'] == 'random' and row['comparator'] == 'random':
                    self.assertEqual([row['mean'], row['lo'], row['hi']], [0, 0, 0])
            for name, model in result['models'].items():
                self.assertNotIn(name, model['world_ids'])
                self.assertEqual(len(model['world_ids']), 2)
            metadata = json.loads((output / 'analysis_metadata.json').read_text())
            for name, checksum in metadata['output_sha256'].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), checksum)
            self.assertEqual(len(metadata['input_record_sha256']), 6)
        self.assertEqual((records, original), before)


if __name__ == '__main__':
    unittest.main()
