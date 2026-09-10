"""Exploratory source-world posterior routing; original pilot files stay intact.

The source-only model and selected-observation interface are separate from the
evaluator below. Run this follow-up only after freezing its code and protocol.
The Gaussian posterior is an uncalibrated surrogate, not a target guarantee.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from learning import (ACTIONS, FAMILIES, COST_KEYS, _candidate_cost, _costs,
                      _json_default, _json_hash, _seed, _selected_task_costs,
                      _setup_generation_seconds, _write_csv)

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / 'world_model_protocol.json'
METHODS = ('random', 'fixed', 'information_gain', 'decision_value')
BUDGETS = (0, 4, 8, 16, 32)
SEEDS = (11, 23, 47)
WORKLOADS = (1, 10, 50, 200)
SETTINGS = {'world_count': 2, 'task_bucket_pseudocount': 8, 'probe_cell_pseudocount': 4,
            'noise_variance_floor': 0.01, 'gauss_hermite_points': 12,
            'methods': list(METHODS), 'pair_budgets': list(BUDGETS), 'pool_seeds': list(SEEDS),
            'future_task_counts': list(WORKLOADS), 'bootstrap_draws': 2000, 'bootstrap_seed': 20260910}
GH_NODES, GH_WEIGHTS = np.polynomial.hermite.hermgauss(12)
GH_WEIGHTS = GH_WEIGHTS / GH_WEIGHTS.sum()


def cell(candidate):
    """Source likelihood axes: family, alternative action, query bucket."""
    family = FAMILIES.index(candidate['family'])
    action, bucket = int(candidate['action']), int(candidate['bucket'])
    if action != candidate['action'] or bucket != candidate['bucket'] or not 1 <= action < 5 or not 0 <= bucket < 4:
        raise ValueError('Candidate action or bucket outside the frozen menu')
    return family, action - 1, bucket


def normalize_log_weights(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (2,) or np.any(np.isnan(values)) or np.any(np.isposinf(values)) or not np.any(np.isfinite(values)):
        raise ValueError('Two finite-or-negative-infinity log weights are required')
    shifted = values - np.max(values)
    return shifted - np.log(np.exp(shifted).sum())


def posterior_after(log_weights, means, variance, outcome):
    """Stable two-world Gaussian update, also used at unbounded GH nodes."""
    log_weights = normalize_log_weights(log_weights)
    means = np.asarray(means, dtype=float)
    if means.shape != (2,) or not np.all(np.isfinite(means)) or not np.isfinite(variance) or variance <= 0 or not np.isfinite(outcome):
        raise ValueError('Finite means/outcome and positive variance are required')
    return normalize_log_weights(log_weights - 0.5 * (outcome - means) ** 2 / variance)


@dataclass
class WorldModel:
    utilities: np.ndarray
    task_mix: np.ndarray
    probe_means: np.ndarray
    probe_variance: np.ndarray
    world_ids: tuple[str, str]
    fitting: dict = field(default_factory=dict)

    def __post_init__(self):
        self.utilities = np.asarray(self.utilities, dtype=float)
        self.task_mix = np.asarray(self.task_mix, dtype=float)
        self.probe_means = np.asarray(self.probe_means, dtype=float)
        self.probe_variance = np.asarray(self.probe_variance, dtype=float)
        for array, shape in ((self.utilities, (2, 4, 5)), (self.task_mix, (4,)),
                             (self.probe_means, (2, 2, 4, 4)), (self.probe_variance, (2, 4, 4))):
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError('Invalid finite source-world array')
        if (np.any((self.utilities < 0) | (self.utilities > 1)) or np.any(self.task_mix < 0)
                or not np.isclose(self.task_mix.sum(), 1) or np.any(self.probe_variance <= 0)):
            raise ValueError('Invalid source-world utility, mix or variance')
        if len(self.world_ids) != 2 or len(set(self.world_ids)) != 2:
            raise ValueError('Exactly two distinct source worlds are required')

    def expected_utilities(self, log_weights):
        return np.einsum('w,wba->ba', np.exp(normalize_log_weights(log_weights)), self.utilities)

    def route(self, buckets, log_weights):
        buckets = np.asarray(buckets)
        if buckets.ndim != 1 or not np.issubdtype(buckets.dtype, np.integer) or np.any((buckets < 0) | (buckets >= 4)):
            raise ValueError('Future query buckets must be integers in [0,3]')
        return self.expected_utilities(log_weights).argmax(axis=1)[buckets]

    def gain(self, candidate, log_weights, method):
        if method not in ('information_gain', 'decision_value'):
            raise ValueError('Unknown model-based acquisition')
        log_weights = normalize_log_weights(log_weights)
        probabilities = np.exp(log_weights)
        index = cell(candidate)
        means, variance = self.probe_means[(slice(None), *index)], self.probe_variance[index]
        outcomes = (means[:, None] + np.sqrt(2 * variance) * GH_NODES[None, :]).ravel()
        likelihoods = -0.5 * (outcomes[:, None] - means[None, :]) ** 2 / variance
        posterior_logs = likelihoods + log_weights[None, :]
        posterior_logs -= posterior_logs.max(axis=1, keepdims=True)
        posterior_logs -= np.log(np.exp(posterior_logs).sum(axis=1, keepdims=True))
        posterior = np.exp(posterior_logs)
        if method == 'information_gain':
            active = np.isfinite(log_weights)
            values = np.sum(posterior[:, active] * (posterior_logs[:, active] - log_weights[None, active]), axis=1)
        else:
            current_actions = self.expected_utilities(log_weights).argmax(axis=1)
            after = np.einsum('nw,wba->nba', posterior, self.utilities)
            current_after = after[:, np.arange(4), current_actions]
            values = (after.max(axis=2) - current_after) @ self.task_mix
        # Each exact integrand is nonnegative. Avoid negative roundoff and avoid
        # subtracting a prior value that finite quadrature need not reproduce.
        weights = (probabilities[:, None] * GH_WEIGHTS[None, :]).ravel()
        return float(weights @ np.maximum(values, 0))

    def responsive_actions(self):
        """Actions reachable by some mixture weight, using source utilities only."""
        result = []
        for bucket in range(4):
            intercept = self.utilities[1, bucket]
            slope = self.utilities[0, bucket] - intercept
            points = {0.0, 1.0}
            for a in range(5):
                for b in range(a):
                    if slope[a] != slope[b]:
                        crossing = (intercept[b] - intercept[a]) / (slope[a] - slope[b])
                        if 0 < crossing < 1:
                            points.add(float(crossing))
            points = sorted(points)
            probes = points + [(left + right) / 2 for left, right in zip(points[:-1], points[1:])]
            result.append(sorted({int(np.argmax(intercept + p * slope)) for p in probes}))
        return result

    def as_dict(self):
        return {'world_ids': list(self.world_ids), 'prior': [0.5, 0.5],
                'utilities': self.utilities.tolist(), 'task_mix': self.task_mix.tolist(),
                'probe_means': self.probe_means.tolist(), 'probe_variance': self.probe_variance.tolist(),
                'probe_cell_axes': ['family', 'alternative_action', 'bucket'],
                'prior_actions_by_bucket': self.expected_utilities(np.log([0.5, 0.5])).argmax(axis=1).tolist(),
                'source_only_reachable_actions_by_bucket': self.responsive_actions(), **self.fitting}


def fit_world_model(source_records):
    """Read source TRAIN labels and source probes only; no other task partition."""
    sources = sorted(source_records, key=lambda record: record['corpus_name'])
    if (len(sources) != 2 or len({r['corpus_name'] for r in sources}) != 2
            or len({r['backend'] for r in sources}) != 1):
        raise ValueError('Fitting needs two disjoint source corpora on one backend')
    utilities, frequencies, task_counts, fit_inputs = [], [], [], {}
    counts = np.zeros((2, 2, 4, 4), dtype=int)
    sums, means = np.zeros_like(counts, dtype=float), np.zeros_like(counts, dtype=float)
    within_ss = np.zeros((2, 4, 4), dtype=float)
    within_df = np.zeros((2, 4, 4), dtype=int)
    family_action_counts = np.zeros((2, 2, 4), dtype=int)
    family_action_means = np.zeros((2, 2, 4), dtype=float)
    for world, record in enumerate(sources):
        train = record['tasks']['train']
        scores, buckets = np.asarray(train['scores'], dtype=float), np.asarray(train['buckets'])
        if (scores.ndim != 2 or scores.shape[1] != 5 or not len(scores) or len(train['ids']) != len(scores)
                or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1))
                or buckets.shape != (len(scores),) or not np.issubdtype(buckets.dtype, np.integer)
                or np.any((buckets < 0) | (buckets >= 4))):
            raise ValueError('Invalid source training utility/bucket cohort')
        bucket_counts = np.bincount(buckets, minlength=4)
        global_mean = scores.mean(axis=0)
        utilities.append(np.array([(scores[buckets == b].sum(axis=0) + 8 * global_mean) / (bucket_counts[b] + 8) for b in range(4)]))
        frequencies.append(bucket_counts / len(scores))
        task_counts.append(bucket_counts.tolist())
        unique = {}
        for _, pool in sorted(record['pools'].items()):
            for candidate in pool['candidates']:
                identifier = str(candidate['id'])
                value = (cell(candidate), float(pool['observations'][candidate['id']]['delta']))
                if not np.isfinite(value[1]) or not -1 <= value[1] <= 1:
                    raise ValueError('Source probe RR delta must be in [-1,1]')
                if identifier in unique and unique[identifier] != value:
                    raise ValueError('Repeated source probe ID has inconsistent cell or outcome')
                unique[identifier] = value
        cell_values = {index: [] for index in np.ndindex(2, 4, 4)}
        for index, value in unique.values():
            cell_values[index].append(value)
        for family in range(2):
            for action in range(4):
                family_values = [value for bucket in range(4) for value in cell_values[family, action, bucket]]
                family_action_counts[world, family, action] = len(family_values)
                fallback = float(np.mean(family_values)) if family_values else 0.0
                family_action_means[world, family, action] = fallback
                for bucket in range(4):
                    index = (family, action, bucket)
                    values = np.asarray(cell_values[index], dtype=float)
                    count = len(values)
                    counts[(world, *index)] = count
                    sums[(world, *index)] = values.sum()
                    means[(world, *index)] = (values.sum() + 4 * fallback) / (count + 4)
                    if count:
                        within_ss[index] += float(np.square(values - values.mean()).sum())
                        within_df[index] += max(count - 1, 0)
        fit_inputs[record['name']] = {'train': {key: train[key] for key in ('ids', 'buckets', 'scores')},
            'unique_probes': {identifier: {'cell': list(index), 'delta': value} for identifier, (index, value) in sorted(unique.items())}}
    raw_variance = within_ss / np.maximum(within_df, 1)
    return WorldModel(np.asarray(utilities), np.mean(frequencies, axis=0), means, np.maximum(raw_variance, 0.01),
        tuple(record['name'] for record in sources), {
            'source_corpora': [record['corpus_name'] for record in sources],
            'source_train_query_ids': {record['name']: list(record['tasks']['train']['ids']) for record in sources},
            'source_train_bucket_counts': task_counts, 'source_probe_cell_counts': counts.tolist(),
            'source_probe_family_action_counts': family_action_counts.tolist(),
            'source_probe_family_action_fallback_means': family_action_means.tolist(),
            'source_probe_cell_sums': sums.tolist(), 'within_source_variance_df': within_df.tolist(),
            'within_source_variance_raw': raw_variance.tolist(),
            'empty_probe_cells_using_family_action_fallback': (counts == 0).tolist(),
            'empty_task_buckets_using_world_global_fallback': (np.asarray(task_counts) == 0).tolist(),
            'source_fit_input_sha256': {name: _json_hash(value) for name, value in fit_inputs.items()},
            'fit_scope': 'Only source train IDs/buckets/action utilities and unique source probe cell/delta pairs. No source calibration/utility/test tasks or target tasks.',
            'extrapolation': 'One global world posterior can move unprobed-bucket decisions. Empty probe cells use family/action means; empty task buckets use world-global means. Likelihoods and world coverage are uncalibrated.'})


def fit_for_target(records, target_corpus, backend):
    return fit_world_model([record for record in records if record['corpus_name'] != target_corpus and record['backend'] == backend])


def run_onboarding(model, candidates, observe, *, method, budget, seed=0):
    """Only selected IDs cross the callback; future target tasks are absent."""
    if method not in METHODS or int(budget) != budget or budget < 0:
        raise ValueError('Invalid selector or pair budget')
    candidates = list(candidates)
    ids = [str(candidate['id']) for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError('Candidate IDs must be unique')
    forbidden = {'delta', 'rr_original', 'rr_alternative', 'observation', 'observations', 'scores', 'recalls', 'qrels', 'utility', 'target_tasks'}
    for candidate in candidates:
        if forbidden.intersection(candidate):
            raise ValueError('Outcome/task fields are forbidden in candidate metadata')
        cell(candidate)
        if _candidate_cost(candidate) <= 0:
            raise ValueError('Selected-probe search costs must be positive')
    log_weights = np.log([0.5, 0.5])
    result = {'selected_ids': [], 'trace': [], 'snapshots': {0: log_weights.copy()}}
    order = np.random.default_rng(seed).permutation(len(candidates))
    priority = {int(index): rank for rank, index in enumerate(order)}
    remaining = list(range(len(candidates)))
    counts = np.zeros((2, 4, 4), dtype=int)
    for step in range(min(int(budget), len(candidates))):
        gain = score = None
        if method == 'random':
            chosen = min(remaining, key=priority.get)
        elif method == 'fixed':
            def fixed_key(index):
                f, a, b = cell(candidates[index])
                return counts[f, a, b], (f, b, a), str(candidates[index]['id'])
            chosen = min(remaining, key=fixed_key)
        else:
            cache = {}
            for index in remaining:
                candidate = candidates[index]
                key = cell(candidate)
                if key not in cache:
                    cache[key] = model.gain(candidate, log_weights, method)
            chosen = max(remaining, key=lambda i: (cache[cell(candidates[i])] / _candidate_cost(candidates[i]), -priority[i]))
            gain = cache[cell(candidates[chosen])]
            score = gain / _candidate_cost(candidates[chosen])
        candidate = candidates[chosen]
        value = float(observe(candidate['id'])['delta'])
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Selected RR delta must be finite and in [-1,1]')
        index = cell(candidate)
        before = log_weights.copy()
        log_weights = posterior_after(log_weights, model.probe_means[(slice(None), *index)], model.probe_variance[index], value)
        costs = _costs(candidate.get('cost_breakdown'))
        costs['search_calls'] = _candidate_cost(candidate)
        result['selected_ids'].append(str(candidate['id']))
        result['trace'].append({'id': str(candidate['id']), 'cell': list(index), 'delta': value,
            'expected_source_model_gain': gain, 'gain_per_search': score, 'costs': costs,
            'log_posterior_before': before.tolist(), 'log_posterior_after': log_weights.tolist(),
            'posterior_after': np.exp(log_weights).tolist()})
        result['snapshots'][step + 1] = log_weights.copy()
        counts[index] += 1
        remaining.remove(chosen)
    return result


def paired_interval(deltas, rng):
    deltas = np.atleast_2d(deltas)
    seed_indices = rng.integers(deltas.shape[0], size=(SETTINGS['bootstrap_draws'], deltas.shape[0]))
    query_indices = rng.integers(deltas.shape[1], size=(SETTINGS['bootstrap_draws'], deltas.shape[1]))
    boot = deltas[seed_indices[:, :, None], query_indices[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {'mean': float(deltas.mean()), 'lo': float(lo), 'hi': float(hi)}


def evaluate_records(records, original_paired, output_dir, *, protocol, input_provenance=None):
    """Evaluator only. Complete source fitting/onboarding before target scoring."""
    if protocol['settings'] != SETTINGS:
        raise ValueError('Follow-up protocol settings differ from this implementation')
    revision_paths = (Path(__file__), ROOT / 'learning.py', PROTOCOL_PATH)
    revisions = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in revision_paths}
    output_dir = Path(output_dir)
    if output_dir.resolve() == (ROOT / 'results').resolve():
        raise ValueError('Follow-up must not overwrite original semantic results')
    records = sorted(list(records), key=lambda record: record['name'])
    names = {record['name'] for record in records}
    if len(names) != len(records) or set(original_paired) != names:
        raise ValueError('Original and follow-up environment sets must match uniquely')
    started = time.perf_counter()
    models, onboarding = {}, {}
    # These two phases do not read any held-out target task fields.
    for record in records:
        model = fit_for_target(records, record['corpus_name'], record['backend'])
        models[record['name']] = model
        for seed, pool in sorted(record['pools'].items()):
            for method in METHODS:
                onboarding[record['name'], seed, method] = run_onboarding(model, pool['candidates'],
                    lambda identifier, pool=pool: pool['observations'][identifier],
                    method=method, budget=max(BUDGETS), seed=_seed('target', seed))
    source_and_onboarding_seconds = time.perf_counter() - started
    adaptation, costs, baseline_rows, paired, traces = [], [], [], {}, {}
    for record in records:
        name, model = record['name'], models[record['name']]
        tasks, original = record['tasks']['test'], original_paired[name]
        scores, recalls = np.asarray(tasks['scores']), np.asarray(tasks['recalls'])
        if (tasks['ids'] != original['query_ids'] or not len(scores) or scores.shape != (len(tasks['ids']), 5)
                or recalls.shape != scores.shape or not np.array_equal(scores, original['action_scores'])
                or not np.array_equal(recalls, original['action_recalls_at_10'])):
            raise ValueError('Follow-up and original task utility/cohort evidence differ')
        indices = np.arange(len(scores))
        zero_actions = model.route(tasks['buckets'], np.log([0.5, 0.5]))
        frozen_actions = np.asarray(original['source_router_actions'])
        if frozen_actions.shape != (len(scores),) or not np.issubdtype(frozen_actions.dtype, np.integer) or np.any((frozen_actions < 0) | (frozen_actions >= 5)):
            raise ValueError('Invalid original frozen-router actions')
        baseline_actions = {'zero_budget': zero_actions, 'original_frozen_router': frozen_actions, 'original_query': np.zeros(len(scores), dtype=int)}
        baseline = {label: {'actions': actions.tolist(), 'ndcg': scores[indices, actions].tolist(),
                           'recall_at_10': recalls[indices, actions].tolist()} for label, actions in baseline_actions.items()}
        for label, actions in baseline_actions.items():
            baseline_rows.append({'environment': name, 'baseline': label, 'n_queries': len(scores),
                'ndcg': float(scores[indices, actions].mean()), 'recall_at_10': float(recalls[indices, actions].mean()),
                **{'mean_task_' + key: value for key, value in _selected_task_costs(tasks, actions).items()}})
        paired[name] = {'query_ids': list(tasks['ids']), 'query_buckets': list(tasks['buckets']), 'action_names': list(ACTIONS),
                        'baselines': baseline, 'runs': {}}
        for seed, pool in sorted(record['pools'].items()):
            for method in METHODS:
                selected = onboarding[name, seed, method]
                traces[f'{name}/{seed}/{method}'] = selected['trace']
                for budget in BUDGETS:
                    count = min(budget, len(selected['selected_ids']))
                    log_weights = selected['snapshots'][count]
                    actions = model.route(tasks['buckets'], log_weights)
                    ndcg, recall = scores[indices, actions], recalls[indices, actions]
                    onboarding_cost = _costs(pool.get('setup_costs')) if budget else _costs()
                    generation_seconds = _setup_generation_seconds(pool.get('setup_costs', {})) if budget else 0.0
                    for observation in selected['trace'][:count]:
                        for key in COST_KEYS:
                            onboarding_cost[key] += observation['costs'][key]
                    serving = _selected_task_costs(tasks, actions)
                    row = {'environment': name, 'corpus_name': record['corpus_name'], 'backend': record['backend'],
                        'seed': seed, 'method': method, 'budget': budget, 'n_queries': len(scores), 'selected_probe_pairs': count,
                        'ndcg': float(ndcg.mean()), 'recall_at_10': float(recall.mean()),
                        'delta_vs_zero_budget': float(np.mean(ndcg - baseline['zero_budget']['ndcg'])),
                        'delta_vs_original_frozen_router': float(np.mean(ndcg - baseline['original_frozen_router']['ndcg'])),
                        'delta_recall_vs_zero_budget': float(np.mean(recall - baseline['zero_budget']['recall_at_10'])),
                        'delta_recall_vs_original_frozen_router': float(np.mean(recall - baseline['original_frozen_router']['recall_at_10'])),
                        'changed_action_fraction_vs_zero_budget': float(np.mean(actions != zero_actions)),
                        'changed_action_fraction_vs_original_frozen_router': float(np.mean(actions != frozen_actions)),
                        'posterior_world_0': float(np.exp(log_weights[0])), 'posterior_world_1': float(np.exp(log_weights[1])),
                        'measured_onboarding_generation_seconds': generation_seconds,
                        **{'onboarding_' + key: value for key, value in onboarding_cost.items()},
                        **{'mean_task_' + key: value for key, value in serving.items()}}
                    adaptation.append(row)
                    paired[name]['runs'][f'{seed}/{method}/{budget}'] = {'actions': actions.tolist(), 'ndcg': ndcg.tolist(),
                        'recall_at_10': recall.tolist(), 'selected_probe_ids': selected['selected_ids'][:count],
                        'log_posterior': log_weights.tolist(), 'posterior': np.exp(log_weights).tolist()}
                    for workload in WORKLOADS:
                        costs.append({'environment': name, 'seed': seed, 'method': method, 'budget': budget,
                            'future_tasks': workload, 'ndcg': float(ndcg.mean()), 'recall_at_10': float(recall.mean()),
                            **{'total_' + key: onboarding_cost[key] + workload * serving[key] for key in COST_KEYS},
                            'measured_onboarding_generation_seconds': generation_seconds,
                            'amortized_setup_generation_seconds_per_future_task': generation_seconds / workload})
    rng = np.random.default_rng(SETTINGS['bootstrap_seed'])
    intervals = []
    for name, data in sorted(paired.items()):
        seeds = sorted(next(record for record in records if record['name'] == name)['pools'])
        for method in METHODS:
            for budget in BUDGETS:
                current = np.array([data['runs'][f'{seed}/{method}/{budget}']['ndcg'] for seed in seeds])
                for comparator in ('zero_budget', 'random', 'original_frozen_router'):
                    reference = (np.array([data['runs'][f'{seed}/random/{budget}']['ndcg'] for seed in seeds])
                                 if comparator == 'random' else np.asarray(data['baselines'][comparator]['ndcg'])[None, :])
                    intervals.append({'environment': name, 'method': method, 'budget': budget, 'comparator': comparator,
                                      **paired_interval(current - reference, rng)})
    macro = []
    for method in METHODS:
        for budget in BUDGETS:
            matching = [row for row in adaptation if row['method'] == method and row['budget'] == budget]
            families = sorted({row['corpus_name'] for row in matching})
            row = {'method': method, 'budget': budget, 'n_families': len(families)}
            for metric in ('ndcg', 'recall_at_10', 'delta_vs_zero_budget', 'delta_vs_original_frozen_router'):
                row['equal_family_' + metric] = float(np.mean([np.mean([r[metric] for r in matching if r['corpus_name'] == family]) for family in families]))
            macro.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum for path, checksum in revisions.items()):
        raise ValueError('Code or protocol changed during follow-up evaluation')
    for filename, values in (('adaptation.csv', adaptation), ('cost_curves.csv', costs), ('baselines.csv', baseline_rows),
                             ('paired_intervals.csv', intervals), ('family_macro.csv', macro)):
        _write_csv(output_dir / filename, values)
    for filename, values in (('models.json', {name: model.as_dict() for name, model in models.items()}),
                             ('paired_outcomes.json', paired), ('posterior_traces.json', traces)):
        (output_dir / filename).write_text(json.dumps(values, default=_json_default, indent=2, allow_nan=False) + '\n')
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'exploratory follow-up; not confirmatory',
        'code_sha256': revisions[str(Path(__file__))], 'protocol': protocol,
        'protocol_canonical_sha256': _json_hash(protocol), 'protocol_file_sha256': revisions[str(PROTOCOL_PATH)],
        'learning_sha256': revisions[str(ROOT / 'learning.py')],
        'input_record_sha256': {record['name']: _json_hash(record) for record in records},
        'original_paired_sha256': _json_hash(original_paired), 'input_provenance': input_provenance or {},
        'numpy': np.__version__, 'python': platform.python_version(),
        'offline_source_fitting_and_simulated_onboarding_seconds': source_and_onboarding_seconds,
        'output_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output_dir.iterdir())
                          if path.is_file() and path.name != 'analysis_metadata.json' and path.suffix in ('.csv', '.json')},
        'interpretation': 'Own-zero-budget differences isolate posterior adaptation; comparison with the original frozen router also includes a different source-only starting policy. Surrogate likelihoods and intervals are not target confidence guarantees.'}
    (output_dir / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return {'adaptation': adaptation, 'baselines': baseline_rows, 'models': {name: model.as_dict() for name, model in models.items()},
            'paired_outcomes': paired, 'posterior_traces': traces, 'paired_intervals': intervals}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'world_model_followup')
    args = parser.parse_args(argv)
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    original_metadata_path = ROOT / 'results' / 'analysis_metadata.json'
    original_metadata = json.loads(original_metadata_path.read_text())
    original_paired_path = ROOT / 'results' / 'paired_outcomes.json'
    original_paired = json.loads(original_paired_path.read_text())
    diagnostics_path = ROOT / 'results' / 'diagnostics_provenance.json'
    original_diagnostics = json.loads(diagnostics_path.read_text())
    if hashlib.sha256(original_paired_path.read_bytes()).hexdigest() != original_diagnostics['input_sha256']['results/paired_outcomes.json']:
        raise ValueError('Original paired outcomes differ from the audited original run')
    main_protocol_path = ROOT / 'protocol.json'
    main_protocol = json.loads(main_protocol_path.read_text())
    for path, expected_hash in ((ROOT / 'learning.py', original_metadata['learning_sha256']),
                                (main_protocol_path, original_metadata['protocol_sha256'])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Original frozen learning/protocol revision drift')
    expected_names = {corpus + '_' + backend for corpus in main_protocol['datasets'] for backend in ('bm25', 'dense')}
    if set(original_paired) != expected_names:
        raise ValueError('All six original environments are required')
    records, paths = [], []
    for name in sorted(expected_names):
        path = ROOT / 'cache' / f'{name}_outcomes.json'
        record = json.loads(path.read_text())
        if _json_hash(record) != original_metadata['input_record_sha256'][name]:
            raise ValueError('Retrieval input differs from original frozen analysis')
        if (len(record['tasks']['train']['ids']) != 128 or set(record['pools']) != {str(seed) for seed in SEEDS}
                or len(record['tasks']['test']['ids']) != main_protocol['query_selection']['test_counts'][record['corpus_name']]):
            raise ValueError('Follow-up requires the declared source/cohort/pool grid')
        records.append(record)
        paths.append(path)
    input_paths = [original_metadata_path, original_paired_path, diagnostics_path, main_protocol_path, PROTOCOL_PATH, *paths]
    provenance = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    evaluate_records(records, original_paired, args.output_dir, protocol=protocol, input_provenance=provenance)
    print(f'Wrote the exploratory source-world follow-up to {args.output_dir}')


if __name__ == '__main__':
    main()
