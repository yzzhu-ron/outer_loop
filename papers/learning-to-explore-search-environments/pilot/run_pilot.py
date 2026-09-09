"""Exploratory, family-held-out ceiling and lexical-probe signal pilot.

Target onboarding is deliberately a separate function without tasks or qrels.
Retrieval caches are experimental infrastructure, not free target observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import eligible_queries, load_dataset, make_split
from engine import (ACTIONS, BM25, DenseRetriever, feature_bucket, generate_probe,
                    ndcg_at_k, query_features, reciprocal_rank, recall_at_k,
                    tokenize, transform)

ROOT = Path(__file__).resolve().parent
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
SEEDS = (11, 23, 47)
BUDGETS = (0, 16, 64, 256)
FAMILIES = ('exact_title', 'body_terms')
WORKLOADS = (1, 10, 50, 200)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generation_contract(args):
    return {'input_code': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ('engine.py', 'data.py', 'protocol.json', 'run_pilot.py')},
            'actions': list(ACTIONS), 'test_queries': args.test_queries,
            'source_queries': args.source_queries, 'probes': args.probes,
            'seeds': list(SEEDS), 'budgets': list(BUDGETS)}


def select_ids(ids, limit, salt):
    return sorted(ids, key=lambda q: digest([salt, str(q)]))[:limit]


class CachedSearch:
    """Cache lookup also records virtual calls; an observation is never uncharged."""

    def __init__(self, backend, path, identity):
        self.backend, self.path = backend, path
        self.identity = identity
        self.ranks = {}
        self.calls = 0
        self.cache_misses = 0
        self.compute_seconds = 0.0
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identity'] == identity:
                self.ranks = saved['ranks']

    def batch(self, queries):
        self.calls += len(queries)
        missing = list(dict.fromkeys(q for q in queries if q not in self.ranks))
        if missing:
            start = time.perf_counter()
            ranks = self.backend.batch_search(missing, k=10)
            self.compute_seconds += time.perf_counter() - start
            self.cache_misses += len(missing)
            self.ranks.update(zip(missing, ranks))
        return [self.ranks[q] for q in queries]

    def save(self):
        write_json(self.path, {'identity': self.identity, 'ranks': self.ranks})


class Profile:
    """Fixed 4 buckets x 3 pairs; contains no document IDs, text, or answers."""

    def __init__(self):
        self.count = np.zeros((4, 3), dtype=int)
        self.sum = np.zeros((4, 3))
        self.sum_sq = np.zeros((4, 3))

    def update(self, bucket, alternative, delta):
        self.count[bucket, alternative - 1] += 1
        self.sum[bucket, alternative - 1] += delta
        self.sum_sq[bucket, alternative - 1] += delta * delta

    def features(self, bucket, action):
        if action == 0:
            return [0.0, 0.0]
        a = action - 1
        return [float(self.sum[:, a].sum() / (self.count[:, a].sum() + 3)),
                float(self.sum[bucket, a] / (self.count[bucket, a] + 3))]

    def global_deltas(self):
        return np.r_[0, self.sum.sum(axis=0) / (self.count.sum(axis=0) + 3)]

    def to_dict(self):
        return {'count': self.count.tolist(), 'sum': self.sum.tolist(),
                'sum_sq': self.sum_sq.tolist()}

    @classmethod
    def from_dict(cls, data):
        profile = cls()
        for name in ('count', 'sum', 'sum_sq'):
            setattr(profile, name, np.asarray(data[name]))
        return profile


def onboard(sampled_documents, search, family, budget):
    """Fixed-coverage diagnostic. No task queries/qrels or unseen outcomes enter.

    A sampled document costs one operation even when its generated probe fails.
    A valid contrastive pair costs two further retrieval operations. Generation
    is deterministic local CPU work, separately timed and counted.
    """
    profile, audit = Profile(), []
    operations = generation_words = 0
    generation_seconds = profile_update_seconds = 0.0
    start = time.perf_counter()
    if budget < 3:
        return {'profile': profile.to_dict(), 'audit': [], 'operations': 0,
                'sampled_documents': 0, 'generation_input_words': 0,
                'llm_tokens': 0, 'llm_api_calls': 0,
                'generation_seconds': 0.0, 'profile_update_seconds': 0.0,
                'profile_bytes': len(json.dumps(profile.to_dict()).encode()),
                'measured_seconds_including_cache': time.perf_counter() - start}
    for index, doc in enumerate(sampled_documents):
        if operations + 3 > budget:
            break
        operations += 1
        generation_start = time.perf_counter()
        generation_words += len(tokenize(doc.get('title', '') + ' ' + doc.get('text', '')))
        query = generate_probe(doc, family)
        generation_seconds += time.perf_counter() - generation_start
        if not query:
            audit.append({'valid': False, 'reason': 'empty_probe'})
            continue
        alternative = 1 + index % (len(ACTIONS) - 1)
        queries = [transform(query, ACTIONS[0]), transform(query, ACTIONS[alternative])]
        ranks = search.batch(queries)
        operations += 2
        rr = [reciprocal_rank(rank, str(doc['_id'])) for rank in ranks]
        profile_start = time.perf_counter()
        profile.update(feature_bucket(query), alternative, rr[1] - rr[0])
        profile_update_seconds += time.perf_counter() - profile_start
        union = set(ranks[0]) | set(ranks[1])
        audit.append({'valid': True, 'action': ACTIONS[alternative],
                      'bucket': feature_bucket(query), 'rr_original': rr[0],
                      'rr_alternative': rr[1], 'identical_actions': queries[0] == queries[1],
                      'empty_results': sum(not r for r in ranks),
                      'overlap': len(set(ranks[0]) & set(ranks[1])) / max(1, len(union))})
    return {'profile': profile.to_dict(), 'audit': audit,
            'operations': operations, 'sampled_documents': len(audit),
            'generation_input_words': generation_words,
            'llm_tokens': 0, 'llm_api_calls': 0,
            'generation_seconds': generation_seconds,
            'profile_update_seconds': profile_update_seconds,
            'profile_bytes': len(json.dumps(profile.to_dict()).encode()),
            'measured_seconds_including_cache': time.perf_counter() - start}


def outcomes(ids, queries, qrels, search):
    transformed = [transform(queries[qid], action) for qid in ids for action in ACTIONS]
    start = time.perf_counter()
    ranks = search.batch(transformed)
    scores = np.asarray([ndcg_at_k(ranks[i * len(ACTIONS) + a], qrels[qid])
                         for i, qid in enumerate(ids) for a in range(len(ACTIONS))])
    scores = scores.reshape(len(ids), len(ACTIONS))
    recalls = np.asarray([recall_at_k(ranks[i * 4 + a], qrels[qid])
                          for i, qid in enumerate(ids) for a in range(4)]).reshape(len(ids), 4)
    return {'ids': ids, 'features': [query_features(queries[q]) for q in ids],
            'buckets': [feature_bucket(queries[q]) for q in ids],
            'scores': scores.tolist(), 'recall': recalls.tolist(),
            'rankings': [ranks[i * 4:(i + 1) * 4] for i in range(len(ids))],
            'identical_action_fraction': float(np.mean([
                len(set(transformed[i * 4:(i + 1) * 4])) < 4 for i in range(len(ids))])),
            'measured_seconds': time.perf_counter() - start}


def rrf(rankings, actions):
    scores = {}
    for action in actions:
        for position, doc_id in enumerate(rankings[action], 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + position)
    return sorted(scores, key=lambda d: (-scores[d], d))[:10]


def build_environment(name, backend_name, args, manifest):
    docs, queries, splits = load_dataset(args.data, name)
    exploration, heldout, split_audit = make_split(docs)
    all_ids = {str(d['_id']) for d in docs}
    held_test, eligibility = eligible_queries(splits['test'], heldout, all_ids)
    held_train, train_eligibility = eligible_queries(splits['train'], heldout, all_ids)
    test_ids = select_ids(held_test, args.test_queries, 'test-20260909')
    train_ids = select_ids(held_train, args.source_queries, 'train-20260909')
    if len(train_ids) < 20 or not test_ids:
        raise ValueError('Insufficient source or target tasks after support separation')
    source_cut = min(192, max(1, int(len(train_ids) * .75)))
    env = name + '_' + backend_name
    corpus_hash = digest(docs)
    start = time.perf_counter()
    if backend_name == 'bm25':
        backend = BM25(docs)
    else:
        backend = DenseRetriever(docs, cache_dir=args.data / 'embeddings',
                                 model_name=MODEL, revision=REVISION)
    index_seconds = time.perf_counter() - start
    identity = digest({'corpus': corpus_hash, 'backend': backend_name, 'revision': REVISION,
                       'model': MODEL, 'dense_embedding_identity': getattr(backend, 'cache_key', None),
                       'engine': hashlib.sha256((ROOT / 'engine.py').read_bytes()).hexdigest()})
    search = CachedSearch(backend, args.cache / (env + '_ranks.json'), identity)
    print(f'{env}: index ready, {len(docs)} documents; {len(test_ids)} test / {len(held_test)} eligible', flush=True)
    by_id = {str(doc['_id']): doc for doc in docs}
    histories = {}
    phase_costs = {}
    phase_start, phase_calls, phase_seconds = search.cache_misses, search.calls, search.compute_seconds
    # Onboarding runs before even supplying real query strings to the method.
    for seed in SEEDS:
        sampled_ids = sorted(exploration)
        random.Random(seed).shuffle(sampled_ids)
        sampled = [by_id[doc_id] for doc_id in sampled_ids[:args.probes]]
        for family in FAMILIES:
            for budget in BUDGETS:
                histories[f'{family}:{seed}:{budget}'] = onboard(sampled, search, family, budget)
    phase_costs['onboarding_grid'] = {'uncached_queries': search.cache_misses - phase_start,
        'virtual_queries': search.calls - phase_calls, 'uncached_seconds': search.compute_seconds - phase_seconds}
    search.save()
    print(f'{env}: onboarding complete', flush=True)
    task_outputs = []
    for phase, ids, labels in [('source_router_training', train_ids[:source_cut], held_train),
                               ('source_calibration', train_ids[source_cut:], held_train),
                               ('target_test_cache', test_ids, held_test)]:
        phase_start, phase_calls, phase_seconds = search.cache_misses, search.calls, search.compute_seconds
        task_outputs.append(outcomes(ids, queries, labels, search))
        phase_costs[phase] = {'uncached_queries': search.cache_misses - phase_start,
            'virtual_queries': search.calls - phase_calls, 'uncached_seconds': search.compute_seconds - phase_seconds}
    training, calibration, testing = task_outputs
    search.save()
    # A genuinely executed timing sample, bypassing the outcome cache.
    timing_queries = [queries[qid] for qid in test_ids[:20]]
    timing_start = time.perf_counter()
    for query in timing_queries:
        backend.search(transform(query, 'original'), k=10)
    timing = (time.perf_counter() - timing_start) / len(timing_queries)
    record = {'name': env, 'family': name, 'backend': backend_name,
              'generation_contract': generation_contract(args),
              'generation_runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'training': training, 'calibration': calibration, 'testing': testing,
              'histories': histories, 'test_qrels': {qid: held_test[qid] for qid in test_ids}}
    write_json(args.cache / (env + '_outcomes.json'), record)
    manifest['environments'][env] = {
        'corpus_documents': len(docs), 'corpus_sha256': corpus_hash,
        'archive_sha256': hashlib.sha256((args.data / (name + '.zip')).read_bytes()).hexdigest(),
        'dataset_url': f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip',
        'split_audit': split_audit, 'test_eligibility': eligibility,
        'source_train_eligibility': train_eligibility,
        'evaluated_query_ids': test_ids, 'source_router_query_ids': train_ids[:source_cut],
        'source_calibration_query_ids': train_ids[source_cut:],
        'index_load_or_build_seconds': index_seconds, 'actual_uncached_queries': search.cache_misses,
        'actual_uncached_search_seconds': search.compute_seconds,
        'phase_costs': phase_costs,
        'selected_test_blank_support_queries': [qid for qid in test_ids if any(
            score > 0 and not (by_id[did]['title'] + by_id[did]['text']).strip()
            for did, score in held_test[qid].items())],
        'selected_source_blank_support_queries': [qid for qid in train_ids if any(
            score > 0 and not (by_id[did]['title'] + by_id[did]['text']).strip()
            for did, score in held_train[qid].items())],
        'mean_serial_search_seconds_20_tasks': timing,
        'identical_action_fraction_test': testing['identical_action_fraction'],
        'mean_test_query_words': float(np.mean([len(tokenize(queries[q])) for q in test_ids])),
        'dense_document_truncation_tokens': 256 if backend_name == 'dense' else None,
    }
    print(f'{env}: task outcomes and timing sample complete', flush=True)
    return record


def x_features(record, split):
    return np.c_[record[split]['features'], np.full(len(record[split]['ids']), record['backend'] == 'dense')]


def score_choices(outcome, choices):
    return np.asarray(outcome['scores'])[np.arange(len(choices)), choices]


def fit_source(source_records):
    router = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    router.fit(np.concatenate([x_features(r, 'training') for r in source_records]),
               np.concatenate([r['training']['scores'] for r in source_records]))
    calibrators = {}
    for family in FAMILIES:
        features = {a: [] for a in range(1, 4)}
        targets = {a: [] for a in range(1, 4)}
        for record in source_records:
            pred = router.predict(x_features(record, 'calibration'))
            truth = np.asarray(record['calibration']['scores'])
            for seed in SEEDS:
                for budget in BUDGETS[1:]:
                    profile = Profile.from_dict(record['histories'][f'{family}:{seed}:{budget}']['profile'])
                    for a in range(1, 4):
                        for i, bucket in enumerate(record['calibration']['buckets']):
                            features[a].append(profile.features(bucket, a))
                            targets[a].append((truth[i, a] - truth[i, 0]) - (pred[i, a] - pred[i, 0]))
        calibrators[family] = {a: Ridge(alpha=10.0, fit_intercept=False).fit(features[a], targets[a])
                               for a in range(1, 4)}
    return router, calibrators


def profile_predictions(base, buckets, profile, calibrators):
    result = base.copy()
    for action in range(1, 4):
        result[:, action] += calibrators[action].predict([profile.features(b, action) for b in buckets])
    return result


def analyze(records, args, manifest):
    headroom_rows, alignment_rows, adaptation_rows, profile_records, paired = [], [], [], {}, {}
    trained_models = {}
    for target_family in ('scifact', 'fiqa'):
        source = [r for r in records if r['family'] != target_family]
        targets = [r for r in records if r['family'] == target_family]
        training_start = time.perf_counter()
        router, calibrators = fit_source(source)
        trained_models[target_family] = {
            'source_family': source[0]['family'],
            'fit_seconds': time.perf_counter() - training_start,
            'query_scaler_mean': router[0].mean_.tolist(), 'query_scaler_scale': router[0].scale_.tolist(),
            'router_coef': router[1].coef_.tolist(), 'router_intercept': router[1].intercept_.tolist(),
            'profile_coefficients': {f: {ACTIONS[a]: m.coef_.tolist() for a, m in c.items()}
                                     for f, c in calibrators.items()}}
        for record in targets:
            env, test = record['name'], record['testing']
            scores = np.asarray(test['scores'])
            base = router.predict(x_features(record, 'testing'))
            base_choices = base.argmax(axis=1)
            base_scores = score_choices(test, base_choices)
            source_same_backend = next(r for r in source if r['backend'] == record['backend'])
            source_means = np.asarray(source_same_backend['training']['scores']).mean(axis=0)
            source_fixed = int(source_means.argmax())
            extra_action = int(np.argmax(source_means[1:])) + 1
            rrf_scores = np.asarray([ndcg_at_k(rrf(ranks, (0, extra_action)), record['test_qrels'][qid])
                                     for qid, ranks in zip(test['ids'], test['rankings'])])
            means = scores.mean(axis=0)
            oracle = scores.max(axis=1)
            headroom_rows.append({
                'environment': env, 'n_queries': len(scores), 'original': means[0],
                'source_fixed': means[source_fixed], 'source_router': base_scores.mean(),
                'target_fixed_oracle': means.max(), 'query_oracle': oracle.mean(),
                'oracle_gap_vs_router': (oracle - base_scores).mean(),
                'oracle_gap_vs_target_fixed': oracle.mean() - means.max(),
                'source_fixed_action': ACTIONS[source_fixed],
                'target_fixed_oracle_action': ACTIONS[int(means.argmax())],
                'rrf_extra': rrf_scores.mean(),
            })
            paired[env] = {'query_ids': test['ids'], 'source_router': base_scores.tolist(),
                           'original': scores[:, 0].tolist(), 'query_oracle': oracle.tolist(),
                           'action_ndcg': scores.tolist(), 'action_recall': test['recall'],
                           'rrf_extra': rrf_scores.tolist(), 'profile_runs': {}}
            for family in FAMILIES:
                for seed in SEEDS:
                    full = record['histories'][f'{family}:{seed}:256']
                    p = Profile.from_dict(full['profile'])
                    for a in range(1, 4):
                        n = int(p.count[:, a - 1].sum())
                        alignment_rows.append({'environment': env, 'family': family, 'seed': seed,
                            'action': ACTIONS[a], 'probe_rr_delta': float(p.sum[:, a - 1].sum() / n) if n else '',
                            'task_ndcg_delta': float(means[a] - means[0]), 'count': n})
                    for budget in BUDGETS:
                        key = f'{family}:{seed}:{budget}'
                        history = record['histories'][key]
                        p = Profile.from_dict(history['profile'])
                        prediction_start = time.perf_counter()
                        adapted = profile_predictions(base, test['buckets'], p, calibrators[family])
                        profile_latency = (time.perf_counter() - prediction_start) / len(scores)
                        choices = adapted.argmax(axis=1)
                        adapted_scores = score_choices(test, choices)
                        probe_scores = scores[:, int(p.global_deltas().argmax())]
                        paired[env]['profile_runs'][key] = {'profile_router': adapted_scores.tolist(),
                            'probe_only': probe_scores.tolist(),
                            'changed_action_fraction': float(np.mean(choices != base_choices))}
                        profile_records[f'{env}:{key}'] = history
                        for method, values, operations, searches in (
                                ('source_router', base_scores, 0, 1),
                                ('profile_router', adapted_scores, history['operations'], 1),
                                ('probe_only', probe_scores, history['operations'], 1),
                                ('rrf_extra', rrf_scores, 0, 2)):
                            for workload in WORKLOADS:
                                adaptation_rows.append({'environment': env, 'family': family, 'seed': seed,
                                    'budget': budget, 'method': method, 'ndcg': float(values.mean()),
                                    'delta_vs_source_router': float((values - base_scores).mean()),
                                    'onboarding_operations': operations, 'task_searches': searches,
                                    'workload': workload, 'total_operations': operations + searches * workload,
                                    'mean_profile_apply_seconds': profile_latency if method == 'profile_router' else 0.0})
    write_csv(args.output / 'headroom.csv', headroom_rows)
    write_csv(args.output / 'alignment.csv', alignment_rows)
    write_csv(args.output / 'adaptation.csv', adaptation_rows)
    write_json(args.output / 'profiles.json', profile_records)
    write_json(args.output / 'paired_outcomes.json', paired)
    write_json(args.output / 'source_models.json', trained_models)
    # Profile-first paired bootstrap, conditional on each tested environment.
    # Three histories only: no population confidence claim is warranted.
    intervals = []
    rng = np.random.default_rng(20260909)
    for env, panel in paired.items():
        base = np.asarray(panel['source_router'])
        for family in FAMILIES:
            runs = np.asarray([panel['profile_runs'][f'{family}:{seed}:256']['profile_router'] for seed in SEEDS])
            draws = []
            for _ in range(2000):
                histories = rng.integers(0, len(SEEDS), len(SEEDS))
                query_indices = rng.integers(0, len(base), len(base))
                draws.append(float((runs[histories][:, query_indices] - base[query_indices]).mean()))
            intervals.append({'environment': env, 'family': family, 'budget': 256,
                'mean_delta': float((runs - base).mean()),
                'conditional_95_low': float(np.quantile(draws, .025)),
                'conditional_95_high': float(np.quantile(draws, .975)),
                'histories': len(SEEDS), 'queries': len(base)})
    write_csv(args.output / 'conditional_intervals.csv', intervals)
    manifest['actions'] = list(ACTIONS)
    manifest['status'] = 'exploratory lexical ceiling-and-signal pilot; no learned selector or agent result'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--cache', type=Path, default=ROOT / 'cache')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--test-queries', type=int, default=150)
    parser.add_argument('--source-queries', type=int, default=256)
    parser.add_argument('--probes', type=int, default=64)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.test_queries < 1 or args.source_queries < 20 or args.probes < 1:
        parser.error('Need positive test/probe counts and at least 20 source queries')
    for path in (args.data, args.cache, args.output):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(args.data / 'hf'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    started = time.perf_counter()
    frozen_contract = generation_contract(args)
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if args.analyze_only else {
        'proposal_commit': 'a24a29a90c8dd6a91e5ee91f525ea0a1a2266392',
        'python': platform.python_version(), 'platform': platform.platform(),
        'model': MODEL, 'model_revision': REVISION, 'environments': {},
        'parameters': {'test_queries': args.test_queries, 'source_queries': args.source_queries,
                       'probes': args.probes, 'seeds': SEEDS, 'budgets': BUDGETS, 'workloads': WORKLOADS},
        'protocol_sha256': hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest()}
    records = []
    for name in ('scifact', 'fiqa'):
        for backend in ('bm25', 'dense'):
            if generation_contract(args) != frozen_contract:
                raise RuntimeError('Generation code/protocol changed while the run was active')
            if args.analyze_only:
                record = json.loads((args.cache / f'{name}_{backend}_outcomes.json').read_text())
                if record.get('generation_contract') != generation_contract(args):
                    raise ValueError('Outcome generation contract changed; regenerate outcomes before analyzing')
                records.append(record)
            else:
                records.append(build_environment(name, backend, args, manifest))
                if records[-1]['generation_contract'] != frozen_contract:
                    raise RuntimeError('Generation code/protocol changed while building an environment')
                write_json(manifest_path, manifest)
    analyze(records, args, manifest)
    manifest['last_analysis_total_seconds'] = time.perf_counter() - started
    manifest['generation_contract'] = records[0]['generation_contract']
    manifest['generation_runner_sha256'] = records[0]['generation_runner_sha256']
    manifest['analysis_code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in ROOT.glob('*.py')}
    write_json(manifest_path, manifest)
    print(json.dumps({'result_directory': str(args.output), 'elapsed_seconds': time.perf_counter() - started}), flush=True)


if __name__ == '__main__':
    main()
