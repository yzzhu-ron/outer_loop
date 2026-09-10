"""Three-corpus semantic probe experiment; generation, retrieval, and analysis stages.

The first lexical pilot is preserved unchanged. LLM-generated passages are
retrieval queries, never evidence. Failed generation is retained and scored as
an unavailable action; no target task is removed for a bad rewrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / 'pilot'
REPO = ROOT.parents[2]
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(REPO / 'packages' / 'searchprobe' / 'src'))
from data import load_dataset, make_split, eligible_queries
from engine import BM25, DenseRetriever, query_features, feature_bucket, tokenize, transform
from engine import ndcg_at_k, reciprocal_rank, recall_at_k
from run_pilot import CachedSearch, select_ids, digest, write_json, rrf
from searchprobe import audit_records

ACTIONS = ('original', 'keywords', 'semantic', 'hyde', 'decomposed')
DATASETS = ('scifact', 'fiqa', 'nfcorpus')
SEEDS = (11, 23, 47)
MODEL_NAME = 'mlx-community/Qwen3-4B-Instruct-2507-4bit'
MODEL_REVISION = '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def code_contract():
    files = [ROOT / 'protocol.json', ROOT / 'generation.py', Path(__file__),
             PILOT / 'data.py', PILOT / 'engine.py']
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def archive_dir(name):
    return ROOT / 'data' if name == 'nfcorpus' else PILOT / 'data'


def model_path():
    return ROOT / 'data' / 'hf' / 'hub' / 'models--mlx-community--Qwen3-4B-Instruct-2507-4bit' / 'snapshots' / MODEL_REVISION


def dataset_inputs(name, args):
    docs, queries, qrels = load_dataset(archive_dir(name), name)
    exploration, heldout, split_audit = make_split(docs)
    all_ids = {doc['_id'] for doc in docs}
    test, test_audit = eligible_queries(qrels['test'], heldout, all_ids)
    train, train_audit = eligible_queries(qrels['train'], heldout, all_ids)
    source_ids = select_ids(train, args.source_queries, 'train-20260909')
    if len(source_ids) < 64:
        raise ValueError(f'Not enough eligible source queries in {name}')
    train_end = int(len(source_ids) * 2 / 3)
    cal_end = train_end + (len(source_ids) - train_end) // 2
    ids = {'train': source_ids[:train_end], 'calibration': source_ids[train_end:cal_end],
           'utility': source_ids[cal_end:],
           'test': select_ids(test, args.test_queries, 'test-20260909')}
    by_id = {doc['_id']: doc for doc in docs}
    pools = {}
    for seed in SEEDS:
        candidates = sorted(exploration)
        random.Random(seed).shuffle(candidates)
        pools[str(seed)] = candidates[:args.probe_documents]
    relevant = {part: {qid: (test if part == 'test' else train)[qid] for qid in selected}
                for part, selected in ids.items()}
    return docs, queries, ids, pools, relevant, {
        'split': split_audit, 'test': test_audit, 'train': train_audit,
        'archive_sha256': hashlib.sha256((archive_dir(name) / f'{name}.zip').read_bytes()).hexdigest(),
        'selected_test_queries': len(ids['test']), 'selected_source_queries': len(source_ids),
        'selected_test_blank_support_ids': [qid for qid in ids['test'] if any(
            score > 0 and not (by_id[d]['title'] + by_id[d]['text']).strip()
            for d, score in test[qid].items())],
    }


def generation_usage(row):
    usage = row.get('usage', {})
    return {'llm_calls': 1, 'input_tokens': int(usage.get('prompt_tokens', 0)),
            'output_tokens': int(usage.get('output_tokens', 0)),
            'seconds': float(usage.get('seconds', 0))}


def add_cost(target, values):
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def action_queries(query, row):
    generated = [[], [], []]
    if row.get('valid'):
        generated = [[row['semantic_query']], [row['hypothetical_document']], row['subqueries']]
    return [[query], [transform(query, 'keywords')], *generated]


def prepare(args):
    from generation import FrozenGenerator
    contract = code_contract()
    generator = FrozenGenerator(model_path(), ROOT / 'cache' / 'generation', batch_size=args.batch_size)
    preparation = {'contract': contract, 'model': MODEL_NAME, 'revision': MODEL_REVISION,
                   'datasets': {}, 'parameters': vars_json(args)}
    started = time.perf_counter()
    for name in DATASETS:
        docs, queries, ids, pools, relevant, audit = dataset_inputs(name, args)
        by_id = {doc['_id']: doc for doc in docs}
        sampled_ids = sorted(set(doc_id for pool in pools.values() for doc_id in pool))
        print(f'{name}: generating question pairs for {len(sampled_ids)} sampled documents', flush=True)
        generated_probes = generator.probe_many([by_id[doc_id] for doc_id in sampled_ids])
        probe_rows = dict(zip(sampled_ids, generated_probes))
        probe_queries, probe_keys = [], []
        for doc_id, row in probe_rows.items():
            for family in ('direct_question', 'indirect_question'):
                if row.get('valid') and row.get(family):
                    probe_keys.append((doc_id, family))
                    probe_queries.append(row[family])
        print(f'{name}: rewriting {len(probe_queries)} probe questions with the query-only generator', flush=True)
        rewritten_probes = generator.rewrite_many(probe_queries)
        probe_rewrites = {doc_id + ':' + family: row for (doc_id, family), row in zip(probe_keys, rewritten_probes)}
        all_task_ids = list(dict.fromkeys(qid for part in ids.values() for qid in part))
        print(f'{name}: rewriting {len(all_task_ids)} real queries (all labels kept outside generator)', flush=True)
        rewritten_tasks = generator.rewrite_many([queries[qid] for qid in all_task_ids])
        data = {'ids': ids, 'pools': pools, 'qrels': relevant, 'audit': audit,
                'queries': {qid: queries[qid] for qid in all_task_ids},
                'probe_questions': probe_rows, 'probe_rewrites': probe_rewrites,
                'task_rewrites': dict(zip(all_task_ids, rewritten_tasks)),
                'probe_audit_examples': [
                    {'document_id': doc_id, 'title': by_id[doc_id]['title'],
                     'body_excerpt': by_id[doc_id]['text'][:650],
                     'generated': probe_rows[doc_id]} for doc_id in sampled_ids[:4]]}
        preparation['datasets'][name] = data
        if contract != code_contract():
            raise RuntimeError('Generation contract changed during preparation')
        write_json(ROOT / 'cache' / 'prepared.json', preparation)
        print(f'{name}: generation saved; valid task rewrites {sum(r["valid"] for r in rewritten_tasks)}/{len(rewritten_tasks)}', flush=True)
    preparation['measured_prepare_seconds'] = time.perf_counter() - started
    write_json(ROOT / 'cache' / 'prepared.json', preparation)
    audit = {'model': MODEL_NAME, 'revision': MODEL_REVISION, 'datasets': {}}
    for name, data in preparation['datasets'].items():
        task_rows = list(data['task_rewrites'].values())
        question_rows = list(data['probe_questions'].values())
        rewrite_rows = list(data['probe_rewrites'].values())
        audit['datasets'][name] = {
            'split': data['audit'], 'task_count': len(task_rows),
            'valid_task_rewrites': sum(row['valid'] for row in task_rows),
            'sampled_documents': len(question_rows),
            'valid_question_pairs': sum(row['valid'] for row in question_rows),
            'valid_probe_rewrites': sum(row['valid'] for row in rewrite_rows),
            'probe_rewrite_count': len(rewrite_rows), 'examples': data['probe_audit_examples'],
            'logical_generation_cost': {
                key: sum(generation_usage(row)[key] for row in task_rows + question_rows + rewrite_rows)
                for key in ('llm_calls', 'input_tokens', 'output_tokens', 'seconds')},
            'actual_generation_cost': {
                key: sum(row.get('usage', {}).get('incurred_' + key, 0) for row in task_rows + question_rows + rewrite_rows)
                for key in ('prompt_tokens', 'output_tokens', 'seconds')},
        }
    write_json(ROOT / 'results' / 'generation_audit.json', audit)
    print(f'Preparation complete in {time.perf_counter() - started:.1f}s', flush=True)


def vars_json(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def search_actions(query, generated, search):
    queries = action_queries(query, generated)
    flattened = [q for group in queries for q in group]
    rankings = search.batch(flattened)
    outputs, costs = [], []
    offset = 0
    for action, group in enumerate(queries):
        ranks = rankings[offset:offset + len(group)]
        offset += len(group)
        result = rrf(ranks, tuple(range(len(ranks)))) if len(ranks) > 1 else ranks[0] if ranks else []
        outputs.append(result)
        costs.append({'search_calls': len(group), 'llm_calls': int(action >= 2),
                      'input_tokens': generation_usage(generated)['input_tokens'] if action >= 2 else 0,
                      'output_tokens': generation_usage(generated)['output_tokens'] if action >= 2 else 0})
    return outputs, costs


def retrieve(args):
    preparation = json.loads((ROOT / 'cache' / 'prepared.json').read_text())
    if preparation['contract'] != code_contract():
        raise ValueError('Prepared generations use a different contract; prepare again (valid generations remain cached)')
    os.environ.setdefault('HF_HOME', str(PILOT / 'data' / 'hf'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    contract = code_contract()
    manifest = {'contract': contract, 'parameters': preparation['parameters'],
                'actions': list(ACTIONS), 'environments': {}, 'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for name in DATASETS:
        docs, _, _, _, _, _ = dataset_inputs(name, args)
        data = preparation['datasets'][name]
        all_task_ids = list(dict.fromkeys(qid for part in data['ids'].values() for qid in part))
        # Frozen query encoder features are deployable and shared by all routers.
        dense_start = time.perf_counter()
        dense = DenseRetriever(docs, cache_dir=PILOT / 'data' / 'embeddings',
                               model_name=ENCODER, revision=ENCODER_REVISION)
        dense_load_seconds = time.perf_counter() - dense_start
        feature_start = time.perf_counter()
        embeddings = dense._encode([data['queries'][qid] for qid in all_task_ids])
        feature_seconds = time.perf_counter() - feature_start
        features = {qid: query_features(data['queries'][qid]) + embeddings[i].astype(float).tolist()
                    for i, qid in enumerate(all_task_ids)}
        for backend_name in ('bm25', 'dense'):
            name_full = name + '_' + backend_name
            index_start = time.perf_counter()
            backend = BM25(docs) if backend_name == 'bm25' else dense
            search = CachedSearch(backend, ROOT / 'cache' / f'{name_full}_ranks.json',
                digest({'corpus': digest(docs), 'encoder_identity': dense.cache_key,
                        'backend': backend_name, 'engine': contract['pilot/engine.py']}))
            record = {'name': name_full, 'corpus_name': name, 'backend': backend_name,
                      'contract': contract, 'tasks': {}, 'pools': {}}
            # Search outcomes for candidate probes are stored by the harness.
            # Deployment policies access these only through a selected-probe callback.
            for seed, sampled_ids in data['pools'].items():
                candidates, observations, traces, excluded = [], {}, [], []
                setup = {'sample_calls': len(sampled_ids), 'llm_calls': 0,
                         'input_tokens': 0, 'output_tokens': 0, 'seconds': 0.0}
                for doc_id in sampled_ids:
                    questions = data['probe_questions'][doc_id]
                    add_cost(setup, generation_usage(questions))
                    for family in ('direct_question', 'indirect_question'):
                        if not questions.get('valid') or not questions.get(family):
                            excluded.append({'document_id': doc_id, 'family': family, 'reason': 'invalid_question_generation'})
                            continue
                        query = questions[family]
                        rewritten = data['probe_rewrites'][doc_id + ':' + family]
                        add_cost(setup, generation_usage(rewritten))
                        groups = action_queries(query, rewritten)
                        ranks, _ = search_actions(query, rewritten, search)
                        for action in range(1, len(ACTIONS)):
                            candidate_id = digest([doc_id, family, action])[:24]
                            left, right = '\n'.join(groups[0]), '\n'.join(groups[action])
                            if not groups[action] or left == right:
                                excluded.append({'document_id': doc_id, 'family': family,
                                                 'action': ACTIONS[action], 'reason': 'invalid_action' if not groups[action] else 'identical_query'})
                                continue
                            cost = len(groups[0]) + len(groups[action])
                            candidates.append({'id': candidate_id, 'action': action, 'family': family,
                                'bucket': feature_bucket(query), 'features': query_features(query), 'cost': cost})
                            rr = [reciprocal_rank(ranks[a], doc_id) for a in (0, action)]
                            union = set(ranks[0]) | set(ranks[action])
                            observations[candidate_id] = {'delta': rr[1] - rr[0], 'rr_original': rr[0],
                                'rr_alternative': rr[1], 'overlap': len(set(ranks[0]) & set(ranks[action])) / max(1, len(union))}
                            rank_positions = [ranks[a].index(doc_id) + 1 if doc_id in ranks[a] else None for a in (0, action)]
                            traces.append({'probe_id': candidate_id, 'family': family,
                                'actions': ['original', ACTIONS[action]], 'queries': [left, right],
                                'bucket': str(feature_bucket(query)),
                                'known_target': {'ranks': rank_positions, 'cutoff': 10},
                                'cost': {'values': [len(groups[0]), len(groups[action])], 'total': cost, 'unit': 'search_calls'}})
                record['pools'][seed] = {'candidates': candidates, 'observations': observations,
                                         'setup_costs': setup, 'excluded': excluded}
                write_json(ROOT / 'results' / 'probe_diagnostics' / f'{name_full}_{seed}.json', audit_records(traces))
                trace_file = ROOT / 'results' / 'probe_traces' / f'{name_full}_{seed}.jsonl'
                trace_file.parent.mkdir(parents=True, exist_ok=True)
                trace_file.write_text(''.join(json.dumps(row) + '\n' for row in traces))
            for partition, ids in data['ids'].items():
                scores, recalls, rankings, costs = [], [], [], []
                for qid in ids:
                    rank, cost = search_actions(data['queries'][qid], data['task_rewrites'][qid], search)
                    label = data['qrels'][partition][qid]
                    scores.append([ndcg_at_k(r, label) for r in rank])
                    recalls.append([recall_at_k(r, label) for r in rank])
                    rankings.append(rank)
                    costs.append(cost)
                record['tasks'][partition] = {'ids': ids, 'features': [features[qid] for qid in ids],
                    'buckets': [feature_bucket(data['queries'][qid]) for qid in ids],
                    'scores': scores, 'recalls': recalls, 'rankings': rankings, 'action_costs': costs}
            search.save()
            if contract != code_contract():
                raise RuntimeError('Retrieval contract changed during the run')
            write_json(ROOT / 'cache' / f'{name_full}_outcomes.json', record)
            manifest['environments'][name_full] = {'data_audit': data['audit'], 'corpus_documents': len(docs),
                'uncached_searches': search.cache_misses, 'uncached_search_seconds': search.compute_seconds,
                'measured_task_and_probe_stage_seconds': time.perf_counter() - index_start,
                'dense_index_load_seconds_shared': dense_load_seconds,
                'query_feature_encoding_seconds_shared': feature_seconds,
                'probe_candidates_by_seed': {seed: len(pool['candidates']) for seed, pool in record['pools'].items()},
                'selected_task_ids': data['ids']}
            write_json(ROOT / 'results' / 'manifest.json', manifest)
            print(f'{name_full}: saved {len(data["ids"]["test"])} test queries; candidates {[len(p["candidates"]) for p in record["pools"].values()]}', flush=True)
        del dense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'retrieve', 'analyze'))
    parser.add_argument('--test-queries', type=int, default=150)
    parser.add_argument('--source-queries', type=int, default=192)
    parser.add_argument('--probe-documents', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    for path in (ROOT / 'data', ROOT / 'cache', ROOT / 'results'):
        path.mkdir(parents=True, exist_ok=True)
    if args.phase == 'prepare':
        prepare(args)
    elif args.phase == 'retrieve':
        retrieve(args)
    else:
        from learning import analyze_environments
        records = [json.loads((ROOT / 'cache' / f'{name}_{backend}_outcomes.json').read_text())
                   for name in DATASETS for backend in ('bm25', 'dense')]
        if any(record['contract'] != code_contract() for record in records):
            raise ValueError('Outcome cache contract changed; rerun preparation/retrieval')
        analyze_environments(records, ROOT / 'results')


if __name__ == '__main__':
    main()
