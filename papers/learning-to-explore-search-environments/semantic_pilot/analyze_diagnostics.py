"""Descriptive paired uncertainty, proxy alignment, and source-selected RRF.

This is evaluator code. Target labels only score frozen decisions. The extra
analysis protocol was written before semantic retrieval outcomes were observed.
Intervals condition on this small exploratory panel; they do not certify new
corpus generalization or correct for comparing many methods and budgets.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'pilot'))
from engine import ndcg_at_k, recall_at_k
from run_pilot import rrf

ACTIONS = ('original', 'keywords', 'semantic', 'hyde', 'decomposed')
METHODS = ('random', 'fixed', 'information_gain', 'learned_value')
BUDGETS = (0, 4, 8, 16, 32)


def validate_inputs(paired, records, prepared):
    """Refuse to combine stale cohorts, action menus, or retrieval contracts."""
    if set(paired) != set(records):
        raise ValueError('Paired and retrieval environment sets differ')
    for name, data in paired.items():
        record = records[name]
        tasks = record['tasks']['test']
        corpus = prepared['datasets'][record['corpus_name']]
        if record['contract'] != prepared['contract']:
            raise ValueError(f'{name}: preparation/retrieval contract mismatch')
        if data['query_ids'] != tasks['ids'] or tasks['ids'] != corpus['ids']['test']:
            raise ValueError(f'{name}: test query ID/order mismatch')
        for partition in ('train', 'calibration', 'utility'):
            if partition in record['tasks'] and record['tasks'][partition]['ids'] != corpus['ids'][partition]:
                raise ValueError(f'{name}: {partition} query ID/order mismatch')
        if list(data['action_names']) != list(ACTIONS):
            raise ValueError(f'{name}: action menu mismatch')
        if not np.array_equal(data['action_scores'], tasks['scores']):
            raise ValueError(f'{name}: paired and retrieval action scores differ')
        if not np.array_equal(data['action_recalls_at_10'], tasks['recalls']):
            raise ValueError(f'{name}: paired and retrieval action recalls differ')
        if len(tasks['ids']) != len(set(tasks['ids'])) or set(tasks['ids']) != set(corpus['qrels']['test']):
            raise ValueError(f'{name}: duplicate IDs or relevance cohort mismatch')
        truth = np.asarray(tasks['scores'])
        recalls = np.asarray(tasks['recalls'])
        baseline = np.asarray(data['source_router_actions'])
        if baseline.shape != (len(truth),) or not np.issubdtype(baseline.dtype, np.integer) or np.any((baseline < 0) | (baseline >= 5)):
            raise ValueError(f'{name}: invalid standalone baseline actions')
        for key, run in data['runs'].items():
            actions = np.asarray(run['actions'])
            if actions.shape != (len(truth),) or not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions >= 5)):
                raise ValueError(f'{name}/{key}: invalid selected actions')
            if not np.array_equal(run['ndcg'], truth[np.arange(len(truth)), actions]):
                raise ValueError(f'{name}/{key}: selected outcomes disagree with retrieval cache')
            if not np.array_equal(run['recall_at_10'], recalls[np.arange(len(recalls)), actions]):
                raise ValueError(f'{name}/{key}: selected recalls disagree with retrieval cache')
            if '/source_router/' in key and not np.array_equal(actions, baseline):
                raise ValueError(f'{name}/{key}: baseline actions differ between saved outputs')


def write_csv(name, rows):
    path = ROOT / 'results' / name
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(deltas, rng, draws=2000):
    """Paired query and shared-seed bootstrap of a seed-by-query difference."""
    deltas = np.atleast_2d(deltas)
    seed_ids = rng.integers(deltas.shape[0], size=(draws, deltas.shape[0]))
    query_ids = rng.integers(deltas.shape[1], size=(draws, deltas.shape[1]))
    boot = deltas[seed_ids[:, :, None], query_ids[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {'mean': float(deltas.mean()), 'lo': float(lo), 'hi': float(hi)}


def main():
    result = ROOT / 'results'
    paired = json.loads((result / 'paired_outcomes.json').read_text())
    prepared = json.loads((ROOT / 'cache' / 'prepared.json').read_text())
    records = {name: json.loads((ROOT / 'cache' / f'{name}_outcomes.json').read_text()) for name in paired}
    validate_inputs(paired, records, prepared)
    analysis_metadata = json.loads((result / 'analysis_metadata.json').read_text())
    for name, record in records.items():
        record_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if record_hash != analysis_metadata['input_record_sha256'][name]:
            raise ValueError(f'{name}: retrieval record differs from the fitted analysis input')
    rng = np.random.default_rng(20260910)
    intervals, action_rows, probe_rows, fusion_rows, headroom_intervals = [], [], [], [], []
    for name in sorted(paired):
        data, record = paired[name], records[name]
        tasks = record['tasks']['test']
        truth = np.asarray(data['action_scores'])
        indices = np.arange(len(truth))
        base = truth[indices, data['source_router_actions']]
        seeds = sorted(record['pools'])
        for method in METHODS:
            for budget in BUDGETS:
                current = np.array([data['runs'][f'{seed}/{method}/{budget}']['ndcg'] for seed in seeds])
                for comparator in ('source_router', 'random'):
                    reference = base[None, :] if comparator == 'source_router' else np.array([
                        data['runs'][f'{seed}/random/{budget}']['ndcg'] for seed in seeds])
                    bounds = interval(current - reference, rng)
                    intervals.append({'environment': name, 'method': method, 'budget': budget,
                                      'comparator': comparator, **bounds})
        oracle = truth.max(axis=1)
        fixed_action = truth.mean(axis=0).argmax()
        for comparator, values in [('source_router', base), ('target_fixed_oracle', truth[:, fixed_action])]:
            headroom_intervals.append({'environment': name, 'comparator': comparator,
                                      **interval(oracle - values, rng)})
        for action, label in enumerate(ACTIONS):
            for bucket in (-1, 0, 1, 2, 3):
                mask = np.ones(len(truth), dtype=bool) if bucket == -1 else np.asarray(tasks['buckets']) == bucket
                if not mask.any():
                    continue
                action_rows.append({'environment': name, 'action': label, 'bucket': bucket,
                    'n_queries': int(mask.sum()), 'mean_ndcg': float(truth[mask, action].mean()),
                    'mean_recall': float(np.asarray(tasks['recalls'])[mask, action].mean()),
                    'mean_delta_vs_original': float((truth[mask, action] - truth[mask, 0]).mean()),
                    'strict_win_fraction_vs_original': float(np.mean(truth[mask, action] > truth[mask, 0]))})
        # Candidate IDs identify the same pair across overlapping seed pools.
        # Deduplicate them here; deployed costs remain charged per pool.
        unique = {}
        for pool in record['pools'].values():
            for candidate in pool['candidates']:
                unique[candidate['id']] = (candidate, pool['observations'][candidate['id']])
        for family in ('direct_question', 'indirect_question'):
            for action in range(1, 5):
                for bucket in (-1, 0, 1, 2, 3):
                    rows = [(c, o) for c, o in unique.values() if c['family'] == family and c['action'] == action
                            and (bucket == -1 or c['bucket'] == bucket)]
                    if not rows:
                        continue
                    probe_rows.append({'environment': name, 'family': family, 'action': ACTIONS[action],
                        'bucket': bucket, 'n_unique_pairs': len(rows),
                        'mean_delta': float(np.mean([o['delta'] for _, o in rows])),
                        'nonzero_fraction': float(np.mean([abs(o['delta']) > 1e-12 for _, o in rows])),
                        'both_rank_one_fraction': float(np.mean([o['rr_original'] == 1 and o['rr_alternative'] == 1 for _, o in rows])),
                        'both_absent_fraction': float(np.mean([o['rr_original'] == 0 and o['rr_alternative'] == 0 for _, o in rows]))})
        source = [r for r in records.values() if r['corpus_name'] != record['corpus_name'] and r['backend'] == record['backend']]
        source_means = np.mean([np.asarray(r['tasks']['train']['scores']).mean(axis=0) for r in source], axis=0)
        alternative = int(np.argmax(source_means[1:])) + 1
        labels = prepared['datasets'][record['corpus_name']]['qrels']['test']
        scores, recalls = [], []
        for qid, rankings in zip(tasks['ids'], tasks['rankings'], strict=True):
            ranking = rrf([rankings[0], rankings[alternative]], (0, 1))
            scores.append(ndcg_at_k(ranking, labels[qid]))
            recalls.append(recall_at_k(ranking, labels[qid]))
        costs = {key: float(np.mean([sum(row[a].get(key, 0) for a in (0, alternative)) for row in tasks['action_costs']]))
                 for key in ('search_calls', 'llm_calls', 'input_tokens', 'output_tokens')}
        fusion_rows.append({'environment': name, 'alternative': ACTIONS[alternative], 'n_queries': len(truth),
            'ndcg': float(np.mean(scores)), 'recall_at_10': float(np.mean(recalls)),
            'delta_vs_source_router': float(np.mean(scores - base)), **{f'mean_task_{k}': v for k, v in costs.items()}})
    for filename, rows in [('paired_intervals.csv', intervals), ('headroom_intervals.csv', headroom_intervals),
                           ('action_utilities.csv', action_rows), ('probe_contrasts.csv', probe_rows), ('fusion_baseline.csv', fusion_rows)]:
        write_csv(filename, rows)
    plot_figures(intervals, headroom_intervals, action_rows, probe_rows, fusion_rows)
    (result / 'diagnostics_provenance.json').write_text(json.dumps({
        'analysis_protocol': json.loads((ROOT / 'analysis_protocol.json').read_text()),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [result / 'paired_outcomes.json', ROOT / 'analysis_protocol.json', ROOT / 'cache' / 'prepared.json',
             *[ROOT / 'cache' / f'{name}_outcomes.json' for name in sorted(records)]]},
        'output_sha256': {name: hashlib.sha256((result / name).read_bytes()).hexdigest() for name in
            ('paired_intervals.csv', 'headroom_intervals.csv', 'action_utilities.csv', 'probe_contrasts.csv', 'fusion_baseline.csv',
             'figures/headroom.svg', 'figures/adaptation.svg', 'figures/alignment.svg')},
        'interpretation': 'Conditional descriptive bootstrap; shared profiles, tasks and corpora are not independent population evidence.'
    }, indent=2) + '\n')


def plot_figures(intervals, headroom, action_rows, probes, fusion):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'semantic-pilot-20260909'})
    out = ROOT / 'results' / 'figures'
    out.mkdir(exist_ok=True)
    names = sorted({r['environment'] for r in intervals})
    colors = {'random': '#666666', 'fixed': '#477A63', 'information_gain': '#497AAB', 'learned_value': '#B35632'}

    def save(fig, name):
        fig.savefig(out / f'{name}.png', dpi=170, bbox_inches='tight')
        fig.savefig(out / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), constrained_layout=True)
    for ax, name in zip(axes.ravel(), names):
        for method in METHODS:
            rows = [r for r in intervals if r['environment'] == name and r['method'] == method and r['comparator'] == 'source_router']
            ax.plot([r['budget'] for r in rows], [r['mean'] for r in rows], marker='o', ms=3, color=colors[method], label=method.replace('_', ' '))
            if method == 'learned_value':
                ax.fill_between([r['budget'] for r in rows], [r['lo'] for r in rows], [r['hi'] for r in rows], color=colors[method], alpha=.12)
        ax.axhline(0, color='#444444', lw=.7)
        ax.set(title=name.replace('_', ' / '), xlabel='Selected probe pairs', ylabel='nDCG@10 change from source router')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Semantic probes: changes in downstream routing quality\nShading: descriptive 95% query/seed bootstrap for learned selection', fontsize=12)
    save(fig, 'adaptation')

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), constrained_layout=True)
    for ax, name in zip(axes.ravel(), names):
        rows = [r for r in action_rows if r['environment'] == name and r['bucket'] == -1]
        ax.bar(range(5), [r['mean_ndcg'] for r in rows], color=['#444444', '#A0A0A0', '#6E8E76', '#B87859', '#7392AC'])
        oracle_rows = [r for r in headroom if r['environment'] == name]
        fixed_mean = max(r['mean_ndcg'] for r in rows)
        oracle = fixed_mean + next(r['mean'] for r in oracle_rows if r['comparator'] == 'target_fixed_oracle')
        ax.axhline(oracle, color='#B35632', ls='--', label='per-query oracle')
        fusion_quality = next(r['ndcg'] for r in fusion if r['environment'] == name)
        ax.axhline(fusion_quality, color='#497AAB', ls=':', label='source-selected RRF')
        ax.set_xticks(range(5), ['original', 'keywords', 'semantic', 'HyDE', 'decompose'], rotation=25, ha='right')
        ax.set(title=name.replace('_', ' / '), ylabel='Mean nDCG@10', ylim=(0, min(1, max(oracle, fusion_quality) + .07)))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Action quality and attainable headroom on the retained test queries', fontsize=12)
    save(fig, 'headroom')

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), constrained_layout=True)
    markers = {'direct_question': 'o', 'indirect_question': '^'}
    for ax, name in zip(axes.ravel(), names):
        for family in markers:
            rows = [r for r in probes if r['environment'] == name and r['family'] == family and r['bucket'] == -1]
            for r in rows:
                actual = next(a['mean_delta_vs_original'] for a in action_rows if a['environment'] == name and a['bucket'] == -1 and a['action'] == r['action'])
                ax.scatter(r['mean_delta'], actual, marker=markers[family], color='#B35632' if family == 'indirect_question' else '#497AAB', s=24)
                if family == 'indirect_question':
                    ax.annotate(r['action'], (r['mean_delta'], actual), fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.axhline(0, color='#999999', lw=.7); ax.axvline(0, color='#999999', lw=.7)
        ax.set(title=name.replace('_', ' / '), xlabel='Known-document RR change', ylabel='Real-query nDCG@10 change')
    fig.suptitle('Proxy action contrasts versus real-task action contrasts\nCircles: direct questions; triangles: indirect questions. Correlated descriptive means.', fontsize=12)
    save(fig, 'alignment')


if __name__ == '__main__':
    main()
