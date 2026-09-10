"""Render the separated source-world intervention with verified saved evidence."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FOLLOWUP = RESULTS / 'world_model_followup'
METHODS = ('random', 'fixed', 'information_gain', 'decision_value')
BUDGETS = (0, 4, 8, 16, 32)
SEEDS = (11, 23, 47)


def read(name):
    with (FOLLOWUP / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_checked(validate_cost_rows, validate_cost_curves):
    metadata = json.loads((FOLLOWUP / 'analysis_metadata.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    names = {c + '_' + b for c in protocol['datasets'] for b in ('bm25', 'dense')}
    required_inputs = {'results/analysis_metadata.json', 'results/paired_outcomes.json',
                       'results/diagnostics_provenance.json', 'protocol.json', 'world_model_protocol.json',
                       *{f'cache/{name}_outcomes.json' for name in names}}
    if set(metadata['input_provenance']) != required_inputs:
        raise ValueError('Incomplete follow-up input provenance')
    follow_protocol = json.loads((ROOT / 'world_model_protocol.json').read_text())
    canonical = hashlib.sha256(json.dumps(follow_protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (metadata['protocol'] != follow_protocol or metadata['protocol_canonical_sha256'] != canonical
            or metadata['protocol_file_sha256'] != metadata['input_provenance']['world_model_protocol.json']):
        raise ValueError('Follow-up protocol metadata mismatch')
    checks = [(ROOT / 'world_model_followup.py', metadata['code_sha256']),
              (ROOT / 'learning.py', metadata['learning_sha256'])]
    checks.extend((ROOT / name, value) for name, value in metadata['input_provenance'].items())
    checks.extend((FOLLOWUP / name, value) for name, value in metadata['output_sha256'].items())
    required_outputs = {'adaptation.csv', 'cost_curves.csv', 'baselines.csv', 'paired_intervals.csv',
                        'family_macro.csv', 'models.json', 'paired_outcomes.json', 'posterior_traces.json'}
    if set(metadata['output_sha256']) != required_outputs:
        raise ValueError('Incomplete follow-up output provenance')
    for path, value in checks:
        if hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise ValueError('Follow-up evidence drift: ' + path.name)
    paired = json.loads((FOLLOWUP / 'paired_outcomes.json').read_text())
    original = json.loads((RESULTS / 'paired_outcomes.json').read_text())
    records = {name: json.loads((ROOT / 'cache' / f'{name}_outcomes.json').read_text()) for name in names}
    adaptation, intervals, baselines = read('adaptation.csv'), read('paired_intervals.csv'), read('baselines.csv')
    if set(paired) != names:
        raise ValueError('Incomplete follow-up environments')
    baseline_names = ('zero_budget', 'original_frozen_router', 'original_query')
    keys = [(r['environment'], r['baseline']) for r in baselines]
    if len(keys) != len(names)*3 or set(keys) != {(n, b) for n in names for b in baseline_names}:
        raise ValueError('Incomplete follow-up baseline grid')
    def check_actions(data, actions):
        if len(actions) != len(data['query_ids']) or any(type(a) is not int or not 0 <= a < 5 for a in actions):
            raise ValueError('Invalid follow-up actions')
    for name in names:
        data, evidence = paired[name], original[name]
        if data['query_ids'] != evidence['query_ids'] or set(data['baselines']) != set(baseline_names):
            raise ValueError('Follow-up query cohort or baselines changed')
        required_runs = {f'{s}/{m}/{b}' for s in SEEDS for m in METHODS for b in BUDGETS}
        if set(data['runs']) != required_runs:
            raise ValueError('Incomplete follow-up paired run grid')
        for baseline in data['baselines'].values():
            check_actions(data, baseline['actions'])
        if (data['baselines']['original_frozen_router']['actions'] != evidence['source_router_actions']
                or data['baselines']['original_query']['actions'] != [0]*len(data['query_ids'])):
            raise ValueError('Follow-up reference baseline changed')
        for run in data['runs'].values():
            check_actions(data, run['actions'])
    for row in baselines:
        data, evidence, record = paired[row['environment']], original[row['environment']], records[row['environment']]
        baseline = data['baselines'][row['baseline']]
        if int(row['n_queries']) != len(data['query_ids']):
            raise ValueError('Follow-up baseline cohort mismatch')
        for metric, source in [('ndcg', 'action_scores'), ('recall_at_10', 'action_recalls_at_10')]:
            expected = [s[a] for s, a in zip(evidence[source], baseline['actions'], strict=True)]
            if baseline[metric] != expected or not math.isclose(float(row[metric]), sum(expected)/len(expected), abs_tol=1e-12):
                raise ValueError('Follow-up baseline quality mismatch')
        for key in ('search_calls', 'sample_calls', 'llm_calls', 'input_tokens', 'output_tokens'):
            expected = sum(float(c[a].get(key, 0)) for c, a in zip(record['tasks']['test']['action_costs'], baseline['actions'], strict=True))/len(baseline['actions'])
            if not math.isclose(float(row['mean_task_' + key]), expected, abs_tol=1e-12):
                raise ValueError('Follow-up baseline cost mismatch')
    required = {(n, s, m, b) for n in names for s in SEEDS for m in METHODS for b in BUDGETS}
    keys = [(r['environment'], int(r['seed']), r['method'], int(r['budget'])) for r in adaptation]
    if len(keys) != len(required) or set(keys) != required:
        raise ValueError('Incomplete follow-up run grid')
    required_intervals = {(n, m, b, c) for n in names for m in METHODS for b in BUDGETS
                          for c in ('zero_budget', 'random', 'original_frozen_router')}
    keys = [(r['environment'], r['method'], int(r['budget']), r['comparator']) for r in intervals]
    if len(keys) != len(required_intervals) or set(keys) != required_intervals:
        raise ValueError('Incomplete follow-up interval grid')
    for row in adaptation:
        data = paired[row['environment']]
        evidence = original[row['environment']]
        if data['query_ids'] != evidence['query_ids']:
            raise ValueError('Follow-up query cohort changed')
        run = data['runs'][f"{row['seed']}/{row['method']}/{row['budget']}"]
        for metric, source in [('ndcg', 'action_scores'), ('recall_at_10', 'action_recalls_at_10')]:
            expected = [scores[a] for scores, a in zip(evidence[source], run['actions'], strict=True)]
            if run[metric] != expected or not math.isclose(float(row[metric]), sum(expected)/len(expected), abs_tol=1e-12):
                raise ValueError('Follow-up quality mismatch')
        zero = data['baselines']['zero_budget']
        if int(row['budget']) == 0 and (run['actions'] != zero['actions'] or run['selected_probe_ids']):
            raise ValueError('Follow-up budget zero differs from own prior')
        changed = sum(a != b for a, b in zip(run['actions'], zero['actions'], strict=True))/len(run['actions'])
        if not math.isclose(float(row['changed_action_fraction_vs_zero_budget']), changed, abs_tol=1e-12):
            raise ValueError('Follow-up action-change mismatch')
    for row in intervals:
        data = paired[row['environment']]
        deltas = []
        for seed in SEEDS:
            run = data['runs'][f"{seed}/{row['method']}/{row['budget']}"]
            baseline = (data['runs'][f"{seed}/random/{row['budget']}"] if row['comparator'] == 'random'
                        else data['baselines'][row['comparator']])
            deltas.extend(a-b for a, b in zip(run['ndcg'], baseline['ndcg'], strict=True))
        if (not math.isclose(float(row['mean']), sum(deltas)/len(deltas), abs_tol=1e-12)
                or not -1 <= float(row['lo']) <= float(row['hi']) <= 1):
            raise ValueError('Follow-up interval contrast mismatch')
    validate_cost_rows(adaptation, paired, records)
    validate_cost_curves(read('cost_curves.csv'), adaptation, protocol, methods=METHODS)
    return sorted(names), adaptation, intervals, baselines


def fragment(table, figure, validate_cost_rows, validate_cost_curves):
    names, adaptation, intervals, baselines = load_checked(validate_cost_rows, validate_cost_curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.hashsalt': 'source-world-followup',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), constrained_layout=True)
    for ax, name in zip(axes.flat, names, strict=True):
        for method, color in zip(METHODS, ('#666666', '#477A63', '#497AAB', '#B35632')):
            rows = [r for r in intervals if r['environment'] == name and r['method'] == method and r['comparator'] == 'zero_budget']
            rows.sort(key=lambda r: int(r['budget']))
            x = [int(r['budget']) for r in rows]
            ax.plot(x, [float(r['mean']) for r in rows], color=color,
                    linestyle='--' if method == 'information_gain' else '-',
                    marker=None if method == 'information_gain' else 'o', ms=3,
                    zorder=3 if method == 'information_gain' else 2,
                    label=method.replace('_', ' '))
            if method == 'decision_value':
                ax.fill_between(x, [float(r['lo']) for r in rows], [float(r['hi']) for r in rows], color=color, alpha=.12)
        ax.axhline(0, color='#444444', lw=.7)
        ax.set(title=name.replace('_', ' / '), xlabel='Selected probe pairs', ylabel='Δ nDCG@10 vs own prior', xticks=BUDGETS)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Exploratory intervention: route through a posterior over source worlds\nShading: conditional query/seed interval for decision-value selection', fontsize=12)
    for suffix in ('svg', 'png'):
        fig.savefig(RESULTS / 'figures' / f'world_model_followup.{suffix}', dpi=170, bbox_inches='tight',
                    **({'metadata': {'Date': None}} if suffix == 'svg' else {}))
    plt.close(fig)
    rows, costs = [], []
    for name in names:
        selected = [r for r in adaptation if r['environment'] == name and r['method'] == 'decision_value' and r['budget'] == '32']
        mean = lambda key: sum(float(r[key]) for r in selected)/len(selected)
        prior = next(r for r in baselines if r['environment'] == name and r['baseline'] == 'zero_budget')
        frozen = next(r for r in baselines if r['environment'] == name and r['baseline'] == 'original_frozen_router')
        comparisons = {r['comparator']: r for r in intervals if r['environment'] == name and r['method'] == 'decision_value' and r['budget'] == '32'}
        signed = lambda v: f"{0.0 if abs(float(v)) < .00005 else float(v):+.4f}"
        label = lambda r: f"{signed(r['mean'])} [{signed(r['lo'])}, {signed(r['hi'])}]"
        rows.append([name, f"{float(frozen['ndcg']):.4f}", f"{float(prior['ndcg']):.4f}", f"{mean('ndcg'):.4f}", label(comparisons['zero_budget']),
                     label(comparisons['random']), f"{100*mean('changed_action_fraction_vs_zero_budget'):.1f}%"])
        costs.append([name, f"{mean('onboarding_sample_calls'):.0f}", f"{mean('onboarding_llm_calls'):.0f}",
                      f"{mean('onboarding_search_calls'):.1f}", f"{mean('onboarding_output_tokens'):.0f}",
                      f"{mean('mean_task_search_calls'):.2f}", f"{mean('mean_task_output_tokens'):.0f}",
                      f"{float(prior['recall_at_10']):.4f} → {mean('recall_at_10'):.4f}"])
    return (
        '<section id="followup"><h2>Intervening on the unresponsive router</h2>'
        '<p>This follow-up was designed after the original semantic result, then fixed before evaluating its target outcomes. It preserves the original run. Two source corpora become two possible worlds; selected probe outcomes update their weights. Each future query uses its bucket to choose the action with greatest posterior-weighted source utility.</p>'
        '<p>Only source router-training queries estimate utilities and task mix. Source probes estimate Gaussian observation likelihoods. Random, fixed coverage, information gain and decision value share that model. The primary comparison below is with this model’s own zero-budget prior router, separating probe effects from a different starting policy.</p>'
        + table(['Environment', 'Original router', 'Own prior', 'Decision / 32', 'Gain over prior [95%]', 'Gain over random [95%]', 'Actions changed'], rows)
        + '<p><strong>The strongest descriptive result is SciFact/dense:</strong> all three decision-value runs choose HyDE for every test query at budget 32, reaching the target-best fixed action. This improves on both starting policies, but does not recover the per-query oracle’s additional headroom. Every own-prior interval above includes zero. The other environments are mixed, and information gain has identical quality to decision value at budget 32 in all six environments despite different selected histories.</p>'
        + figure('world_model_followup', 'A separate exploratory intervention. Dashed information-gain curves overlay decision-value curves where they coincide. Every selector and budget is shown; intervals do not establish population transfer or adjust for model development.')
        + '<p>The Gaussian likelihoods are uncalibrated. Repeated probes share documents and queries, violating conditional independence; posterior concentration is not verified confidence. All probes remain in long-query buckets, so short-query decisions extrapolate through the assumed global world.</p>'
        '<p>With two worlds and shared Gaussian variance, equal-cost channels are ordered by standardized mean separation. Exact information gain and Bayes decision value prefer the same most informative equal-cost channel. Differences here can come from costs, ties or quadrature; this intervention cannot establish a benefit from separating nuisance information in a richer world model.</p>'
        + table(['Environment', 'Inspections', 'Setup LLM calls', 'Setup searches', 'Setup output tokens', 'Searches / task', 'Output tokens / task', 'Recall@10 prior → 32'], costs)
        + '<p>Costs above are for decision value at 32 pairs, averaged over seeds. Setup includes the full candidate pool. The <a href="semantic_pilot/results/world_model_followup/cost_curves.csv">complete ledger</a> retains all resources and four workloads. See the <a href="semantic_pilot/world_model_protocol.json">follow-up protocol</a> and <a href="semantic_pilot/results/world_model_followup/paired_intervals.csv">all paired comparisons</a>, including the original frozen router.</p></section>')
