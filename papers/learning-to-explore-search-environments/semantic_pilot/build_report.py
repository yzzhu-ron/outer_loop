"""Build a self-contained report from saved, verified experiment results."""
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import re

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def read_csv(name):
    with (RESULTS / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def num(value):
    return f'{float(value):.4f}'


def validate_cost_rows(adaptation, paired, records):
    """Reconstruct logical costs from the hashed retrieval records and choices."""
    keys = ('search_calls', 'sample_calls', 'llm_calls', 'input_tokens', 'output_tokens')
    for row in adaptation:
        data, record = paired[row['environment']], records[row['environment']]
        saved = data['runs'][f"{row['seed']}/{row['method']}/{row['budget']}"]
        pool = record['pools'][row['seed']]
        candidates = {candidate['id']: candidate for candidate in pool['candidates']}
        selected = saved['selected_probe_ids']
        if (len(selected) != len(set(selected)) or not set(selected) <= set(candidates)
                or len(selected) > int(row['budget'])
                or ((row['method'] == 'source_router' or int(row['budget']) == 0) and selected)):
            raise ValueError('Invalid selected probes in report cost evidence')
        setup = pool.get('setup_costs', {}) if int(row['budget']) > 0 and row['method'] != 'source_router' else {}
        for key in keys:
            onboarding = float(setup.get(key, 0))
            for identifier in selected:
                candidate = candidates[identifier]
                onboarding += float(candidate.get('cost_breakdown', {}).get(key, candidate['cost'] if key == 'search_calls' else 0))
            serving = sum(float(costs[action].get(key, 0)) for costs, action in
                          zip(record['tasks']['test']['action_costs'], saved['actions'], strict=True)) / len(saved['actions'])
            if (not math.isclose(float(row[f'onboarding_{key}']), onboarding, abs_tol=1e-12)
                    or not math.isclose(float(row[f'mean_task_{key}']), serving, abs_tol=1e-12)):
                raise ValueError('Report costs disagree with selected probes or serving actions')


def validate_cost_curves(rows, adaptation, protocol):
    """Require every frozen workload and connect its totals to the run ledger."""
    environments = {name + '_' + backend for name in protocol['datasets'] for backend in ('bm25', 'dense')}
    methods = ('source_router', 'random', 'fixed', 'information_gain', 'learned_value')
    required_runs = {(name, seed, method, budget) for name in environments for seed in protocol['probes']['seeds']
                     for method in methods for budget in protocol['probes']['pair_budgets']}
    run_key = lambda row: (row['environment'], int(row['seed']), row['method'], int(row['budget']))
    by_run = {run_key(row): row for row in adaptation}
    if len(adaptation) != len(required_runs) or set(by_run) != required_runs:
        raise ValueError('Cost curves require the complete adaptation grid')
    cost_keys = ('search_calls', 'sample_calls', 'llm_calls', 'input_tokens', 'output_tokens')
    for name in environments:
        baselines = [row for row in adaptation if row['environment'] == name and row['method'] == 'source_router']
        for row in baselines:
            if (any(float(row['onboarding_' + key]) != 0 for key in cost_keys)
                    or any(not math.isclose(float(row[key]), float(baselines[0][key]), abs_tol=1e-12)
                           for key in ('ndcg', 'recall_at_10', *['mean_task_' + key for key in cost_keys]))):
                raise ValueError('Source-router quality or costs vary across seeds or budgets')
    for row in adaptation:
        if any(not math.isfinite(float(row[prefix + key])) or float(row[prefix + key]) < 0
               for prefix in ('onboarding_', 'mean_task_') for key in cost_keys):
            raise ValueError('Adaptation costs must be finite and nonnegative')
    required = {(*key, workload) for key in required_runs for workload in (1, 10, 50, 200)}
    actual = [(*run_key(row), int(row['future_tasks'])) for row in rows]
    if len(actual) != len(required) or set(actual) != required:
        raise ValueError('Partial, duplicate or unexpected cost-curve grid')
    for row in rows:
        run = by_run[run_key(row)]
        workload = int(row['future_tasks'])
        for metric in ('ndcg', 'recall_at_10'):
            value = float(row[metric])
            if not 0 <= value <= 1 or not math.isclose(value, float(run[metric]), abs_tol=1e-12):
                raise ValueError('Cost-curve quality disagrees with adaptation')
        for key in cost_keys:
            total = float(row['total_' + key])
            expected = float(run['onboarding_' + key]) + workload * float(run['mean_task_' + key])
            if not math.isfinite(total) or total < 0 or not math.isclose(total, expected, abs_tol=1e-12):
                raise ValueError('Cost-curve total disagrees with onboarding plus serving')
    return sorted(environments)


def validate_results(headroom, adaptation, intervals):
    """Require the full frozen grid and cross-check assembled evidence."""
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    expected = {name + '_' + backend for name in protocol['datasets'] for backend in ('bm25', 'dense')}
    paired = json.loads((RESULTS / 'paired_outcomes.json').read_text())
    if set(paired) != expected or {r['environment'] for r in headroom} != expected or len(headroom) != len(expected):
        raise ValueError('Report requires every frozen environment exactly once')
    budgets = protocol['probes']['pair_budgets']
    seeds = protocol['probes']['seeds']
    methods = ('source_router', 'random', 'fixed', 'information_gain', 'learned_value')
    actual_keys = [(r['environment'], int(r['seed']), r['method'], int(r['budget'])) for r in adaptation]
    required = {(name, seed, method, budget) for name in expected for seed in seeds for method in methods for budget in budgets}
    if len(actual_keys) != len(required) or set(actual_keys) != required:
        raise ValueError('Partial, duplicate or unexpected adaptation grid')
    required_intervals = {(name, method, budget, comparator) for name in expected for method in methods[1:] for budget in budgets for comparator in ('source_router', 'random')}
    actual_intervals = [(r['environment'], r['method'], int(r['budget']), r['comparator']) for r in intervals]
    if len(actual_intervals) != len(required_intervals) or set(actual_intervals) != required_intervals:
        raise ValueError('Partial or duplicate uncertainty grid')
    for row in adaptation:
        saved = paired[row['environment']]['runs'][f"{row['seed']}/{row['method']}/{row['budget']}"]
        for metric in ('ndcg', 'recall_at_10'):
            if not math.isclose(float(row[metric]), sum(saved[metric]) / len(saved[metric]), abs_tol=1e-12):
                raise ValueError('Report metric disagrees with paired outcomes')
        changed = sum(action != baseline for action, baseline in
                      zip(saved['actions'], paired[row['environment']]['source_router_actions'], strict=True)) / len(saved['actions'])
        if not math.isclose(float(row['changed_action_fraction']), changed, abs_tol=1e-12):
            raise ValueError('Report action-change fraction disagrees with paired outcomes')
    for row in headroom:
        data = paired[row['environment']]
        scores = data['action_scores']
        recalls = data['action_recalls_at_10']
        base = [s[a] for s, a in zip(scores, data['source_router_actions'], strict=True)]
        base_recalls = [s[a] for s, a in zip(recalls, data['source_router_actions'], strict=True)]
        values = {'original': sum(s[0] for s in scores)/len(scores), 'source_router': sum(base)/len(base),
                  'query_oracle': sum(max(s) for s in scores)/len(scores),
                  'target_fixed_oracle': max(sum(s[a] for s in scores)/len(scores) for a in range(5)),
                  'original_recall_at_10': sum(s[0] for s in recalls)/len(recalls),
                  'source_router_recall_at_10': sum(base_recalls)/len(base_recalls)}
        values['oracle_gap_vs_router'] = values['query_oracle'] - values['source_router']
        values['oracle_gap_vs_target_fixed'] = values['query_oracle'] - values['target_fixed_oracle']
        if int(row['n_queries']) != len(scores) or any(not math.isclose(float(row[k]), v, abs_tol=1e-12) for k, v in values.items()):
            raise ValueError('Headroom disagrees with paired outcomes')
    metadata = json.loads((RESULTS / 'analysis_metadata.json').read_text())
    diagnostics = json.loads((RESULTS / 'diagnostics_provenance.json').read_text())
    sensitivity_provenance = json.loads((RESULTS / 'router_sensitivity_provenance.json').read_text())
    cost_provenance = json.loads((RESULTS / 'cost_plot_provenance.json').read_text())
    record_inputs = {f'cache/{name}_outcomes.json' for name in expected}
    diagnostic_inputs = {'results/paired_outcomes.json', 'analysis_protocol.json', 'cache/prepared.json'} | record_inputs
    diagnostic_outputs = {'paired_intervals.csv', 'headroom_intervals.csv', 'action_utilities.csv', 'probe_contrasts.csv',
                          'fusion_baseline.csv', 'figures/headroom.svg', 'figures/adaptation.svg', 'figures/alignment.svg'}
    sensitivity_inputs = {'results/models.json', 'results/paired_outcomes.json', 'results/analysis_metadata.json'} | record_inputs
    sensitivity_outputs = {'router_sensitivity.csv', 'router_sensitivity_queries.json'}
    if sensitivity_provenance['uses_unselected_probe_outcomes']:
        sensitivity_outputs.add('full_pool_sensitivity.csv')
    cost_inputs = {'cost_curves.csv', 'fusion_baseline.csv', 'adaptation.csv'}
    cost_outputs = {f'figures/cost_{category}.{suffix}' for category in ('search_calls', 'output_tokens') for suffix in ('png', 'svg')}
    if (set(diagnostics['input_sha256']) != diagnostic_inputs or set(diagnostics['output_sha256']) != diagnostic_outputs
            or set(sensitivity_provenance['input_sha256']) != sensitivity_inputs
            or set(sensitivity_provenance['output_sha256']) != sensitivity_outputs
            or set(cost_provenance['input_sha256']) != cost_inputs or set(cost_provenance['output_sha256']) != cost_outputs
            or cost_provenance['future_tasks'] != 50):
        raise ValueError('Incomplete or unexpected report evidence provenance')
    checks = [(ROOT / 'learning.py', metadata['learning_sha256']), (ROOT / 'protocol.json', metadata['protocol_sha256']),
              (ROOT / 'analyze_diagnostics.py', diagnostics['code_sha256']),
              (ROOT / 'audit_router_sensitivity.py', sensitivity_provenance['code_sha256']),
              (ROOT / 'plot_costs.py', cost_provenance['code_sha256']),
              (Path(__file__), cost_provenance['validation_sha256']),
              (ROOT / 'protocol.json', cost_provenance['protocol_sha256'])]
    checks.extend((ROOT / name, value) for name, value in diagnostics['input_sha256'].items())
    checks.extend((RESULTS / name, value) for name, value in diagnostics['output_sha256'].items())
    checks.extend((ROOT / name, value) for name, value in sensitivity_provenance['input_sha256'].items())
    checks.extend((RESULTS / name, value) for name, value in sensitivity_provenance['output_sha256'].items())
    checks.extend((RESULTS / name, value) for name, value in cost_provenance['input_sha256'].items())
    checks.extend((RESULTS / name, value) for name, value in cost_provenance['output_sha256'].items())
    for path, value in checks:
        if hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise ValueError(f'Report evidence drift: {path.name}')
    records = {name: json.loads((ROOT / 'cache' / f'{name}_outcomes.json').read_text()) for name in expected}
    validate_cost_rows(adaptation, paired, records)
    validate_cost_curves(read_csv('cost_curves.csv'), adaptation, protocol)
    sensitivity = read_csv('router_sensitivity.csv')
    actual_sensitivity = [(r['environment'], int(r['pair_budget'])) for r in sensitivity]
    required_sensitivity = {(name, budget) for name in expected for budget in budgets}
    if len(actual_sensitivity) != len(required_sensitivity) or set(actual_sensitivity) != required_sensitivity:
        raise ValueError('Partial, duplicate or unexpected sensitivity grid')
    for row in sensitivity:
        fixed = float(row['certified_unchangeable_fraction'])
        changeable = float(row['changeable_query_fraction_upper_bound'])
        if (int(row['n_queries']) != len(paired[row['environment']]['query_ids'])
                or not 0 <= fixed <= 1 or not 0 <= changeable <= 1
                or not math.isclose(fixed + changeable, 1, rel_tol=0, abs_tol=1e-12)):
            raise ValueError('Sensitivity cohort or fraction mismatch')


def table(headings, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + html.escape(h) + '</th>' for h in headings) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(value)) + '</td>' for value in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def figure(name, caption):
    svg = (RESULTS / 'figures' / (name + '.svg')).read_text()
    svg = svg[svg.index('<svg'):]
    # Namespace matplotlib IDs because multiple independent SVGs share one DOM.
    ids = re.findall(r'\bid="([^"]+)"', svg)
    for identifier in sorted(set(ids), key=len, reverse=True):
        svg = svg.replace(f'id="{identifier}"', f'id="{name}_{identifier}"')
        svg = svg.replace(f'href="#{identifier}"', f'href="#{name}_{identifier}"')
        svg = svg.replace(f'url(#{identifier})', f'url(#{name}_{identifier})')
    svg = svg.replace('<svg ', f'<svg role="img" aria-label="{html.escape(caption, quote=True)}" ', 1)
    return '<figure>' + svg + '<figcaption>' + html.escape(caption) + '</figcaption></figure>'


def main():
    headroom = read_csv('headroom.csv')
    adaptation = read_csv('adaptation.csv')
    intervals = read_csv('paired_intervals.csv')
    fusion = {r['environment']: r for r in read_csv('fusion_baseline.csv')}
    generation = json.loads((RESULTS / 'generation_audit.json').read_text())
    diagnostics = json.loads((RESULTS / 'selector_diagnostics.json').read_text())
    validate_results(headroom, adaptation, intervals)
    replacements = {}
    replacements['HEADROOM'] = table(['Environment', 'Queries', 'Original', 'Source router', 'Best fixed oracle', 'Per-query oracle', 'Oracle − fixed', 'Oracle − router'], [
        [r['environment'], r['n_queries'], num(r['original']), num(r['source_router']), num(r['target_fixed_oracle']), num(r['query_oracle']), num(r['oracle_gap_vs_target_fixed']), num(r['oracle_gap_vs_router'])]
        for r in headroom])
    rows = []
    for h in headroom:
        name = h['environment']
        candidates = [r for r in adaptation if r['environment'] == name and r['method'] == 'learned_value' and int(r['budget']) == 32]
        mean = lambda key: sum(float(r[key]) for r in candidates) / len(candidates)
        bounds = next(r for r in intervals if r['environment'] == name and r['method'] == 'learned_value' and r['budget'] == '32' and r['comparator'] == 'source_router')
        rows.append([name, num(mean('ndcg')), f"{num(bounds['mean'])} [{num(bounds['lo'])}, {num(bounds['hi'])}]", f"{100 * mean('changed_action_fraction'):.1f}%", num(fusion[name]['ndcg'])])
    replacements['ADAPTATION'] = table(['Environment', 'Learned / 32 pairs', 'Change [95% interval]', 'Actions changed', 'Source-selected RRF'], rows)
    replacements['RECALL'] = table(['Environment', 'Original', 'Source router', 'Learned / 32 pairs', 'RRF'], [
        [h['environment'], num(h['original_recall_at_10']), num(h['source_router_recall_at_10']),
         num(sum(float(r['recall_at_10']) for r in adaptation if r['environment'] == h['environment'] and r['method'] == 'learned_value' and r['budget'] == '32') / 3),
         num(fusion[h['environment']]['recall_at_10'])] for h in headroom])
    budget32 = [r for r in intervals if r['method'] == 'learned_value' and r['budget'] == '32' and r['comparator'] == 'source_router']
    mean_gain = sum(float(r['mean']) for r in budget32) / len(budget32)
    nonzero = sum(abs(float(r['mean'])) > 1e-12 for r in budget32)
    replacements['MEASURED_SUMMARY'] = html.escape(
        f"The semantic action menu leaves {min(float(r['oracle_gap_vs_target_fixed']) for r in headroom):.3f}–{max(float(r['oracle_gap_vs_target_fixed']) for r in headroom):.3f} nDCG@10 of per-query oracle headroom over the best fixed action. "
        f"At 32 probes, learned selection changes quality in {nonzero}/{len(headroom)} configurations, with a descriptive equal-family/backend average change of {mean_gain:+.4f} against the source router. The complete results appear below.")
    replacements['GENERATION'] = table(['Corpus', 'Valid question pairs', 'Valid probe rewrites', 'Valid task rewrites', 'Retained test / original'], [
        [name, f"{r['valid_question_pairs']}/{r['sampled_documents']}", f"{r['valid_probe_rewrites']}/{r['probe_rewrite_count']}", f"{r['valid_task_rewrites']}/{r['task_count']}", {'scifact': '249/300 → 150 evaluated', 'fiqa': '402/648 → 150 evaluated', 'nfcorpus': '65/323 → 65 evaluated'}[name]]
        for name, r in generation['datasets'].items()])
    replacements['SELECTOR_SIGNAL'] = table(['Target fold', 'Source utility queries', 'Training rows', 'Nonzero one-step labels', 'Positive labels'], [
        [name, ' + '.join(map(str, d['unique_utility_queries'].values())), d['training_examples'], f"{100*d['gain_nonzero_fraction']:.2f}%", f"{100*d['gain_positive_fraction']:.2f}%"]
        for name, d in diagnostics.items() if name != 'contract'])
    sensitivity = read_csv('router_sensitivity.csv')
    replacements['SENSITIVITY'] = table(['Environment', 'Provably unchanged at 32 pairs', 'Potentially changeable (upper bound)'], [
        [r['environment'], f"{100*float(r['certified_unchangeable_fraction']):.1f}%", f"{100*float(r['changeable_query_fraction_upper_bound']):.1f}%"]
        for r in sensitivity if r['pair_budget'] == '32'])
    costs = []
    for h in headroom:
        rows = [r for r in adaptation if r['environment'] == h['environment'] and r['method'] == 'learned_value' and r['budget'] == '32']
        avg = lambda key: sum(float(r[key]) for r in rows) / len(rows)
        costs.append([h['environment'], f"{avg('onboarding_sample_calls'):.0f}", f"{avg('onboarding_search_calls'):.1f}", f"{avg('onboarding_llm_calls'):.1f}", f"{avg('onboarding_output_tokens'):.0f}", f"{avg('mean_task_search_calls'):.2f}", f"{avg('mean_task_output_tokens'):.0f}"])
    replacements['COSTS'] = table(['Environment', 'Inspections', 'Probe searches', 'Setup LLM calls', 'Setup output tokens', 'Searches / task', 'Output tokens / task'], costs)
    for name, caption in [('headroom', 'Five shared actions, a target-label oracle, and source-selected reciprocal-rank fusion.'),
                          ('adaptation', 'All four selectors use the same source-trained router and calibration; bands are conditional descriptive intervals.'),
                          ('alignment', 'Synthetic known-document reward and real-query relevance are different endpoints; displayed means are correlated.'),
                          ('cost_search_calls', 'Quality versus logical search requests for onboarding plus 50 future tasks, averaged over three seeds.'),
                          ('cost_output_tokens', 'Quality versus generated output tokens for onboarding plus 50 future tasks; this is a separate resource axis, not a complete price or latency measure.')]:
        replacements['FIGURE_' + name.upper()] = figure(name, caption)
    content = (ROOT / 'report_template.html').read_text()
    for key, value in replacements.items():
        content = content.replace('@@' + key + '@@', value)
    if re.search(r'@@[A-Z_]+@@', content):
        raise ValueError('Unresolved report placeholder')
    (ROOT.parent / 'searchprobe-research-report.html').write_text(content)
    (RESULTS / 'report_provenance.json').write_text(json.dumps({
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'template_sha256': hashlib.sha256((ROOT / 'report_template.html').read_bytes()).hexdigest(),
        'report_sha256': hashlib.sha256(content.encode()).hexdigest(),
        'input_sha256': {str(p.relative_to(RESULTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RESULTS.rglob('*'))
                         if p.is_file() and p.suffix in ('.csv', '.svg', '.json') and p.name != 'report_provenance.json'}
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
