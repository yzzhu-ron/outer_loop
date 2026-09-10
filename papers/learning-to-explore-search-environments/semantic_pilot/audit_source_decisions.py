"""Apply SearchProbe to the two measured source worlds in every frozen fold.

Specified before semantic retrieval: use only the 128 router-training queries
of each source corpus; equally weight queries within a world. This diagnostic
does not change the learned policy or use target test labels. Fixed actions
are the five supplied policies, so its oracle is not a per-query oracle.
"""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / 'packages' / 'searchprobe' / 'src'))
from searchprobe.decisions import audit_decision_model

ACTIONS = ['original', 'keywords', 'semantic', 'hyde', 'decomposed']
CORPORA = ('scifact', 'fiqa', 'nfcorpus')


def source_model(records, target, backend):
    sources = sorted((r for r in records if r['corpus_name'] != target and r['backend'] == backend), key=lambda r: r['name'])
    if len(sources) != 2 or {r['corpus_name'] for r in sources} != set(CORPORA) - {target}:
        raise ValueError('Each model requires exactly the other two source corpora')
    utilities = []
    for record in sources:
        training = record['tasks']['train']
        scores = training['scores']
        if len(scores) != 128 or len(set(training['ids'])) != 128 or len(training['ids']) != len(scores):
            raise ValueError('Expected 128 unique source training queries')
        if any(len(row) != len(ACTIONS) for row in scores):
            raise ValueError('Wrong action menu in source scores')
        rational = [[Fraction(str(v)) for v in row] for row in scores]
        if any(not 0 <= value <= 1 for row in rational for value in row):
            raise ValueError('Source nDCG must be bounded')
        utilities.append([str(sum(row[a] for row in rational) / len(scores)) for a in range(len(ACTIONS))])
    return {
        'schema_version': 1, 'kind': 'source_utility_model',
        'model_id': f'semantic-training-sources-for-{target}-{backend}',
        'actions': ACTIONS, 'world_ids': [r['name'] for r in sources], 'utilities': utilities,
        'provenance': {
            'utility_source': 'Measured BEIR nDCG@10 for the five frozen query actions; input hashes and IDs in source_decision_provenance.json.',
            'source_split': 'Only router-training partitions of the two other corpora; no source calibration/utility or target-task labels.',
            'task_distribution': 'Uniform over 128 selected source training queries within each world. Cohorts differ by world. The minimax prior is adversarial, not an empirical corpus frequency.',
            'synthetic': False,
        },
        'assumptions': [
            'Conditional finite diagnostic: the two source worlds define this model, without asserting target coverage or observational indistinguishability.',
            'Recorded floating-point nDCG values are converted to exact decimal rationals; exact optimization does not remove measurement or sampling uncertainty.',
            'Each policy uses one fixed query action on every task; the oracle chooses the best fixed policy within each source world.',
            'A conflict can motivate exploration only if permitted observations can distinguish its decision-relevant worlds; this audit does not establish that link.',
        ],
    }


def main():
    inputs = [ROOT / 'cache' / f'{corpus}_{backend}_outcomes.json' for corpus in CORPORA for backend in ('bm25', 'dense')]
    records = [json.loads(p.read_text()) for p in inputs]
    results = ROOT / 'results'
    directory = results / 'source_decision_models'
    directory.mkdir(exist_ok=True)
    rows, outputs = [], []
    for target in CORPORA:
        for backend in ('bm25', 'dense'):
            model = source_model(records, target, backend)
            report = audit_decision_model(model)
            for suffix, data in [('model', model), ('audit', report)]:
                path = directory / f'{target}_{backend}_{suffix}.json'
                path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
                outputs.append(path)
            rows.append({'target_fold': f'{target}_{backend}',
                'world_ids': '|'.join(model['world_ids']),
                'decision_radius': report['decision_radius']['value'],
                'exact_decision_radius': report['decision_radius']['exact'],
                'common_optimal_actions': '|'.join(report['common_optimal_actions']),
                'target_regret_guarantee': report['target_regret_guarantee']})
    summary = results / 'source_decisions.csv'
    with summary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(summary)
    (results / 'source_decision_provenance.json').write_text(json.dumps({
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'source_training_ids': {r['name']: r['tasks']['train']['ids'] for r in records},
        'output_sha256': {str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        'uses_target_test_labels': False,
        'scope': 'Six conditional source-model certificates, not a new policy, uncertainty interval, or target transfer guarantee.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
