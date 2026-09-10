"""Quality against two unpriced deployment-cost categories at 50 future tasks."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_report import validate_cost_curves

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
METHODS = ('random', 'fixed', 'information_gain', 'learned_value')
COLORS = ('#666666', '#477A63', '#497AAB', '#B35632')


def main():
    def read_csv(name):
        with (RESULTS / name).open(newline='') as handle:
            return list(csv.DictReader(handle))
    rows, fusion, adaptation = [read_csv(name) for name in ('cost_curves.csv', 'fusion_baseline.csv', 'adaptation.csv')]
    protocol_path = ROOT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    names = validate_cost_curves(rows, adaptation, protocol)
    if len(fusion) != len(names) or {row['environment'] for row in fusion} != set(names):
        raise ValueError('Cost plot requires every fusion environment exactly once')
    for row in fusion:
        corpus = row['environment'].rsplit('_', 1)[0]
        if (int(row['n_queries']) != protocol['query_selection']['test_counts'][corpus]
                or not 0 <= float(row['ndcg']) <= 1 or any(not math.isfinite(float(row['mean_task_' + category]))
                or float(row['mean_task_' + category]) < 0 for category in ('search_calls', 'output_tokens'))):
            raise ValueError('Fusion cohort, quality or plotted costs are invalid')
    rows = [r for r in rows if int(r['future_tasks']) == 50]
    (RESULTS / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'searchprobe-cost-50'})
    paths = []
    for category, label in [('search_calls', 'Logical search requests / 50 tasks'), ('output_tokens', 'LLM output tokens / 50 tasks')]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), constrained_layout=True)
        for ax, name in zip(axes.flat, names, strict=True):
            for method, color in zip(METHODS, COLORS):
                points = []
                for budget in (0, 4, 8, 16, 32):
                    subset = [r for r in rows if r['environment'] == name and r['method'] == method and int(r['budget']) == budget]
                    points.append((np.mean([float(r['total_' + category]) for r in subset]), np.mean([float(r['ndcg']) for r in subset])))
                ax.plot(*np.array(points).T, color=color, marker='o', ms=3, label=method.replace('_', ' '))
            base = next(r for r in rows if r['environment'] == name and r['method'] == 'source_router' and int(r['budget']) == 0)
            ax.scatter(float(base['total_' + category]), float(base['ndcg']), color='black', marker='*', s=75, label='source router', zorder=4)
            extra = next(r for r in fusion if r['environment'] == name)
            ax.scatter(50 * float(extra['mean_task_' + category]), float(extra['ndcg']), color='#795B8D', marker='D', s=32, label='source-selected RRF', zorder=4)
            ax.set(title=name.replace('_', ' / '), xlabel=label, ylabel='Mean nDCG@10')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
        fig.suptitle('Onboarding plus serving cost for 50 future tasks\nSeparate cost categories; no implied dollar/latency exchange rate', fontsize=12)
        for suffix in ('png', 'svg'):
            path = RESULTS / 'figures' / f'cost_{category}.{suffix}'
            fig.savefig(path, dpi=170, bbox_inches='tight', **({'metadata': {'Date': None}} if suffix == 'svg' else {}))
            paths.append(path)
        plt.close(fig)
    (RESULTS / 'cost_plot_provenance.json').write_text(json.dumps({
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'validation_sha256': hashlib.sha256(Path(validate_cost_curves.__code__.co_filename).read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'future_tasks': 50,
        'input_sha256': {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in ('cost_curves.csv', 'fusion_baseline.csv', 'adaptation.csv')},
        'output_sha256': {str(p.relative_to(RESULTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope': 'Each panel averages three profiles on one corpus/backend. Horizontal axes are separate cost categories, not a scalar complete cost. Input tokens, sampling calls and missing serving latency also matter.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
