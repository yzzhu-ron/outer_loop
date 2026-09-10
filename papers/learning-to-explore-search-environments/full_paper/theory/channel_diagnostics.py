"""Post-hoc source-model diagnostics; never reads target utilities or outcomes.

For row-stochastic channels A and B, A Blackwell-dominates B exactly when
B = A G for a row-stochastic garbling G. This script minimizes ||A G - B||_inf.
It also verifies a separating functional without trusting LP optimality:

  distance >= [<W,B> - sum_i max_j (A^T W)[i,j]] / sum(abs(W)).

The latter calculation uses exact rational arithmetic after row-normalizing
the saved decimal probabilities. It is a statement about these fitted source
channels, never an uncertainty bound on the underlying retrieval process.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
NUMERIC_THRESHOLD = 1e-9
ACTION_NAMES = ('original', 'keywords', 'semantic', 'hyde', 'decomposed')
CHANNEL_NAMES = tuple(f'{family}/{action}' for family in
                      ('direct_question', 'indirect_question')
                      for action in ACTION_NAMES[1:])


def validate_channel(channel):
    a = np.asarray(channel, dtype=float)
    if (a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all()
            or np.min(a) < 0 or not np.allclose(a.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError('Expected a finite nonnegative row-stochastic channel')
    return a


def rational_channel(channel):
    """Exact row normalization of saved decimal probabilities, explicitly defined."""
    rows = [[Fraction(str(float(x))) for x in row] for row in channel]
    return [[x / sum(row) for x in row] for row in rows]


def separation_certificate(a, b, weights):
    """Every nonzero W gives a valid distance lower bound, optimal or otherwise."""
    aa, bb = rational_channel(a), rational_channel(b)
    w = [[Fraction(str(float(x))).limit_denominator(10000) for x in row]
         for row in weights]
    norm = sum(abs(x) for row in w for x in row)
    if norm == 0:
        lower = Fraction(0)
    else:
        target = sum(w[r][j] * bb[r][j] for r in range(len(bb))
                     for j in range(len(bb[0])))
        support = sum(max(sum(aa[r][i] * w[r][j] for r in range(len(aa)))
                          for j in range(len(bb[0]))) for i in range(len(aa[0])))
        lower = max(Fraction(0), (target - support) / norm)
    return {'lower_bound_fraction': str(lower), 'lower_bound': float(lower),
            'positive': lower > 0,
            'weights_fraction': [[str(x) for x in row] for row in w]}


def garbling_distance(source, target):
    a, b = validate_channel(source), validate_channel(target)
    if a.shape[0] != b.shape[0]:
        raise ValueError('Channels must share the same ordered worlds')
    worlds, source_outcomes = a.shape
    target_outcomes = b.shape[1]
    size = source_outcomes * target_outcomes
    transform = np.zeros((worlds * target_outcomes, size))
    for world in range(worlds):
        for outcome in range(target_outcomes):
            transform[world * target_outcomes + outcome, outcome:size:target_outcomes] = a[world]
    inequalities = np.vstack((np.column_stack((transform, -np.ones(len(transform)))),
                              np.column_stack((-transform, -np.ones(len(transform))))))
    limits = np.concatenate((b.ravel(), -b.ravel()))
    equality = np.zeros((source_outcomes, size + 1))
    for outcome in range(source_outcomes):
        equality[outcome, outcome * target_outcomes:(outcome + 1) * target_outcomes] = 1
    objective = np.zeros(size + 1)
    objective[-1] = 1
    result = linprog(objective, A_ub=inequalities, b_ub=limits,
                     A_eq=equality, b_eq=np.ones(source_outcomes),
                     bounds=[(0, None)] * (size + 1), method='highs',
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
    if not result.success:
        raise RuntimeError(f'Garbling LP failed: status={result.status}, {result.message}')
    g = result.x[:-1].reshape(source_outcomes, target_outcomes)
    residual = float(np.max(np.abs(a @ g - b)))
    row_error = float(np.max(np.abs(g.sum(axis=1) - 1)))
    if (not np.isfinite(g).all() or g.min() < -1e-9 or row_error > 1e-9
            or abs(residual - result.fun) > 1e-8):
        raise RuntimeError('Returned garbling fails independent primal verification')
    count = len(transform)
    weights = (result.ineqlin.marginals[:count] - result.ineqlin.marginals[count:]).reshape(b.shape)
    certificate = separation_certificate(a, b, weights)
    if certificate['lower_bound'] > residual + 1e-8:
        raise RuntimeError('Separation lower bound exceeds primal upper witness')
    return {'lp_linf_distance': float(result.fun), 'verified_primal_linf_residual': residual,
            'garbling_row_sum_max_error': row_error, 'garbling_min_entry': float(g.min()),
            'numerical_garbling_exists': residual <= NUMERIC_THRESHOLD,
            'solver_status': int(result.status), 'garbling': g.tolist(),
            'rational_separation': certificate}


def decision_labels(model):
    utility = np.asarray(model['utilities'], dtype=float)
    if (utility.ndim != 3 or utility.shape[0] != len(model['names'])
            or not np.isfinite(utility).all()):
        raise ValueError('Invalid world-by-bucket-by-action utilities')
    labels = np.argmax(utility, axis=2)
    unique, counts = np.unique(labels, axis=0, return_counts=True)
    return {'worlds': len(labels), 'distinct_argmax_vectors': len(unique),
            'world_to_bucket_actions': {name: row.tolist() for name, row in zip(model['names'], labels)},
            'uniform_prior_label_masses': (counts / len(labels)).tolist(),
            'unique_vectors': unique.tolist(),
            'label_entropy_equals_world_entropy_for_every_posterior': len(unique) == len(labels)}


def summarize_channels(likelihood):
    channels = np.asarray(likelihood, dtype=float)
    if channels.shape != (4, 8, 5):
        raise ValueError('This diagnostic expects four worlds, eight channels, five outcomes')
    normalization_change = max(
        abs(Fraction(str(float(channels[world, channel, outcome]))) - value)
        for channel in range(8)
        for world, row in enumerate(rational_channel(channels[:, channel]))
        for outcome, value in enumerate(row))
    comparisons = {}
    for source, target in itertools.permutations(range(8), 2):
        comparisons[f'{source}->{target}'] = garbling_distance(channels[:, source], channels[:, target])
    pairs = []
    for a, b in itertools.combinations(range(8), 2):
        ab, ba = comparisons[f'{a}->{b}'], comparisons[f'{b}->{a}']
        pairs.append({'channels': [a, b],
                      'numerically_incomparable': not ab['numerical_garbling_exists'] and not ba['numerical_garbling_exists'],
                      'rational_separation_both_directions': ab['rational_separation']['positive'] and ba['rational_separation']['positive']})
    return {'ordered_comparisons': comparisons, 'unordered_pairs': pairs,
            'exact_row_normalization_max_entry_change': float(normalization_change),
            'numerically_incomparable_pairs': sum(p['numerically_incomparable'] for p in pairs),
            'rationally_separated_incomparable_pairs': sum(p['rational_separation_both_directions'] for p in pairs),
            'unordered_pair_count': len(pairs),
            'minimum_positive_rational_distance_bound': min(
                (c['rational_separation']['lower_bound'] for c in comparisons.values()
                 if c['rational_separation']['positive']), default=None)}


def main():
    path = ROOT.parent / 'natural/results/models.json'
    models = json.loads(path.read_text())
    rows, fits = [], {}
    for name, model in models.items():
        if not name.endswith('/standard'):
            continue
        serial = json.dumps(model, sort_keys=True, separators=(',', ':'))
        fit_id = hashlib.sha256(serial.encode()).hexdigest()
        if fit_id not in fits:
            fits[fit_id] = {'source_names': model['names'], **summarize_channels(model['likelihood'])}
        rows.append({'model': name, 'fit_sha256': fit_id, **decision_labels(model)})
    output = {
        'status': 'Post-hoc diagnostics of frozen fitted source models. No target outcomes read.',
        'interpretation': 'A one-to-one argmax label map makes decision entropy and world entropy identical at every posterior. Channel incomparability only means no universal ordering across all decision problems; it need not improve these particular retrieval decisions.',
        'numerical_comparison': {'problem': 'minimize max-entry |A G - B| over G >= 0 with each row summing to one',
                                 'threshold': NUMERIC_THRESHOLD, 'solver': 'scipy.optimize.linprog / HiGHS'},
        'rational_certificate': {
            'channel_definition': 'Each saved decimal probability is converted to Fraction and its row normalized exactly. Normalization differs from the floating fit only at rounding scale.',
            'formula': 'For any nonzero W, distance_inf(B, {A G}) >= (<W,B> - sum_i max_j (A^T W)[i,j]) / sum(abs(W)). Positive bounds exclude every stochastic garbling.',
            'weights': 'LP inequality marginal difference, rounded to rationals of denominator at most 10000; validity does not require dual feasibility or LP optimality.',
            'limit': 'A certificate about fitted source distributions, not statistical evidence of population incomparability, target transfer, or retrieval-specific value.'},
        'model_rows': rows, 'distinct_source_fits': len(fits), 'fits': fits,
        'channel_names': list(CHANNEL_NAMES), 'action_names': list(ACTION_NAMES),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'input_sha256': {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (path, ROOT / 'channel_diagnostics.py')},
    }
    destination = ROOT / 'channel_diagnostics.json'
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for row in rows:
        fit = fits[row['fit_sha256']]
        print(row['model'], 'unique labels:', row['distinct_argmax_vectors'],
              'incomparable pairs:', fit['rationally_separated_incomparable_pairs'], '/', fit['unordered_pair_count'])
    print(destination)


if __name__ == '__main__':
    main()
