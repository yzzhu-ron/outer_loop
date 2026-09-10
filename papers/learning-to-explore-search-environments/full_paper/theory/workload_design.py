"""Exact small examples for acquisition before the future workload is known.

These are finite decision calculations, not a fitted retrieval method. All
design columns must represent plans satisfying the same complete cost budget.
The minimax identity is standard zero-sum game theory; no novelty is claimed.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'theory'))
from finite_decisions import (bayes_regret, deterministic_likelihood, distribution,
                              expected_regret_after, matrix, posteriors, solve_linear)


def context_utilities(utilities):
    worlds = tuple(matrix(world) for world in utilities)
    if not worlds or any(len(world) != len(worlds[0]) or len(world[0]) != len(worlds[0][0])
                         for world in worlds):
        raise ValueError('World/context/action dimensions must agree')
    if any(not 0 <= u <= 1 for world in worlds for row in world for u in row):
        raise ValueError('Utilities must lie in [0,1]')
    return tuple(tuple(world[x] for world in worlds) for x in range(len(worlds[0])))


def risks(utilities, prior):
    return tuple(bayes_regret(context, prior) for context in context_utilities(utilities))


def residuals(utilities, prior, likelihood):
    return tuple(expected_regret_after(context, prior, likelihood)
                 for context in context_utilities(utilities))


def risk_matrix(utilities, prior, channels):
    columns = tuple(residuals(utilities, prior, channel) for channel in channels)
    if not columns:
        raise ValueError('At least one channel is required')
    return tuple(tuple(column[x] for column in columns) for x in range(len(columns[0])))


def worst_workload(context_risks, workloads):
    values = tuple(F(str(value)) for value in context_risks)
    mixtures = tuple(distribution(mu, len(values)) for mu in workloads)
    if not mixtures:
        raise ValueError('At least one workload is required')
    return max(sum(m * v for m, v in zip(mu, values)) for mu in mixtures)


@dataclass(frozen=True)
class Design:
    risk: F
    mixture: tuple[F, ...]
    workload_witness: tuple[F, ...]


def minimax_design(context_by_design, workloads):
    """Exact primal/dual support enumeration, suitable only for tiny games.

    Nature chooses a fixed workload before seeing design randomization or its
    observations. Returned witness weights mix the supplied workload vertices.
    """
    rows = matrix(context_by_design)
    if any(not 0 <= value <= 1 for row in rows for value in row):
        raise ValueError('Residual risks must lie in [0,1]')
    mixtures = tuple(distribution(mu, len(rows)) for mu in workloads)
    if not mixtures:
        raise ValueError('At least one workload is required')
    loss = tuple(tuple(sum(mu[x] * rows[x][q] for x in range(len(rows)))
                       for q in range(len(rows[0]))) for mu in mixtures)
    n, m = len(loss), len(loss[0])
    primal = dual = None
    for k in range(1, min(n, m) + 1):
        for active_rows in combinations(range(n), k):
            for active_columns in combinations(range(m), k):
                system = [[1] * k + [0]] + [
                    [loss[i][j] for j in active_columns] + [-1] for i in active_rows]
                answer = solve_linear(system, [1] + [0] * k)
                if answer is not None and all(p >= 0 for p in answer[:-1]):
                    p = tuple(answer[active_columns.index(j)] if j in active_columns else F(0)
                              for j in range(m))
                    value = max(sum(p[j] * row[j] for j in range(m)) for row in loss)
                    if value == answer[-1] and (primal is None or value < primal[0]):
                        primal = value, p
                system = [[1] * k + [0]] + [
                    [loss[i][j] for i in active_rows] + [-1] for j in active_columns]
                answer = solve_linear(system, [1] + [0] * k)
                if answer is not None and all(w >= 0 for w in answer[:-1]):
                    w = tuple(answer[active_rows.index(i)] if i in active_rows else F(0)
                              for i in range(n))
                    value = min(sum(w[i] * loss[i][j] for i in range(n)) for j in range(m))
                    if value == answer[-1] and (dual is None or value > dual[0]):
                        dual = value, w
    if primal is None or dual is None or primal[0] != dual[0]:
        raise AssertionError('Exact minimax certificates failed')
    return Design(primal[0], primal[1], dual[1])


def universally_best(context_by_design):
    """Designs minimizing every context risk; exact for full-simplex workloads."""
    rows = matrix(context_by_design)
    return tuple(q for q in range(len(rows[0])) if all(row[q] == min(row) for row in rows))


def crossing_pairs(context_by_design):
    """Witness contexts where two designs rank in opposite orders."""
    rows = matrix(context_by_design)
    found = []
    for a, b in combinations(range(len(rows[0])), 2):
        better_a = [x for x, row in enumerate(rows) if row[a] < row[b]]
        better_b = [x for x, row in enumerate(rows) if row[b] < row[a]]
        if better_a and better_b:
            found.append((a, b, better_a[0], better_b[0]))
    return tuple(found)


def bit_problem(context_count, nuisance_count=0):
    if type(context_count) is not int or context_count < 1 or type(nuisance_count) is not int or nuisance_count < 0:
        raise ValueError('Positive context count and nonnegative nuisance count required')
    worlds = tuple(product((0, 1), repeat=context_count + nuisance_count))
    prior = tuple(F(1, len(worlds)) for _ in worlds)
    utility = tuple(tuple((int(world[x] == 0), int(world[x] == 1))
                          for x in range(context_count)) for world in worlds)
    channels = [deterministic_likelihood([world[x] for world in worlds]) for x in range(context_count)]
    if nuisance_count:
        channels.append(deterministic_likelihood([world[context_count:] for world in worlds]))
    return utility, prior, tuple(channels)


def timing_risk(context_count, budget, messages=1):
    """Sharp construction: independent fair decision bits, one bit per probe.

    A deterministic message partitions future context IDs before acquisition;
    Nature chooses the context before randomization. Noiseless unit-cost probes.
    """
    if any(type(v) is not int for v in (context_count, budget, messages)) or context_count < 1 or budget < 0 or messages < 1:
        raise ValueError('Positive context/messages and nonnegative integer budget required')
    largest_cell = (context_count + messages - 1) // messages
    return max(F(0), F(1, 2) * (1 - F(budget, largest_cell)))


def robust_conditioning_example():
    # The signal reveals which context has uncertainty, not its correct action.
    utility = (((1, 0), (1, 0)), ((0, 1), (1, 0)),
               ((1, 0), (1, 0)), ((1, 0), (0, 1)))
    prior = (F(1, 4),) * 4
    channel = deterministic_likelihood((0, 0, 1, 1))
    workloads = ((1, 0), (0, 1))
    before = worst_workload(risks(utility, prior), workloads)
    fixed = worst_workload(residuals(utility, prior, channel), workloads)
    reacting = sum(mass * worst_workload(risks(utility, posterior), workloads)
                   for mass, posterior in posteriors(prior, channel))
    return {'before': before, 'fixed_workload_after': fixed,
            'outcome_reacting_workload_after': reacting,
            'incorrect_fixed_workload_voi': before - reacting}
