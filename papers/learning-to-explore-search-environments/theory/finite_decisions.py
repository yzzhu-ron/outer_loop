"""Exact finite decision experiments; Python standard library only.

Utilities are environment-by-action matrices. An action may represent a complete
query-conditioned router. These examples are mathematical constructions, not
measurements of real retrieval. Vertex enumeration is for small models only.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from math import log2
from typing import Sequence


def fraction(x) -> F:
    return x if isinstance(x, F) else F(str(x))


def matrix(rows: Sequence[Sequence]) -> tuple[tuple[F, ...], ...]:
    result = tuple(tuple(map(fraction, row)) for row in rows)
    if not result or not result[0] or any(len(r) != len(result[0]) for r in result):
        raise ValueError("A nonempty rectangular matrix is required")
    return result


def losses(utilities):
    rows = matrix(utilities)
    if any(v < 0 or v > 1 for row in rows for v in row):
        raise ValueError("Utilities must lie in [0, 1]")
    return tuple(tuple(max(row) - v for v in row) for row in rows)


def solve_linear(coefficients, rhs):
    """Solve a square system by exact elimination; return None if singular."""
    coefficients, rhs = tuple(coefficients), tuple(rhs)
    n = len(rhs)
    if not n or len(coefficients) != n or any(len(row) != n for row in coefficients):
        raise ValueError("A nonempty square system is required")
    aug = [list(map(fraction, row)) + [fraction(b)] for row, b in zip(coefficients, rhs)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col]), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        aug[col] = [v / denom for v in aug[col]]
        for r in range(n):
            if r != col:
                multiple = aug[r][col]
                aug[r] = [v - multiple * w for v, w in zip(aug[r], aug[col])]
    return tuple(row[-1] for row in aug)


@dataclass(frozen=True)
class Radius:
    value: F
    action_mixture: tuple[F, ...]
    witness_prior: tuple[F, ...]


def minimax_radius(utilities) -> Radius:
    """Solve primal and dual separately and verify their exact equality.

    The primal minimizes the worst environment's expected regret. The dual
    finds the prior with largest Bayes regret. There exist optimal primal and
    dual solutions with support at most min(number of worlds, number of
    actions); other optima in degenerate games may have larger support.
    """
    loss = losses(utilities)
    n, m = len(loss), len(loss[0])
    primal, dual = None, None
    for k in range(1, min(n, m) + 1):
        for worlds in combinations(range(n), k):
            for actions in combinations(range(m), k):
                system = [[1] * k + [0]] + [
                    [loss[e][a] for a in actions] + [-1] for e in worlds
                ]
                sol = solve_linear(system, [1] + [0] * k)
                if sol is not None and all(v >= 0 for v in sol[:-1]):
                    p = tuple(sol[actions.index(a)] if a in actions else F(0) for a in range(m))
                    value = max(sum(p[a] * row[a] for a in range(m)) for row in loss)
                    if value == sol[-1] and (primal is None or value < primal[0]):
                        primal = value, p
                system = [[1] * k + [0]] + [
                    [loss[e][a] for e in worlds] + [-1] for a in actions
                ]
                sol = solve_linear(system, [1] + [0] * k)
                if sol is not None and all(v >= 0 for v in sol[:-1]):
                    w = tuple(sol[worlds.index(e)] if e in worlds else F(0) for e in range(n))
                    value = min(sum(w[e] * loss[e][a] for e in range(n)) for a in range(m))
                    if value == sol[-1] and (dual is None or value > dual[0]):
                        dual = value, w
    if primal is None or dual is None or primal[0] != dual[0]:
        raise AssertionError("Exact primal/dual certificate failed")
    return Radius(primal[0], primal[1], dual[1])


def pairwise_incompatibility(utilities) -> F:
    """Maximum minimum SUM of regrets for a pair, without dividing by two."""
    loss = losses(utilities)
    return max(min(x + y for x, y in zip(a, b)) for a in loss for b in loss)


def distribution(weights, n):
    weights = tuple(map(fraction, weights))
    if len(weights) != n or any(w < 0 for w in weights) or sum(weights) != 1:
        raise ValueError("Probability weights must be nonnegative and sum to one")
    return weights


def bayes_regret(utilities, prior) -> F:
    loss = losses(utilities)
    prior = distribution(prior, len(loss))
    return min(sum(p * row[a] for p, row in zip(prior, loss)) for a in range(len(loss[0])))


def posteriors(prior, likelihoods):
    """Return (outcome mass, posterior) for every positive-mass observation."""
    rows = matrix(likelihoods)
    prior = distribution(prior, len(rows))
    for row in rows:
        distribution(row, len(row))
    answer = []
    for z in range(len(rows[0])):
        joint = tuple(prior[e] * rows[e][z] for e in range(len(rows)))
        mass = sum(joint)
        if mass:
            answer.append((mass, tuple(v / mass for v in joint)))
    return answer


def expected_regret_after(utilities, prior, likelihoods) -> F:
    return sum(mass * bayes_regret(utilities, posterior) for mass, posterior in posteriors(prior, likelihoods))


def probe_value(utilities, prior, likelihoods) -> F:
    return bayes_regret(utilities, prior) - expected_regret_after(utilities, prior, likelihoods)


def entropy(prior) -> float:
    return -sum(float(p) * log2(float(p)) for p in prior if p)


def environment_information(prior, likelihoods) -> float:
    return entropy(prior) - sum(float(mass) * entropy(post) for mass, post in posteriors(prior, likelihoods))


def deterministic_likelihood(outcomes):
    alphabet = list(dict.fromkeys(outcomes))
    return tuple(tuple(F(int(z == outcome)) for z in alphabet) for outcome in outcomes)


def robust_after_partition(utilities, outcomes) -> F:
    """Exact worst-case radius for a fixed noiseless probe (or whole transcript)."""
    rows = matrix(utilities)
    if len(outcomes) != len(rows):
        raise ValueError("One observation per environment is required")
    return max(minimax_radius([row for row, z in zip(rows, outcomes) if z == label]).value for label in dict.fromkeys(outcomes))


def gaussian_variance_reduction(covariance, probe, contrast, noise_variance) -> F:
    """Exact covariance identity; this is a regret surrogate, not a guarantee."""
    covariance = matrix(covariance)
    probe, contrast = tuple(map(fraction, probe)), tuple(map(fraction, contrast))
    n = len(covariance)
    if any(len(r) != n for r in covariance) or len(probe) != n or len(contrast) != n:
        raise ValueError("Incompatible covariance/vector shapes")
    noise_variance = fraction(noise_variance)
    if noise_variance < 0:
        raise ValueError("Noise variance must be nonnegative")
    if any(covariance[i][j] != covariance[j][i] for i in range(n) for j in range(n)):
        raise ValueError("Covariance must be symmetric")
    # Exact Schur-complement PSD check. In a PSD matrix a zero diagonal
    # requires its entire row/column to be zero, including singular cases.
    residual = [list(row) for row in covariance]
    for i in range(n):
        pivot = residual[i][i]
        if pivot < 0 or (pivot == 0 and any(residual[i][j] for j in range(i + 1, n))):
            raise ValueError("Covariance must be positive semidefinite")
        if pivot:
            for j in range(i + 1, n):
                for k in range(i + 1, n):
                    residual[j][k] -= residual[j][i] * residual[i][k] / pivot
    sigma_probe = [sum(row[j] * probe[j] for j in range(n)) for row in covariance]
    denominator = noise_variance + sum(probe[j] * sigma_probe[j] for j in range(n))
    if denominator <= 0:
        raise ValueError("Observation variance must be positive")
    numerator = sum(contrast[j] * sigma_probe[j] for j in range(n)) ** 2
    return numerator / denominator


def select_action_from_contrasts(nonreference_contrasts) -> int:
    """Argmax with reference action 0 included at score zero; ties favor it.

    Input contrast j is the utility of action j+1 minus that of action 0.
    Omitting the reference candidate would invalidate the argmax error bound
    whenever all alternatives are worse than the reference.
    """
    scores = (F(0),) + tuple(map(fraction, nonreference_contrasts))
    return max(range(len(scores)), key=scores.__getitem__)
