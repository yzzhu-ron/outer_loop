"""Optional, labeled-source finite-model decision audit.

This module never infers utilities from query logs or certifies target regret.
Its exact vertex-enumeration solver is adapted from this repository's
papers/learning-to-explore-search-environments/theory/finite_decisions.py.
The mathematics is standard finite minimax decision theory / LP duality,
not a new theorem or algorithm. See that note for assumptions and derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import json
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MAX_VERTEX_SYSTEMS = 20_000


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _names(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or not all(_nonempty_text(x) for x in value):
        raise ValueError(f"{field}: expected a nonempty array of nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field}: entries must be unique")
    return tuple(value)


def _keys(value: Mapping, allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed, key=str)
    if unknown:
        raise ValueError(f"{field}: unknown fields {', '.join(map(str, unknown))}")


def _utility(value: Any, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Fraction)):
        raise ValueError(f"{field}: expected a number or rational string in [0, 1]")
    try:
        rational = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(f"{field}: expected a finite number or rational string in [0, 1]") from None
    if not 0 <= rational <= 1:
        raise ValueError(f"{field}: utility must lie in [0, 1]")
    return rational


def _validate_model(model: Any) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[Fraction, ...], ...]]:
    if not isinstance(model, Mapping):
        raise ValueError("model: expected a JSON object")
    _keys(model, {"schema_version", "kind", "model_id", "actions", "world_ids", "utilities", "provenance", "assumptions"}, "model")
    if type(model.get("schema_version")) is not int or model["schema_version"] != 1:
        raise ValueError("schema_version: expected integer 1")
    if model.get("kind") != "source_utility_model":
        raise ValueError("kind: expected 'source_utility_model'; this command does not accept label-free probe logs")
    if not _nonempty_text(model.get("model_id")):
        raise ValueError("model_id: expected a nonempty string")
    actions, worlds = _names(model.get("actions"), "actions"), _names(model.get("world_ids"), "world_ids")
    rows = model.get("utilities")
    if not isinstance(rows, (list, tuple)) or len(rows) != len(worlds):
        raise ValueError("utilities: expected one row per world_id")
    utilities = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != len(actions):
            raise ValueError(f"utilities[{i}]: expected one value per action")
        utilities.append(tuple(_utility(x, f"utilities[{i}][{j}]") for j, x in enumerate(row)))
    provenance = model.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance: required source-data description")
    _keys(provenance, {"utility_source", "source_split", "task_distribution", "synthetic"}, "provenance")
    for name in ("utility_source", "source_split", "task_distribution"):
        if not _nonempty_text(provenance.get(name)):
            raise ValueError(f"provenance.{name}: expected a nonempty source-data description")
    if not isinstance(provenance.get("synthetic"), bool):
        raise ValueError("provenance.synthetic: explicitly declare true or false")
    assumptions = model.get("assumptions")
    if not isinstance(assumptions, (list, tuple)) or not assumptions or not all(_nonempty_text(x) for x in assumptions):
        raise ValueError("assumptions: provide at least one explicit modeling assumption")
    return actions, worlds, tuple(utilities)


def _solve_linear(coefficients: Sequence[Sequence[Fraction]], rhs: Sequence[int]) -> tuple[Fraction, ...] | None:
    n = len(rhs)
    if not n or len(coefficients) != n or any(len(row) != n for row in coefficients):
        raise ValueError("A nonempty square system is required")
    augmented = [[Fraction(x) for x in row] + [Fraction(b)] for row, b in zip(coefficients, rhs)]
    for column in range(n):
        pivot = next((r for r in range(column, n) if augmented[r][column]), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [v / scale for v in augmented[column]]
        for row in range(n):
            if row != column:
                scale = augmented[row][column]
                augmented[row] = [v - scale * w for v, w in zip(augmented[row], augmented[column])]
    return tuple(row[-1] for row in augmented)


@dataclass(frozen=True)
class _Radius:
    value: Fraction
    action_mixture: tuple[Fraction, ...]
    witness_prior: tuple[Fraction, ...]
    vertex_systems: int


def _radius(utilities: tuple[tuple[Fraction, ...], ...], max_vertex_systems: int) -> _Radius:
    if type(max_vertex_systems) is not int or max_vertex_systems < 1:
        raise ValueError("max_vertex_systems must be a positive integer")
    loss = tuple(tuple(max(row) - x for x in row) for row in utilities)
    n, m = len(loss), len(loss[0])
    # Vandermonde's identity: sum_k C(n,k)C(m,k) = C(n+m,n).
    systems = 2 * (comb(n + m, min(n, m)) - 1)
    if systems > max_vertex_systems:
        raise ValueError(f"Exact solver would enumerate {systems} linear systems, exceeding limit {max_vertex_systems}. "
                         "Use a smaller explicitly justified model or an external LP solver; raising the limit may be expensive.")
    primal, dual = None, None
    for k in range(1, min(n, m) + 1):
        for worlds in combinations(range(n), k):
            for actions in combinations(range(m), k):
                system = [[1] * k + [0]] + [[loss[e][a] for a in actions] + [-1] for e in worlds]
                solution = _solve_linear(system, [1] + [0] * k)
                if solution is not None and all(x >= 0 for x in solution[:-1]):
                    mixture = [Fraction(0)] * m
                    for i, a in enumerate(actions):
                        mixture[a] = solution[i]
                    value = max(sum(p * x for p, x in zip(mixture, row)) for row in loss)
                    if value == solution[-1] and (primal is None or value < primal[0]):
                        primal = value, tuple(mixture)
                system = [[1] * k + [0]] + [[loss[e][a] for e in worlds] + [-1] for a in actions]
                solution = _solve_linear(system, [1] + [0] * k)
                if solution is not None and all(x >= 0 for x in solution[:-1]):
                    prior = [Fraction(0)] * n
                    for i, e in enumerate(worlds):
                        prior[e] = solution[i]
                    value = min(sum(prior[e] * loss[e][a] for e in range(n)) for a in range(m))
                    if value == solution[-1] and (dual is None or value > dual[0]):
                        dual = value, tuple(prior)
    if primal is None or dual is None or primal[0] != dual[0]:
        raise ArithmeticError("Exact primal/dual certificate failed; no radius is reported")
    return _Radius(primal[0], primal[1], dual[1], systems)


def _reported(value: Fraction) -> dict[str, Any]:
    return {"exact": str(value), "value": float(value)}


def audit_decision_model(model: Mapping[str, Any], *, max_vertex_systems: int = DEFAULT_MAX_VERTEX_SYSTEMS) -> dict[str, Any]:
    """Compute a conditional minimax regret certificate from source utilities.

    This API requires the explicit source_utility_model contract. It does not
    accept the paired-probe JSONL contract, infer utilities, update world
    weights from observations, or certify target-world coverage. The exact
    solver is intentionally limited to small models and uses no dependencies.
    """
    actions, worlds, utilities = _validate_model(model)
    result = _radius(utilities, max_vertex_systems)
    loss = tuple(tuple(max(row) - x for x in row) for row in utilities)
    per_world = tuple(sum(p * x for p, x in zip(result.action_mixture, row)) for row in loss)
    per_action = tuple(sum(result.witness_prior[e] * loss[e][a] for e in range(len(worlds))) for a in range(len(actions)))
    pairwise = (max(min(x + y for x, y in zip(left, right)) for left in loss for right in loss)
                if result.value else Fraction(0))
    common_optimal = [a for i, a in enumerate(actions) if all(row[i] == 0 for row in loss)]
    diagnostics = []
    if pairwise == 0 and result.value > 0:
        diagnostics.append({"code": "multiway_conflict", "message": "Every pair of source worlds has a common optimal action, but all worlds together do not. Pairwise agreement misses this conflict."})
    if result.value == 0:
        diagnostics.append({"code": "source_common_optimum", "message": "The supplied source worlds share an optimal action. This says nothing about worlds omitted from the model or the accuracy of the utility estimates."})
    return {
        "schema_version": 1,
        "audit_kind": "source_model_decision_radius",
        "status": "computed",
        "model_id": model["model_id"],
        "target_regret_guarantee": False,
        "scope": "Exact conditional regret certificate for the supplied finite source utility matrix; not a target-regret guarantee.",
        "provenance": dict(model["provenance"]),
        "declared_assumptions": list(model["assumptions"]),
        "model": {"actions": list(actions), "world_ids": list(worlds), "utilities_exact": [[str(x) for x in row] for row in utilities]},
        "decision_radius": _reported(result.value),
        "pairwise_sum_incompatibility": _reported(pairwise),
        "pairwise_radius_lower_bound": _reported(pairwise / 2),
        "common_optimal_actions": common_optimal,
        "minimax_action_mixture": [{"action": action, "probability": _reported(p)} for action, p in zip(actions, result.action_mixture)],
        "worst_case_source_prior": [{"world_id": world, "weight": _reported(w)} for world, w in zip(worlds, result.witness_prior)],
        "certificate": {
            "primal_max_regret": _reported(max(per_world)),
            "dual_min_expected_regret": _reported(min(per_action)),
            "duality_gap": _reported(max(per_world) - min(per_action)),
            "witness_support": sum(w > 0 for w in result.witness_prior),
            "witness_support_upper_bound": min(len(worlds), len(actions)),
            "per_world_mixture_regret": [{"world_id": world, "regret": _reported(r)} for world, r in zip(worlds, per_world)],
            "per_action_witness_regret": [{"action": action, "regret": _reported(r)} for action, r in zip(actions, per_action)],
        },
        "diagnostics": diagnostics,
        "required_interpretation": [
            "Each action denotes the same available policy/router in every world; quality is expected over the declared task distribution.",
            "Utilities are treated as exact supplied numbers; source-label sampling error and model misspecification are not covered by this certificate.",
            "The witness prior is an adversarial certificate, not a learned posterior or an estimate of target-world frequency.",
            "The action mixture is a model solution, not a recommendation to deploy it on an unvalidated target.",
            "A radius of zero does not certify target performance, source-family coverage, useful probes, or source-to-target transfer.",
        ],
        "what_this_model_cannot_establish": [
            "Whether the target world or its workload is represented by the supplied source worlds.",
            "Whether real probe transcripts distinguish worlds or produce calibrated posteriors/confidence sets.",
            "Whether source labels, utility estimates, declared splits, and task-mix assumptions are correct or free of leakage.",
            "Whether any exploration policy reduces actual target regret or repays its cost.",
        ],
        "method": {"solver": "Exact rational primal/dual vertex enumeration", "vertex_systems_enumerated": result.vertex_systems,
                   "max_vertex_systems": max_vertex_systems, "precision": "Integer/rational/decimal inputs treated exactly; display values use floating point.",
                   "credit": "Adapted from outer_loop theory/finite_decisions.py; standard finite minimax statistical decision theory and LP duality."},
    }


def audit_decision_json(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Read one source utility model JSON object and compute its certificate."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(token):
        raise ValueError(f"Non-JSON number {token}")

    model = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    return audit_decision_model(model, **kwargs)
