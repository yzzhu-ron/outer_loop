"""Exact, conditional argmax stability under caller-supplied score bounds.

This diagnostic reads no query text, labels, or outcomes. It cannot establish
that the declared bounds cover a real router. Its certificate concerns a closed
axis-aligned score box and the lowest-action-index tie rule, not target regret
or generalization. Coupled real corrections may occupy only part of the box.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .decisions import _keys, _names, _nonempty_text


def _rational(value: Any, field: str, *, nonnegative: bool = False) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal, Fraction)):
        raise ValueError(f"{field}: expected a finite number or decimal/rational string")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(f"{field}: expected a finite number or decimal/rational string") from None
    if nonnegative and result < 0:
        raise ValueError(f"{field}: correction bounds must be nonnegative")
    return result


def _reported(value: Fraction) -> dict[str, Any]:
    try:
        approximate = float(value)
    except OverflowError:
        approximate = None
    if approximate is not None and not math.isfinite(approximate):
        approximate = None
    return {"exact": str(value), "value": approximate}


def _validate_model(model: Any):
    if not isinstance(model, Mapping):
        raise ValueError("model: expected a JSON object")
    _keys(model, {"schema_version", "kind", "model_id", "actions", "row_ids", "base_scores",
                  "correction_bounds", "provenance", "assumptions"}, "model")
    if type(model.get("schema_version")) is not int or model["schema_version"] != 1:
        raise ValueError("schema_version: expected integer 1")
    if model.get("kind") != "score_box_model":
        raise ValueError("kind: expected 'score_box_model'")
    if not _nonempty_text(model.get("model_id")):
        raise ValueError("model_id: expected a nonempty string")
    actions, row_ids = _names(model.get("actions"), "actions"), _names(model.get("row_ids"), "row_ids")
    rows = model.get("base_scores")
    if not isinstance(rows, (list, tuple)) or len(rows) != len(row_ids):
        raise ValueError("base_scores: expected one row per row_id")
    scores = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != len(actions):
            raise ValueError(f"base_scores[{i}]: expected one score per action")
        scores.append(tuple(_rational(value, f"base_scores[{i}][{j}]") for j, value in enumerate(row)))
    values = model.get("correction_bounds")
    if not isinstance(values, (list, tuple)) or len(values) != len(actions):
        raise ValueError("correction_bounds: required shared vector with one bound per action")
    bounds = tuple(_rational(value, f"correction_bounds[{j}]", nonnegative=True) for j, value in enumerate(values))
    provenance = model.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance: required score and bound descriptions")
    _keys(provenance, {"score_source", "bound_source", "row_population", "synthetic"}, "provenance")
    for field in ("score_source", "bound_source", "row_population"):
        if not _nonempty_text(provenance.get(field)):
            raise ValueError(f"provenance.{field}: expected a nonempty description")
    if not isinstance(provenance.get("synthetic"), bool):
        raise ValueError("provenance.synthetic: explicitly declare true or false")
    assumptions = model.get("assumptions")
    if not isinstance(assumptions, (list, tuple)) or not assumptions or not all(_nonempty_text(x) for x in assumptions):
        raise ValueError("assumptions: provide at least one explicit bound assumption")
    return actions, row_ids, tuple(scores), bounds


def _beats(left: Fraction, left_index: int, right: Fraction, right_index: int) -> bool:
    return left > right or (left == right and left_index < right_index)


def audit_response_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Certify unchanged argmax rows for additive |correction[j]| <= bound[j].

    A winner is unchanged iff its lower score wins against every challenger's
    upper score under the fixed tie rule. Potential challengers are alternative
    actions that can actually win somewhere in the full independent score box:
    their upper score must beat every other action's lower score. Thus a
    pairwise threat blocked by a third action is not called a potential winner.
    """
    actions, row_ids, scores, bounds = _validate_model(model)
    rows = []
    for row_id, score in zip(row_ids, scores):
        winner = max(range(len(actions)), key=lambda index: score[index])
        low = tuple(value - bound for value, bound in zip(score, bounds))
        high = tuple(value + bound for value, bound in zip(score, bounds))
        certified = all(_beats(low[winner], winner, high[j], j) for j in range(len(actions)) if j != winner)
        challengers = [j for j in range(len(actions)) if j != winner and
                       all(_beats(high[j], j, low[k], k) for k in range(len(actions)) if k != j)]
        rows.append({
            "row_id": row_id,
            "original_action": actions[winner],
            "original_action_index": winner,
            "certified_unchanged": certified,
            "potential_challengers": [actions[j] for j in challengers],
            "potential_challenger_indices": challengers,
            "winner_lower_score": _reported(low[winner]),
            "action_lower_scores_exact": [str(value) for value in low],
            "action_upper_scores_exact": [str(value) for value in high],
        })
    unchanged = sum(row["certified_unchanged"] for row in rows)
    responsive = len(rows) - unchanged
    fraction = Fraction(unchanged, len(rows))
    change_bound = Fraction(responsive, len(rows))
    return {
        "schema_version": 1,
        "audit_kind": "conditional_score_box_responsiveness",
        "status": "computed",
        "model_id": model["model_id"],
        "scope": "Exact conditional argmax stability in the supplied independent score box; not a target-regret or generalization guarantee.",
        "bound_validity_verified": False,
        "target_regret_guarantee": False,
        "generalization_guarantee": False,
        "labels_or_outcomes_used": False,
        "provenance": dict(model["provenance"]),
        "declared_assumptions": list(model["assumptions"]),
        "model": {
            "actions": list(actions),
            "row_ids": list(row_ids),
            "base_scores_exact": [[str(value) for value in score] for score in scores],
            "correction_bounds_exact": [str(value) for value in bounds],
        },
        "counts": {"rows": len(rows), "actions": len(actions), "certified_unchanged": unchanged, "potentially_changed": responsive},
        "unchanged_fraction": _reported(fraction),
        "potentially_changed_fraction": _reported(change_bound),
        "mean_utility_change_bound": {
            "absolute_upper_bound": _reported(change_bound),
            "signed_interval_exact": [str(-change_bound), str(change_bound)],
            "utility_range": [0, 1],
            "interpretation": "For fixed per-row/action utilities in [0,1], the absolute change of their unweighted mean is at most the potentially changed row fraction. This is not observed utility change or regret.",
        },
        "rows": rows,
        "method": {
            "tie_rule": "Highest score wins; lowest action index wins exact ties. Action indices are zero-based and follow input order.",
            "box": "For every row and action j, corrected_score[j] = base_score[j] + delta[j], with -bound[j] <= delta[j] <= bound[j].",
            "certificate": "Compare the original winner's lower score with every alternative's upper score, including the tie rule. Potential challengers are exact alternative winners within the independent box.",
            "precision": "Exact rational arithmetic; JSON decimal numbers retain their written decimal value. Python float inputs use their decimal string value. Display values are approximate and become null if not finite.",
        },
        "required_interpretation": [
            "The caller supplies and justifies the bounds. The tool cannot verify that they cover every correction a real router can produce.",
            "The result applies only to the supplied rows, action order, additive bounds, and deterministic argmax tie rule.",
            "A potentially changed row need not change in practice. Coupled corrections, reachable histories, or a smaller feasible set can make the real router more stable than this independent box.",
            "The utility-change bound assumes fixed action utilities in [0,1] and equal row weights. It says nothing about whether the original decisions were good.",
            "Labels, target regret, probe quality, correction attainability, and generalization are not established by this certificate.",
        ],
    }


def audit_response_json(path: str | Path) -> dict[str, Any]:
    """Read one strict score-box JSON object, preserving decimal precision."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON number {value}")

    model = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                       parse_constant=reject_constant, parse_float=Decimal)
    return audit_response_model(model)
