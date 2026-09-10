"""Deterministic transcript checks; no relevance labels or model calls required."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


OUTCOME_TOLERANCE = 1e-12


def canonical_query(query: str) -> tuple[str, ...]:
    """A token multiset, preserving repetitions; a heuristic, not API semantics."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    return tuple(sorted(re.findall(r"\w+", normalized, flags=re.UNICODE)))


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _sum(values: Iterable[float]) -> float | None:
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


def validate_record(record: Any) -> list[dict[str, str]]:
    """Return field-addressed errors without mutating a record.

    Empty list means structurally valid. Cross-record duplicate IDs are checked
    by audit_records. Queries may be equal: that is a diagnostic, not a schema
    violation. Cost and observations are caller-reported, never independently
    verified by this validator.
    """
    errors: list[dict[str, str]] = []

    def error(field: str, message: str, code: str = "invalid_record") -> None:
        errors.append({"code": code, "field": field, "message": message})

    def keys(value: Mapping, allowed: set[str], field: str = "") -> None:
        for key in sorted(set(value) - allowed, key=str):
            error(f"{field}.{key}" if field else str(key), "Unknown field; check the contract or a misspelled name.")

    if not isinstance(record, Mapping):
        error("$", "A probe must be a JSON object.")
        return errors
    keys(record, {"probe_id", "family", "actions", "queries", "cost", "bucket", "known_target", "outcomes"})
    for field in ("probe_id", "family"):
        if not _text(record.get(field)):
            error(field, "Required nonempty string.")
    if "bucket" in record and not _text(record["bucket"]):
        error("bucket", "If supplied, bucket must be a nonempty string.")
    for field in ("actions", "queries"):
        value = record.get(field)
        if not _pair(value) or not all(_text(item) for item in value):
            error(field, "Required pair of nonempty strings, in the same left/right order throughout the record.")
    if _pair(record.get("actions")) and record["actions"][0] == record["actions"][1]:
        error("actions", "Use two distinct action names; a probe compares two actions.")

    cost = record.get("cost")
    if not isinstance(cost, Mapping):
        error("cost", "Required object with values, total, and unit.", "accounting_violation")
    else:
        keys(cost, {"values", "total", "unit"}, "cost")
        values = cost.get("values")
        values_valid = _pair(values) and all(_number(x) and x >= 0 for x in values)
        total_valid = _number(cost.get("total")) and cost["total"] >= 0
        if not values_valid:
            error("cost.values", "Required pair of finite, nonnegative numbers.", "accounting_violation")
        if not total_valid:
            error("cost.total", "Required finite, nonnegative number.", "accounting_violation")
        if not _text(cost.get("unit")):
            error("cost.unit", "Required nonempty unit, e.g. search_calls, milliseconds, or USD.", "accounting_violation")
        if values_valid and total_valid:
            pair_total = _sum(values)
            if pair_total is None or not math.isclose(pair_total, cost["total"], rel_tol=1e-9, abs_tol=1e-12):
                error("cost.total", "Must equal the sum of cost.values (relative tolerance 1e-9, absolute tolerance 1e-12).", "accounting_violation")

    if "known_target" in record and "outcomes" in record:
        error("$", "Supply known_target or outcomes, not both; one record must have an unambiguous observed endpoint.")
    if "known_target" in record:
        target = record["known_target"]
        if not isinstance(target, Mapping):
            error("known_target", "Must be an object with ranks and cutoff.")
        else:
            keys(target, {"id", "ranks", "cutoff"}, "known_target")
            if "id" in target and not _text(target["id"]):
                error("known_target.id", "If supplied, target id must be a nonempty string.")
            cutoff = target.get("cutoff")
            if not _positive_integer(cutoff):
                error("known_target.cutoff", "Required positive integer retrieval depth.")
            ranks = target.get("ranks")
            if not _pair(ranks):
                error("known_target.ranks", "Required pair: one-based rank or null if absent at cutoff.")
            else:
                for i, rank in enumerate(ranks):
                    if rank is not None and (not _positive_integer(rank) or (_positive_integer(cutoff) and rank > cutoff)):
                        error(f"known_target.ranks[{i}]", "Must be null or a positive integer no greater than cutoff; null means absent at cutoff.")
    if "outcomes" in record:
        outcomes = record["outcomes"]
        if not isinstance(outcomes, Mapping):
            error("outcomes", "Must be an object with paired numeric values.")
        else:
            keys(outcomes, {"values", "bounds", "higher_is_better"}, "outcomes")
            values = outcomes.get("values")
            values_valid = _pair(values) and all(_number(x) for x in values)
            if not values_valid:
                error("outcomes.values", "Required pair of finite numbers measuring the same endpoint.")
            if "higher_is_better" in outcomes and not isinstance(outcomes["higher_is_better"], bool):
                error("outcomes.higher_is_better", "Must be true or false; defaults to true.")
            if "bounds" in outcomes:
                bounds = outcomes["bounds"]
                bounds_valid = _pair(bounds) and all(_number(x) for x in bounds) and bounds[0] < bounds[1]
                if not bounds_valid:
                    error("outcomes.bounds", "Must be a finite [minimum, maximum] pair with minimum < maximum.")
                elif values_valid and not all(bounds[0] <= x <= bounds[1] for x in values):
                    error("outcomes.values", "Values must lie within the declared bounds.")
    return errors


@dataclass(frozen=True)
class _MalformedJSON:
    line: int
    message: str


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}.")
        result[key] = value
    return result


def _reject_nonstandard_number(token: str) -> None:
    raise ValueError(f"Non-JSON number {token}.")


def audit_jsonl(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Read UTF-8 JSONL, retaining malformed-line errors in the audit report.

    Blank lines are ignored. JSON NaN/Infinity and duplicate object keys are
    rejected. Record indices count nonblank lines; parse errors also include
    their original physical line number. File/encoding errors propagate.
    """
    def records() -> Iterable[Any]:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line, object_pairs_hook=_strict_object,
                                     parse_constant=_reject_nonstandard_number)
                except (ValueError, RecursionError) as exc:
                    yield _MalformedJSON(line_number, str(exc))
    return audit_records(records(), **kwargs)


def _observation(record: Mapping[str, Any]) -> tuple[bool, bool, bool] | None:
    """Return (zero contrast, both best-bound, both worst-bound), or None."""
    if "known_target" in record:
        left, right = record["known_target"]["ranks"]
        return left == right, left == right == 1, left is None and right is None
    if "outcomes" not in record:
        return None
    outcomes = record["outcomes"]
    left, right = outcomes["values"]
    equal = lambda x, y: math.isclose(x, y, rel_tol=0.0, abs_tol=OUTCOME_TOLERANCE)
    if "bounds" not in outcomes:
        return equal(left, right), False, False
    low, high = outcomes["bounds"]
    best, worst = (high, low) if outcomes.get("higher_is_better", True) else (low, high)
    # Bound attainment is exact: a tolerance must not label a narrow interval
    # as simultaneously at its success ceiling and failure floor.
    return equal(left, right), left == right == best, left == right == worst


def audit_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_action_pairs: Sequence[Sequence[str]] | None = None,
    expected_buckets: Sequence[str] | None = None,
    min_per_cell: int = 3,
) -> dict[str, Any]:
    """Audit query contrast, observed contrast, coverage, and reported cost.

    Pure Python, deterministic, and network-free. Expected pairs are unordered;
    coverage cells are family × pair × bucket. Each observed family is checked
    against the explicit expected grid, or against the union of observed pairs
    and buckets when an expected axis is omitted. A minimum of 3 is only an
    editable coverage heuristic, not a power calculation or adequacy guarantee.
    """
    if not _positive_integer(min_per_cell):
        raise ValueError("min_per_cell must be a positive integer")
    expected_pairs: set[tuple[str, str]] = set()
    if expected_action_pairs is not None:
        for pair in expected_action_pairs:
            if not _pair(pair) or not all(_text(x) for x in pair) or pair[0] == pair[1]:
                raise ValueError("expected_action_pairs must contain pairs of distinct, nonempty action names")
            expected_pairs.add(tuple(sorted(pair)))
    if expected_buckets is not None and (isinstance(expected_buckets, str) or not all(_text(x) for x in expected_buckets)):
        raise ValueError("expected_buckets must be a sequence of nonempty strings")

    counts: Counter[str] = Counter()
    cells: dict[tuple, Counter] = defaultdict(Counter)
    invalid_records = []
    examples: dict[str, list[str]] = defaultdict(list)
    costs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    families, observed_pairs, observed_buckets, seen_ids = set(), set(), set(), set()

    def count(name: str, record: Mapping, cell: Counter) -> None:
        counts[name] += 1
        cell[name] += 1
        if len(examples[name]) < 5:
            examples[name].append(record["probe_id"])

    for record_index, record in enumerate(records, 1):
        counts["records"] += 1
        if isinstance(record, _MalformedJSON):
            counts["invalid_records"] += 1
            invalid_records.append({"record_index": record_index, "line": record.line, "probe_id": None,
                                    "errors": [{"code": "invalid_json", "field": "$", "message": record.message}]})
            continue
        errors = validate_record(record)
        probe_id = record.get("probe_id") if isinstance(record, Mapping) else None
        if _text(probe_id):
            if probe_id in seen_ids:
                errors.append({"code": "duplicate_probe_id", "field": "probe_id", "message": "Probe id already appeared; use a unique id for every trial, including repetitions."})
            seen_ids.add(probe_id)
        if errors:
            counts["invalid_records"] += 1
            if any(error["code"] == "accounting_violation" for error in errors):
                counts["accounting_violation_records"] += 1
            invalid_records.append({"record_index": record_index, "probe_id": probe_id if _text(probe_id) else None, "errors": errors})
            continue

        counts["valid_records"] += 1
        family, pair, bucket = record["family"], tuple(sorted(record["actions"])), record.get("bucket")
        families.add(family)
        observed_pairs.add(pair)
        observed_buckets.add(bucket)
        cell = cells[(family, pair, bucket)]
        cell["records"] += 1
        left, right = record["queries"]
        exact = left == right
        bag = canonical_query(left) == canonical_query(right)
        if exact:
            count("exact_identical_query_pairs", record, cell)
        if bag:
            count("token_bag_identical_query_pairs", record, cell)
        if not exact:
            count("exact_distinct_query_pairs", record, cell)
        if not bag:
            count("token_bag_distinct_query_pairs", record, cell)

        cost = record["cost"]
        costs[cost["unit"]]["total"].append(cost["total"])
        if exact:
            costs[cost["unit"]]["on_exact_identical_queries"].append(cost["total"])
        observation = _observation(record)
        if observation is None:
            count("preflight_records", record, cell)
            continue
        count("observed_records", record, cell)
        zero, best, worst = observation
        if zero:
            count("zero_contrast_pairs", record, cell)
            costs[cost["unit"]]["on_zero_contrast_outcomes"].append(cost["total"])
        else:
            count("nonzero_contrast_pairs", record, cell)
            if exact:
                count("exact_identical_with_nonzero_outcomes", record, cell)
        if best:
            count("saturated_success_pairs", record, cell)
        if worst:
            count("floor_pairs", record, cell)

    all_count_names = (
        "records", "valid_records", "invalid_records", "accounting_violation_records",
        "preflight_records", "observed_records", "exact_identical_query_pairs",
        "token_bag_identical_query_pairs", "exact_distinct_query_pairs", "token_bag_distinct_query_pairs",
        "zero_contrast_pairs", "nonzero_contrast_pairs", "saturated_success_pairs", "floor_pairs",
        "exact_identical_with_nonzero_outcomes",
    )
    pairs = observed_pairs | expected_pairs
    buckets = observed_buckets | set(expected_buckets or [])
    if not buckets:
        buckets = {None}
    coverage_cells = []
    for family in sorted(families):
        for pair in sorted(pairs):
            for bucket in sorted(buckets, key=lambda x: (x is not None, x or "")):
                cell = cells[(family, pair, bucket)]
                coverage_cells.append({"family": family, "actions": list(pair), "bucket": bucket,
                                       "records": cell["records"], "exact_distinct_query_pairs": cell["exact_distinct_query_pairs"],
                                       "token_bag_distinct_query_pairs": cell["token_bag_distinct_query_pairs"],
                                       "observed_records": cell["observed_records"], "nonzero_contrast_pairs": cell["nonzero_contrast_pairs"]})

    diagnostics: list[dict[str, Any]] = []

    def diagnostic(code: str, severity: str, evidence: str, interpretation: str, action: str,
                   count_name: str | None = None) -> None:
        item = {"code": code, "severity": severity, "evidence": evidence, "interpretation": interpretation, "action": action}
        if count_name:
            item["count"] = counts[count_name]
            item["example_probe_ids"] = examples[count_name]
        diagnostics.append(item)

    if not counts["valid_records"]:
        diagnostic("no_valid_probes", "error", "No valid probe records were available.",
                   "The audit has no usable transcript evidence, including no evidence of declared coverage.",
                   "Provide at least one valid record; preflight records may omit outcomes.")
    if counts["invalid_records"]:
        diagnostic("invalid_records", "error", f"{counts['invalid_records']} records failed validation and were excluded from all other summaries.",
                   "Reported costs and coverage describe valid records only; excluded costs are unknown.",
                   "Fix the field-addressed errors in invalid_records and rerun the audit.")
    if counts["accounting_violation_records"]:
        diagnostic("accounting_violations", "error", f"{counts['accounting_violation_records']} records have missing, invalid, or inconsistent paired costs.",
                   "The transcript cannot support a complete budget claim until these entries are corrected.",
                   "Record both marginal costs, their total, and a consistent unit; do not hide generation or setup costs inside an unlogged budget.")
    if counts["exact_identical_query_pairs"]:
        diagnostic("exact_query_collision", "warning", "The two submitted query strings are byte-for-byte identical in these pairs.",
                   "With deterministic retrieval and equal non-query settings, these pairs cannot identify a query-transformation effect. Stochastic or configuration effects remain possible.",
                   "Check transformations before retrieval. Regenerate collapsed pairs, or explicitly treat them as repeated-query controls.", "exact_identical_query_pairs")
    bag_only = counts["token_bag_identical_query_pairs"] - counts["exact_identical_query_pairs"]
    if bag_only:
        diagnostic("token_bag_collision", "warning", f"{bag_only} additional pairs differ as strings but have the same normalized token multiset.",
                   "Such pairs collapse for order-insensitive backends using this normalization. Phrase syntax, case, tokenization, and semantic models may distinguish them.",
                   "Check the backend's actual analyzer; include probes that change relevant content when testing bag-of-words search.")
    if counts["zero_contrast_pairs"]:
        diagnostic("observed_zero_contrast", "warning", "The paired recorded endpoints are equal (numeric outcomes use absolute tolerance 1e-12).",
                   "These observations provide no signed preference between the paired actions on this endpoint. They do not establish permanent incapacity or equal result lists.",
                   "Inspect endpoint coarseness and probe difficulty; use informative targets and repeat stochastic cases.", "zero_contrast_pairs")
    if counts["saturated_success_pairs"]:
        diagnostic("saturated_success", "warning", "Both actions reach rank 1, or the declared best outcome bound.",
                   "The chosen endpoint has no upward room on these trials; this does not measure future-task usefulness.",
                   "Try indirect, paraphrased, or harder known-target probes and check that both easy and difficult cases remain represented.", "saturated_success_pairs")
    if counts["floor_pairs"]:
        diagnostic("observed_floor", "warning", "Both targets are absent at cutoff, or both outcomes reach the declared worst bound.",
                   "Censoring or an overly difficult probe may hide differences. Absence at cutoff says nothing about deeper ranks.",
                   "Inspect target validity and query difficulty; consider a deeper cutoff or a more sensitive endpoint.", "floor_pairs")
    if counts["exact_identical_with_nonzero_outcomes"]:
        diagnostic("same_query_different_outcomes", "warning", "Some exact-identical query pairs have different recorded endpoints.",
                   "The log is inconsistent with deterministic retrieval under equal non-query settings, but is not necessarily invalid.",
                   "Check randomness, changing indexes, cache/version differences, request parameters, and logging order before interpreting an action effect.", "exact_identical_with_nonzero_outcomes")
    sparse = [cell for cell in coverage_cells if cell["records"] < min_per_cell]
    sparse_distinct = [cell for cell in coverage_cells if cell["exact_distinct_query_pairs"] < min_per_cell]
    if sparse:
        diagnostic("sparse_coverage", "warning", f"{len(sparse)} of {len(coverage_cells)} family/action-pair/bucket cells have fewer than {min_per_cell} valid records.",
                   "Coverage is uneven or missing in the declared/observed grid. The chosen minimum is a heuristic, not a sample-size guarantee.",
                   "Review whether the crossed grid is meaningful, declare expected pairs/buckets, and fill relevant missing cells.")
    if sparse_distinct and len(sparse_distinct) > len(sparse):
        diagnostic("sparse_distinct_query_coverage", "warning", f"{len(sparse_distinct)} cells have fewer than {min_per_cell} pairs with distinct query strings.",
                   "Raw sample counts overstate opportunities to distinguish query transformations in deterministic fixed-setting retrieval.",
                   "Regenerate collapsed pairs in affected cells before increasing retrieval budget.")
    if counts["preflight_records"]:
        diagnostic("outcomes_not_observed", "info", f"{counts['preflight_records']} records contain no endpoint observations.",
                   "Their query structure and declared costs can be checked; saturation and observed contrast cannot yet be assessed.",
                   "Run structurally useful probes, record the paired outcomes, and audit the resulting log.")

    cost_summary = {}
    for unit in sorted(costs):
        cost_summary[unit] = {field: _sum(costs[unit][field]) for field in ("total", "on_exact_identical_queries", "on_zero_contrast_outcomes")}
    if any(value is None for unit in cost_summary.values() for value in unit.values()):
        diagnostic("cost_aggregation_overflow", "error", "At least one cost aggregate exceeds finite floating-point range and is null.",
                   "Individual records passed validation, but a numeric budget total cannot be represented.",
                   "Use a larger cost unit and rerun the audit.")
    valid, observed = counts["valid_records"], counts["observed_records"]
    return {
        "schema_version": 1,
        "status": "error" if any(d["severity"] == "error" for d in diagnostics) else "warning" if any(d["severity"] == "warning" for d in diagnostics) else "ok",
        "counts": {name: counts[name] for name in all_count_names},
        "rates": {
            "exact_query_collision_among_valid": counts["exact_identical_query_pairs"] / valid if valid else None,
            "token_bag_collision_among_valid": counts["token_bag_identical_query_pairs"] / valid if valid else None,
            "zero_contrast_among_observed": counts["zero_contrast_pairs"] / observed if observed else None,
            "saturated_success_among_observed": counts["saturated_success_pairs"] / observed if observed else None,
            "nonzero_contrast_among_observed": counts["nonzero_contrast_pairs"] / observed if observed else None,
        },
        "coverage": {"min_per_cell": min_per_cell, "expected_action_pairs": [list(pair) for pair in sorted(expected_pairs)],
                     "expected_buckets": sorted(set(expected_buckets or [])), "cells": coverage_cells,
                     "sparse_cells": len(sparse), "sparse_distinct_query_cells": len(sparse_distinct),
                     "scope": "Observed families crossed with the union of observed and declared pairs/buckets; undeclared, unobserved families cannot be detected."},
        "costs_by_unit": cost_summary,
        "cost_scope": "Valid records only; reported paired marginal costs. Generation/setup costs outside these records are not inferred. Cost categories overlap and must not be added together.",
        "invalid_records": invalid_records,
        "diagnostics": diagnostics,
        "what_this_log_cannot_establish": [
            "Whether probe preferences predict future-task preferences, improve relevance, or generalize to another environment.",
            "Whether unsaturated or nonzero-contrast probes are causally informative, statistically reliable, or worth their cost.",
            "Whether equal query strings imply equal full API requests; request settings and retrieval determinism must be established separately.",
            "Whether self-reported costs, target identities, outcome bounds, and observations match the actual service.",
            "Whether probe generation leaked held-out labels, or unlogged probes/costs were omitted.",
        ],
        "method": {"query_canonicalization": "Unicode NFKC, casefold, Unicode \\w+ tokens, sorted with multiplicity",
                   "outcome_absolute_tolerance": OUTCOME_TOLERANCE, "target_endpoint": "Reciprocal rank at declared cutoff; null is censored absence, not an unbounded rank.",
                   "no_future_task_labels_used": True, "example_ids_per_diagnostic_limit": 5},
    }
