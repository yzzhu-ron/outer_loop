"""Label-free decision-change bounds for the frozen semantic profile router.

For B paired observations and profile pseudocount s, each of the four inputs to
an action's correction model lies in [-B/(B+s), B/(B+s)]. Consequently
|correction_a| <= ||coef_a||_1 B/(B+s). The original action has zero correction.
The bound ignores shared counts and dependencies between global/bucket means;
it is conservative. A baseline winner is certified unchanged if its margin to
every challenger exceeds the sum of their correction bounds. An uncertified
query is only possibly changeable; this is not a reachability certificate.

The optional full-pool calculation uses every observed probe as an evaluator
diagnostic. It never enters selection. Full-pool aggregation can cancel useful
subsets and is not an oracle ceiling. Neither calculation reads task utilities
or relevance labels. Full input hashes are checked only for cache integrity.

Run after analysis:
  python audit_router_sensitivity.py --include-full-pool
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from learning import ACTIONS, BUDGETS, SHRINKAGE, Profile


ROOT = Path(__file__).resolve().parent
ROUNDING_SLACK = 1e-12


def coefficient_matrix(model_metadata):
    if model_metadata.get("calibrator_intercept") is not False:
        raise ValueError("Certificate requires the frozen zero-intercept calibrator")
    matrix = np.zeros((len(ACTIONS), 4), dtype=float)
    given = model_metadata["calibrator_coefficients"]
    if "original" in given and np.any(np.asarray(given["original"]) != 0):
        raise ValueError("Original action must have exactly zero correction")
    for action, name in enumerate(ACTIONS[1:], 1):
        values = np.asarray(given[name], dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("Each alternative needs four finite correction coefficients")
        matrix[action] = values
    return matrix


def base_predictions(model_metadata, features):
    """Reconstruct the saved fixed or StandardScaler+Ridge source router."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not len(features) or not np.all(np.isfinite(features)):
        raise ValueError("Deployment features must be a nonempty finite matrix")
    parameters = model_metadata["base_router_parameters"]
    if parameters["type"] == "fixed":
        scores = np.asarray(parameters["scores"], dtype=float)
        predictions = np.tile(scores, (len(features), 1))
    elif parameters["type"] == "standard_scaler_ridge":
        mean = np.asarray(parameters["scaler_mean"], dtype=float)
        scale = np.asarray(parameters["scaler_scale"], dtype=float)
        coefficients = np.asarray(parameters["coefficients"], dtype=float)
        intercept = np.asarray(parameters["intercept"], dtype=float)
        if mean.shape != (features.shape[1],) or scale.shape != mean.shape or np.any(scale <= 0) or coefficients.shape != (len(ACTIONS), features.shape[1]):
            raise ValueError("Saved scaler/linear model dimensions are inconsistent")
        predictions = ((features - mean) / scale) @ coefficients.T + intercept
    else:
        raise ValueError("Unknown saved base-router representation")
    if predictions.shape != (len(features), len(ACTIONS)) or not np.all(np.isfinite(predictions)):
        raise ValueError("Saved router must produce five finite action predictions")
    return predictions


def change_certificate(base, coefficients, pair_budget, *, shrinkage=4.0):
    """Return an upper bound on changeable queries, without task outcomes.

    The guarantee is mathematical for the stated linear profile model. Positive
    bounds are rounded upward and strict comparisons include 1e-12 slack; this
    implementation is not a formally verified interval-arithmetic library.
    Zero-bound pairs preserve the existing deterministic argmax, including ties.
    """
    base, coefficients = np.asarray(base, dtype=float), np.asarray(coefficients, dtype=float)
    if base.ndim != 2 or base.shape[1] != len(ACTIONS) or not len(base) or not np.all(np.isfinite(base)):
        raise ValueError("Base predictions must be a finite nonempty n-by-five matrix")
    if coefficients.shape != (len(ACTIONS), 4) or not np.all(np.isfinite(coefficients)) or np.any(coefficients[0] != 0):
        raise ValueError("Certificate requires five coefficient rows with zero original row")
    if isinstance(pair_budget, bool) or int(pair_budget) != pair_budget or pair_budget < 0 or not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("Pair budget must be a nonnegative integer and shrinkage positive")
    feature_bound = float(pair_budget / (pair_budget + shrinkage))
    coefficient_l1 = np.abs(coefficients).sum(axis=1)
    bounds = coefficient_l1 * feature_bound
    positive = bounds > 0
    bounds[positive] = np.nextafter(bounds[positive], np.inf)
    winners = base.argmax(axis=1)
    margins = base[np.arange(len(base)), winners, None] - base
    pair_bounds = bounds[winners, None] + bounds[None, :]
    # A zero-bound pair cannot change, even if the baseline predictions tie.
    stable_pairs = (pair_bounds == 0) | (margins > pair_bounds + ROUNDING_SLACK)
    stable_pairs[np.arange(len(base)), winners] = True
    certified = stable_pairs.all(axis=1)
    challenger_mask = ~stable_pairs
    display_margins = margins.copy()
    display_margins[np.arange(len(base)), winners] = np.inf
    slack = margins - pair_bounds
    slack[np.arange(len(base)), winners] = np.inf
    return {
        "pair_budget": int(pair_budget), "profile_feature_abs_bound": feature_bound,
        "coefficient_l1": coefficient_l1.tolist(), "action_correction_abs_bounds": bounds.tolist(),
        "n_queries": len(base), "baseline_actions": winners.tolist(),
        "baseline_smallest_margins": display_margins.min(axis=1).tolist(),
        "certificate_slacks": slack.min(axis=1).tolist(),
        "certified_unchangeable": certified.tolist(), "potential_challengers": challenger_mask.tolist(),
        "certified_unchangeable_fraction": float(certified.mean()),
        "changeable_query_fraction_upper_bound": float((~certified).mean()),
        "mean_bounded_utility_change_abs_upper_bound": float((~certified).mean()),
    }


def audit_environment(record, model_metadata, *, expected_baseline=None, budgets=BUDGETS, include_full_pool=False):
    """Only query IDs/features/buckets and optional probe outcomes are accessed."""
    tasks = record["tasks"]["test"]
    ids = list(tasks["ids"])
    predictions = base_predictions(model_metadata, tasks["features"])
    if len(ids) != len(predictions):
        raise ValueError("Query IDs and deployment feature rows differ")
    if expected_baseline is not None and not np.array_equal(predictions.argmax(axis=1), expected_baseline):
        raise ValueError("Reconstructed base-router actions differ from saved analysis")
    coefficients = coefficient_matrix(model_metadata)
    shrinkage = float(model_metadata["profile_shrinkage"])
    if shrinkage != SHRINKAGE:
        raise ValueError("Saved profile shrinkage differs from the frozen implementation")
    bounds, queries = [], {}
    for budget in budgets:
        certificate = change_certificate(predictions, coefficients, budget, shrinkage=shrinkage)
        row = {"environment": record["name"], "pair_budget": int(budget), "n_queries": len(ids),
               "profile_feature_abs_bound": certificate["profile_feature_abs_bound"],
               "certified_unchangeable_fraction": certificate["certified_unchangeable_fraction"],
               "changeable_query_fraction_upper_bound": certificate["changeable_query_fraction_upper_bound"],
               "mean_bounded_utility_change_abs_upper_bound": certificate["mean_bounded_utility_change_abs_upper_bound"]}
        row.update({f"{action}_correction_abs_bound": certificate["action_correction_abs_bounds"][i] for i, action in enumerate(ACTIONS)})
        bounds.append(row)
        queries[str(budget)] = {"query_ids": ids, **certificate}
    full_pool = []
    if include_full_pool:
        # The real policy never receives these aggregate, unselected outcomes.
        for seed, pool in sorted(record["pools"].items()):
            profile = Profile()
            for candidate in pool["candidates"]:
                profile.update(candidate, pool["observations"][candidate["id"]])
            corrected = predictions.copy()
            for action in range(1, len(ACTIONS)):
                corrected[:, action] += profile.routing_features(action, tasks["buckets"]) @ coefficients[action]
            count = len(pool["candidates"])
            certificate = change_certificate(predictions, coefficients, count, shrinkage=shrinkage)
            row = {"environment": record["name"], "seed": str(seed), "all_pool_pair_count": count,
                   "logical_search_calls_to_observe_all_pairs": sum(int(c["cost"]) for c in pool["candidates"]),
                   "observed_full_pool_changed_action_fraction": float(np.mean(corrected.argmax(axis=1) != predictions.argmax(axis=1))),
                   "certified_unchangeable_fraction_at_full_pair_count": certificate["certified_unchangeable_fraction"],
                   "evaluator_only": True}
            row.update({f"{name}_observed_correction_max_abs": float(np.max(np.abs(corrected[:, action] - predictions[:, action]))) for action, name in enumerate(ACTIONS)})
            full_pool.append(row)
    return {"bounds": bounds, "queries": queries, "full_pool": full_pool,
            "coefficients": coefficients.tolist(), "profile_shrinkage": shrinkage}


def _write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--include-full-pool", action="store_true")
    args = parser.parse_args(argv)
    models = json.loads((args.results_dir / "models.json").read_text())
    paired = json.loads((args.results_dir / "paired_outcomes.json").read_text())
    provenance = json.loads((args.results_dir / "analysis_metadata.json").read_text())
    learning_hash = hashlib.sha256((ROOT / "learning.py").read_bytes()).hexdigest()
    if learning_hash != provenance["learning_sha256"]:
        raise ValueError("Profile implementation differs from the saved analysis revision")
    bounds, full_pool, details = [], [], {}
    input_paths = {f"results/{name}": args.results_dir / name for name in ("models.json", "paired_outcomes.json", "analysis_metadata.json")}
    for environment, saved in paired.items():
        path = args.cache_dir / f"{environment}_outcomes.json"
        record = json.loads(path.read_text())
        actual_hash = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if actual_hash != provenance["input_record_sha256"][environment]:
            raise ValueError("Outcome record differs from the input used to fit the saved model")
        fold = f"{record['corpus_name']}::{record['backend']}"
        model = models[fold]
        if model["analysis_learning_sha256"] != provenance["learning_sha256"] or model["analysis_protocol_sha256"] != provenance["protocol_sha256"]:
            raise ValueError("Saved model and analysis contracts differ")
        if record["tasks"]["test"]["ids"] != saved["query_ids"]:
            raise ValueError("Deployment query ID order differs from saved analysis")
        result = audit_environment(record, model, expected_baseline=saved["source_router_actions"], include_full_pool=args.include_full_pool)
        bounds.extend(result["bounds"])
        full_pool.extend(result["full_pool"])
        details[environment] = {"queries": result["queries"], "coefficients": result["coefficients"], "profile_shrinkage": result["profile_shrinkage"]}
        input_paths[f"cache/{environment}_outcomes.json"] = path
    _write_csv(args.results_dir / "router_sensitivity.csv", bounds)
    _write_csv(args.results_dir / "full_pool_sensitivity.csv", full_pool)
    if not full_pool:
        (args.results_dir / "full_pool_sensitivity.csv").unlink(missing_ok=True)
    (args.results_dir / "router_sensitivity_queries.json").write_text(json.dumps(details, indent=2) + "\n")
    metadata = {
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in input_paths.items()},
        "output_sha256": {name: hashlib.sha256((args.results_dir / name).read_bytes()).hexdigest() for name in
                          ["router_sensitivity.csv", "router_sensitivity_queries.json", *(["full_pool_sensitivity.csv"] if full_pool else [])]},
        "uses_target_task_utility_labels": False, "uses_unselected_probe_outcomes": bool(args.include_full_pool),
        "uses_future_query_features_for_frozen_evaluation": True,
        "learning_sha256": learning_hash,
        "guarantee": "For any history of at most B paired probes with RR differences in [-1,1], each linear correction is bounded by coefficient_l1*B/(B+shrinkage). Original correction is zero. Certified queries cannot change action in this frozen profile router; all other queries are only potentially changeable.",
        "conservatism": "The four feature extremes and the winner/challenger bounds need not be jointly attainable under a shared total probe budget. The test intentionally overestimates how much decisions could change.",
        "bounded_utility_statement": "If per-query utilities lie in [0,1], the absolute mean utility change is at most the uncertified query fraction. This upper bound requires no task labels and is usually loose.",
        "floating_point": f"Positive bounds rounded upward once; strict margin comparisons add {ROUNDING_SLACK} slack. This is not formally verified interval arithmetic.",
        "full_pool_warning": "Optional hindsight diagnostic only. Combining every probe can cancel useful subsets, so unchanged full-pool actions do not establish that all subsets are useless. It is not a selector oracle or statistical generalization claim.",
        "cost_scope": "Full-pool search counts are logical probe-pair searches; sampling/generation setup remains separately recorded in the main cost ledger.",
    }
    (args.results_dir / "router_sensitivity_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(bounds)} label-free budget certificates; {len(full_pool)} optional full-pool diagnostics.")


if __name__ == "__main__":
    main()
