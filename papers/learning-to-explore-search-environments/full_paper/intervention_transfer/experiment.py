"""Source-only utility transfer; prediction and test-label evaluation are separate stages."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
ACTIONS = ("original", "keywords", "semantic", "hyde", "decomposed")
POLICIES = ((0,), (1,), (2,), (3,), (4,), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
ALL_POLICIES = POLICIES + ((0, 1, 2, 3, 4),)
LAMBDAS = (0, .125, .25, .5, .75, .875, 1)
COST_KEYS = ("logical_search_calls", "underlying_search_calls", "llm_calls", "input_tokens", "output_tokens", "sample_calls")
P = len(POLICIES)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def zero_cost():
    return dict.fromkeys(COST_KEYS, 0)


def add_cost(total, extra):
    for key in COST_KEYS:
        total[key] += extra.get(key, 0)


def choose(prediction):
    if len(prediction) != P or not np.isfinite(prediction).all():
        raise ValueError("Expected eleven finite policy predictions")
    def order(i):
        subset = POLICIES[i]
        calls = sum(2 if action == 4 else 1 for action in subset)
        return (-float(prediction[i]), calls, int(any(a >= 2 for a in subset)), tuple(ACTIONS[a] for a in subset))
    return min(range(P), key=order)


def profile(observations):
    if len(observations) != 8:
        raise ValueError("Exactly eight sampled documents are required, including unavailable probes")
    rr = np.asarray([row["rr"] for row in observations], dtype=float)
    if rr.shape != (8, P) or not np.isfinite(rr).all() or np.any(rr < 0) or np.any(rr > 1):
        raise ValueError("Invalid reciprocal-rank observations")
    return np.mean(rr - rr[:, :1], axis=0)


def predict(model, x=None):
    mu = np.asarray(model["mu_D"], dtype=float)
    if model["eta"] == 0:
        result = mu.copy()
    else:
        if x is None:
            raise ValueError("A responsive model requires observed target probes")
        result = mu + model["eta"] * np.asarray(model["beta"]) * (np.asarray(x) - np.asarray(model["mu_X"]))
    result[0] = 0.0
    return result


def _arrays(sources, shuffle_seed=None):
    train = np.asarray([source["train_scores"] for source in sources], dtype=float)[..., :P]
    cal = np.asarray([source["calibration_scores"] for source in sources], dtype=float)[..., :P]
    x = np.asarray([source["fit_profiles"] for source in sources], dtype=float)
    cal_x = np.asarray([source["calibration_profiles"] for source in sources], dtype=float)
    if train.shape[1] != 7 or x.shape != (len(sources), 3, 7, P) or cal_x.shape != (len(sources), 2, 7, P):
        raise ValueError("Wrong source lambda/panel/policy dimensions")
    permutations = []
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for c in range(len(sources)):
            order = rng.permutation(7)
            train[c] = train[c, order].copy()
            cal[c] = cal[c, order].copy()
            permutations.append(order.tolist())
    return train, cal, x, cal_x, permutations


def fit_bridge(sources, mode="within", shuffle_seed=None):
    """Sources contain train/calibration labels only; never a target record."""
    if mode not in ("within", "pooled"):
        raise ValueError("Unknown slope mode")
    train, cal, x, cal_x, permutations = _arrays(sources, shuffle_seed)
    d = (train - train[..., :1]).mean(axis=2)
    # Preserve the exact floating-point static prior under source shuffling,
    # including tie behavior, rather than resumming a permuted lambda axis.
    prior_train = np.asarray([source["train_scores"] for source in sources], dtype=float)[..., :P]
    prior_d = (prior_train - prior_train[..., :1]).mean(axis=2)
    if mode == "within":
        cx = x - x.mean(axis=2, keepdims=True)
        cd = d - d.mean(axis=1, keepdims=True)
    else:
        cx = x - x.mean(axis=(0, 1, 2), keepdims=True)
        cd = d - d.mean(axis=(0, 1), keepdims=True)
    variance = np.mean(cx * cx, axis=(0, 1, 2))
    covariance = np.mean(cx * cd[:, None, :, :], axis=(0, 1, 2))
    beta = np.divide(covariance, 1.1 * variance, out=np.zeros(P), where=variance > 1e-6)
    beta[0] = 0.0
    mu_d = prior_d.mean(axis=(0, 1)); mu_d[0] = 0.0
    mu_x = x.mean(axis=(0, 1, 2)); mu_x[0] = 0.0
    model = {"mode": mode, "mu_D": mu_d.tolist(), "mu_X": mu_x.tolist(), "beta": beta.tolist(),
             "eta": 0.0, "variance": variance.tolist(), "degenerate": (variance <= 1e-6).tolist(),
             "lambda_source_D": d.mean(axis=0).tolist(), "shuffle_seed": shuffle_seed,
             "source_permutations": permutations}
    choices = []
    cal_utilities = cal.mean(axis=2)
    for eta in (0.0, 0.5, 1.0):
        trial = model | {"eta": eta}
        values = [cal_utilities[c, j, choose(predict(trial, cal_x[c, r, j]))]
                  for c in range(len(sources)) for r in range(2) for j in range(7)]
        choices.append({"eta": eta, "mean_ndcg": float(np.mean(values))})
    best = max(row["mean_ndcg"] for row in choices)
    model["eta"] = next(row["eta"] for row in choices if best - row["mean_ndcg"] <= 1e-12)
    model["calibration"] = choices
    return model


def decide_target(model, document_ids, observe):
    """Only a current-backend selected-document callback crosses this boundary."""
    if len(document_ids) != 8 or len(set(document_ids)) != 8:
        raise ValueError("Expected eight distinct predetermined documents")
    costs = zero_cost(); observations = []
    if model["eta"] != 0:
        for doc in document_ids:
            row = observe(doc)
            observations.append({"document_id": doc, "rr": list(row["rr"]), "cost": dict(row["cost"]),
                                 "action_rankings": row.get("action_rankings")})
            add_cost(costs, row["cost"])
    x = profile(observations) if observations else None
    prediction = predict(model, x)
    return {"prediction": prediction.tolist(), "policy": choose(prediction), "profile": None if x is None else x.tolist(),
            "onboarding_cost": costs, "observations": observations}


def source_view(record):
    """Explicit source whitelist: the caller must never pass a target to fitting."""
    worlds = record["worlds"]
    def profiles(kind):
        return [[profile([worlds[str(j)]["probes"][doc] for doc in panel]).tolist()
                 for j in range(7)] for panel in record["panels"][kind]]
    return {"train_scores": [worlds[str(j)]["train"]["scores"] for j in range(7)],
            "calibration_scores": [worlds[str(j)]["calibration"]["scores"] for j in range(7)],
            "fit_profiles": profiles("fit"), "calibration_profiles": profiles("calibration")}


def make_decisions(evidence, shuffle_seeds=range(20260910, 20260930)):
    """This stage accepts evidence with no test-label or test-score field."""
    if evidence.get("stage") != "inputs_without_test_utilities":
        raise ValueError("Decision stage requires label-separated input evidence")
    output = {"stage": "locked_predictions_before_test_utilities", "folds": {}, "schema_version": 1}
    records = evidence["corpora"]
    for target_name, target in records.items():
        sources = [source_view(record) for name, record in records.items() if name != target_name]
        if len(sources) != 2:
            raise ValueError("Exactly two source families required")
        within = fit_bridge(sources, "within")
        pooled = fit_bridge(sources, "pooled")
        combined_d = []
        for source in sources:
            train = np.asarray(source["train_scores"])[..., :P]
            cal = np.asarray(source["calibration_scores"])[..., :P]
            mean = (train.sum(axis=1) + cal.sum(axis=1)) / (train.shape[1] + cal.shape[1])
            combined_d.append(mean - mean[:, :1])
        combined_prior = np.mean(combined_d, axis=(0, 1)).tolist()
        models = {"within": within, "pooled": pooled, "source_fixed": within | {"eta": 0.0},
                  "source_fixed_train_cal": within | {"eta": 0.0, "mu_D": combined_prior},
                  "within_unshrunk": within | {"eta": 1.0}}
        for seed in shuffle_seeds:
            models[f"shuffle_{seed}"] = fit_bridge(sources, "within", seed)
        fold = {"models": models, "runs": {}}
        for j in range(7):
            fold["runs"][str(j)] = {}
            # The privileged comparator is explicitly constructed by evaluator.
            known = within | {"eta": 0.0, "mu_D": within["lambda_source_D"][j]}
            for seed, documents in target["panels"]["target"].items():
                current = target["worlds"][str(j)]["probes"]
                def observe(doc, current=current):
                    return current[doc]
                fold["runs"][str(j)][seed] = {name: decide_target(model, documents, observe)
                                              for name, model in (models | {"known_lambda": known}).items()}
        output["folds"][target_name] = fold
    return output


def paired_macro_interval(differences, draws=2000, seed=20260910):
    """Each array is[lambda,panel,query]; query resampling preserves other axes."""
    rng = np.random.default_rng(seed)
    means = np.zeros(draws)
    for values in differences.values():
        values = np.asarray(values, dtype=float)
        q = values.mean(axis=(0, 1))
        means += np.mean(q[rng.integers(0, len(q), size=(draws, len(q)))], axis=1) / len(differences)
    point = float(np.mean([np.mean(values) for values in differences.values()]))
    return {"mean": point, "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975))}


def validate_decision_lock(decisions, inputs):
    if decisions.get("stage") != "locked_predictions_before_test_utilities" or set(decisions.get("folds", {})) != set(inputs["corpora"]):
        raise ValueError("Missing complete target decision lock")
    for family, target in inputs["corpora"].items():
        fold = decisions["folds"][family]
        for j in range(7):
            for seed in target["panels"]["target"]:
                runs = fold["runs"][str(j)][seed]
                if set(runs) != set(fold["models"]) | {"known_lambda"}:
                    raise ValueError("Missing registered method in decision lock")
                for run in runs.values():
                    if choose(run["prediction"]) != run["policy"]:
                        raise ValueError("Locked policy differs from its predictions")


def alias_audit(fold, seeds, true_utility):
    """Evaluator-only: compare compressed profile aliases after predictions lock."""
    result = {}
    for seed in seeds:
        groups = {}
        for j in range(7):
            run = fold["runs"][str(j)][seed]["within_unshrunk"]
            key = tuple(np.round(run["profile"], 12))
            groups.setdefault(key, []).append(j)
        rows = []
        for group in groups.values():
            optimal = [set(np.flatnonzero(true_utility[j] >= true_utility[j].max() - 1e-12)) for j in group]
            responses = [fold["runs"][str(j)][seed]["within_unshrunk"]["observations"] for j in group]
            rr_states = {json.dumps([row["rr"] for row in observation]) for observation in responses}
            full_states = {json.dumps([row["action_rankings"] for row in observation]) for observation in responses}
            rows.append({"lambda_indices": group, "optimal_policy_sets": [sorted(int(i) for i in values) for values in optimal],
                "no_common_optimal_policy": not bool(set.intersection(*optimal)),
                "distinct_document_rr_states": len(rr_states), "distinct_action_ranklist_states": len(full_states),
                "adaptation_headroom": float(true_utility[group].max(axis=1).mean() - true_utility[group].mean(axis=0).max())})
        result[seed] = rows
    return {"grouping": "Round each mean-profile coordinate to12decimal places; original coordinate is zero.", "panels": result}


def evaluate(decisions, labels, inputs):
    if decisions.get("stage") != "locked_predictions_before_test_utilities" or labels.get("stage") != "test_utilities_after_decision_lock":
        raise ValueError("Evaluation requires locked decisions and separately exported test utilities")
    validate_decision_lock(decisions, inputs)
    summary = {"families": {}, "macro": {}, "intervals": {}, "response_transport": {}, "costs": {}, "profile_aliases": {}, "fixed_policy_frontier": {}}
    differences = {}; combined_differences = {}; macro = {}; response_totals = {}
    for family, fold in decisions["folds"].items():
        target = labels["corpora"][family]
        for world in target["worlds"].values():
            if world["ids"] != inputs["corpora"][family]["test_meta"]["ids"]:
                raise ValueError("Test utility query IDs do not match locked metadata")
        utility = np.asarray([target["worlds"][str(j)]["scores"] for j in range(7)])
        recall = np.asarray([target["worlds"][str(j)]["recalls"] for j in range(7)])
        seeds = sorted(inputs["corpora"][family]["panels"]["target"])
        methods = list(fold["runs"]["0"][seeds[0]])
        values = {}; family_rows = {}; family_costs = {}; response = {}
        relative = utility[..., :P].mean(axis=1); relative -= relative[:, :1]
        true_centered = relative - relative.mean(axis=0, keepdims=True)
        response["zero_response_mse"] = float(np.mean(true_centered[:, 1:] ** 2))
        for method in methods:
            arr = np.empty((7, len(seeds), utility.shape[1]))
            rec = np.empty_like(arr); predictions = np.empty((7, len(seeds), P))
            costs = zero_cost(); serving = zero_cost()
            changes = 0
            for j in range(7):
                for r, seed in enumerate(seeds):
                    run = fold["runs"][str(j)][seed][method]; policy = run["policy"]
                    arr[j, r] = utility[j, :, policy]; rec[j, r] = recall[j, :, policy]
                    predictions[j, r] = run["prediction"]
                    add_cost(costs, run["onboarding_cost"])
                    for row in inputs["corpora"][family]["test_meta"]["policy_costs"]:
                        add_cost(serving, row[policy])
                    changes += policy != fold["runs"][str(j)][seed]["source_fixed"]["policy"]
            values[method] = arr
            family_rows[method] = {"ndcg": float(arr.mean()), "recall": float(rec.mean()),
                "by_lambda": arr.mean(axis=(1, 2)).tolist(), "by_panel": arr.mean(axis=(0, 2)).tolist(),
                "interior_ndcg": float(arr[1:-1].mean()), "changed_fraction": changes / (7 * len(seeds))}
            family_costs[method] = {"onboarding_per_environment_panel": {k: v / (7 * len(seeds)) for k, v in costs.items()},
                "serving_per_query": {k: v / (7 * len(seeds) * utility.shape[1]) for k, v in serving.items()}}
            centered = predictions - predictions.mean(axis=0, keepdims=True)
            response[method] = {"centered_mse": float(np.mean((centered[:, :, 1:] - true_centered[:, None, 1:]) ** 2)),
                "centered_mse_by_policy": np.mean((centered - true_centered[:, None, :]) ** 2, axis=(0, 1)).tolist(),
                "mean_residual_by_policy": (predictions - relative[:, None, :]).mean(axis=(0, 1)).tolist()}
            macro.setdefault(method, []).append(family_rows[method]["ndcg"])
            response_totals.setdefault(method, []).append(response[method]["centered_mse"])
        for name, index in (("original", 0), ("hyde", 3), ("rrf_all", P)):
            family_rows[name] = {"ndcg": float(utility[:, :, index].mean()), "recall": float(recall[:, :, index].mean())}
            macro.setdefault(name, []).append(family_rows[name]["ndcg"])
            static_cost = zero_cost()
            for row in inputs["corpora"][family]["test_meta"]["policy_costs"]:
                add_cost(static_cost, row[index])
            family_costs[name] = {"onboarding_per_environment_panel": zero_cost(),
                "serving_per_query": {key: value / utility.shape[1] for key, value in static_cost.items()}}
        for method_cost in family_costs.values():
            method_cost["amortized_per_query"] = {str(workload): {key: method_cost["serving_per_query"][key] +
                method_cost["onboarding_per_environment_panel"][key] / workload for key in COST_KEYS}
                for workload in (10, 100, 1000)}
        mean_u = utility[..., :P].mean(axis=1)
        summary["fixed_policy_frontier"][family] = []
        for i, subset in enumerate(ALL_POLICIES):
            cost = zero_cost()
            for row in inputs["corpora"][family]["test_meta"]["policy_costs"]:
                add_cost(cost, row[i])
            summary["fixed_policy_frontier"][family].append({"policy_index": i,
                "actions": [ACTIONS[a] for a in subset], "ndcg": float(utility[..., i].mean()),
                "recall": float(recall[..., i].mean()), "ndcg_by_lambda": utility[..., i].mean(axis=1).tolist(),
                "serving_per_query": {key: value / utility.shape[1] for key, value in cost.items()}})
        family_rows["oracle_best_fixed"] = float(mean_u.mean(axis=0).max())
        family_rows["oracle_best_by_lambda"] = float(mean_u.max(axis=1).mean())
        family_rows["lambda_adaptation_headroom"] = family_rows["oracle_best_by_lambda"] - family_rows["oracle_best_fixed"]
        differences[family] = values["within"] - values["source_fixed"]
        combined_differences[family] = values["within"] - values["source_fixed_train_cal"]
        summary["families"][family] = family_rows
        summary["costs"][family] = family_costs
        summary["response_transport"][family] = response
        summary["profile_aliases"][family] = alias_audit(fold, seeds, mean_u)
    summary["macro"] = {name: float(np.mean(rows)) for name, rows in macro.items()}
    summary["intervals"]["within_minus_source_fixed"] = paired_macro_interval(differences)
    summary["intervals"]["within_minus_source_fixed_train_cal"] = paired_macro_interval(combined_differences)
    summary["intervals"]["by_family"] = {name: paired_macro_interval({name: values}) for name, values in differences.items()}
    summary["intervals"]["by_family_train_cal"] = {name: paired_macro_interval({name: values}) for name, values in combined_differences.items()}
    summary["within_family_deltas"] = {name: float(values.mean()) for name, values in differences.items()}
    summary["within_family_deltas_train_cal"] = {name: float(values.mean()) for name, values in combined_differences.items()}
    delta = summary["intervals"]["within_minus_source_fixed"]["mean"]
    deltas = list(summary["within_family_deltas"].values())
    second_delta = summary["intervals"]["within_minus_source_fixed_train_cal"]["mean"]
    second_deltas = list(summary["within_family_deltas_train_cal"].values())
    summary["advancement_gate"] = {"passed": bool(delta >= .005 and sum(x > 0 for x in deltas) >= 2 and min(deltas) >= -.005
        and second_delta >= .005 and sum(x > 0 for x in second_deltas) >= 2 and min(second_deltas) >= -.005),
        "required_against_both": ["source_fixed", "source_fixed_train_cal"],
        "macro_gain_at_least": .005, "positive_families_at_least": 2, "maximum_family_loss": .005}
    summary["response_macro_mse"] = {name: float(np.mean(rows)) for name, rows in response_totals.items()}
    summary["response_macro_mse"]["zero_response"] = float(np.mean([row["zero_response_mse"] for row in summary["response_transport"].values()]))
    shuffle_values = [value for name, value in summary["macro"].items() if name.startswith("shuffle_")]
    summary["attribution_diagnostics"] = {"within_minus_pooled": summary["macro"]["within"] - summary["macro"]["pooled"],
        "within_minus_best_shuffle": summary["macro"]["within"] - max(shuffle_values) if shuffle_values else None,
        "within_response_mse_reduction": summary["response_macro_mse"]["zero_response"] - summary["response_macro_mse"]["within"]}
    return summary


def create_freeze(output):
    adapter_audit = json.loads((ROOT.parent / "interventions" / "endpoint_audit.v1.json").read_text())
    files = [ROOT / name for name in ("experiment.py", "export_evidence.py", "test_experiment.py", "protocol.v1.json")]
    files += [ROOT.parent / "interventions" / name for name in ("adapter.py", "operator.v1.json", "freeze.v1.json")]
    files += [PAPER / "pilot" / "engine.py"]
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "before_interior_outcomes",
               "runtime": {"python": sys.version, "numpy": np.__version__},
               "files": {str(path.relative_to(PAPER)): digest(path) for path in files},
               "source_inputs": adapter_audit["input_files"]}
    write_json(output, payload)


def verify_freeze(path, verify_sources=False, cache_dir=None):
    frozen = json.loads(Path(path).read_text())
    for rel, expected in frozen["files"].items():
        if digest(PAPER / rel) != expected:
            raise ValueError("Frozen implementation changed: " + rel)
    if verify_sources:
        for rel, expected in frozen["source_inputs"].items():
            source_path = Path(cache_dir) / Path(rel).name if cache_dir is not None else PAPER / rel
            if digest(source_path) != expected["sha256"]:
                raise ValueError("Frozen source input changed: " + str(source_path))
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "decide", "evaluate"))
    parser.add_argument("--freeze", type=Path, default=ROOT / "freeze.v1.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs.v1.json")
    parser.add_argument("--decisions", type=Path, default=ROOT / "decisions.v1.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "test_labels.v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        create_freeze(args.output or args.freeze); return
    verify_freeze(args.freeze)
    inputs = json.loads(args.inputs.read_text())
    if inputs["freeze_sha256"] != digest(args.freeze):
        raise ValueError("Input evidence belongs to a different freeze")
    if args.phase == "decide":
        result = make_decisions(inputs)
        result.update({"inputs_sha256": digest(args.inputs), "freeze_sha256": digest(args.freeze),
                       "prediction_lock_created_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(args.output or args.decisions, result)
    else:
        decisions = json.loads(args.decisions.read_text()); labels = json.loads(args.labels.read_text())
        if labels["decisions_sha256"] != digest(args.decisions) or decisions["inputs_sha256"] != digest(args.inputs):
            raise ValueError("Decision/label/input provenance mismatch")
        if decisions["freeze_sha256"] != digest(args.freeze) or labels["freeze_sha256"] != digest(args.freeze) or labels["inputs_sha256"] != digest(args.inputs):
            raise ValueError("Decision/label freeze provenance mismatch")
        result = evaluate(decisions, labels, inputs)
        result["provenance"] = {"inputs": digest(args.inputs), "decisions": digest(args.decisions), "labels": digest(args.labels), "freeze": digest(args.freeze)}
        write_json(args.output or ROOT / "results.v1.json", result)


if __name__ == "__main__":
    main()
