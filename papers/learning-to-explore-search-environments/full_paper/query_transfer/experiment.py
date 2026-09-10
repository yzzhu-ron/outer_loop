"""Source-calibrated query-only routing on previously inspected development data.

No target score enters model fitting, calibration, or selection. This diagnostic
has no onboarding, synthetic probes, learned selector, or new retrieval calls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
LEXICAL_DIMENSIONS = 8
FEATURE_DIMENSIONS = 392


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def candidates():
    yield {"name": "global_mean", "kind": "fixed"}
    for alpha in (10, 100):
        yield {"name": f"ridge_{alpha}", "kind": "ridge", "alpha": alpha}
    for k in (16, 64, 128):
        for shrinkage in (0.0, 0.5):
            for balance in (True, False):
                yield {"name": f"knn_k{k}_shrink{shrinkage:g}_{'balanced' if balance else 'pooled'}",
                       "kind": "knn", "k": k, "shrinkage": shrinkage, "balanced": balance}


def normalized_embeddings(features):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSIONS or not np.isfinite(features).all():
        raise ValueError("Expected finite n-by-392 query features")
    embeddings = features[:, LEXICAL_DIMENSIONS:]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero query embeddings are invalid cosine features")
    return embeddings / norms


class QueryModel:
    def __init__(self, specification):
        self.specification = specification

    def fit(self, training):
        """Only source-training arrays enter fitting; no calibration or targets."""
        self.training = training
        self.prior = np.mean([part["scores"].mean(axis=0) for part in training], axis=0)
        x = np.concatenate([part["features"] for part in training])
        y = np.concatenate([part["scores"] for part in training])
        kind = self.specification["kind"]
        if kind == "ridge":
            self.estimator = make_pipeline(StandardScaler(), Ridge(alpha=self.specification["alpha"]))
            self.estimator.fit(x, y)
        if kind == "knn":
            self.embedding_groups = [normalized_embeddings(part["features"]) for part in training]
            self.pooled_embeddings = np.concatenate(self.embedding_groups)
            self.pooled_scores = y
        return self

    def predict(self, features):
        kind = self.specification["kind"]
        if kind == "fixed":
            return np.tile(self.prior, (len(features), 1))
        if kind == "ridge":
            return self.estimator.predict(features)
        query = normalized_embeddings(features)
        k = self.specification["k"]
        if self.specification["balanced"]:
            group_predictions = []
            for embeddings, part in zip(self.embedding_groups, self.training):
                count = min(max(1, k // len(self.training)), len(embeddings))
                neighbors = np.argsort(-(query @ embeddings.T), axis=1, kind="stable")[:, :count]
                group_predictions.append(part["scores"][neighbors].mean(axis=1))
            local = np.mean(group_predictions, axis=0)
        else:
            neighbors = np.argsort(-(query @ self.pooled_embeddings.T), axis=1, kind="stable")[:, :min(k, len(self.pooled_embeddings))]
            local = self.pooled_scores[neighbors].mean(axis=1)
        shrinkage = self.specification["shrinkage"]
        return (1 - shrinkage) * local + shrinkage * self.prior

    @property
    def needs_embedding(self):
        return self.specification["kind"] != "fixed"


def selected_scores(scores, choices):
    return np.asarray(scores)[np.arange(len(choices)), choices]


def quality(scores, predictions):
    return float(selected_scores(scores, predictions.argmax(axis=1)).mean())


def fit_and_select(training, calibration):
    """Equal-family held-out-query calibration; not held-out-family validation."""
    models, rows = {}, []
    for specification in candidates():
        model = QueryModel(specification).fit(training)
        values = [quality(part["scores"], model.predict(part["features"])) for part in calibration]
        models[specification["name"]] = model
        rows.append({**specification, "source_family_calibration_ndcg": values,
                     "source_calibration_macro_ndcg": float(np.mean(values))})
    def winner(allowed):
        eligible = [row for row in rows if row["kind"] in allowed]
        return max(eligible, key=lambda row: row["source_calibration_macro_ndcg"])["name"]
    return models, rows, {"source_selected_query": winner({"fixed", "ridge", "knn"}),
                          "source_selected_knn": winner({"knn"}),
                          "reconstructed_existing_router": winner({"fixed", "ridge"})}


def load_partition(evidence, environment, split):
    item = evidence["environments"][environment]
    query = evidence["families"][item["family"]]["partitions"][split]
    labels = item["partitions"][split]
    return {"ids": list(query["ids"]), "features": np.asarray(query["features"], dtype=float),
            "scores": np.asarray(labels["scores"], dtype=float), "recalls": np.asarray(labels["recalls"], dtype=float)}


def validate_evidence(evidence):
    if evidence["actions"] != ["original", "keywords", "semantic", "hyde", "decomposed"]:
        raise ValueError("Action order differs from the registered cache")
    for family, record in evidence["families"].items():
        seen = set()
        for split in ("train", "calibration", "test"):
            part = record["partitions"][split]
            expected_count = 128 if split == "train" else 32 if split == "calibration" else 65 if family == "nfcorpus" else 150
            if len(part["ids"]) != expected_count:
                raise ValueError(f"Unexpected {split} count for {family}")
            if len(part["ids"]) != len(set(part["ids"])) or seen & set(part["ids"]):
                raise ValueError(f"Query IDs overlap partitions for {family}")
            seen.update(part["ids"])
            embeddings = normalized_embeddings(part["features"])
            if len(embeddings) != len(part["ids"]):
                raise ValueError("Query features are misaligned")
    for name, environment in evidence["environments"].items():
        for split in ("train", "calibration"):
            part = load_partition(evidence, name, split)
            if part["scores"].shape != (len(part["ids"]), 5) or not np.isfinite(part["scores"]).all() or np.any((part["scores"] < 0) | (part["scores"] > 1)):
                raise ValueError("Malformed action scores")
            if part["recalls"].shape != part["scores"].shape or not np.isfinite(part["recalls"]).all() or np.any((part["recalls"] < 0) | (part["recalls"] > 1)):
                raise ValueError("Malformed action recall array")
        test_count = len(evidence["families"][environment["family"]]["partitions"]["test"]["ids"])
        archived = np.asarray(environment["archived_source_router_actions"])
        rrf = np.asarray(environment["rrf_all_actions_ndcg"], dtype=float)
        if archived.shape != (test_count,) or not np.issubdtype(archived.dtype, np.integer) or np.any((archived < 0) | (archived >= 5)):
            raise ValueError("Malformed archived choices")
        if rrf.shape != (test_count,) or not np.isfinite(rrf).all() or np.any((rrf < 0) | (rrf > 1)):
            raise ValueError("Malformed imported fusion vector")
        costs = environment["partitions"]["test"]["action_costs"]
        if len(costs) != test_count or any(len(row) != 5 for row in costs):
            raise ValueError("Malformed action cost rows")
        if any(not np.isfinite(value) or value < 0 for row in costs for action in row for value in action.values()):
            raise ValueError("Invalid action resource count")


def resource_means(costs, choices=None):
    keys = ("search_calls", "llm_calls", "input_tokens", "output_tokens")
    if choices is None:
        # The original rewrite generator returns semantic, HyDE and decomposed
        # text together. Its same call/token charges appear in three action
        # entries. Fusion executes all searches but generates that bundle once.
        for row in costs:
            for key in keys[1:]:
                if len({row[action].get(key, 0) for action in (2, 3, 4)}) != 1:
                    raise ValueError("RRF generation cost entries no longer describe one shared rewrite bundle")
        return {key: float(np.mean([sum(action.get(key, 0) for action in row) if key == "search_calls" else max(action.get(key, 0) for action in row) for row in costs])) for key in keys}
    return {key: float(np.mean([row[int(choice)].get(key, 0) for row, choice in zip(costs, choices)])) for key in keys}


def freeze_identity():
    return {name: sha256(ROOT / name) for name in ("protocol.v1.json", "experiment.py", "export_evidence.py", "test_experiment.py", "evidence.json")}


def freeze():
    identity = freeze_identity()
    path = ROOT / "freeze.v1.json"
    if path.exists():
        if json.loads(path.read_text())["sha256"] != identity:
            raise RuntimeError("Frozen files changed; create a versioned follow-up")
        return
    write_json(path, {"version": "1.0.0", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                      "stage": "Exploratory query-only transfer on previously inspected targets", "sha256": identity})


def run():
    frozen = json.loads((ROOT / "freeze.v1.json").read_text())
    if frozen["sha256"] != freeze_identity():
        raise RuntimeError("Frozen code, protocol, or evidence hash mismatch")
    evidence = json.loads((ROOT / "evidence.json").read_text())
    validate_evidence(evidence)
    started = time.perf_counter()
    results, grids, selections, paired, reconstruction = [], [], {}, {}, []
    output = ROOT / "results.v1"
    for target, item in sorted(evidence["environments"].items()):
        sources = [name for name, candidate in sorted(evidence["environments"].items())
                   if candidate["family"] != item["family"] and candidate["backend"] == item["backend"]]
        if len(sources) != 2:
            raise ValueError("Every fold requires exactly two source families with matching backend")
        training = [load_partition(evidence, source, "train") for source in sources]
        calibration = [load_partition(evidence, source, "calibration") for source in sources]
        fit_start = time.perf_counter()
        models, calibration_grid, winners = fit_and_select(training, calibration)
        selections[target] = {"source_environments": sources, "selected_models": winners,
                              "fitting_and_selection_seconds": time.perf_counter() - fit_start}
        grids.extend({"target_environment": target, **row} for row in calibration_grid)
        # Persist the source-only choices before loading or scoring this target.
        write_json(output / "selections.json", selections)
        write_json(output / "source_calibration_grid.json", grids)
        test = load_partition(evidence, target, "test")
        scores, recalls = test["scores"], test["recalls"]
        n = len(scores)
        if scores.shape != (len(test["ids"]), 5) or recalls.shape != scores.shape or not np.isfinite(scores).all() or not np.isfinite(recalls).all() or np.any((scores < 0) | (scores > 1)) or np.any((recalls < 0) | (recalls > 1)):
            raise ValueError("Malformed target evaluation arrays")
        query_info = evidence["families"][item["family"]]
        encoding_average = query_info["original_batched_query_encoding_seconds"] / query_info["original_encoded_query_count"]
        predictions, inference_seconds = {}, {}
        for name, model in models.items():
            inference_start = time.perf_counter()
            predictions[name] = model.predict(test["features"])
            inference_seconds[name] = (time.perf_counter() - inference_start) / n
        choices = {name: value.argmax(axis=1) for name, value in predictions.items()}
        named = {"original": np.zeros(n, dtype=int), "source_fixed": choices["global_mean"],
                 "hyde": np.full(n, 3, dtype=int),
                 "archived_existing_router": np.asarray(item["archived_source_router_actions"], dtype=int)}
        for method, model_name in winners.items():
            named[method] = choices[model_name]
        # The complete frozen grid is reported, but target scores select no model.
        named.update({"candidate/" + name: action for name, action in choices.items()})
        prior_actions = choices["global_mean"]
        panel = {"query_ids": test["ids"], "action_names": evidence["actions"], "methods": {}}
        for method, action in named.items():
            values, recall = selected_scores(scores, action), selected_scores(recalls, action)
            model_name = winners.get(method, method.removeprefix("candidate/") if method.startswith("candidate/") else None)
            if method == "archived_existing_router":
                model_name = "global_mean" if item["archived_source_router_model"] == "fixed" else item["archived_source_router_model"]
            model = models.get(model_name)
            needs_embedding = bool(model and model.needs_embedding)
            if method == "archived_existing_router":
                needs_embedding = item["archived_source_router_model"] != "fixed"
            resources = resource_means(item["partitions"]["test"]["action_costs"], action)
            row = {"environment": target, "family": item["family"], "backend": item["backend"],
                   "method": method, "selected_model": model_name, "n_queries": n,
                   "ndcg": float(values.mean()), "recall_at_10": float(recall.mean()),
                   "oracle_regret": float((scores.max(axis=1) - values).mean()),
                   "changed_from_source_fixed": float(np.mean(action != prior_actions)),
                   "action_counts": dict(zip(evidence["actions"], np.bincount(action, minlength=5).tolist())),
                   "onboarding_operations": 0, "query_encoder_calls_per_task": int(needs_embedding),
                   "amortized_original_batch_encoding_seconds": encoding_average if needs_embedding else 0.0,
                   "router_prediction_seconds_per_query": inference_seconds.get(model_name, 0.0),
                   "prediction_timing_basis": "reconstructed matching specification" if method == "archived_existing_router" else "current fitted candidate" if model else "constant action", **resources}
            results.append(row)
            panel["methods"][method] = {"actions": action.tolist(), "ndcg": values.tolist(), "recall": recall.tolist()}
        rrf_values = np.asarray(item["rrf_all_actions_ndcg"], dtype=float)
        results.append({"environment": target, "family": item["family"], "backend": item["backend"],
                        "method": "rrf_all_actions", "selected_model": None, "n_queries": n,
                        "ndcg": float(rrf_values.mean()), "onboarding_operations": 0,
                        "query_encoder_calls_per_task": 0, **resource_means(item["partitions"]["test"]["action_costs"])})
        panel["methods"]["rrf_all_actions"] = {"ndcg": rrf_values.tolist()}
        for name, values in (("target_best_fixed_oracle", scores[:, int(scores.mean(axis=0).argmax())]),
                             ("per_query_action_oracle", scores.max(axis=1))):
            results.append({"environment": target, "family": item["family"], "backend": item["backend"],
                            "method": name, "n_queries": n, "ndcg": float(values.mean()), "privileged_target_labels": True})
            panel["methods"][name] = {"ndcg": values.tolist()}
        reconstruction.append({"environment": target, "model": winners["reconstructed_existing_router"],
                               "action_match_fraction": float(np.mean(named["reconstructed_existing_router"] == named["archived_existing_router"])),
                               "ndcg_difference": float(np.mean(selected_scores(scores, named["reconstructed_existing_router"]) - selected_scores(scores, named["archived_existing_router"])))})
        paired[target] = panel
        print(json.dumps({"target": target, "selected": winners}), flush=True)
    # Conditional paired-query intervals describe these six reused environments.
    # They are not new-family confidence intervals or a model-selection guarantee.
    rng = np.random.default_rng(20260910)
    intervals = []
    for environment, panel in paired.items():
        candidate = np.asarray(panel["methods"]["source_selected_query"]["ndcg"])
        draws = rng.integers(0, len(candidate), size=(2000, len(candidate)))
        for comparator in ("source_fixed", "archived_existing_router", "original", "hyde", "rrf_all_actions"):
            delta = candidate - np.asarray(panel["methods"][comparator]["ndcg"])
            bootstrap = delta[draws].mean(axis=1)
            intervals.append({"environment": environment, "method": "source_selected_query", "comparator": comparator,
                              "mean_delta": float(delta.mean()), "conditional_95_low": float(np.quantile(bootstrap, .025)),
                              "conditional_95_high": float(np.quantile(bootstrap, .975))})
    primary_methods = ("source_selected_query", "source_selected_knn", "reconstructed_existing_router", "archived_existing_router", "original", "source_fixed", "hyde", "rrf_all_actions", "target_best_fixed_oracle", "per_query_action_oracle")
    primary = [row for row in results if row["method"] in primary_methods]
    macro = [{"method": method, "equal_family_backend_macro_ndcg": float(np.mean([row["ndcg"] for row in primary if row["method"] == method]))} for method in primary_methods]
    write_json(output / "all_results.json", results)
    write_json(output / "primary_results.json", primary)
    write_json(output / "paired.json", paired)
    write_json(output / "conditional_intervals.json", intervals)
    write_json(output / "reconstruction_audit.json", reconstruction)
    write_json(output / "summary.json", {"stage": "Exploratory; target families previously inspected", "freeze": frozen,
        "seconds": time.perf_counter() - started, "candidates_per_fold": len(list(candidates())),
        "selection": "Source held-out-query calibration in known source corpora, not held-out-family validation",
        "family_macro": macro, "all_existing_router_actions_reproduced": all(row["action_match_fraction"] == 1 for row in reconstruction),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        freeze() if arguments.freeze else run()
