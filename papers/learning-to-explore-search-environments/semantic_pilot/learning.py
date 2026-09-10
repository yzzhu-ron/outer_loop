"""Source-only routing and myopic probe-value learning for the semantic pilot.

The onboarding boundary accepts candidate metadata and an observation callback.
It never receives future task queries, task labels, or unselected observations.
Source utility labels train a supervised one-step acquisition heuristic; this
is not a claim of globally optimal or statistically calibrated exploration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ACTIONS = ("original", "keywords", "semantic", "hyde", "decomposed")
FAMILIES = ("direct_question", "indirect_question")
BUDGETS = (0, 4, 8, 16, 32)
METHODS = ("source_router", "random", "fixed", "information_gain", "learned_value")
COST_KEYS = ("search_calls", "sample_calls", "llm_calls", "input_tokens", "output_tokens")
SHRINKAGE = 4.0


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _cell(candidate):
    family = FAMILIES.index(candidate["family"])
    bucket, action = int(candidate["bucket"]), int(candidate["action"])
    if not 0 <= bucket < 4 or not 1 <= action < 5:
        raise ValueError("Candidate bucket/action is outside the frozen menu")
    return family, bucket, action - 1


def _candidate_cost(candidate):
    value = candidate.get("cost_breakdown", {}).get("search_calls", candidate["cost"])
    if float(value) != int(value) or value < 0:
        raise ValueError("Probe search cost must be a nonnegative integer")
    return int(value)


def _costs(value=None):
    value = value or {}
    result = {key: float(value.get(key, 0.0)) for key in COST_KEYS}
    if any(not np.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("Costs must be finite and nonnegative")
    return result


def _setup_generation_seconds(setup):
    """Recorded generation batch-share time, excluding serving/search latency."""
    value = float(setup.get("generation_seconds", setup.get("seconds", 0.0)))
    if not np.isfinite(value) or value < 0:
        raise ValueError("Measured setup generation time must be finite and nonnegative")
    return value


@dataclass
class Profile:
    count: np.ndarray = field(default_factory=lambda: np.zeros((2, 4, 4), dtype=int))
    total: np.ndarray = field(default_factory=lambda: np.zeros((2, 4, 4), dtype=float))
    square: np.ndarray = field(default_factory=lambda: np.zeros((2, 4, 4), dtype=float))

    def copy(self):
        return Profile(self.count.copy(), self.total.copy(), self.square.copy())

    def update(self, candidate, observation):
        delta = float(observation["delta"])
        if not np.isfinite(delta) or not -1.0000001 <= delta <= 1.0000001:
            raise ValueError("Paired reciprocal-rank contrast must be finite and in [-1,1]")
        cell = _cell(candidate)
        self.count[cell] += 1
        self.total[cell] += delta
        self.square[cell] += delta * delta

    def routing_features(self, action, buckets):
        """Four features: shrunk global/bucket means for each probe family."""
        action = int(action) - 1
        buckets = np.asarray(buckets, dtype=int)
        if np.any((buckets < 0) | (buckets > 3)):
            raise ValueError("Task bucket is outside [0,3]")
        columns = []
        for family in range(2):
            global_mean = self.total[family, :, action].sum() / (self.count[family, :, action].sum() + SHRINKAGE)
            bucket_mean = self.total[family, :, action] / (self.count[family, :, action] + SHRINKAGE)
            columns.extend([np.full(len(buckets), global_mean), bucket_mean[buckets]])
        return np.column_stack(columns)

    def candidate_features(self, candidate):
        f, b, a = _cell(candidate)
        raw = np.asarray(candidate["features"], dtype=float)
        if raw.shape != (8,) or not np.all(np.isfinite(raw)):
            raise ValueError("Candidate features must contain exactly eight finite values")
        means = self.total / (self.count + SHRINKAGE)
        variance = np.maximum(0, self.square / np.maximum(self.count, 1) - (self.total / np.maximum(self.count, 1)) ** 2)
        return np.concatenate([
            np.eye(2)[f], np.eye(4)[a], np.eye(4)[b], raw,
            np.log1p(self.count).ravel(), means.ravel(), variance.ravel(),
            [np.log1p(self.count.sum()), float(candidate["cost"])],
        ])

    def as_dict(self):
        return {"count": self.count.tolist(), "sum": self.total.tolist(), "sumsq": self.square.tolist()}


@dataclass
class Acquisition:
    prior_variance: np.ndarray = field(default_factory=lambda: np.full((2, 4, 4), 0.05))
    noise_variance: np.ndarray = field(default_factory=lambda: np.full((2, 4, 4), 0.05))
    regressor: object = None

    def score(self, candidate, profile, method):
        cost = max(_candidate_cost(candidate), 1)
        if method == "information_gain":
            cell = _cell(candidate)
            noise = self.noise_variance[cell]
            posterior = 1.0 / (1.0 / self.prior_variance[cell] + profile.count[cell] / noise)
            return float(0.5 * np.log1p(posterior / noise) / cost)
        if method == "learned_value":
            if self.regressor is None:
                return 0.0
            return float(self.regressor.predict(profile.candidate_features(candidate)[None, :])[0] / cost)
        raise ValueError(f"Unsupported model-based acquisition method: {method}")

    def score_candidates(self, candidates, profile, method):
        if method == "learned_value" and self.regressor is not None:
            features = np.stack([profile.candidate_features(c) for c in candidates])
            values = self.regressor.predict(features)
            return [float(value / max(_candidate_cost(c), 1)) for c, value in zip(candidates, values)]
        return [self.score(c, profile, method) for c in candidates]


def run_onboarding(candidates, observe, *, method, budget, acquisition=None, seed=0, max_search_calls=None):
    """Select at most `budget` paired probes; call observe(id) only when selected.

    max_search_calls optionally caps selected-probe searches, excluding pool
    construction. Positive-budget analysis adds every pool setup cost separately.
    Snapshots contain only numerical statistics. Candidate metadata may not
    contain outcome fields; real target tasks are absent from this interface.
    """
    if method not in METHODS or int(budget) != budget or budget < 0:
        raise ValueError("Unknown method or invalid pair budget")
    if max_search_calls is not None and max_search_calls < 0:
        raise ValueError("Search-call cap must be nonnegative")
    forbidden = {"delta", "rr_original", "rr_alternative", "overlap", "observation", "scores", "utility"}
    candidates = list(candidates)
    ids = [str(c["id"]) for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique within a pool")
    for candidate in candidates:
        if forbidden.intersection(candidate):
            raise ValueError("Outcome fields are forbidden in pre-observation metadata")
        _cell(candidate)
        _candidate_cost(candidate)
    acquisition = acquisition or Acquisition()
    profile = Profile()
    result = {"profile": profile, "selected_ids": [], "search_calls": 0, "costs": _costs(), "trace": [], "snapshots": {0: profile.copy()}}
    if method == "source_router":
        return result
    permutation = np.random.default_rng(seed).permutation(len(candidates))
    random_priority = {int(i): rank for rank, i in enumerate(permutation)}
    remaining = list(range(len(candidates)))
    for step in range(int(budget)):
        affordable = [i for i in remaining if max_search_calls is None or result["search_calls"] + _candidate_cost(candidates[i]) <= max_search_calls]
        if not affordable:
            break
        if method == "random":
            chosen = min(affordable, key=random_priority.get)
            score = None
        elif method == "fixed":
            chosen = min(affordable, key=lambda i: (profile.count[_cell(candidates[i])], _cell(candidates[i]), str(candidates[i]["id"])))
            score = None
        else:
            values = acquisition.score_candidates([candidates[i] for i in affordable], profile, method)
            scores = dict(zip(affordable, values))
            chosen = max(affordable, key=lambda i: (scores[i], -random_priority[i]))
            score = scores[chosen]
        candidate = candidates[chosen]
        observation = observe(candidate["id"])
        profile.update(candidate, observation)
        cost = _candidate_cost(candidate)
        breakdown = _costs(candidate.get("cost_breakdown"))
        breakdown["search_calls"] = cost
        for key in COST_KEYS:
            result["costs"][key] += breakdown[key]
        result["search_calls"] += cost
        result["selected_ids"].append(str(candidate["id"]))
        result["trace"].append({"id": str(candidate["id"]), "score_before_observation_per_search": score, "costs": breakdown})
        result["snapshots"][step + 1] = profile.copy()
        remaining.remove(chosen)
    return result


class BaseRouter:
    def __init__(self, fixed_scores, estimator=None, label="fixed"):
        self.fixed_scores = np.asarray(fixed_scores, dtype=float)
        self.estimator, self.label = estimator, label

    def predict(self, tasks):
        features = np.asarray(tasks["features"], dtype=float)
        if self.estimator is None:
            return np.tile(self.fixed_scores, (len(features), 1))
        return np.asarray(self.estimator.predict(features), dtype=float)


class Calibrator:
    def __init__(self, models=None):
        self.models = models or {}

    def predict(self, base, buckets, profile):
        output = np.asarray(base, dtype=float).copy()
        for action, estimator in self.models.items():
            output[:, action] += estimator.predict(profile.routing_features(action, buckets))
        return output


def _task_scores(tasks):
    scores = np.asarray(tasks["scores"], dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(ACTIONS) or not np.all(np.isfinite(scores)):
        raise ValueError("Task scores must be finite n-by-five arrays")
    return scores


def _task_recalls(tasks):
    recalls = np.asarray(tasks["recalls"], dtype=float)
    if recalls.shape != _task_scores(tasks).shape or not np.all(np.isfinite(recalls)) or np.any((recalls < 0) | (recalls > 1)):
        raise ValueError("Recall@10 must be a finite n-by-five array in [0,1]")
    return recalls


def _quality(scores, predicted):
    actions = np.asarray(predicted).argmax(axis=1)
    return float(np.mean(scores[np.arange(len(scores)), actions]))


def fit_base_router(source_records):
    train = [r["tasks"]["train"] for r in source_records]
    fixed_scores = np.mean([_task_scores(t).mean(axis=0) for t in train], axis=0)
    models = [BaseRouter(fixed_scores)]
    features = np.concatenate([np.asarray(t["features"], dtype=float) for t in train])
    labels = np.concatenate([_task_scores(t) for t in train])
    for alpha in (10.0, 100.0):
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        estimator.fit(features, labels)
        models.append(BaseRouter(fixed_scores, estimator, f"ridge_{int(alpha)}"))
    values = [np.mean([_quality(_task_scores(r["tasks"]["calibration"]), model.predict(r["tasks"]["calibration"])) for r in source_records]) for model in models]
    best = models[int(np.argmax(values))]
    return best, {"selected": best.label, "source_calibration_ndcg": {m.label: float(v) for m, v in zip(models, values)}, "source_fixed_action": int(np.argmax(fixed_scores)), "source_fixed_mean_scores": fixed_scores.tolist()}


def _source_histories(records):
    histories = []
    for record in records:
        for pool_seed, pool in sorted(record["pools"].items()):
            candidates = list(pool["candidates"])
            order = np.random.default_rng(_seed("source_history", record["name"], pool_seed)).permutation(len(candidates))
            for budget in BUDGETS:
                selected = set(int(i) for i in order[:budget])
                profile = Profile()
                for i in sorted(selected):
                    candidate = candidates[i]
                    profile.update(candidate, pool["observations"][candidate["id"]])
                histories.append((record, pool, budget, selected, profile))
    return histories


def fit_calibrator(source_records, base_router, histories):
    models = {}
    counts = {r["name"]: sum(h[0]["name"] == r["name"] for h in histories) for r in source_records}
    for action in range(1, len(ACTIONS)):
        xs, ys, weights = [], [], []
        for record, _, _, _, profile in histories:
            tasks = record["tasks"]["calibration"]
            base, actual = base_router.predict(tasks), _task_scores(tasks)
            xs.append(profile.routing_features(action, tasks["buckets"]))
            ys.append((actual[:, action] - actual[:, 0]) - (base[:, action] - base[:, 0]))
            # Each task keeps unit total weight across repeated histories.
            weights.extend([1.0 / counts[record["name"]]] * len(actual))
        models[action] = Ridge(alpha=10.0, fit_intercept=False).fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.asarray(weights))
    return Calibrator(models)


def fit_acquisition(source_records, base_router, calibrator, histories):
    acquisition = Acquisition()
    cells = {}
    env_means = {}
    for record in source_records:
        local = {}
        for pool in record["pools"].values():
            for candidate in pool["candidates"]:
                cell = _cell(candidate)
                delta = float(pool["observations"][candidate["id"]]["delta"])
                cells.setdefault(cell, []).append(delta)
                local.setdefault(cell, []).append(delta)
        for cell, values in local.items():
            env_means.setdefault(cell, []).append(float(np.mean(values)))
    for cell, values in cells.items():
        acquisition.noise_variance[cell] = max(float(np.var(values)), 0.01)
        acquisition.prior_variance[cell] = max(float(np.var(env_means[cell])), 0.01)
    x, y, source_names = [], [], []
    for record, pool, budget, selected, profile in histories:
        tasks = record["tasks"]["utility"]
        # Utility queries are separate from base fitting/calibration queries.
        base, truth = base_router.predict(tasks), _task_scores(tasks)
        before = _quality(truth, calibrator.predict(base, tasks["buckets"], profile))
        for index, candidate in enumerate(pool["candidates"]):
            if index in selected:
                continue
            x.append(profile.candidate_features(candidate))
            after_profile = profile.copy()
            after_profile.update(candidate, pool["observations"][candidate["id"]])
            after = _quality(truth, calibrator.predict(base, tasks["buckets"], after_profile))
            y.append(after - before)
            source_names.append(record["name"])
    if x:
        names, counts = np.unique(source_names, return_counts=True)
        weights_by_name = {name: len(x) / (len(names) * count) for name, count in zip(names, counts)}
        sample_weight = np.asarray([weights_by_name[name] for name in source_names])
        acquisition.regressor = ExtraTreesRegressor(n_estimators=64, max_depth=5, min_samples_leaf=20, max_features=1.0, random_state=1927, n_jobs=1)
        acquisition.regressor.fit(np.asarray(x), np.asarray(y), sample_weight=sample_weight)
    diagnostics = {
        "training_examples": len(y), "source_environments": sorted(set(source_names)),
        "unique_utility_queries": {r["name"]: len(r["tasks"]["utility"]["ids"]) for r in source_records},
        "gain_mean": float(np.mean(y)) if y else 0.0,
        "gain_positive_fraction": float(np.mean(np.asarray(y) > 1e-12)) if y else 0.0,
        "gain_negative_fraction": float(np.mean(np.asarray(y) < -1e-12)) if y else 0.0,
        "gain_nonzero_fraction": float(np.mean(np.abs(y) > 1e-12)) if y else 0.0,
        "gain_range": [float(min(y)), float(max(y))] if y else [0.0, 0.0],
        "calibration_histories": len(histories),
        "repeated_labels_warning": "Histories reuse source calibration and utility labels; training rows are not independent task or environment evidence.",
        "information_gain_model": "Independent Gaussian cell-mean surrogate, source marginal RR-contrast variance as noise and between-environment mean variance as prior; each floored at .01. This is a heuristic, not calibrated coverage.",
        "extra_trees": {"n_estimators": 64, "max_depth": 5, "min_samples_leaf": 20, "random_state": 1927},
        "prior_variance": acquisition.prior_variance.tolist(), "noise_variance": acquisition.noise_variance.tolist(),
    }
    return acquisition, diagnostics


def _selected_task_costs(tasks, selected):
    values = tasks.get("action_costs", {})
    # Also accept metric -> n-by-action arrays, useful for compact caches.
    if isinstance(values, dict):
        arrays = {k: np.asarray(values.get(k, np.zeros((len(selected), len(ACTIONS)))), dtype=float) for k in COST_KEYS}
        return {k: float(arrays[k][np.arange(len(selected)), selected].mean()) for k in COST_KEYS}
    return {key: float(np.mean([_costs(values[i][int(action)])[key] for i, action in enumerate(selected)])) for key in COST_KEYS}


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()).hexdigest()


def _analysis_provenance(records):
    root = Path(__file__).resolve().parent
    protocol_path = root / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "learning_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "protocol": json.loads(protocol_bytes),
        "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        "input_record_sha256": {r["name"]: _json_hash(r) for r in records},
        "retrieval_contracts": {r["name"]: r.get("contract", {}) for r in records},
        "source_fitting_seconds": {},
        "timing_scope": "measured_onboarding_generation_seconds sums the pool's recorded generation batch-wall-time shares. It excludes model load, retrieval, and serving. Task generation/search latency is unavailable; no total latency or total generation-seconds claim is made.",
        "recall_scope": "Recall@10 is secondary and never used to fit or select models. Oracle recall columns evaluate the actions selected by the nDCG oracle, not a separate recall oracle.",
    }


def _save_model_bundle(directory, fold, base_router, calibrator, acquisition, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle = {"base_router": base_router, "calibrator": calibrator, "acquisition": acquisition,
              "learning_sha256": provenance["learning_sha256"], "protocol_sha256": provenance["protocol_sha256"]}
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".joblib", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        filename = fold.replace("::", "_").replace("/", "_") + "-" + checksum[:16] + ".joblib"
        destination = directory / filename
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination.resolve()), "sha256": checksum, "format": "joblib", "contains": ["base_router", "calibrator", "acquisition"], "loading": "Local trusted artifact; import learning from this analysis revision before joblib.load. Check the recorded checksum and library versions."}


def _base_model_parameters(router):
    if router.estimator is None:
        return {"type": "fixed", "scores": router.fixed_scores.tolist()}
    scaler, ridge = router.estimator.steps[0][1], router.estimator.steps[1][1]
    return {"type": "standard_scaler_ridge", "alpha": ridge.alpha,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coefficients": ridge.coef_.tolist(), "intercept": ridge.intercept_.tolist()}


def analyze_environments(records, output_dir, *, model_cache_dir=None):
    """Cross-family evaluation; only evaluator code reads target test outcomes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    provenance = _analysis_provenance(records)
    analysis_started = time.perf_counter()
    model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else Path(__file__).resolve().parent / "cache" / "fitted_models"
    corpus_names = sorted({r["corpus_name"] for r in records})
    if len(corpus_names) < 3:
        raise ValueError("This pilot requires at least three corpus families")
    headroom, adaptation, cost_rows = [], [], []
    paired, profiles, model_metadata, selector_diagnostics = {}, {}, {}, {}
    for target_corpus in corpus_names:
        source = [r for r in records if r["corpus_name"] != target_corpus]
        target = [r for r in records if r["corpus_name"] == target_corpus]
        for backend in sorted({r["backend"] for r in target}):
            sources = [r for r in source if r["backend"] == backend]
            if len({r["corpus_name"] for r in sources}) < 2:
                raise ValueError("Each target backend requires two disjoint source families")
            fitting_started = time.perf_counter()
            base_router, metadata = fit_base_router(sources)
            histories = _source_histories(sources)
            calibrator = fit_calibrator(sources, base_router, histories)
            acquisition, diagnostic = fit_acquisition(sources, base_router, calibrator, histories)
            fold = f"{target_corpus}::{backend}"
            provenance["source_fitting_seconds"][fold] = time.perf_counter() - fitting_started
            metadata.update({"target_corpus": target_corpus, "backend": backend, "source_corpora": sorted({r["corpus_name"] for r in sources}), "calibrator_alpha": 10.0, "calibrator_intercept": False, "profile_shrinkage": SHRINKAGE, "calibrator_coefficients": {ACTIONS[a]: m.coef_.tolist() for a, m in calibrator.models.items()}})
            metadata["base_router_parameters"] = _base_model_parameters(base_router)
            metadata["serialized_model"] = _save_model_bundle(model_cache_dir, fold, base_router, calibrator, acquisition, provenance)
            metadata["analysis_learning_sha256"] = provenance["learning_sha256"]
            metadata["analysis_protocol_sha256"] = provenance["protocol_sha256"]
            if acquisition.regressor is not None:
                metadata["selector_feature_importances"] = acquisition.regressor.feature_importances_.tolist()
            model_metadata[fold], selector_diagnostics[fold] = metadata, diagnostic
            for record in (r for r in target if r["backend"] == backend):
                environment = record["name"]
                tasks = record["tasks"]["test"]
                truth, recalls, base = _task_scores(tasks), _task_recalls(tasks), base_router.predict(tasks)
                base_actions, fixed_action = base.argmax(axis=1), int(base_router.fixed_scores.argmax())
                base_quality = _quality(truth, base)
                fixed_means = truth.mean(axis=0)
                base_recall = float(recalls[np.arange(len(recalls)), base_actions].mean())
                headroom_row = {
                    "environment": environment, "corpus_name": target_corpus, "backend": backend, "n_queries": len(truth),
                    "original": float(truth[:, 0].mean()), "source_fixed": float(truth[:, fixed_action].mean()),
                    "source_router": base_quality, "target_fixed_oracle": float(fixed_means.max()),
                    "query_oracle": float(truth.max(axis=1).mean()),
                    "oracle_gap_vs_router": float(truth.max(axis=1).mean() - base_quality),
                    "oracle_gap_vs_target_fixed": float(truth.max(axis=1).mean() - fixed_means.max()),
                    "source_fixed_action": ACTIONS[fixed_action], "target_fixed_oracle_action": ACTIONS[int(fixed_means.argmax())],
                    "original_recall_at_10": float(recalls[:, 0].mean()), "source_fixed_recall_at_10": float(recalls[:, fixed_action].mean()), "source_router_recall_at_10": base_recall,
                    "ndcg_target_fixed_oracle_recall_at_10": float(recalls[:, int(fixed_means.argmax())].mean()),
                    "ndcg_query_oracle_recall_at_10": float(recalls[np.arange(len(recalls)), truth.argmax(axis=1)].mean()),
                }
                for baseline, actions in (("original", np.zeros(len(truth), dtype=int)), ("source_fixed", np.full(len(truth), fixed_action, dtype=int)), ("source_router", base_actions)):
                    headroom_row.update({f"{baseline}_mean_task_{key}": value for key, value in _selected_task_costs(tasks, actions).items()})
                headroom.append(headroom_row)
                paired[environment] = {"query_ids": list(tasks["ids"]), "action_names": ACTIONS, "action_scores": truth.tolist(), "action_recalls_at_10": recalls.tolist(), "source_router_actions": base_actions.tolist(), "runs": {}}
                for seed, pool in sorted(record["pools"].items()):
                    for method in METHODS:
                        # The observation dictionary remains behind this callback.
                        onboarding = run_onboarding(pool["candidates"], lambda candidate_id, pool=pool: pool["observations"][candidate_id], method=method, budget=max(BUDGETS), acquisition=acquisition, seed=_seed("target", seed))
                        for budget in BUDGETS:
                            count = min(budget, len(onboarding["selected_ids"]))
                            profile = onboarding["snapshots"][count]
                            predicted = calibrator.predict(base, tasks["buckets"], profile)
                            actions = predicted.argmax(axis=1)
                            quality = _quality(truth, predicted)
                            recall = float(recalls[np.arange(len(recalls)), actions].mean())
                            setup = _costs(pool.get("setup_costs")) if budget > 0 and method != "source_router" else _costs()
                            generation_seconds = _setup_generation_seconds(pool.get("setup_costs", {})) if budget > 0 and method != "source_router" else 0.0
                            onboard_cost = setup.copy()
                            for item in onboarding["trace"][:count]:
                                for key in COST_KEYS:
                                    onboard_cost[key] += item["costs"][key]
                            task_cost = _selected_task_costs(tasks, actions)
                            row = {"environment": environment, "seed": str(seed), "budget": budget, "method": method, "ndcg": quality, "delta_vs_source_router": quality - base_quality, "recall_at_10": recall, "delta_recall_at_10_vs_source_router": recall - base_recall, "changed_action_fraction": float(np.mean(actions != base_actions)), "measured_onboarding_generation_seconds": generation_seconds}
                            row.update({f"onboarding_{key}": onboard_cost[key] for key in COST_KEYS})
                            row.update({f"mean_task_{key}": task_cost[key] for key in COST_KEYS})
                            row.update({"corpus_name": target_corpus, "backend": backend, "selected_probe_pairs": count})
                            adaptation.append(row)
                            key = f"{seed}/{method}/{budget}"
                            paired[environment]["runs"][key] = {"actions": actions.tolist(), "ndcg": truth[np.arange(len(truth)), actions].tolist(), "recall_at_10": recalls[np.arange(len(recalls)), actions].tolist(), "selected_probe_ids": onboarding["selected_ids"][:count]}
                            profiles[f"{environment}/{key}"] = {**profile.as_dict(), "selected_probe_ids": onboarding["selected_ids"][:count], "selection_trace": onboarding["trace"][:count]}
                            for workload in (1, 10, 50, 200):
                                cost_row = {"environment": environment, "seed": str(seed), "budget": budget, "method": method, "future_tasks": workload, "ndcg": quality, "recall_at_10": recall,
                                            "measured_onboarding_generation_seconds": generation_seconds,
                                            "amortized_setup_generation_seconds_per_future_task": generation_seconds / workload}
                                cost_row.update({f"total_{key}": onboard_cost[key] + workload * task_cost[key] for key in COST_KEYS})
                                cost_rows.append(cost_row)
    macro = []
    for method in METHODS:
        for budget in BUDGETS:
            family_values, family_recalls = [], []
            for corpus in corpus_names:
                family_rows = [r for r in adaptation if r["method"] == method and r["budget"] == budget and r["corpus_name"] == corpus]
                family_values.append(float(np.mean([r["ndcg"] for r in family_rows])))
                family_recalls.append(float(np.mean([r["recall_at_10"] for r in family_rows])))
            macro.append({"method": method, "budget": budget, "equal_family_ndcg": float(np.mean(family_values)), "equal_family_recall_at_10": float(np.mean(family_recalls)), "n_families": len(corpus_names), "family_ndcg": json.dumps(dict(zip(corpus_names, family_values)), sort_keys=True), "family_recall_at_10": json.dumps(dict(zip(corpus_names, family_recalls)), sort_keys=True)})
    selector_diagnostics["contract"] = {
        "folding": "Hold out one entire corpus family and every backend configuration; fit using the other families with the same documented backend.",
        "budgets": list(BUDGETS), "budget_unit": "paired probes, not tool calls",
        "zero_budget": "No pool generation or selected probes charged; all methods reduce exactly to source router.",
        "positive_budget": "Charge the complete candidate pool setup, including unselected generations, plus selected-probe operations.",
        "baseline_source_router": "Zero onboarding cost at every displayed budget; budget is a comparison axis, not spending.",
        "inference": "Queries share profiles and corpus configurations; these are not independent adaptation replicates. Three corpus families are pilot evidence only.",
        "myopic_warning": "Learned value approximates one-step source utility; no global optimality or complementarity guarantee.",
        "analysis_learning_sha256": provenance["learning_sha256"], "analysis_protocol_sha256": provenance["protocol_sha256"],
    }
    if provenance["learning_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or provenance["protocol_sha256"] != hashlib.sha256((Path(__file__).resolve().parent / "protocol.json").read_bytes()).hexdigest():
        raise RuntimeError("Analysis implementation or frozen protocol changed during analysis")
    provenance["analysis_wall_seconds"] = time.perf_counter() - analysis_started
    _write_csv(output_dir / "headroom.csv", headroom)
    _write_csv(output_dir / "adaptation.csv", adaptation)
    _write_csv(output_dir / "cost_curves.csv", cost_rows)
    _write_csv(output_dir / "family_macro.csv", macro)
    for filename, value in (("paired_outcomes.json", paired), ("profiles.json", profiles), ("models.json", model_metadata), ("selector_diagnostics.json", selector_diagnostics), ("analysis_metadata.json", provenance)):
        (output_dir / filename).write_text(json.dumps(value, indent=2) + "\n")
    return {"headroom": headroom, "adaptation": adaptation, "family_macro": macro, "selector_diagnostics": selector_diagnostics}
