"""Exact, constructed acquisition comparisons; these are not retrieval results.

Run tests, then ``python experiment.py --freeze`` and ``python experiment.py --run``.
The frozen protocol and executable hashes must match before any registered grid
is evaluated. Probabilities, utilities and evaluation are rational; entropy uses
floating-point log2 of the exact probabilities. No Monte Carlo is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import cache
import hashlib
from itertools import combinations, product
import json
from math import log2
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
METHODS = ("no_probe", "source_fixed", "random", "environment_ig",
           "decision_region_entropy", "decision_myopic", "decision_myopic_forced",
           "decision_lookahead_2", "decision_full_horizon", "world_oracle")
WEIGHTS = (F(0), F(1, 4), F(1, 2), F(3, 4), F(1))
BUDGETS = (0, 1, 2, 3, 4)
SHUFFLE_SEEDS = (101, 211, 307, 401, 503)


def entropy(probabilities):
    return -sum(float(p) * log2(float(p)) for p in probabilities if p)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def measure(value):
    return {"exact": str(value), "float": float(value)}


class FiniteModel:
    """A source-supplied finite observation/utility model with an explicit mix.

    World bits have independent uniform priors. Each candidate may be selected
    once. Conditional observation noise is independent across probes and output
    bits; the proxy's 1/4 base error compounds with ``extra_noise``.
    """
    def __init__(self, scenario, weight=F(3, 4), extra_noise=F(0), shuffle_seed=None, probe_priority="canonical"):
        if scenario not in ("xor", "gated"):
            raise ValueError("scenario must be xor or gated")
        self.scenario, self.weight = scenario, F(weight)
        self.extra_noise = F(extra_noise)
        if not 0 <= self.weight <= 1 or not 0 <= self.extra_noise <= F(1, 2):
            raise ValueError("Invalid task weight or noise")
        self.bit_names = ("d0", "d1", "r", "s") + (("gate",) if scenario == "gated" else ()) + ("n0", "n1")
        self.worlds = tuple(product((0, 1), repeat=len(self.bit_names)))
        self.probes = ("d0", "d1", "r", "s") + (("gate",) if scenario == "gated" else ()) + ("nuisance", "proxy")
        indices = tuple(range(len(self.probes)))
        self.probe_priority = {"canonical": indices, "reverse": indices[::-1], "rotate3": indices[3:] + indices[:3]}[probe_priority]
        self.weights = ((1 - self.weight) / 2, (1 - self.weight) / 2, self.weight)
        truth = []
        signatures = []
        for bits in self.worlds:
            world = dict(zip(self.bit_names, bits))
            target = world["r"] ^ world["s"] if scenario == "xor" else world["s"] if world["gate"] else world["r"]
            truth.append((world["d0"], world["d1"], target))
            signatures.append(tuple((world["n0"], world["n1"]) if q == "nuisance" else (target,) if q == "proxy" else (world[q],) for q in self.probes))
        self.true_labels = tuple(truth)
        labels = list(truth)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(labels)
        self.labels = tuple(labels)
        self.shuffle_seed = shuffle_seed
        self.label_sha256 = hashlib.sha256(json.dumps(self.labels).encode()).hexdigest()
        self.alphabets = tuple(tuple(product((0, 1), repeat=2 if q == "nuisance" else 1)) for q in self.probes)
        self.likelihoods = []
        for q, name in enumerate(self.probes):
            error = F(1, 4) + self.extra_noise / 2 if name == "proxy" else self.extra_noise
            rows = []
            for signature in signatures:
                rows.append(tuple(self._likelihood(signature[q], z, error) for z in self.alphabets[q]))
            self.likelihoods.append(tuple(rows))
        self.likelihoods = tuple(self.likelihoods)
        self.prior = (F(1, len(self.worlds)),) * len(self.worlds)

    @staticmethod
    def _likelihood(expected, observed, error):
        probability = F(1)
        for a, b in zip(expected, observed):
            probability *= 1 - error if a == b else error
        return probability

    @staticmethod
    def extend(history, probe, outcome):
        if any(q == probe for q, _ in history):
            raise ValueError("Probes cannot be repeated")
        return tuple(sorted((*history, (probe, outcome))))

    @cache
    def posterior(self, history=()):
        if not history:
            return self.prior
        probe, outcome = history[-1]
        prior = self.posterior(history[:-1])
        joint = tuple(p * row[outcome] for p, row in zip(prior, self.likelihoods[probe]))
        mass = sum(joint)
        if not mass:
            raise ValueError("Observation history has zero probability under the supplied model")
        return tuple(p / mass for p in joint)

    def available(self, history):
        used = {q for q, _ in history}
        return tuple(q for q in self.probe_priority if q not in used)

    @cache
    def branches(self, history, probe):
        prior = self.posterior(history)
        branches = []
        for z in range(len(self.alphabets[probe])):
            mass = sum(p * row[z] for p, row in zip(prior, self.likelihoods[probe]))
            if mass:
                branches.append((z, mass, self.extend(history, probe, z)))
        return tuple(branches)

    @cache
    def marginals(self, history=()):
        posterior = self.posterior(history)
        return tuple(sum(p for p, labels in zip(posterior, self.labels) if labels[context]) for context in range(3))

    @cache
    def action(self, history=()):
        return sum((1 << context) for context, probability in enumerate(self.marginals(history))
                   if self.weights[context] and probability > F(1, 2))

    @cache
    def value(self, history=()):
        return sum(weight * max(p, 1 - p) for weight, p in zip(self.weights, self.marginals(history)))

    def utility_of(self, history, action):
        return sum(weight * (p if action & (1 << context) else 1 - p)
                   for context, (weight, p) in enumerate(zip(self.weights, self.marginals(history))))

    @cache
    def world_entropy(self, history=()):
        return entropy(self.posterior(history))

    @cache
    def region_entropy(self, history=()):
        """Entropy of optimal router classes, ignoring zero-weight contexts.

        This is a stronger relevance-aware information baseline than full-world
        IG, but weights within the positive-weight classes are deliberately not
        converted to utility. It is not EC2 and has no greedy guarantee here.
        """
        masses = defaultdict(F)
        for p, labels in zip(self.posterior(history), self.labels):
            code = tuple(label for label, weight in zip(labels, self.weights) if weight)
            masses[code] += p
        return entropy(masses.values())

    @cache
    def one_step_value(self, history, probe):
        return sum(mass * self.value(child) for _, mass, child in self.branches(history, probe)) - self.value(history)

    @cache
    def information(self, history, probe, region=False):
        objective = self.region_entropy if region else self.world_entropy
        return objective(history) - sum(float(mass) * objective(child) for _, mass, child in self.branches(history, probe))

    @cache
    def dynamic_program(self, history, budget):
        """Exact model-based terminal-utility DP, with STOP and cost tie-breaking."""
        best = (self.value(history), F(0), None)
        if budget <= 0:
            return best
        for q in self.available(history):
            value, cost = F(0), F(1)
            for _, mass, child in self.branches(history, q):
                child_value, child_cost, _ = self.dynamic_program(child, budget - 1)
                value += mass * child_value
                cost += mass * child_cost
            if value > best[0] or value == best[0] and cost < best[1]:
                best = value, cost, q
        return best

    @cache
    def subset_value(self, history, subset):
        if not subset:
            return self.value(history)
        return sum(mass * self.subset_value(child, subset[1:]) for _, mass, child in self.branches(history, subset[0]))

    @cache
    def fixed_subset(self, budget):
        """Strong source-optimal nonadaptive subset; no observed outcome selects it."""
        best_value, best_subset = self.value(), ()
        for size in range(1, min(budget, len(self.probes)) + 1):
            for subset in combinations(range(len(self.probes)), size):
                value = self.subset_value((), subset)
                if value > best_value:
                    best_value, best_subset = value, subset
        return best_subset

    def choices(self, method, history, remaining, initial_budget):
        """Return a distribution over next probes; empty is STOP."""
        available = self.available(history)
        if remaining <= 0 or not available or method in ("no_probe", "world_oracle"):
            return ()
        if method == "random":
            return tuple((q, F(1, len(available))) for q in available)
        if method == "source_fixed":
            todo = [q for q in self.fixed_subset(initial_budget) if q in available]
            return ((todo[0], F(1)),) if todo else ()
        if method in ("decision_full_horizon", "decision_lookahead_2"):
            depth = min(remaining, 2) if method == "decision_lookahead_2" else remaining
            q = self.dynamic_program(history, depth)[2]
            return ((q, F(1)),) if q is not None else ()
        if method in ("decision_myopic", "decision_myopic_forced"):
            values = [self.one_step_value(history, q) for q in available]
            maximum = max(values)
            if maximum == 0 and method == "decision_myopic":
                return ()
            return ((available[values.index(maximum)], F(1)),)
        if method in ("environment_ig", "decision_region_entropy"):
            values = [self.information(history, q, region=method == "decision_region_entropy") for q in available]
            maximum = max(values)
            if maximum <= 1e-12:
                return ()
            # The same fixed candidate order resolves analytical/numeric ties.
            q = next(q for q, value in zip(available, values) if maximum - value <= 1e-12)
            return ((q, F(1)),)
        raise ValueError(f"Unknown method {method}")

    @staticmethod
    def clear_caches():
        # functools caches otherwise retain each finished condition/model.
        for name in ("posterior", "branches", "marginals", "action", "value", "world_entropy", "region_entropy", "one_step_value", "information", "dynamic_program", "subset_value", "fixed_subset"):
            getattr(FiniteModel, name).cache_clear()


def evaluate(source, truth, method, budget):
    """Integrate the deployed source-model policy under the true observation law.

    Acquisition never receives true labels, true task weights or true channel
    errors. Only the evaluator calls ``truth``. All baseline routers otherwise
    share the same source posterior and action rule.
    """
    @cache
    def visit(history, remaining):
        choices = source.choices(method, history, remaining, budget)
        if not choices:
            if method == "world_oracle":
                return F(1), F(0), 0.0
            return truth.utility_of(history, source.action(history)), F(0), truth.world_entropy(history)
        utility, cost, residual_entropy = F(0), F(1), 0.0
        for q, selection_probability in choices:
            for _, mass, child in truth.branches(history, q):
                child_utility, child_cost, child_entropy = visit(child, remaining - 1)
                probability = selection_probability * mass
                utility += probability * child_utility
                cost += probability * child_cost
                residual_entropy += float(probability) * child_entropy
        return utility, cost, residual_entropy

    started = time.perf_counter()
    utility, cost, residual_entropy = visit((), budget)
    first = {source.probes[q]: str(p) for q, p in source.choices(method, (), budget, budget)}
    return {
        "method": method, "budget": budget,
        "expected_utility_exact": str(utility), "expected_utility": float(utility),
        "expected_regret_exact": str(1 - utility), "expected_regret": float(1 - utility),
        "expected_operations_exact": str(cost), "expected_operations": float(cost),
        "expected_residual_environment_entropy_bits": residual_entropy,
        "first_probe_distribution": first,
        "evaluation_states": visit.cache_info().currsize,
        "elapsed_seconds": time.perf_counter() - started,
    }


def incomparability_witness(model, q, other):
    """A row-equality witness disproves an observation-kernel garbling.

    If q has equal rows at worlds i,j but other differs there, other cannot be
    generated by postprocessing q. Finding a witness both ways proves these two
    finite experiments are Blackwell incomparable without a numerical LP.
    """
    groups = {}
    for index, row in enumerate(model.likelihoods[q]):
        if row in groups:
            previous = groups[row]
            if model.likelihoods[other][previous] != model.likelihoods[other][index]:
                return [previous, index]
        else:
            groups[row] = index
    return None


def diagnostics(scenario):
    model = FiniteModel(scenario, F(3, 4))
    pairs = []
    for q, other in combinations(range(len(model.probes)), 2):
        pairs.append({"first": model.probes[q], "second": model.probes[other],
                      "second_not_garbling_of_first": incomparability_witness(model, q, other),
                      "first_not_garbling_of_second": incomparability_witness(model, other, q)})
    candidates = [{"probe": name, "one_step_value": measure(model.one_step_value((), q)),
                   "environment_information_bits": model.information((), q),
                   "decision_region_information_bits": model.information((), q, True)}
                  for q, name in enumerate(model.probes)]
    result = {"scenario": scenario, "world_count": len(model.worlds), "bit_order": model.bit_names,
              "probe_order": model.probes, "prior_quality": measure(model.value()),
              "candidate_diagnostics_at_weight_3_4": candidates, "blackwell_pairs": pairs,
              "all_probe_pairs_incomparable": all(p["second_not_garbling_of_first"] is not None and p["first_not_garbling_of_second"] is not None for p in pairs)}
    FiniteModel.clear_caches()
    return result


def conditions():
    for scenario in ("xor", "gated"):
        for weight in WEIGHTS:
            yield {"scenario": scenario, "control": "matched", "source_weight": weight, "target_weight": weight}
        yield {"scenario": scenario, "control": "hidden_task_mix_shift", "source_weight": F(3, 4), "target_weight": F(1, 4)}
        yield {"scenario": scenario, "control": "announced_task_mix_shift", "source_weight": F(1, 4), "target_weight": F(1, 4)}
        for noise in (F(1, 10), F(1, 4), F(1, 2)):
            yield {"scenario": scenario, "control": "channel_model_mismatch", "source_weight": F(3, 4), "target_weight": F(3, 4), "target_extra_noise": noise}
        for seed in SHUFFLE_SEEDS:
            yield {"scenario": scenario, "control": "source_utility_observation_shuffle", "source_weight": F(3, 4), "target_weight": F(3, 4), "shuffle_seed": seed}
        for weight in (F(3, 4), F(1)):
            for priority in ("reverse", "rotate3"):
                yield {"scenario": scenario, "control": "probe_order_sensitivity", "source_weight": weight, "target_weight": weight, "probe_priority": priority}


def frozen_identity():
    return {path.name: sha256(path) for path in (ROOT / "protocol.v1.json", ROOT / "experiment.py", ROOT / "test_experiment.py")}


def freeze():
    path = ROOT / "freeze.v1.json"
    identity = frozen_identity()
    if path.exists():
        if json.loads(path.read_text())["sha256"] != identity:
            raise RuntimeError("Frozen files differ; create an explicitly versioned follow-up, do not overwrite the freeze")
        return
    write_json(path, {"protocol_version": "1.0.0", "stage": "constructed development experiment", "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": identity})


def run():
    freeze_path = ROOT / "freeze.v1.json"
    frozen = json.loads(freeze_path.read_text())
    if frozen["sha256"] != frozen_identity():
        raise RuntimeError("Executable/protocol hashes differ from the preregistered development freeze")
    started = time.perf_counter()
    rows, condition_timings = [], []
    output = ROOT / "results.v1"
    for condition_index, condition in enumerate(conditions(), 1):
        condition_start = time.perf_counter()
        source = FiniteModel(condition["scenario"], condition["source_weight"], shuffle_seed=condition.get("shuffle_seed"), probe_priority=condition.get("probe_priority", "canonical"))
        truth = FiniteModel(condition["scenario"], condition["target_weight"], extra_noise=condition.get("target_extra_noise", F(0)))
        metadata = {key: str(value) if isinstance(value, F) else value for key, value in condition.items()}
        metadata["context"] = "parity" if source.scenario == "xor" else "gate_selected_bit"
        metadata["source_utility_label_sha256"] = source.label_sha256
        order_sensitivity = condition["control"] == "probe_order_sensitivity"
        budgets = BUDGETS[1:] if order_sensitivity else BUDGETS
        methods = ("environment_ig", "decision_region_entropy", "decision_myopic", "decision_myopic_forced") if order_sensitivity else METHODS
        for budget in budgets:
            for method in methods:
                rows.append({**metadata, **evaluate(source, truth, method, budget)})
            if condition["control"] == "channel_model_mismatch":
                # This comparator knows the target channel error, but observes
                # neither the latent world nor target task outcomes. In this
                # control source and target utilities/mixes coincide exactly.
                calibrated = evaluate(truth, truth, "decision_full_horizon", budget)
                calibrated["method"] = "channel_calibrated_dp"
                rows.append({**metadata, **calibrated})
        stats = {**metadata, "elapsed_seconds": time.perf_counter() - condition_start,
                 "dp_cached_states": FiniteModel.dynamic_program.cache_info().currsize,
                 "posterior_cached_states": FiniteModel.posterior.cache_info().currsize}
        condition_timings.append(stats)
        write_json(output / "table.json", rows)
        write_json(output / "condition_timings.json", condition_timings)
        print(json.dumps({"condition": condition_index, **stats}), flush=True)
        FiniteModel.clear_caches()
    model_diagnostics = [diagnostics(scenario) for scenario in ("xor", "gated")]
    write_json(output / "model_diagnostics.json", model_diagnostics)
    highlights = [row for row in rows if row["control"] == "matched" and row["source_weight"] in ("3/4", "1") and row["budget"] in (1, 2, 4)]
    summary = {
        "stage": "Exact constructed development suite; not retrieval evidence or a novel VOI theorem",
        "protocol": "protocol.v1.json", "freeze": frozen, "row_count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "numerics": "Exact rational expected utility, regret, and operations; floating entropy and entropy tie tolerance 1e-12",
        "source_models": "Supplied exact finite source models, not fitted from retrieval data",
        "all_probe_pairs_blackwell_incomparable": all(item["all_probe_pairs_incomparable"] for item in model_diagnostics),
        "highlights": highlights,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"rows": len(rows), "seconds": summary["elapsed_seconds"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
