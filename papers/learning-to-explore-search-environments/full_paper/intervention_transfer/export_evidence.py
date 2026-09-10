"""Export portable replay evidence after a code freeze, with test utilities sealed separately."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from experiment import (ROOT, PAPER, POLICIES, ALL_POLICIES, LAMBDAS, P, add_cost,
                        digest, verify_freeze, write_json, zero_cost, validate_decision_lock)
sys.path.insert(0, str(ROOT.parent / "interventions"))
from adapter import FrozenHybridBackend, action_queries, fuse_subqueries

CACHE = PAPER / "semantic_pilot" / "cache"


def generation_cost(row):
    result = zero_cost()
    if row is not None:
        usage = row.get("usage", {})
        result.update(llm_calls=1, input_tokens=int(usage.get("prompt_tokens", 0)),
                      output_tokens=int(usage.get("output_tokens", 0)))
    return result


def policy_rankings(action_rankings, include_reference=True):
    policies = ALL_POLICIES if include_reference else POLICIES
    return [list(action_rankings[p[0]]) if len(p) == 1 else fuse_subqueries([action_rankings[a] for a in p])
            for p in policies]


def policy_costs(groups, generated):
    result = []
    for subset in ALL_POLICIES:
        cost = zero_cost()
        cost["logical_search_calls"] = sum(len(groups[action]) for action in subset)
        cost["underlying_search_calls"] = 2 * cost["logical_search_calls"]
        if any(action >= 2 for action in subset):
            add_cost(cost, generation_cost(generated))
        result.append(cost)
    return result


def document_observation(data, document_id, backend):
    """Charge one sampled prefix document and only its actually attempted setup."""
    cost = zero_cost(); cost["sample_calls"] = 1
    question = data["probe_questions"][document_id]
    add_cost(cost, generation_cost(question))
    if not question.get("valid") or not question.get("indirect_question"):
        return {"rr": [0.0] * P, "action_rankings": [[] for _ in range(5)], "cost": cost, "status": "unavailable_question"}
    key = document_id + ":indirect_question"
    generated = data["probe_rewrites"].get(key)
    if generated is None:
        raise ValueError("Valid indirect question lacks its frozen rewrite attempt")
    add_cost(cost, generation_cost(generated))
    groups = action_queries(question["indirect_question"], generated)
    results = [backend.search_group(group) for group in groups]
    for result in results:
        cost["logical_search_calls"] += result.logical_search_calls
        cost["underlying_search_calls"] += result.underlying_search_calls
    action_ranks = [result.ranking for result in results]
    rankings = policy_rankings(action_ranks, include_reference=False)
    rr = [1 / (ranking.index(document_id) + 1) if document_id in ranking else 0.0 for ranking in rankings]
    return {"rr": rr, "action_rankings": action_ranks, "cost": cost,
            "status": "available" if generated.get("valid") else "unavailable_generated_actions"}


def split_panels(corpus, data):
    unique = set(doc for panel in data["pools"].values() for doc in panel)
    def order(doc):
        key = "intervention-probe-split-v1:" + corpus + ":" + doc
        return hashlib.sha256(key.encode()).hexdigest(), doc
    documents = sorted(unique, key=order)
    if len(documents) < 40:
        raise ValueError("Forty source documents required for disjoint fitting/calibration panels")
    target = {seed: list(data["pools"][seed][:8]) for seed in ("11", "23", "47")}
    if any(len(panel) != 8 or len(set(panel)) != 8 for panel in target.values()):
        raise ValueError("Target prefix must contain eight distinct sampled documents")
    return {"fit": [documents[i:i + 8] for i in (0, 8, 16)],
            "calibration": [documents[i:i + 8] for i in (24, 32)], "target": target}


def ndcg(ranking, qrels):
    ideal = sorted((max(0, score) for score in qrels.values()), reverse=True)[:10]
    def dcg(values):
        return sum((2 ** score - 1) / np.log2(i + 2) for i, score in enumerate(values))
    denominator = dcg(ideal)
    return float(dcg([max(0, qrels.get(doc, 0)) for doc in ranking[:10]]) / denominator) if denominator else 0.0


def recall(ranking, qrels):
    relevant = {doc for doc, score in qrels.items() if score > 0}
    return len(set(ranking[:10]) & relevant) / len(relevant) if relevant else 0.0


def task_table(data, partition, backend):
    scores, recalls = [], []
    for qid in data["ids"][partition]:
        groups = action_queries(data["queries"][qid], data["task_rewrites"][qid])
        results = [backend.search_group(group).ranking for group in groups]
        rankings = policy_rankings(results)
        label = data["qrels"][partition][qid]
        scores.append([ndcg(ranking, label) for ranking in rankings])
        recalls.append([recall(ranking, label) for ranking in rankings])
    return {"ids": list(data["ids"][partition]), "scores": scores, "recalls": recalls}


def load_rank_maps(corpus, cache_dir):
    return {name: json.loads((cache_dir / f"{corpus}_{name}_ranks.json").read_text())["ranks"] for name in ("bm25", "dense")}


def export_inputs(preparation, cache_dir=CACHE):
    result = {"schema_version": 1, "stage": "inputs_without_test_utilities", "corpora": {},
              "policies": [list(policy) for policy in ALL_POLICIES], "lambda_grid": list(LAMBDAS)}
    for corpus, data in preparation["datasets"].items():
        panels = split_panels(corpus, data)
        used_docs = sorted(set(doc for kind in ("fit", "calibration") for panel in panels[kind] for doc in panel)
                           | set(doc for panel in panels["target"].values() for doc in panel))
        maps = load_rank_maps(corpus, cache_dir)
        test_meta = {"ids": list(data["ids"]["test"]), "policy_costs": []}
        for qid in test_meta["ids"]:
            groups = action_queries(data["queries"][qid], data["task_rewrites"][qid])
            test_meta["policy_costs"].append(policy_costs(groups, data["task_rewrites"][qid]))
        record = {"panels": panels, "worlds": {}, "test_meta": test_meta}
        for j, weight in enumerate(LAMBDAS):
            backend = FrozenHybridBackend(maps["bm25"], maps["dense"], weight)
            record["worlds"][str(j)] = {"train": task_table(data, "train", backend),
                "calibration": task_table(data, "calibration", backend),
                "probes": {doc: document_observation(data, doc, backend) for doc in used_docs}}
        result["corpora"][corpus] = record
    return result


def export_test_labels(preparation, decisions, inputs, cache_dir=CACHE):
    if decisions.get("stage") != "locked_predictions_before_test_utilities":
        raise ValueError("Test utilities cannot be exported before decisions are locked")
    if set(decisions["folds"]) != set(preparation["datasets"]):
        raise ValueError("All target families must have locked decisions")
    validate_decision_lock(decisions, inputs)
    result = {"schema_version": 1, "stage": "test_utilities_after_decision_lock", "corpora": {}}
    for corpus, data in preparation["datasets"].items():
        if inputs["corpora"][corpus]["test_meta"]["ids"] != data["ids"]["test"]:
            raise ValueError("Target query IDs differ from locked input metadata")
        maps = load_rank_maps(corpus, cache_dir)
        worlds = {str(j): task_table(data, "test", FrozenHybridBackend(maps["bm25"], maps["dense"], weight))
                  for j, weight in enumerate(LAMBDAS)}
        result["corpora"][corpus] = {"worlds": worlds}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "test"))
    parser.add_argument("--freeze", type=Path, default=ROOT / "freeze.v1.json")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs.v1.json")
    parser.add_argument("--decisions", type=Path, default=ROOT / "decisions.v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify_freeze(args.freeze, verify_sources=True, cache_dir=args.cache_dir)
    # The provider owns the original archive, which physically contains all
    # partitions. The inputs stage indexes train/calibration qrels only and
    # emits no test labels/rankings/scores; the decision process never reads it.
    preparation = json.loads((args.cache_dir / "prepared.json").read_text())
    if preparation["contract"]["pilot/engine.py"] != digest(PAPER / "pilot" / "engine.py"):
        raise ValueError("Imported keyword engine differs from prepared-generation contract")
    if args.stage == "inputs":
        started = datetime.now(timezone.utc).isoformat()
        result = export_inputs(preparation, args.cache_dir)
        result["freeze_sha256"] = digest(args.freeze)
        result["input_export_started_at_utc"] = started
        result["input_export_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["provenance"] = {"prepared_sha256": digest(args.cache_dir / "prepared.json"),
            "exporter_sha256": digest(Path(__file__))}
        write_json(args.output or args.inputs, result)
    else:
        inputs = json.loads(args.inputs.read_text()); decisions = json.loads(args.decisions.read_text())
        if decisions["inputs_sha256"] != digest(args.inputs) or decisions["freeze_sha256"] != digest(args.freeze):
            raise ValueError("Decisions do not match the frozen inputs")
        started = datetime.now(timezone.utc).isoformat()
        result = export_test_labels(preparation, decisions, inputs, args.cache_dir)
        result.update(decisions_sha256=digest(args.decisions), inputs_sha256=digest(args.inputs), freeze_sha256=digest(args.freeze))
        result["test_metric_computation_started_at_utc"] = started
        result["test_metric_computation_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output or ROOT / "test_labels.v1.json", result)


if __name__ == "__main__":
    main()
