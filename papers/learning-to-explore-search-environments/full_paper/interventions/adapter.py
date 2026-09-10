"""Replay a frozen truncated hybrid backend without fitting or utility evaluation.

Only the evaluator constructs this object or sees both endpoint caches and lambda.
A deployed policy must receive selected-probe responses through a restricted API.
The CLI audits existing endpoints only; it never evaluates an interior mixture.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
PILOT = PAPER / "pilot"
DEFAULT_CACHE = PAPER / "semantic_pilot" / "cache"
sys.path.insert(0, str(PILOT))
from engine import feature_bucket, query_features, transform

ACTIONS = ("original", "keywords", "semantic", "hyde", "decomposed")
FAMILIES = ("direct_question", "indirect_question")
LAMBDA_GRID = (0, .125, .25, .5, .75, .875, 1)
RRF_CONSTANT = 60
CUTOFF = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_id(document_id: str, family: str, action: int) -> str:
    payload = json.dumps([document_id, family, action], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _validate_ranking(ranking: Sequence[str]) -> None:
    if isinstance(ranking, (str, bytes)) or len(ranking) > CUTOFF:
        raise ValueError("An endpoint ranking must be a sequence of at most ten IDs")
    if any(not isinstance(doc, str) or not doc for doc in ranking):
        raise ValueError("Document IDs must be nonempty strings")
    if len(set(ranking)) != len(ranking):
        raise ValueError("An endpoint ranking must not contain duplicate document IDs")


def weighted_rrf(bm25: Sequence[str], dense: Sequence[str], dense_weight: float) -> list[str]:
    """WRRF60 over two top10 lists, excluding every zero-contribution document.

    Absent documents contribute zero; endpoint order is exactly preserved. At an
    interior weight, scores are (1-lambda)/(60+BM25 rank)+lambda/(60+dense rank).
    Ties use the lexicographic document ID, and truncation follows fusion.
    """
    if not math.isfinite(dense_weight) or not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be finite and in [0, 1]")
    _validate_ranking(bm25)
    _validate_ranking(dense)
    if dense_weight == 0:
        return list(bm25)
    if dense_weight == 1:
        return list(dense)
    scores: dict[str, float] = {}
    for weight, ranking in ((1 - dense_weight, bm25), (dense_weight, dense)):
        for position, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + weight / (RRF_CONSTANT + position)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))[:CUTOFF]


def fuse_subqueries(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Match the existing semantic pilot's ordinary RRF after each query's fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        _validate_ranking(ranking)
        for position, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (RRF_CONSTANT + position)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))[:CUTOFF]


def action_queries(query: str, row: Mapping) -> list[list[str]]:
    """Reconstruct frozen request strings and boundaries; never regenerate text."""
    generated = [[], [], []]
    if row.get("valid"):
        generated = [[row["semantic_query"]], [row["hypothetical_document"]], list(row["subqueries"])]
    return [[query], [transform(query, "keywords")], *generated]


@dataclass(frozen=True)
class ActionResult:
    ranking: list[str]
    logical_search_calls: int
    underlying_search_calls: int


class FrozenHybridBackend:
    """Evaluator-owned cached backend; no task labels or relevance metrics.

    `both` charges both component searches at every lambda (the default hidden
    backend contract). `active` skips a zero-weight endpoint and is suitable only
    when that varying cost cannot leak hidden identity to the policy.
    """

    def __init__(self, bm25: Mapping[str, Sequence[str]], dense: Mapping[str, Sequence[str]],
                 dense_weight: float, execution: str = "both"):
        weighted_rrf([], [], dense_weight)
        if execution not in ("both", "active"):
            raise ValueError("execution must be 'both' or 'active'")
        if set(bm25) != set(dense):
            raise ValueError("Endpoint query-key sets differ")
        for ranking in (*bm25.values(), *dense.values()):
            _validate_ranking(ranking)
        self._bm25 = {query: tuple(ranking) for query, ranking in bm25.items()}
        self._dense = {query: tuple(ranking) for query, ranking in dense.items()}
        self._dense_weight = dense_weight
        self._components = 2 if execution == "both" or 0 < dense_weight < 1 else 1

    def search(self, query: str) -> list[str]:
        # A cache miss is an error, never an empty retrieval result.
        return weighted_rrf(self._bm25[query], self._dense[query], self._dense_weight)

    def search_group(self, queries: Sequence[str]) -> ActionResult:
        rankings = [self.search(query) for query in queries]
        ranking = fuse_subqueries(rankings) if len(rankings) > 1 else rankings[0] if rankings else []
        return ActionResult(ranking, len(queries), self._components * len(queries))


def task_requests(data: Mapping):
    """Yield only task query IDs/strings/groups, without exposing qrels."""
    for partition, ids in data["ids"].items():
        for qid in ids:
            yield partition, qid, action_queries(data["queries"][qid], data["task_rewrites"][qid])


def probe_candidates(data: Mapping, seed: str) -> list[dict]:
    """Restore candidate identity, known document, and both comparison requests."""
    candidates = []
    for doc_id in data["pools"][seed]:
        questions = data["probe_questions"][doc_id]
        for family in FAMILIES:
            if not questions.get("valid") or not questions.get(family):
                continue
            query = questions[family]
            groups = action_queries(query, data["probe_rewrites"][doc_id + ":" + family])
            for action in range(1, len(ACTIONS)):
                if not groups[action] or "\n".join(groups[0]) == "\n".join(groups[action]):
                    continue
                candidates.append({
                    "id": candidate_id(doc_id, family, action), "document_id": doc_id,
                    "action": action, "family": family, "bucket": feature_bucket(query),
                    "features": query_features(query), "cost": len(groups[0]) + len(groups[action]),
                    "original_requests": groups[0], "alternative_requests": groups[action],
                })
    return candidates


def probe_response(backend: FrozenHybridBackend, candidate: Mapping) -> dict:
    """Selected document-only observation; includes charged comparison costs.

    This mirrors the frozen observation schema, adding full response lists for
    future observability audits. It computes no task-label utility.
    """
    left = backend.search_group(candidate["original_requests"])
    right = backend.search_group(candidate["alternative_requests"])
    doc = candidate["document_id"]
    rr_left = 1 / (left.ranking.index(doc) + 1) if doc in left.ranking else 0.0
    rr_right = 1 / (right.ranking.index(doc) + 1) if doc in right.ranking else 0.0
    union = set(left.ranking) | set(right.ranking)
    return {
        "delta": rr_right - rr_left, "rr_original": rr_left, "rr_alternative": rr_right,
        "overlap": len(set(left.ranking) & set(right.ranking)) / max(1, len(union)),
        "original_ranking": left.ranking, "alternative_ranking": right.ranking,
        "logical_search_calls": left.logical_search_calls + right.logical_search_calls,
        "underlying_search_calls": left.underlying_search_calls + right.underlying_search_calls,
    }


def audit_endpoints(cache_dir: Path = DEFAULT_CACHE) -> dict:
    """Check request coverage/schema and exact archived endpoint behavior only."""
    prepared_path = cache_dir / "prepared.json"
    preparation = json.loads(prepared_path.read_text())
    expected_engine = preparation["contract"]["pilot/engine.py"]
    if sha256(PILOT / "engine.py") != expected_engine:
        raise ValueError("Keyword/feature engine differs from prepared input contract")
    inputs = [prepared_path]
    report = {"schema_version": 1, "status": "endpoint_validation_only",
              "interior_mixture_outcomes_computed": False,
              "utility_models_fitted": False, "task_utility_metrics_computed": False,
              "prepared_contract": preparation["contract"], "corpora": {}}
    for corpus, data in preparation["datasets"].items():
        cache_paths = {backend: cache_dir / f"{corpus}_{backend}_ranks.json" for backend in ("bm25", "dense")}
        outcome_paths = {backend: cache_dir / f"{corpus}_{backend}_outcomes.json" for backend in ("bm25", "dense")}
        inputs.extend([*cache_paths.values(), *outcome_paths.values()])
        caches = {backend: json.loads(path.read_text()) for backend, path in cache_paths.items()}
        outcomes = {backend: json.loads(path.read_text()) for backend, path in outcome_paths.items()}
        for backend, outcome in outcomes.items():
            if outcome["contract"] != preparation["contract"]:
                raise ValueError(f"{corpus}/{backend}: outcome contract mismatch")
        requests = list(task_requests(data))
        pools = {seed: probe_candidates(data, seed) for seed in data["pools"]}
        needed = {q for _, _, groups in requests for group in groups for q in group}
        needed.update(q for pool in pools.values() for row in pool
                      for field in ("original_requests", "alternative_requests") for q in row[field])
        stats = {"task_queries_by_partition": {part: len(ids) for part, ids in data["ids"].items()},
                 "distinct_required_query_strings": len(needed),
                 "probe_candidates_by_seed": {seed: len(pool) for seed, pool in pools.items()},
                 "endpoint_checks": {}}
        for backend, weight in (("bm25", 0), ("dense", 1)):
            rank_map = caches[backend]["ranks"]
            missing = sorted(needed - set(rank_map))
            if missing:
                raise ValueError(f"{corpus}/{backend}: {len(missing)} required query strings missing")
            hybrid = FrozenHybridBackend(caches["bm25"]["ranks"], caches["dense"]["ranks"], weight)
            # Check every cached raw request, not only the downstream used subset.
            for query, expected in rank_map.items():
                if hybrid.search(query) != expected:
                    raise AssertionError(f"{corpus}/{backend}: raw endpoint mismatch")
            archived = outcomes[backend]
            by_task = {part: {qid: i for i, qid in enumerate(rows["ids"])}
                       for part, rows in archived["tasks"].items()}
            task_count = 0
            for partition, qid, groups in requests:
                idx = by_task[partition][qid]
                rankings = [hybrid.search_group(group).ranking for group in groups]
                if rankings != archived["tasks"][partition]["rankings"][idx]:
                    raise AssertionError(f"{corpus}/{backend}/{partition}/{qid}: action endpoint mismatch")
                if [len(group) for group in groups] != [cost["search_calls"] for cost in
                        archived["tasks"][partition]["action_costs"][idx]]:
                    raise AssertionError(f"{corpus}/{backend}/{partition}/{qid}: task cost mismatch")
                task_count += len(groups)
            probe_count = 0
            for seed, pool in pools.items():
                archived_pool = archived["pools"][seed]
                candidate_fields = ("id", "action", "family", "bucket", "features", "cost")
                reconstructed = [{key: row[key] for key in candidate_fields} for row in pool]
                if reconstructed != archived_pool["candidates"]:
                    raise AssertionError(f"{corpus}/{backend}/{seed}: candidate schema mismatch")
                for row in pool:
                    response = probe_response(hybrid, row)
                    expected = archived_pool["observations"][row["id"]]
                    if {key: response[key] for key in expected} != expected:
                        raise AssertionError(f"{corpus}/{backend}/{seed}: probe endpoint mismatch")
                    if response["logical_search_calls"] != row["cost"]:
                        raise AssertionError("Probe logical cost mismatch")
                    probe_count += 1
            stats["endpoint_checks"][backend] = {
                "rank_cache_identity": caches[backend]["identity"], "raw_ranklists_exact": len(rank_map),
                "empty_ranklists": sum(not ranking for ranking in rank_map.values()),
                "task_action_ranklists_exact": task_count, "probe_observations_exact": probe_count,
                "missing_required_queries": 0, "candidate_schema_mismatches": 0,
            }
        report["corpora"][corpus] = stats
    report["input_files"] = {str(path.relative_to(PAPER)): {"sha256": sha256(path), "bytes": path.stat().st_size}
                             for path in inputs}
    report["adapter_sha256"] = sha256(Path(__file__))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=HERE / "endpoint_audit.v1.json")
    args = parser.parse_args()
    report = audit_endpoints(args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": str(args.output), "status": report["status"], "corpora": list(report["corpora"])}))


if __name__ == "__main__":
    main()
