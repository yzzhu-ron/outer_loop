"""POSTHOC evaluator-only capacity audit; never fits or exports a deployable policy.

Reads sealed target utilities and the pre-existing label-free query buckets.
Every oracle is an upper envelope of these finite observed utilities, not an
estimate of achievable future performance. Run again with a new --output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
STUDY = ROOT.parent / "intervention_transfer"
QUERY = ROOT.parent / "query_transfer"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def envelopes(scores, buckets):
    """Input dimensions: lambda x query x policy; queries have equal weight."""
    scores = np.asarray(scores, dtype=float)
    buckets = np.asarray(buckets, dtype=int)
    if scores.ndim != 3 or scores.shape[1] != len(buckets) or not len(buckets):
        raise ValueError("Scores and bucket query axes must agree and be nonempty")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected finite nDCG values in [0,1]")
    if np.any((buckets < 0) | (buckets > 3)):
        raise ValueError("Expected the existing four buckets")
    # These are query-weighted contributions, including zero for empty buckets.
    sums = np.stack([scores[:, buckets == b].sum(axis=1) / len(buckets)
                     for b in range(4)])
    global_by_lambda = scores.mean(axis=1)
    values = {
        "best_fixed_global": float(global_by_lambda.mean(axis=0).max()),
        "best_per_lambda_global": float(global_by_lambda.max(axis=1).mean()),
        "best_fixed_four_bucket": float(sums.mean(axis=1).max(axis=1).sum()),
        "best_per_lambda_four_bucket": float(sums.max(axis=2).sum(axis=0).mean()),
        "best_fixed_per_query": float(scores.mean(axis=0).max(axis=1).mean()),
        "best_per_lambda_per_query": float(scores.max(axis=2).mean()),
    }
    for low, high in [
        ("best_fixed_global", "best_per_lambda_global"),
        ("best_fixed_global", "best_fixed_four_bucket"),
        ("best_per_lambda_global", "best_per_lambda_four_bucket"),
        ("best_fixed_four_bucket", "best_per_lambda_four_bucket"),
        ("best_fixed_four_bucket", "best_fixed_per_query"),
        ("best_fixed_per_query", "best_per_lambda_per_query"),
        ("best_per_lambda_four_bucket", "best_per_lambda_per_query"),
    ]:
        if values[low] > values[high] + 1e-12:
            raise AssertionError(f"Policy-class containment failed: {low}, {high}")
    counts = np.bincount(buckets, minlength=4)
    choices = [[int(p) for p in sums[b].argmax(axis=1)] if counts[b] else None
               for b in range(4)]
    return {"values": values,
            "gains": {
                "lambda_global_over_fixed_global": values["best_per_lambda_global"] - values["best_fixed_global"],
                "lambda_bucket_over_lambda_global": values["best_per_lambda_four_bucket"] - values["best_per_lambda_global"],
                "lambda_query_over_lambda_global": values["best_per_lambda_per_query"] - values["best_per_lambda_global"],
                "lambda_bucket_over_fixed_bucket": values["best_per_lambda_four_bucket"] - values["best_fixed_four_bucket"],
                "lambda_bucket_over_fixed_global": values["best_per_lambda_four_bucket"] - values["best_fixed_global"],
            },
            "query_count": len(buckets), "bucket_counts": counts.tolist(),
            "best_fixed_global_policy_index": int(global_by_lambda.mean(axis=0).argmax()),
            "best_global_policy_indices_by_lambda": global_by_lambda.argmax(axis=1).tolist(),
            "best_bucket_policy_indices_by_lambda": choices}


def alias_summary(fold, scores):
    """Finite observable equivalence classes; full ranklists remain permitted."""
    utilities = scores.mean(axis=1)
    summaries = {}
    for seed in sorted(fold["runs"]["0"]):
        groups = {}
        document_states, ranklist_states = {}, {}
        for j in range(len(scores)):
            run = fold["runs"][str(j)][seed]["within_unshrunk"]
            rr = np.array([o["rr"] for o in run["observations"]])
            profile = (rr - rr[:, :1]).mean(axis=0)
            if not np.allclose(profile, run["profile"], atol=1e-14, rtol=0):
                raise AssertionError("Saved profile differs from paid RR observations")
            key = tuple(np.round(profile, 12))
            groups.setdefault(key, []).append(j)
            document_states[j] = json.dumps([o["rr"] for o in run["observations"]])
            ranklist_states[j] = json.dumps([o["action_rankings"] for o in run["observations"]])
        aliases = []
        weighted_loss = 0.0
        for indices in groups.values():
            if len(indices) < 2:
                continue
            subset = utilities[indices]
            loss = float(subset.max(axis=1).mean() - subset.mean(axis=0).max())
            weighted_loss += len(indices) / len(scores) * loss
            aliases.append({"lambda_indices": indices, "within_alias_headroom": loss,
                            "distinct_document_rr_states": len({document_states[j] for j in indices}),
                            "distinct_action_ranklist_states": len({ranklist_states[j] for j in indices})})
        summaries[seed] = {"repeated_mean_profile_groups": aliases,
                           "loss_from_mean_profile_grouping": weighted_loss,
                           "profile_oracle": float(utilities.max(axis=1).mean() - weighted_loss)}
    return summaries


def self_check():
    # One lambda, 2 queries in bucket0 and 1 in bucket1; two policies.
    # Fixed global = 2/3. Both occupied buckets choose perfectly => 1.
    # Equal weighting of buckets (including empty ones) would fail this check.
    out = envelopes([[[1, 0], [1, 0], [0, 1]]], [0, 0, 1])
    assert abs(out["values"]["best_fixed_global"] - 2 / 3) < 1e-12
    assert out["values"]["best_per_lambda_four_bucket"] == 1
    assert out["best_bucket_policy_indices_by_lambda"][2:] == [None, None]
    # Backend changes alone can require different global decisions.
    out = envelopes([[[1, 0]], [[0, 1]]], [0])
    assert out["values"]["best_fixed_global"] == .5
    assert out["values"]["best_per_lambda_global"] == 1


def audit():
    self_check()
    paths = {"labels": STUDY / "test_labels.v1.json", "decisions": STUDY / "decisions.v1.json",
             "inputs": STUDY / "inputs.v1.json", "results": STUDY / "results.v1.json",
             "freeze": STUDY / "freeze.v1.json", "protocol": STUDY / "protocol.v1.json",
             "query_evidence": QUERY / "evidence.json", "query_freeze": QUERY / "freeze.v1.json"}
    hashes = {k: digest(p) for k, p in paths.items()}
    labels, decisions, results = (read(paths[k]) for k in ("labels", "decisions", "results"))
    query, protocol = (read(paths[k]) for k in ("query_evidence", "protocol"))
    for key in ("decisions", "inputs", "freeze"):
        assert labels[key + "_sha256"] == hashes[key]
    for key in ("inputs", "freeze"):
        assert decisions[key + "_sha256"] == hashes[key]
    for key in ("labels", "decisions", "inputs", "freeze"):
        assert results["provenance"][key] == hashes[key]
    assert read(paths["query_freeze"])["sha256"]["evidence.json"] == hashes["query_evidence"]
    for path, expected in read(paths["freeze"])["files"].items():
        assert digest(PAPER / path) == expected, path
    policies = protocol["policy_menu"]["ordered_policies"]
    assert len(policies) == 11
    families = {}
    for family, corpus in labels["corpora"].items():
        worlds = [corpus["worlds"][str(j)] for j in range(len(protocol["lambda_grid"]))]
        query_part = query["families"][family]["partitions"]["test"]
        assert len(set(query_part["ids"])) == len(query_part["ids"])
        assert all(world["ids"] == query_part["ids"] for world in worlds)
        # Reconstruct the old rule directly from its saved label-free features.
        features = np.array(query_part["features"])
        expected_buckets = 2 * (features[:, 0] > np.log1p(8)) + (features[:, 4] > 0)
        assert np.array_equal(expected_buckets, query_part["buckets"])
        full_scores = np.array([world["scores"] for world in worlds])
        assert full_scores.shape[2] == 12  # Exclude six-search RRF-all reference.
        scores = full_scores[:, :, :len(policies)]
        out = envelopes(scores, query_part["buckets"])
        fold = decisions["folds"][family]
        chosen = {run["source_fixed_train_cal"]["policy"]
                  for world in fold["runs"].values() for run in world.values()}
        assert len(chosen) == 1
        fixed_index = chosen.pop()
        fixed_value = float(scores[:, :, fixed_index].mean())
        assert abs(fixed_value - out["values"]["best_fixed_global"]) < 1e-12
        out["fixed160_matches_best_fixed_global"] = True
        out["fixed160_policy_index"] = fixed_index
        out["fixed160_ndcg"] = fixed_value
        out["calibrated_models"] = {
            method: {key: fold["models"][method][key] for key in ("eta", "calibration", "beta", "degenerate")}
            for method in ("within", "pooled")}
        out["mean_profile_aliases"] = alias_summary(fold, scores)
        families[family] = out
    macro = {kind: {k: float(np.mean([f[kind][k] for f in families.values()]))
                    for k in next(iter(families.values()))[kind]}
             for kind in ("values", "gains")}
    gate = protocol["evaluation"]["advancement_gate"]["macro_gain_vs_source_fixed_at_least"]
    return {"stage": "POSTHOC EVALUATOR-ONLY; not a new registered result or deployable policy",
            "scope": "Finite observed target utilities on three reused families and seven fixed lambda values; no future-population bound or learnability claim.",
            "weighting": "Equal families, equal lambdas, equal queries within family; bucket contributions weighted by observed query count.",
            "policy_menu": policies, "lambda_grid": protocol["lambda_grid"],
            "query_buckets": "Existing label-free rule: 2*(surface token count>8)+any identifier-like token; verified from saved lexical features.",
            "oracle_selection": "Uses target test utilities for evaluation only. Representative ties use lowest index; empty buckets have no chosen policy.",
            "source_sha256": {str(paths[k].relative_to(PAPER)): v for k, v in hashes.items()},
            "script_sha256": digest(Path(__file__)), "families": families, "macro": macro,
            "registered_macro_gain_gate": gate,
            "global_policy_gate_unattainable_on_observed_utilities": macro["gains"]["lambda_global_over_fixed_global"] < gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "posthoc_intervention_headroom.v1.json")
    args = parser.parse_args()
    result = audit()
    # This audit never overwrites any evidence or result, including its own output.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "macro": result["macro"]}, indent=2))


if __name__ == "__main__":
    main()
