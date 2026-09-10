"""CPU-only replay of committed outcome summaries and SearchProbe certificates.

This reads recorded target outcomes for evaluation. It does not run retrieval,
generation, training, or probe selection, and does not regenerate bootstrap
interval endpoints. Python's standard library and SearchProbe are sufficient.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import urllib.request

from searchprobe.decisions import audit_decision_json
from searchprobe.responsiveness import audit_response_json


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "evidence_manifest.json"
METHODS = ("source_router", "random", "fixed", "information_gain", "learned_value")
FOLLOW_METHODS = ("random", "fixed", "information_gain", "decision_value")
BASELINES = ("zero_budget", "original_frozen_router", "original_query")
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), f"Invalid CSV header: {path}")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), f"Malformed CSV rows: {path}")
    return rows


def near(actual, expected, label):
    require(not isinstance(actual, bool), f"Boolean in numeric field: {label}")
    actual = float(actual)
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=TOLERANCE),
            f"Recomputed value differs for {label}: recorded={actual}, replayed={expected}")


def mean(values):
    return math.fsum(values) / len(values)


def grid(rows, fields, required, label):
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(len(keys) == len(required) and set(keys) == required, f"Incomplete, duplicated, or unexpected {label} grid")
    return dict(zip(keys, rows))


def selected(data, actions, label):
    n, width = len(data["query_ids"]), len(data["action_names"])
    require(len(actions) == n and all(type(a) is int and 0 <= a < width for a in actions), f"Invalid actions: {label}")
    return ([row[a] for row, a in zip(data["action_scores"], actions)],
            [row[a] for row, a in zip(data["action_recalls_at_10"], actions)])


def verify_run(data, run, label):
    scores, recalls = selected(data, run["actions"], label)
    for key, expected in (("ndcg", scores), ("recall_at_10", recalls)):
        require(len(run[key]) == len(expected), f"Outcome length mismatch: {label}/{key}")
        for index, (actual, value) in enumerate(zip(run[key], expected)):
            near(actual, value, f"{label}/{key}/{index}")
    return mean(scores), mean(recalls)


def verify_selected_ids(run, budget, label):
    ids = run["selected_probe_ids"]
    require(isinstance(ids, list) and all(isinstance(value, str) and value for value in ids)
            and len(set(ids)) == len(ids) and len(ids) <= budget, f"Invalid recorded selected-probe IDs: {label}")


def expected_files(names):
    fixed = {"paired_outcomes.json", "headroom.csv", "adaptation.csv", "paired_intervals.csv",
             "world_model_followup/paired_outcomes.json", "world_model_followup/adaptation.csv",
             "world_model_followup/baselines.csv", "world_model_followup/paired_intervals.csv",
             "response_audits.csv", "source_decisions.csv"}
    return fixed | {f"response_models/{name}.{kind}.json" for name in names for kind in ("model", "audit")} | {
        f"source_decision_models/{name}_{kind}.json" for name in names for kind in ("model", "audit")}


def load_manifest(path=DEFAULT_MANIFEST):
    manifest = read_json(path)
    require(manifest.get("schema_version") == 1, "Unsupported replay manifest schema")
    contract = manifest["population"]
    names = set(contract["query_counts"])
    require(len(names) == 6 and set(manifest["files"]) == expected_files(names), "Replay manifest must cover the complete six-environment evidence set")
    require(len(contract["actions"]) == 5 and len(set(contract["actions"])) == 5, "Invalid action contract")
    require(contract["seeds"] == [11, 23, 47] and contract["budgets"] == [0, 4, 8, 16, 32], "Unexpected frozen seed/budget contract")
    require(all(type(n) is int and n > 0 for n in contract["query_counts"].values()), "Invalid query-count contract")
    require(len(manifest["commit"]) == 40 and all(c in "0123456789abcdef" for c in manifest["commit"]), "Manifest needs a full immutable commit hash")
    for name, checksum in manifest["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and len(checksum) == 64
                and all(c in "0123456789abcdef" for c in checksum), "Unsafe input path or invalid checksum")
    return manifest


def download_evidence(destination, manifest):
    """Fetch pinned files only when requested; never replace existing evidence."""
    destination = Path(destination)
    prefix = f"https://raw.githubusercontent.com/yzzhu-ron/outer_loop/{manifest['commit']}/papers/learning-to-explore-search-environments/semantic_pilot/results/"
    for name, expected in sorted(manifest["files"].items()):
        path = destination / name
        if path.exists():
            require(sha256(path) == expected, f"Existing evidence differs: {name}; choose an empty --evidence-dir")
            continue
        require(not path.is_symlink(), f"Refusing broken evidence symlink: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(prefix + name, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            require(sha256(temporary) == expected, f"Downloaded checksum differs: {name}")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def replay(evidence_dir, manifest_path=DEFAULT_MANIFEST):
    """Verify pinned bytes and recalculate metrics from recorded action outcomes."""
    root, manifest = Path(evidence_dir), load_manifest(manifest_path)
    for name, expected in manifest["files"].items():
        require(sha256(root / name) == expected, f"Evidence checksum mismatch: {name}")
    contract = manifest["population"]
    names, seeds, budgets = sorted(contract["query_counts"]), contract["seeds"], contract["budgets"]
    original = read_json(root / "paired_outcomes.json")
    follow = read_json(root / "world_model_followup/paired_outcomes.json")
    require(set(original) == set(follow) == set(names), "Paired environment cohort mismatch")
    heads = grid(read_csv(root / "headroom.csv"), ("environment",), {(name,) for name in names}, "headroom")
    main_rows = grid(read_csv(root / "adaptation.csv"), ("environment", "seed", "method", "budget"),
                     {(name, str(seed), method, str(budget)) for name in names for seed in seeds for method in METHODS for budget in budgets}, "main adaptation")
    follow_rows = grid(read_csv(root / "world_model_followup/adaptation.csv"), ("environment", "seed", "method", "budget"),
                       {(name, str(seed), method, str(budget)) for name in names for seed in seeds for method in FOLLOW_METHODS for budget in budgets}, "follow-up adaptation")
    baseline_rows = grid(read_csv(root / "world_model_followup/baselines.csv"), ("environment", "baseline"),
                         {(name, baseline) for name in names for baseline in BASELINES}, "follow-up baseline")
    summaries = []
    for name in names:
        data, other = original[name], follow[name]
        n = contract["query_counts"][name]
        require(len(data["query_ids"]) == n and len(set(data["query_ids"])) == n and data["query_ids"] == other["query_ids"], f"Query cohort/order mismatch: {name}")
        require(data["action_names"] == other["action_names"] == contract["actions"], f"Action order mismatch: {name}")
        for field in ("action_scores", "action_recalls_at_10"):
            values = data[field]
            require(len(values) == n and all(len(row) == 5 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in row) for row in values), f"Invalid recorded utilities: {name}/{field}")
        base, base_recall = selected(data, data["source_router_actions"], name + "/source-router")
        base_mean, recall_mean = mean(base), mean(base_recall)
        head = heads[(name,)]
        require(int(head["n_queries"]) == n, f"Headroom query count mismatch: {name}")
        action_means = [mean([row[j] for row in data["action_scores"]]) for j in range(5)]
        fixed = max(range(5), key=lambda j: action_means[j])
        oracle = mean([max(row) for row in data["action_scores"]])
        for field, value in {"original": action_means[0], "source_router": base_mean, "target_fixed_oracle": action_means[fixed],
                             "query_oracle": oracle, "oracle_gap_vs_router": oracle - base_mean,
                             "oracle_gap_vs_target_fixed": oracle - action_means[fixed]}.items():
            near(head[field], value, name + "/headroom/" + field)
        require(head["target_fixed_oracle_action"] == contract["actions"][fixed], f"Target fixed-action argmax mismatch: {name}")
        required_main = {f"{seed}/{method}/{budget}" for seed in seeds for method in METHODS for budget in budgets}
        require(set(data["runs"]) == required_main, f"Main recorded run grid mismatch: {name}")
        for key, run in data["runs"].items():
            seed, method, budget = key.split("/")
            row = main_rows[(name, seed, method, budget)]
            quality, recall = verify_run(data, run, name + "/" + key)
            verify_selected_ids(run, int(budget), name + "/" + key)
            if method == "source_router" or budget == "0":
                require(run["actions"] == data["source_router_actions"] and not run["selected_probe_ids"], f"Zero-probe baseline mismatch: {name}/{key}")
            changes = sum(a != b for a, b in zip(run["actions"], data["source_router_actions"])) / n
            for field, value in {"ndcg": quality, "recall_at_10": recall, "delta_vs_source_router": quality - base_mean,
                                 "delta_recall_at_10_vs_source_router": recall - recall_mean, "changed_action_fraction": changes}.items():
                near(row[field], value, name + "/" + key + "/" + field)
            require(int(row["selected_probe_pairs"]) == len(run["selected_probe_ids"]), f"Selected-pair count mismatch: {name}/{key}")
        require(set(other["baselines"]) == set(BASELINES), f"Follow-up baseline grid mismatch: {name}")
        require(other["baselines"]["original_frozen_router"]["actions"] == data["source_router_actions"]
                and other["baselines"]["original_query"]["actions"] == [0] * n, f"Follow-up reference action mismatch: {name}")
        for baseline, run in other["baselines"].items():
            quality, recall = verify_run(data, run, name + "/baseline/" + baseline)
            row = baseline_rows[(name, baseline)]
            require(int(row["n_queries"]) == n, f"Follow-up baseline cohort mismatch: {name}")
            near(row["ndcg"], quality, name + "/baseline/" + baseline)
            near(row["recall_at_10"], recall, name + "/baseline-recall/" + baseline)
        prior = other["baselines"]["zero_budget"]
        prior_mean, prior_recall = mean(prior["ndcg"]), mean(prior["recall_at_10"])
        require(set(other["runs"]) == {f"{seed}/{method}/{budget}" for seed in seeds for method in FOLLOW_METHODS for budget in budgets}, f"Follow-up run grid mismatch: {name}")
        for key, run in other["runs"].items():
            seed, method, budget = key.split("/")
            row = follow_rows[(name, seed, method, budget)]
            quality, recall = verify_run(data, run, name + "/follow-up/" + key)
            verify_selected_ids(run, int(budget), name + "/follow-up/" + key)
            require(int(row["n_queries"]) == n and int(row["selected_probe_pairs"]) == len(run["selected_probe_ids"]), f"Follow-up counts mismatch: {name}/{key}")
            if budget == "0":
                require(run["actions"] == prior["actions"] and not run["selected_probe_ids"], f"Follow-up zero-budget mismatch: {name}/{key}")
            values = {"ndcg": quality, "recall_at_10": recall, "delta_vs_zero_budget": quality - prior_mean,
                      "delta_vs_original_frozen_router": quality - base_mean, "delta_recall_vs_zero_budget": recall - prior_recall,
                      "delta_recall_vs_original_frozen_router": recall - recall_mean,
                      "changed_action_fraction_vs_zero_budget": sum(a != b for a, b in zip(run["actions"], prior["actions"])) / n,
                      "changed_action_fraction_vs_original_frozen_router": sum(a != b for a, b in zip(run["actions"], data["source_router_actions"])) / n}
            for field, value in values.items():
                near(row[field], value, name + "/follow-up/" + key + "/" + field)
        decision = mean([mean(other["runs"][f"{seed}/decision_value/32"]["ndcg"]) for seed in seeds])
        random = mean([mean(other["runs"][f"{seed}/random/32"]["ndcg"]) for seed in seeds])
        learned = mean([mean(data["runs"][f"{seed}/learned_value/32"]["ndcg"]) for seed in seeds])
        summaries.append({"environment": name, "query_backend_rows": n, "original_query_ndcg": action_means[0],
                          "frozen_router_ndcg": base_mean, "best_fixed_ndcg": action_means[fixed], "per_query_oracle_ndcg": oracle,
                          "oracle_headroom_over_best_fixed": oracle - action_means[fixed], "original_learned_B32_delta": learned - base_mean,
                          "followup_prior_ndcg": prior_mean, "followup_decision_B32_ndcg": decision,
                          "followup_decision_B32_delta_vs_prior": decision - prior_mean, "followup_decision_B32_delta_vs_random": decision - random})

    contrast_count = 0
    for followup, path, methods, comparators in ((False, "paired_intervals.csv", METHODS[1:], ("source_router", "random")),
                                               (True, "world_model_followup/paired_intervals.csv", FOLLOW_METHODS, ("zero_budget", "random", "original_frozen_router"))):
        rows = read_csv(root / path)
        grid(rows, ("environment", "method", "budget", "comparator"),
             {(name, method, str(budget), comparator) for name in names for method in methods for budget in budgets for comparator in comparators}, "paired contrast")
        for row in rows:
            name, method, budget, comparator = (row[key] for key in ("environment", "method", "budget", "comparator"))
            data = follow[name] if followup else original[name]
            deltas = []
            for seed in seeds:
                run = data["runs"][f"{seed}/{method}/{budget}"]
                reference = (data["runs"][f"{seed}/random/{budget}"]["ndcg"] if comparator == "random" else
                             data["baselines"][comparator]["ndcg"] if followup else selected(original[name], original[name]["source_router_actions"], name)[0])
                deltas.extend(a - b for a, b in zip(run["ndcg"], reference))
            near(row["mean"], mean(deltas), path + "/paired mean")
            require(-1 <= float(row["lo"]) <= float(row["hi"]) <= 1, f"Invalid stored interval bounds: {path}")
            contrast_count += 1

    response_rows = grid(read_csv(root / "response_audits.csv"), ("environment",), {(name,) for name in names}, "response certificate")
    source_rows = grid(read_csv(root / "source_decisions.csv"), ("target_fold",),
                       {(name,) for name in names}, "source certificate")
    certificates = []
    for name in names:
        response = audit_response_json(root / f"response_models/{name}.model.json")
        require(response == read_json(root / f"response_models/{name}.audit.json"), f"Response API/report mismatch: {name}")
        row = response_rows[(name,)]
        require(response["model"]["row_ids"] == original[name]["query_ids"] and response["model"]["actions"] == contract["actions"], f"Response cohort/action mismatch: {name}")
        require([item["original_action_index"] for item in response["rows"]] == original[name]["source_router_actions"], f"Response baseline action mismatch: {name}")
        for field, expected in (("n_queries", response["counts"]["rows"]), ("certified_unchanged", response["counts"]["certified_unchanged"]),
                                ("potentially_changed", response["counts"]["potentially_changed"])):
            require(int(row[field]) == expected, f"Response summary count mismatch: {name}/{field}")
        require(row["pair_budget"] == "32", f"Response budget mismatch: {name}")
        near(row["unchanged_fraction"], response["unchanged_fraction"]["value"], name + "/unchanged fraction")
        near(row["mean_utility_change_abs_upper_bound"], response["mean_utility_change_bound"]["absolute_upper_bound"]["value"], name + "/utility-change bound")
        source = audit_decision_json(root / f"source_decision_models/{name}_model.json")
        require(source == read_json(root / f"source_decision_models/{name}_audit.json"), f"Source decision API/report mismatch: {name}")
        corpus, backend = name.rsplit("_", 1)
        row = source_rows[(name,)]
        require(row["world_ids"] == "|".join(source["model"]["world_ids"])
                and row["common_optimal_actions"] == "|".join(source["common_optimal_actions"])
                and source["model"]["actions"] == contract["actions"], f"Source certificate action/world mismatch: {name}")
        require(row["exact_decision_radius"] == source["decision_radius"]["exact"] and row["target_regret_guarantee"] == "False", f"Source certificate summary mismatch: {name}")
        near(row["decision_radius"], source["decision_radius"]["value"], name + "/source radius")
        certificates.append({"environment": name, "certified_unchanged": response["counts"]["certified_unchanged"],
                             "rows": response["counts"]["rows"], "source_decision_radius_exact": source["decision_radius"]["exact"]})
    return {
        "schema_version": 1, "status": "replayed", "milestone_commit": manifest["commit"],
        "manifest_sha256": sha256(manifest_path), "input_files_verified": len(manifest["files"]),
        "calculation_code_sha256": {"replay.py": sha256(Path(__file__)),
                                    "searchprobe.decisions.py": sha256(Path(sys.modules["searchprobe.decisions"].__file__)),
                                    "searchprobe.responsiveness.py": sha256(Path(sys.modules["searchprobe.responsiveness"].__file__))},
        "scope": "Analysis replay of recorded per-action/per-query outcomes and supplied finite-model certificates. This is not retrieval or training regeneration.",
        "uses_recorded_target_outcomes": True, "uses_target_outcomes_for_fitting_or_selection": False,
        "counts": {"environments": len(names), "query_backend_rows": sum(contract["query_counts"].values()),
                   "main_run_rows_recomputed": len(main_rows), "followup_run_rows_recomputed": len(follow_rows),
                   "followup_baseline_rows_recomputed": len(baseline_rows), "paired_contrast_means_recomputed": contrast_count,
                   "response_certificates_recomputed": len(certificates), "source_decision_certificates_recomputed": len(certificates)},
        "environments": summaries, "certificates": certificates,
        "not_replayed": ["Document retrieval or relevance-metric calculation from ranked documents/qrels.",
                         "LLM generation, source fitting, acquisition decisions, or posterior updates.",
                         "Bootstrap sampling and interval endpoints; stored endpoints are hash-checked and checked only for valid ordering/range.",
                         "Full deployment costs, wall time, or the original report builder; these require additional ignored caches or runtime context.",
                         "Whether declared score bounds cover all floating-point router errors, source-world coverage, or target generalization."],
        "numerical_comparison": f"Recorded float summaries checked with absolute tolerance {TOLERANCE}; means use math.fsum. SearchProbe certificate arithmetic is exact for supplied rational/decimal inputs.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path("searchprobe-evidence"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download", action="store_true", help="Fetch missing checksum-pinned evidence files from GitHub; never overwrite differing files")
    parser.add_argument("--output", default="-", help="JSON report path, or - for stdout")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        protected = [args.manifest, *[args.evidence_dir / name for name in manifest["files"]]]
        if args.output != "-":
            output = Path(args.output)
            require(all(output.resolve() != path.resolve() and not (output.exists() and path.exists() and output.samefile(path)) for path in protected), "Output must not overwrite evidence or its manifest")
        if args.download:
            download_evidence(args.evidence_dir, manifest)
        report = replay(args.evidence_dir, args.manifest)
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output == "-":
            sys.stdout.write(rendered)
        else:
            Path(args.output).write_text(rendered, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(f"Replayed {report['counts']['main_run_rows_recomputed']} original and {report['counts']['followup_run_rows_recomputed']} follow-up run means; 12 exact supplied-model certificates. No retrieval, training, or bootstrap rerun.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
