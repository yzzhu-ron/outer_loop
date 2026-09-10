#!/usr/bin/env python3
"""Render the frozen development results without target-selected plot slices.

Outputs are new v1 figures and a manifest containing the plotted numbers,
selection rules, input hashes, query counts, and image alt text. Optional
query-transfer results are summarized in the manifest when available; they do
not silently change the three prespecified figure comparisons.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile


ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = (
    "scifact_bm25", "scifact_dense", "fiqa_bm25", "fiqa_dense",
    "nfcorpus_bm25", "nfcorpus_dense",
)
ENV_LABELS = {
    "scifact_bm25": "SciFact · BM25", "scifact_dense": "SciFact · dense",
    "fiqa_bm25": "FiQA · BM25", "fiqa_dense": "FiQA · dense",
    "nfcorpus_bm25": "NFCorpus · BM25", "nfcorpus_dense": "NFCorpus · dense",
}
METHODS = (
    ("no_probe", "No probes", "#A0A0A0"),
    ("environment_ig", "Environment information gain", "#E69F00"),
    ("decision_region_entropy", "Decision-region entropy", "#B8924A"),
    ("decision_myopic", "Myopic decision value", "#D55E00"),
    ("source_fixed", "Best source-fixed probe plan", "#009E73"),
    ("decision_lookahead_2", "Two-step decision lookahead", "#0072B2"),
    ("decision_full_horizon", "Full-horizon decision planning", "#334D68"),
)
INPUTS = (
    "controlled/results.v1/table.json", "controlled/results.v1/summary.json",
    "controlled/protocol.v1.json", "controlled/freeze.v1.json",
    "natural/results/runs.json", "natural/results/baselines.json",
    "natural/results/paired.json", "natural/results/metadata.json",
    "natural/protocol.json", "natural/capacity_audit.json",
    "fusion_frontier/results.json", "fusion_frontier/protocol.json",
)
QUERY_TRANSFER_INPUTS = (
    "query_transfer/protocol.v1.json", "query_transfer/freeze.v1.json",
    "query_transfer/results.v1/summary.json", "query_transfer/results.v1/primary_results.json",
    "query_transfer/results.v1/all_results.json", "query_transfer/results.v1/source_calibration_grid.json",
    "query_transfer/results.v1/conditional_intervals.json", "query_transfer/results.v1/paired.json",
    "query_transfer/results.v1/selections.json", "query_transfer/results.v1/reconstruction_audit.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite plotted value: {value!r}")
    return result


def one(rows: list[dict], **conditions) -> dict:
    matches = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {conditions}; found {len(matches)}")
    return matches[0]


def build_data(inputs: dict) -> dict:
    table = inputs["controlled/results.v1/table.json"]
    controlled = []
    for scenario in ("xor", "gated"):
        for method, label, _ in METHODS:
            row = one(table, scenario=scenario, method=method, budget=2, control="matched",
                      source_weight="3/4", target_weight="3/4")
            utility = finite(row["expected_utility"])
            assert math.isclose(utility, float(Fraction(row["expected_utility_exact"]))), row
            controlled.append({
                "scenario": scenario, "method": method, "label": label,
                "utility": utility, "utility_exact": row["expected_utility_exact"],
                "expected_operations": finite(row["expected_operations"]),
            })

    runs = inputs["natural/results/runs.json"]
    baselines = inputs["natural/results/baselines.json"]
    paired = inputs["natural/results/paired.json"]
    capacity = inputs["natural/capacity_audit.json"]["rows"]
    natural = []
    for environment in ENVIRONMENTS:
        selected = [row for row in runs if row["environment"] == environment
                    and row["condition"] == "standard" and row["method"] == "lookahead2"
                    and row["budget"] == 64]
        if sorted(str(row["seed"]) for row in selected) != ["11", "23", "47"]:
            raise ValueError(f"Expected three unique primary seeds for {environment}")
        reference = {
            key: finite(one(baselines, environment=environment, baseline=key)["ndcg"])
            for key in ("source_query_bucket", "original", "rrf_all_actions")
        }
        audit = one(capacity, environment=environment)
        lower = finite(audit["verified_witness_lower_bound"]["ndcg"])
        upper = finite(audit["relaxed_upper_bound"]["ndcg"])
        if lower > upper + 1e-7:
            raise ValueError(f"Reversed capacity bracket for {environment}")
        query_ids = paired[environment]["query_ids"]
        n = one(baselines, environment=environment, baseline="original")["n"]
        if len(query_ids) != n or len(set(query_ids)) != n:
            raise ValueError(f"Query-count mismatch for {environment}")
        natural.append({
            "environment": environment, "n_queries": n, **reference,
            "lookahead64": mean(finite(row["ndcg"]) for row in selected),
            "lookahead_seed_values": {str(row["seed"]): finite(row["ndcg"]) for row in selected},
            "capacity_lower": lower, "capacity_upper": upper,
        })
    family_counts = {}
    for family in ("scifact", "fiqa", "nfcorpus"):
        sparse = paired[f"{family}_bm25"]["query_ids"]
        dense = paired[f"{family}_dense"]["query_ids"]
        if sparse != dense:
            raise ValueError(f"Backend query reuse differs for {family}")
        family_counts[family] = len(sparse)

    fusion_input = inputs["fusion_frontier/results.json"]
    fusion_rows = fusion_input["rows"]
    fusion = []
    for backend_known in (True, False):
        for generation_allowed in (False, True):
            for cap in range(1, 7):
                selected = [row for row in fusion_rows if row["backend_known"] == backend_known
                            and row["generation_allowed"] == generation_allowed
                            and row["maximum_search_calls"] == cap]
                if sorted(row["environment"] for row in selected) != sorted(ENVIRONMENTS):
                    raise ValueError("Fusion macro cell must contain exactly all six environments")
                fusion.append({
                    "backend_known": backend_known, "generation_allowed": generation_allowed,
                    "maximum_search_calls": cap,
                    "actual_mean_search_calls": mean(finite(row["per_task_cost"]["search_calls"])
                                                     for row in selected),
                    "macro_ndcg": mean(finite(row["ndcg"]) for row in selected),
                    "actual_mean_llm_calls": mean(finite(row["per_task_cost"]["llm_calls"])
                                                  for row in selected),
                    "actual_mean_input_tokens": mean(finite(row["per_task_cost"]["input_tokens"])
                                                     for row in selected),
                    "actual_mean_output_tokens": mean(finite(row["per_task_cost"]["output_tokens"])
                                                      for row in selected),
                    "environments": len(selected),
                })
    for environment in ENVIRONMENTS:
        if fusion_input["paired"][environment]["query_ids"] != paired[environment]["query_ids"]:
            raise ValueError(f"Fusion/natural query identities differ for {environment}")

    return {
        "controlled": controlled, "natural": natural, "fusion": fusion,
        "rrf_all_macro": mean(row["rrf_all_actions"] for row in natural),
        "counts": {
            "controlled_total_result_rows": len(table), "controlled_plotted_rows": len(controlled),
            "natural_total_run_rows": len(runs), "natural_baseline_rows": len(baselines),
            "natural_primary_run_rows": 18, "capacity_audit_rows": len(capacity),
            "fusion_result_rows": len(fusion_rows), "fusion_macro_cells": len(fusion),
            "development_families": 3, "backends_per_family": 2,
            "queries_by_family": family_counts,
            "family_specific_query_identities": sum(family_counts.values()),
            "query_backend_evaluations_per_method": sum(row["n_queries"] for row in natural),
        },
    }


def axes_style(ax, *, axis="x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B5B5B5")
    ax.tick_params(color="#A0A0A0", length=3)
    ax.grid(axis=axis, color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)


def summarize_query_transfer(inputs: dict) -> dict | None:
    summary_key = "query_transfer/results.v1/summary.json"
    if summary_key not in inputs:
        return None
    primary = inputs["query_transfer/results.v1/primary_results.json"]
    paired = inputs["query_transfer/results.v1/paired.json"]
    for environment in ENVIRONMENTS:
        if paired[environment]["query_ids"] != inputs["natural/results/paired.json"][environment]["query_ids"]:
            raise ValueError(f"Query-transfer identities differ for {environment}")
    macros = []
    for method in sorted({row["method"] for row in primary}):
        rows = [row for row in primary if row["method"] == method]
        if sorted(row["environment"] for row in rows) != sorted(ENVIRONMENTS):
            raise ValueError(f"Incomplete query-transfer primary method: {method}")
        for row in rows:
            if row["n_queries"] != len(paired[row["environment"]]["query_ids"]):
                raise ValueError("Query-transfer row count disagrees with paired identities")
        ndcg = mean(finite(row["ndcg"]) for row in rows)
        archived = one(inputs[summary_key]["family_macro"], method=method)
        if not math.isclose(ndcg, archived["equal_family_backend_macro_ndcg"], abs_tol=1e-12):
            raise ValueError(f"Query-transfer summary disagrees with primary rows for {method}")
        costs = {}
        for key, output_key in (("search_calls", "mean_search_calls"), ("llm_calls", "mean_llm_calls"),
                                ("query_encoder_calls_per_task", "mean_query_encoder_calls")):
            if all(key in row for row in rows):
                costs[output_key] = mean(finite(row[key]) for row in rows)
            elif "oracle" in method:
                costs[output_key] = None
            else:
                raise ValueError(f"Missing deployment cost for query-transfer method {method}: {key}")
        macros.append({"method": method, "macro_ndcg": ndcg, **costs})
    return {
        "stage": inputs[summary_key]["stage"],
        "selection": inputs[summary_key]["selection"],
        "backend_contract": "Backend identity is known; this differs from the natural acquisition experiment.",
        "primary_rows": len(primary),
        "all_result_rows": len(inputs["query_transfer/results.v1/all_results.json"]),
        "source_calibration_rows": len(inputs["query_transfer/results.v1/source_calibration_grid.json"]),
        "conditional_interval_rows": len(inputs["query_transfer/results.v1/conditional_intervals.json"]),
        "family_macro": macros,
        "all_existing_router_actions_reproduced": inputs[summary_key]["all_existing_router_actions_reproduced"],
        "reconstruction_audit": inputs["query_transfer/results.v1/reconstruction_audit.json"],
        "selected_models": inputs["query_transfer/results.v1/selections.json"],
        "display_rule": "Reported as a separate development diagnostic; no additional figure or macro CI.",
    }


def save(fig, plt, output: Path, name: str) -> list[str]:
    paths = []
    plt.rcParams["svg.hashsalt"] = name
    for extension in ("svg", "png"):
        path = output / f"{name}.{extension}"
        metadata = {"Creator": "make_publication_figures.py", "Date": None} if extension == "svg" else None
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight", metadata=metadata)
        paths.append(path.name)
    plt.close(fig)
    return paths


def controlled_figure(plt, data: dict, output: Path) -> dict:
    rows = data["controlled"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    titles = {"xor": "XOR · a fixed probe pair suffices", "gated": "Gated · the second probe is adaptive"}
    for ax, scenario in zip(axes, ("xor", "gated")):
        for index, (method, _, color) in enumerate(METHODS):
            row = one(rows, scenario=scenario, method=method)
            ax.barh(index, row["utility"], height=0.68, color=color, zorder=2)
            ax.text(row["utility"] + 0.016, index, f"{row['utility']:.4f}".rstrip("0").rstrip("."),
                    va="center", fontsize=10)
        ax.set_title(titles[scenario], loc="left", fontsize=11.5, pad=14)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel("Exact expected task utility")
        ax.axvline(.5, color="#666666", linewidth=.8, linestyle=(0, (3, 3)), zorder=1)
        axes_style(ax)
    axes[0].set_yticks(range(len(METHODS)), [label for _, label, _ in METHODS])
    axes[0].invert_yaxis()
    fig.suptitle("Decision-relevant planning in two controlled mechanisms", x=.025, ha="left", y=.99)
    fig.text(.025, .915, "Matched source/target workload weight = 0.75 · maximum probe budget = 2",
             color="#555555", fontsize=10.5)
    fig.text(.025, .025,
             "Exact expectations from finite constructed worlds; no sampling intervals. The no-probe policy spends zero operations.\n"
             "Lookahead equals the best fixed plan in XOR (0.875), and exceeds it in the gated mechanism (0.875 versus 0.750).",
             fontsize=9, color="#555555", va="bottom")
    fig.subplots_adjust(left=.28, right=.985, top=.80, bottom=.20, wspace=.13)
    alt = (
        "Two horizontal bar charts compare seven methods at probe budget two and workload weight three quarters. "
        "In both constructed mechanisms, no probes score 0.5, information gain 0.5625, decision-region entropy 0.625, "
        "and myopic value 0.75. Two-step lookahead and full-horizon planning score 0.875 in both. "
        "The best source-fixed plan scores 0.875 for XOR and 0.75 for the gated mechanism. "
        "These are exact expected utilities, not empirical confidence intervals."
    )
    return {"files": save(fig, plt, output, "controlled_mechanism_v1"), "alt_text": alt,
            "selection": "control=matched; budget=2; source_weight=target_weight=3/4; seven named methods; both scenarios",
            "plotted_rows": rows}


def natural_figure(plt, Line2D, data: dict, output: Path) -> dict:
    rows = data["natural"]
    fig, ax = plt.subplots(figsize=(11.5, 6.9))
    series = (
        ("source_query_bucket", "Source prior", "#777777", "o", -.25),
        ("original", "Original query", "#222222", "D", -.125),
        ("lookahead64", "Lookahead, budget 64 · seed mean", "#0072B2", "o", 0),
        ("rrf_all_actions", "RRF, all five actions", "#009E73", "^", .125),
    )
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        ax.axhline(y, color="#EEEEEE", linewidth=.7, zorder=0)
        for key, _, color, marker, offset in series:
            ax.scatter(row[key], y + offset, color=color, marker=marker,
                       s=49 if key == "lookahead64" else 34, zorder=3,
                       edgecolors="white", linewidths=.5)
        # These are feasible-policy lower witnesses and relaxed capacity upper
        # bounds, not standard errors or confidence intervals.
        lo, hi = row["capacity_lower"], row["capacity_upper"]
        bracket_y = y + .27
        ax.hlines(bracket_y, lo, hi, color="#D55E00", linewidth=2, zorder=3)
        ax.vlines([lo, hi], bracket_y-.045, bracket_y+.045, color="#D55E00", linewidth=1.8, zorder=3)
        delta = row["lookahead64"] - row["source_query_bucket"]
        ax.text(1.035, y, f"{delta:+.4f}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=10, color="#333333")
    ax.text(1.025, 1.04, "Δ vs. prior", transform=ax.transAxes, fontsize=10, color="#555555")
    ax.set_yticks(range(len(rows)-1, -1, -1),
                  [f"{ENV_LABELS[row['environment']]}\n{row['n_queries']} queries" for row in rows])
    ax.set_xlim(.19, .71)
    ax.set_ylim(-.5, len(rows)-.5)
    ax.set_xlabel("Task nDCG@10")
    axes_style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", label=label)
               for _, label, color, marker, _ in series]
    handles.append(Line2D([], [], color="#D55E00", marker="|", linewidth=2,
                          label="Target-label capacity bracket · not a CI"))
    fig.suptitle("Natural retrieval: probe acquisition and router capacity", x=.025, ha="left", y=.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .925),
               ncol=2, frameon=False, fontsize=9.5, columnspacing=2)
    fig.text(.025, .023,
             "Three development families × two backends: 365 family-specific queries, reused in 730 query–backend evaluations.\n"
             "Lookahead averages three probe seeds. Orange brackets use target labels and floating LP bounds; they are not deployable policies.\n"
             "Quality comparison only: onboarding and serving costs differ. Seeds and backends share queries; no macro confidence interval is shown.",
             fontsize=8.7, color="#555555", va="bottom")
    fig.subplots_adjust(left=.20, right=.875, top=.735, bottom=.19)
    changes = "; ".join(f"{ENV_LABELS[row['environment']]} {row['lookahead64']-row['source_query_bucket']:+.4f}"
                        for row in rows)
    alt = (
        "A six-environment dot plot compares the source prior, original query, primary two-step lookahead at budget 64, "
        "all-action reciprocal-rank fusion, and a target-label router-capacity bracket. "
        f"Lookahead minus prior: {changes}. RRF exceeds lookahead in all six environments and exceeds the capacity upper bound "
        "in five; NFCorpus dense is the exception. The SciFact BM25 capacity bracket is 0.61846 to 0.62445; "
        "the other five brackets collapse to a verified value within numerical tolerance. "
        "The plots reuse 365 queries across two backends and report descriptive seed means, with no macro confidence interval."
    )
    return {"files": save(fig, plt, output, "natural_capacity_v1"), "alt_text": alt,
            "selection": "all six environments; condition=standard; method=lookahead2; budget=64; seeds11,23,47",
            "plotted_rows": rows}


def fusion_figure(plt, Line2D, data: dict, output: Path) -> dict:
    rows = data["fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7), sharex=True, sharey=True)
    for ax, known in zip(axes, (True, False)):
        for allowed, color, marker in ((True, "#0072B2", "o"), (False, "#D55E00", "s")):
            selected = [row for row in rows if row["backend_known"] == known
                        and row["generation_allowed"] == allowed]
            points: dict[tuple[float, float], list[int]] = defaultdict(list)
            for row in selected:
                points[row["actual_mean_search_calls"], row["macro_ndcg"]].append(row["maximum_search_calls"])
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    color=color, marker=marker, linewidth=1.7, markersize=5,
                    markeredgecolor="white", markeredgewidth=.5)
            if allowed:
                for (x, y), caps in points.items():
                    label = ",".join(map(str, caps))
                    # Cap labels identify chosen configurations; x is measured
                    # mean serving calls, never the nominal cap itself.
                    ax.annotate(label, (x, y), xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=8.5, color=color)
        ax.scatter(6, data["rrf_all_macro"], marker="*", s=120, color="#009E73",
                   edgecolors="white", linewidths=.5, zorder=4)
        ax.set_title("Backend identity known" if known else "Backend identity hidden",
                     loc="left", fontsize=11.5, pad=13)
        ax.set(xlim=(.75, 6.3), ylim=(.379, .426))
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xlabel("Actual mean search calls per task")
        axes_style(ax, axis="y")
    axes[0].set_ylabel("Equal-family/backend macro nDCG@10")
    handles = [
        Line2D([], [], color="#0072B2", marker="o", label="Source-selected subset · generation allowed"),
        Line2D([], [], color="#D55E00", marker="s", label="Source-selected subset · no generation"),
        Line2D([], [], color="#009E73", marker="*", linestyle="none", markersize=10,
               label="RRF over all five actions · six searches"),
    ]
    fig.suptitle("Source-selected fusion: quality versus serving searches", x=.025, ha="left", y=.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .93),
               ncol=1, frameon=False, fontsize=9.5)
    fig.text(.025, .023,
             "Three development families × two correlated backends; equal means across six environments, not pooled-query means.\n"
             "Blue point labels are allowed search caps 1–6; positions use actual mean calls. Lines connect measured source selections, not a target-selected frontier.\n"
             "Search cost omits generation and tokens (reported in the manifest). These methods do not acquire probes; no macro confidence interval is shown.",
             fontsize=8.7, color="#555555", va="bottom")
    fig.subplots_adjust(left=.085, right=.985, top=.70, bottom=.23, wspace=.16)
    alt = (
        "Two line charts show equal-family/backend mean retrieval quality against actual mean search calls per task, "
        "with backend identity known or hidden. With generation allowed and the six-call cap, known-backend source selection "
        "achieves 0.41402 nDCG with 4.333 mean searches; hidden-backend selection achieves 0.41285 with 4.667. "
        "All-action RRF achieves 0.42118 with six searches. Source selection without generation uses at most 1.333 mean searches "
        "and stays near 0.39 nDCG. Curves connect the recorded source-selected configurations, with no target Pareto filtering "
        "or macro confidence intervals. Generation and token costs are separate from the plotted search axis."
    )
    return {"files": save(fig, plt, output, "fusion_frontier_v1"), "alt_text": alt,
            "selection": "all144 environment configurations; macro by backend_known,generation_allowed,maximum_search_calls",
            "plotted_macro_cells": rows, "rrf_all_macro": data["rrf_all_macro"]}


def markdown_summary(manifest: dict) -> str:
    counts = manifest["counts"]
    lines = ["# Full-paper figure notes", "", "These figures describe controlled mechanisms and development experiments; they do not establish untouched-family transfer.", "",
             f"Counts: {counts['controlled_total_result_rows']:,} controlled result rows; 14 plotted canonical rows. "
             f"Natural retrieval has {counts['natural_total_run_rows']:,} run rows, 60 baseline rows, and 18 primary lookahead runs. "
             "Fusion has 144 environment configurations, displayed as 24 equal-environment macro cells.", "",
             "SciFact and FiQA each contribute 150 query identities; NFCorpus contributes 65. "
             "The same queries run against two backends: 365 family-specific identities and 730 query–backend evaluations per method. "
             "Seeds and backend configurations are correlated; no macro confidence intervals are plotted.", ""]
    for name, item in manifest["figures"].items():
        lines += [f"## {name}", "", f"Files: {', '.join(item['files'])}.", "", item["alt_text"], ""]
    query = manifest.get("query_transfer")
    if query:
        scores = {row["method"]: row["macro_ndcg"] for row in query["family_macro"]}
        lines += ["## Query-only transfer diagnostic", "",
                  f"The separate known-backend query-transfer run contains {query['primary_rows']} primary rows, "
                  f"{query['all_result_rows']} total result rows, and {query['source_calibration_rows']} source calibration rows. "
                  "It uses the same 365 queries/730 query–backend evaluations. Candidate selection holds out source queries "
                  "within known source corpora; it is not held-out-family validation.", "",
                  f"Equal-family/backend macro nDCG@10: source-selected query router {scores['source_selected_query']:.5f}; "
                  f"source-selected kNN {scores['source_selected_knn']:.5f}; reconstructed existing router "
                  f"{scores['reconstructed_existing_router']:.5f}; original {scores['original']:.5f}; "
                  f"all-action RRF {scores['rrf_all_actions']:.5f}. "
                  "The reconstructed router reproduces every archived action. These known-backend scores are a separate "
                  "diagnostic from the hidden-backend natural acquisition comparison.", ""]
    lines += ["## Provenance", "", "Exact input hashes, plotted values, selection rules, and generation/token costs are in `publication_figure_manifest.v1.json`. "
              "The capacity audit is a post-hoc target-label diagnostic: floating-LP upper bounds and verified deterministic-policy lower witnesses, not statistical intervals.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    inputs = {name: json.loads((ROOT / name).read_text()) for name in INPUTS}
    if (ROOT / "query_transfer/results.v1/summary.json").exists():
        inputs.update({name: json.loads((ROOT / name).read_text()) for name in QUERY_TRANSFER_INPUTS})
    data = build_data(inputs)
    query_transfer = summarize_query_transfer(inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="searchprobe-figures-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10,
            "figure.titlesize": 15, "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5, "ytick.labelsize": 10,
            "text.color": "#222222", "axes.labelcolor": "#333333",
            "svg.fonttype": "none",
        })
        figures = {
            "Controlled mechanisms": controlled_figure(plt, data, args.output),
            "Natural retrieval and capacity": natural_figure(plt, Line2D, data, args.output),
            "Fusion quality and serving searches": fusion_figure(plt, Line2D, data, args.output),
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Descriptive development figures. No new retrieval, no target-selected figure slices, no population-level intervals.",
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(ROOT / name) for name in inputs},
        "counts": data["counts"], "figures": figures,
        "query_transfer": query_transfer,
        "output_sha256": {filename: sha256(args.output / filename)
                          for figure in figures.values() for filename in figure["files"]},
    }
    manifest_path = args.output / "publication_figure_manifest.v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "FIGURE_SUMMARIES.v1.md").write_text(markdown_summary(manifest))
    print(json.dumps({"output_directory": str(args.output), "figure_files": len(manifest["output_sha256"]),
                      "counts": data["counts"]}, indent=2))


if __name__ == "__main__":
    main()
