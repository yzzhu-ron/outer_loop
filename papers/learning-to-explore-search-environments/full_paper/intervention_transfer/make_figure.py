#!/usr/bin/env python3
"""Plot the sealed intervention-transfer result; never rerun retrieval or fitting.

Uses all registered families, lambda settings, and target panels. The figure is
an outcome-informed descriptive presentation, not a new inferential analysis.
New output names leave all frozen study and earlier figure artifacts untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile


ROOT = Path(__file__).resolve().parent
FAMILIES = ("fiqa", "nfcorpus", "scifact")
LABELS = {"fiqa": "FiQA", "nfcorpus": "NFCorpus", "scifact": "SciFact"}
STEM = "intervention_transfer_v1"
INPUT_NAMES = ("results.v1.json", "protocol.v1.json", "freeze.v1.json",
               "inputs.v1.json", "decisions.v1.json", "test_labels.v1.json")
BLUE = "#0072B2"
ORANGE = "#C77900"
GRAY = "#606B73"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Figure consistency check failed: {a} != {b}")


def build_data():
    hashes = {name: sha256(ROOT / name) for name in INPUT_NAMES}
    files = {name: json.loads((ROOT / name).read_text()) for name in INPUT_NAMES}
    result = files["results.v1.json"]
    inputs = files["inputs.v1.json"]
    decisions = files["decisions.v1.json"]
    labels = files["test_labels.v1.json"]
    protocol = files["protocol.v1.json"]
    for relative, expected in files["freeze.v1.json"]["files"].items():
        if sha256(ROOT.parents[1] / relative) != expected:
            raise ValueError("Frozen implementation mismatch: " + relative)
    links = {"decisions": "decisions.v1.json", "freeze": "freeze.v1.json",
             "inputs": "inputs.v1.json", "labels": "test_labels.v1.json"}
    for key, filename in links.items():
        if result["provenance"][key] != hashes[filename]:
            raise ValueError("Results provenance mismatch: " + filename)
    if decisions["inputs_sha256"] != hashes["inputs.v1.json"] or decisions["freeze_sha256"] != hashes["freeze.v1.json"]:
        raise ValueError("Decision provenance mismatch")
    if labels["decisions_sha256"] != hashes["decisions.v1.json"]:
        raise ValueError("Test-label lock mismatch")
    if sorted(result["families"]) != sorted(FAMILIES) or len(protocol["lambda_grid"]) != 7:
        raise ValueError("Expected all three registered families and seven settings")
    plotted = {
        "lambda_grid": protocol["lambda_grid"],
        "scope": "Three repeatedly explored development families; seven engineered hybrid settings within each family.",
        "selection": "All registered families, weights, and target panels; no outcome-based omission. Presentation chosen after outcomes.",
        "metric": "nDCG@10; equal query mean within family, then equal lambda/panel/family means.",
        "difference_reference": "Source-fixed160, selected from 128 training plus 32 calibration queries per source family.",
        "oracle_definition": "Evaluator-only: best of the eleven cap2 policies for each family/lambda, averaged over the entire target query set; not per-query routing.",
        "uncertainty": protocol["evaluation"]["uncertainty"],
        "families": {},
    }
    for family in FAMILIES:
        row = result["families"][family]
        qids = inputs["corpora"][family]["test_meta"]["ids"]
        if len(set(qids)) != len(qids):
            raise ValueError("Duplicate target query identities")
        for j in range(7):
            if labels["corpora"][family]["worlds"][str(j)]["ids"] != qids:
                raise ValueError("Target query identities must be reused across lambda")
        cap = [x for x in result["fixed_policy_frontier"][family] if x["policy_index"] < 11]
        if len(cap) != 11:
            raise ValueError("Expected complete cap2 policy menu")
        fixed = list(row["source_fixed_train_cal"]["by_lambda"])
        learned = list(row["within"]["by_lambda"])
        oracle = [max(x["ndcg_by_lambda"][j] for x in cap) for j in range(7)]
        best_fixed = max(cap, key=lambda x: x["ndcg"])
        same(mean(fixed), row["source_fixed_train_cal"]["ndcg"])
        same(mean(learned), row["within"]["ndcg"])
        same(mean(oracle), row["oracle_best_by_lambda"])
        same(best_fixed["ndcg"], row["oracle_best_fixed"])
        same(best_fixed["ndcg"], mean(fixed))
        headroom = mean(oracle) - best_fixed["ndcg"]
        same(headroom, row["lambda_adaptation_headroom"])
        learned_delta = [a - b for a, b in zip(learned, fixed)]
        interval = result["intervals"]["by_family_train_cal"][family]
        same(mean(learned_delta), interval["mean"])
        costs = {method: result["costs"][family][method]
                 for method in ("within", "source_fixed_train_cal", "rrf_all")}
        for method in ("within", "source_fixed_train_cal"):
            serving = costs[method]["serving_per_query"]
            if not (1 <= serving["logical_search_calls"] <= 2 and 0 <= serving["llm_calls"] <= 1):
                raise ValueError("Unexpected cap2 serving cost")
            same(serving["underlying_search_calls"], 2 * serving["logical_search_calls"])
        plotted["families"][family] = {
            "label": LABELS[family], "n_queries": len(qids), "n_panels": 3,
            "eta": decisions["folds"][family]["models"]["within"]["eta"],
            "source_fixed160_ndcg_by_lambda": fixed,
            "within_ndcg_by_lambda": learned,
            "oracle_per_lambda_ndcg": oracle,
            "within_minus_source_fixed160_by_lambda": learned_delta,
            "oracle_minus_source_fixed160_by_lambda": [max(0.0, a - b) for a, b in zip(oracle, fixed)],
            "oracle_best_fixed_policy": best_fixed["actions"],
            "source_fixed160_is_target_best_fixed": True,
            "oracle_adaptation_headroom": headroom,
            "within_minus_source_fixed160_interval": interval,
            "means": {method: row[method]["ndcg"] for method in ("within", "source_fixed_train_cal", "rrf_all")},
            "oracle_mean": mean(oracle), "costs": costs,
        }
    plotted["macro"] = {
        **{method: result["macro"][method] for method in ("within", "source_fixed_train_cal", "rrf_all")},
        "oracle_per_lambda": mean(row["oracle_mean"] for row in plotted["families"].values()),
        "oracle_adaptation_headroom": mean(row["oracle_adaptation_headroom"] for row in plotted["families"].values()),
        "headroom_scope": "Finite-benchmark upper opportunity for choosing one global cap2 policy per setting over the fixed menu, not a future-population guarantee or a bound on query routing.",
        "within_minus_source_fixed160_interval": result["intervals"]["within_minus_source_fixed_train_cal"],
        "registered_advancement_gain_threshold": result["advancement_gate"]["macro_gain_at_least"],
        "advancement_gate_passed": result["advancement_gate"]["passed"],
    }
    same(plotted["macro"]["within"] - plotted["macro"]["source_fixed_train_cal"],
         plotted["macro"]["within_minus_source_fixed160_interval"]["mean"])
    return plotted, hashes


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BDC4C8")
    ax.tick_params(color="#A4ADB2", length=3)
    ax.set_axisbelow(True)


def render(plt, Line2D, data, output):
    fig = plt.figure(figsize=(13.1, 8.8), facecolor="white")
    fig.text(.065, .965, "Little adaptation headroom; no mean gain from the learned bridge",
             fontsize=17, fontweight="bold", va="top")
    fig.text(.065, .923, "Three development families  ·  Seven engineered hybrid settings per family  ·  Two-wrapper serving cap",
             color="#53606A", fontsize=11.2)
    handles = [Line2D([0], [0], color=BLUE, marker="o", lw=2, markersize=5, label="Learned within-corpus bridge"),
               Line2D([0], [0], color=ORANGE, marker="^", lw=1.8, ls=(0, (4, 2)), markersize=5,
                      label="Target-label oracle per setting (cap 2)"),
               Line2D([0], [0], color=GRAY, lw=1.3, label="Source-fixed160 (zero reference)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .90), frameon=False,
               ncol=3, fontsize=10, columnspacing=2.3, handlelength=2.7)
    xs = data["lambda_grid"]
    for i, family in enumerate(FAMILIES):
        row = data["families"][family]
        ax = fig.add_axes([.075 + i * .305, .525, .268, .29])
        axis_style(ax)
        ax.axhline(0, color=GRAY, lw=1.15, zorder=1)
        ax.plot(xs, row["oracle_minus_source_fixed160_by_lambda"], color=ORANGE,
                marker="^", lw=1.8, ls=(0, (4, 2)), markersize=5.3, zorder=3)
        ax.plot(xs, row["within_minus_source_fixed160_by_lambda"], color=BLUE,
                marker="o", lw=2, markersize=4.2, zorder=4)
        ax.set_xlim(-.035, 1.035)
        ax.set_ylim(-.0066, .0100)
        ax.set_xticks(xs, ["0\nBM25", ".125", ".25", ".5", ".75", ".875", "1\ndense"])
        ax.set_yticks([-.005, 0, .005, .010], ["−0.005", "0", "+0.005", "+0.010"])
        ax.grid(axis="y", color="#E5E8EA", lw=.7)
        ax.set_title(f"{'abc'[i]}  {row['label']}  ({row['n_queries']} queries)", loc="left",
                     fontsize=11.8, fontweight="bold", pad=10)
        ax.set_xlabel("Hidden dense weight λ", labelpad=2)
        if i == 0:
            ax.set_ylabel("Δ nDCG@10 vs. source-fixed160", labelpad=8)
        else:
            ax.tick_params(labelleft=False)
        annotation = "η = 0; no onboarding" if row["eta"] == 0 else "η = 1; eight-document panel"
        ax.text(.03, .95, annotation, transform=ax.transAxes, va="top", fontsize=9.4,
                color="#51616D")
        fig.text(.075 + i * .305, .433,
                 f"Mean oracle headroom  +{row['oracle_adaptation_headroom']:.5f}",
                 fontsize=10.2, color=ORANGE, fontweight="bold")

    fig.text(.065, .378, "d  Mean change across all seven settings", fontsize=11.7, fontweight="bold")
    ax = fig.add_axes([.15, .16, .39, .18])
    axis_style(ax)
    ax.axvline(0, color=GRAY, lw=1.15)
    rows = [(data["families"][f]["within_minus_source_fixed160_interval"],
             data["families"][f]["oracle_adaptation_headroom"]) for f in FAMILIES]
    rows += [(data["macro"]["within_minus_source_fixed160_interval"], data["macro"]["oracle_adaptation_headroom"])]
    for y, (interval, headroom) in zip((3, 2, 1, 0), rows):
        ax.errorbar(interval["mean"], y + .10,
                    xerr=[[interval["mean"] - interval["lo"]], [interval["hi"] - interval["mean"]]],
                    fmt="o", color=BLUE, markersize=4.9, capsize=3, lw=1.5)
        ax.plot(headroom, y - .12, marker="^", color=ORANGE, markersize=5.8)
    ax.set_yticks([3, 2, 1, 0], ["FiQA", "NFCorpus", "SciFact", "Equal-family mean"])
    ax.set_ylim(-.55, 3.65)
    ax.set_xlim(-.0098, .0071)
    ax.set_xticks([-.005, 0, .005], ["−0.005", "0", "+0.005"])
    ax.grid(axis="x", color="#E5E8EA", lw=.7)
    ax.set_xlabel("Δ nDCG@10 vs. source-fixed160", labelpad=3)
    fig.text(.15, .092, "Blue intervals: 95% paired-query bootstrap; orange: oracle headroom.",
             fontsize=9.0, color="#53606A")

    fig.text(.585, .378, "Equal-family mean nDCG@10", fontsize=11.7, fontweight="bold")
    entries = [
        ("Source-fixed160", "source_fixed_train_cal", GRAY, "2 wrappers / 4 components"),
        ("Learned bridge", "within", BLUE, "≤2 wrappers / ≤4 components"),
        ("Oracle per setting", "oracle_per_lambda", ORANGE, "≤2 wrappers; target labels"),
        ("RRF-all reference", "rrf_all", "#495057", "6 wrappers / 12 components"),
    ]
    for j, (label, key, color, cost) in enumerate(entries):
        y = .338 - j * .031
        fig.text(.585, y, label, fontsize=10, color=color)
        fig.text(.74, y, f"{data['macro'][key]:.6f}", fontsize=10, color=color, fontweight="bold")
        fig.text(.816, y, cost, fontsize=8.7, color="#53606A")
    fig.text(.585, .193,
             "Learned onboarding: FiQA 0; NFCorpus / SciFact 48 wrappers,\n"
             "96 components, 16 generation attempts and 8 samples per panel.\n"
             "Serving uses at most one shared generation bundle per query.",
             fontsize=9.3, color="#53606A", va="top", linespacing=1.5)

    fig.text(.065, .059,
             "The source-fixed160 choice equals each family’s target-best fixed policy. The oracle selects one cap2 policy per setting, using target labels.",
             fontsize=9.1, color="#53606A")
    fig.text(.065, .038,
             "Lines connect tested settings; means weight them equally. Intervals condition on these three families, source fits and panels; settings are not independent domains.",
             fontsize=9.1, color="#53606A")
    fig.text(.065, .017,
             "The +0.002307 macro oracle opportunity applies to these finite queries/settings and the global cap2 menu; it is not a population or query-routing bound.",
             fontsize=9.1, color="#53606A")
    svg = output / (STEM + ".svg")
    png = output / (STEM + ".png")
    fig.savefig(svg, format="svg", metadata={"Creator": "SearchProbe intervention-transfer figure generator"})
    fig.savefig(png, format="png", dpi=190, facecolor="white")
    plt.close(fig)
    return [svg, png]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "figures")
    args = parser.parse_args()
    output = args.output
    names = [STEM + ".svg", STEM + ".png", "intervention_transfer_plotted_data.v1.json",
             "intervention_transfer_figure_manifest.v1.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Figure outputs already exist; choose a new explicit output directory")
    output.mkdir(parents=True, exist_ok=True)
    data, hashes = build_data()
    with tempfile.TemporaryDirectory(prefix="intervention-transfer-figure-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "text.color": "#25313A", "axes.labelcolor": "#34434D",
                             "xtick.labelsize": 9, "ytick.labelsize": 9.4,
                             "svg.fonttype": "none", "svg.hashsalt": STEM})
        figures = render(plt, Line2D, data, output)
    data_path = output / "intervention_transfer_plotted_data.v1.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    alt = (
        "Four-panel scientific figure of the sealed intervention-transfer development test. "
        "Three upper panels show nDCG@10 changes relative to the source-fixed160 cap2 policy "
        "at seven engineered dense-mixture weights from zero to one for FiQA, NFCorpus and SciFact. "
        "Each family reuses its 150, 65 or 150 test queries across all settings and three probe panels. "
        "The source-fixed160 choice is also the target-best single fixed cap2 policy in all three families. "
        "The evaluator-only per-setting oracle has mean adaptation headroom of 0.000562545, "
        "0.003089213 and 0.003270668. The learned bridge ties fixed160 on FiQA, and its mean changes "
        "are minus 0.000680953 on NFCorpus and minus 0.000862434 on SciFact. "
        "The lower panel shows these means with the existing 95 percent paired-query bootstrap intervals "
        "and oracle headroom markers. Macro bridge quality is 0.428980783 versus fixed160 0.429495245; "
        "the paired difference is minus 0.000514462, interval [minus 0.003510523, plus 0.002125033]. "
        "The target-label per-setting cap2 oracle scores 0.431802720. RRF-all scores 0.433357037 "
        "at six wrapper and twelve component searches per query, versus at most two and four for the deployed "
        "cap2 methods. FiQA selects eta zero and incurs no onboarding; the other two families pay "
        "48 wrapper searches, 96 component searches, 16 generation attempts and eight document samples "
        "per target panel. Serving uses at most one shared generation bundle per query; learned NFCorpus "
        "occasionally chooses an ungenerated singleton. The macro oracle headroom of 0.002307475 is a "
        "finite-benchmark upper opportunity for one global cap2 policy per setting, not a future-population "
        "guarantee or a bound on per-query routing. The seven settings are engineered configurations within "
        "three repeatedly explored development families, not seven independent new domains."
    )
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure": STEM, "title": "Little adaptation headroom; no mean gain from the learned bridge",
        "alt_text": alt, "script": "intervention_transfer/make_figure.py",
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {"intervention_transfer/" + name: value for name, value in hashes.items()},
        "path_convention": {"input_sha256": "relative to full_paper", "output_sha256": "relative to figures"},
        "figures": {"intervention": {
            "label": "Intervention adaptation headroom", "alt_text": alt,
            "files": [path.name for path in figures], "selection": data["selection"],
            "plotted_data": data_path.name,
        }},
        "files": [path.name for path in figures] + [data_path.name],
        "output_sha256": {path.name: sha256(path) for path in [*figures, data_path]},
        "design_notes": [
            "All families and lambda settings are shown; a common delta scale removes unrelated between-family levels.",
            "The oracle uses target labels only for evaluation and is limited to one of eleven cap2 policies per setting, not per query.",
            "Figure layout was selected after seeing registered outcomes; no new inference, retrieval or model selection is performed.",
            "Intervals are taken unchanged from the registered evaluator; no lambda or panel bootstrap is added.",
            "Near-zero negative oracle deltas below numerical precision are drawn as zero; absolute inputs retain their original values.",
            "No prior figure manifest or frozen study artifact is modified.",
        ],
    }
    (output / "intervention_transfer_figure_manifest.v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "files": names, "input_hashes_verified": len(hashes),
                      "oracle_macro_headroom": data["macro"]["oracle_adaptation_headroom"]}, indent=2))


if __name__ == "__main__":
    main()
