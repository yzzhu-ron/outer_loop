#!/usr/bin/env python3
"""Render the recorded pilot CSVs as static scientific figures.

Usage:
    python plot_results.py --results results

All plotted values come from the CSVs. Seed means in the adaptation figure are
descriptive; seeds are repeated document samples, not independent environments.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import mean
import tempfile


ENVIRONMENTS = ("scifact_bm25", "scifact_dense", "fiqa_bm25", "fiqa_dense")
ENVIRONMENT_LABELS = {
    "scifact_bm25": "SciFact · BM25",
    "scifact_dense": "SciFact · dense",
    "fiqa_bm25": "FiQA · BM25",
    "fiqa_dense": "FiQA · dense",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
METHODS = {
    "source_router": ("Source router", "#545454", "o", "--"),
    "profile_router": ("Profile router", "#0072B2", "o", "-"),
    "probe_only": ("Probe-only choice", "#D55E00", "s", "-"),
    "rrf_extra": ("Extra task searches (RRF)", "#009E73", "^", "-"),
}
FAMILIES = {
    "exact_title": ("Exact-title control", "o"),
    "body_terms": ("Body-term probes", "^"),
}


def read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    unknown = {row["environment"] for row in rows} - set(ENVIRONMENTS)
    if unknown:
        raise ValueError(f"{path}: unsupported environments {', '.join(sorted(unknown))}")
    return rows


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key} in row: {row}")
    return value


def style_axes(ax, *, both_axes: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B0B0B0")
    ax.grid(axis="both" if both_axes else "y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color="#A0A0A0")


def save_figure(fig, output: Path, name: str) -> None:
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
        print(path)


def plot_headroom(plt, rows: list[dict[str, str]], output: Path) -> None:
    by_environment = {row["environment"]: row for row in rows}
    if len(by_environment) != len(rows):
        raise ValueError("headroom.csv must contain one row per environment")
    series = (
        ("original", "Original query", "#A5A5A5", None),
        ("source_router", "Source router", "#0072B2", None),
        ("target_fixed_oracle", "Best fixed action · target-label oracle", "#E69F00", "//"),
        ("query_oracle", "Best action per query · target-label oracle", "#CC79A7", "xx"),
    )
    fig, ax = plt.subplots(figsize=(10.7, 5.3))
    width = 0.19
    for index, (key, label, color, hatch) in enumerate(series):
        for position, environment in enumerate(ENVIRONMENTS):
            if environment not in by_environment:
                continue
            value = number(by_environment[environment], key)
            bar = ax.bar(
                position + (index - 1.5) * width,
                value,
                width,
                color=color,
                edgecolor="#555555" if hatch else "white",
                linewidth=0.5,
                hatch=hatch,
                label=label if position == next(
                    p for p, e in enumerate(ENVIRONMENTS) if e in by_environment
                ) else None,
            )
            ax.bar_label(bar, fmt="%.3f", padding=3, fontsize=8)
    labels = []
    for environment in ENVIRONMENTS:
        row = by_environment.get(environment)
        count = f"{int(number(row, 'n_queries')):,} queries" if row else "no data"
        labels.append(f"{ENVIRONMENT_LABELS[environment]}\n{count}")
    ax.set_xticks(range(len(ENVIRONMENTS)), labels)
    ax.set_ylabel("Task nDCG@10")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    style_axes(ax)
    fig.suptitle("Action headroom on full released corpora", x=0.07, ha="left", y=0.99)
    fig.legend(loc="upper center", bbox_to_anchor=(0.52, 0.925), ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.07, 0.025,
        "Hatched bars use target relevance labels and are upper-bound diagnostics, not eligible methods.\n"
        "Pilot action library: lexical transformations; each corpus retains its full retrieval candidate set.",
        fontsize=8.5, color="#555555", va="bottom",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.19)
    save_figure(fig, output, "headroom")
    plt.close(fig)


def plot_alignment(plt, Line2D, rows: list[dict[str, str]], output: Path) -> None:
    recorded = rows
    # An unavailable control is encoded with count=0 and an empty score. It is
    # missing evidence, not an observation of zero reciprocal rank.
    rows = [row for row in recorded if number(row, "count") > 0]
    if not rows:
        raise ValueError("alignment.csv has no probes with a positive observation count")
    unavailable_titles = [
        ENVIRONMENT_LABELS[environment]
        for environment in ENVIRONMENTS
        if any(row["environment"] == environment and row["family"] == "exact_title" for row in recorded)
        and not any(row["environment"] == environment and row["family"] == "exact_title" for row in rows)
    ]
    missing_caption = (
        "\nExact-title controls unavailable: " + ", ".join(unavailable_titles) + "."
        if unavailable_titles else ""
    )
    actions = sorted({row["action"] for row in rows})
    if len(actions) > len(COLORS):
        raise ValueError("alignment.csv has more actions than the accessible color palette supports")
    unknown_families = {row["family"] for row in rows} - set(FAMILIES)
    if unknown_families:
        raise ValueError(f"Unknown probe families: {sorted(unknown_families)}")
    action_colors = dict(zip(actions, COLORS))
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 8.1), sharex=True, sharey=True)
    x_limit = max(0.01, max(abs(number(row, "probe_rr_delta")) for row in rows) * 1.12)
    y_limit = max(0.005, max(abs(number(row, "task_ndcg_delta")) for row in rows) * 1.12)
    for ax, environment in zip(axes.flat, ENVIRONMENTS):
        subset = [row for row in rows if row["environment"] == environment]
        for action, color in action_colors.items():
            for family, (_, marker) in FAMILIES.items():
                points = [row for row in subset if row["action"] == action and row["family"] == family]
                if points:
                    ax.scatter(
                        [number(row, "probe_rr_delta") for row in points],
                        [number(row, "task_ndcg_delta") for row in points],
                        s=36, color=color, marker=marker, alpha=0.72,
                        edgecolors="white", linewidths=0.35,
                    )
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set(xlim=(-x_limit, x_limit), ylim=(-y_limit, y_limit))
        ax.set_title(ENVIRONMENT_LABELS[environment], loc="left", fontsize=11)
        if not subset:
            ax.text(0.5, 0.5, "No data recorded", transform=ax.transAxes, ha="center")
        style_axes(ax, both_axes=True)
    for ax in axes[1]:
        ax.set_xlabel("Probe Δ reciprocal rank (vs. original)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Task Δ nDCG@10 (vs. original)")
    action_handles = [
        Line2D([], [], color=color, marker="o", linestyle="none", label=action.replace("_", " "))
        for action, color in action_colors.items()
    ]
    family_handles = [
        Line2D([], [], color="#555555", marker=marker, linestyle="none", label=label)
        for label, marker in FAMILIES.values()
    ]
    fig.suptitle("Do document-rediscovery probes track task utility?", x=0.08, ha="left", y=0.99)
    fig.legend(handles=action_handles, loc="upper center", bbox_to_anchor=(0.53, 0.95),
               ncol=min(4, len(action_handles)), frameon=False, fontsize=9)
    fig.legend(handles=family_handles, loc="upper center", bbox_to_anchor=(0.53, 0.88),
               ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.08, 0.015,
        "Exploratory lexical diagnostic on full corpora. Points are action × probe family × sampling seed.\n"
        "The axes measure different outcomes: known-document rediscovery and task relevance.\n"
        "Repeated seeds share task labels; these points are not independent environment replications."
        + missing_caption,
        fontsize=8.5, color="#555555", va="bottom",
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.79, bottom=0.17, hspace=0.26, wspace=0.14)
    save_figure(fig, output, "alignment")
    plt.close(fig)


def plot_adaptation(plt, Line2D, rows: list[dict[str, str]], output: Path) -> None:
    rows = [row for row in rows if row["family"] == "body_terms" and number(row, "workload") == 50]
    if not rows:
        raise ValueError("adaptation.csv has no body_terms rows for workload 50")
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.7))
    present = set()
    for ax, environment in zip(axes.flat, ENVIRONMENTS):
        subset = [row for row in rows if row["environment"] == environment]
        for method, (_, color, marker, linestyle) in METHODS.items():
            method_rows = [row for row in subset if row["method"] == method]
            by_budget: dict[float, list[dict[str, str]]] = {}
            for row in method_rows:
                by_budget.setdefault(number(row, "budget"), []).append(row)
            points = []
            for budget_rows in by_budget.values():
                # Average within each seed first, so repeated rows cannot give one
                # seed greater weight in the displayed descriptive mean.
                by_seed: dict[str, list[dict[str, str]]] = {}
                for row in budget_rows:
                    by_seed.setdefault(row["seed"], []).append(row)
                points.append((
                    mean(mean(number(row, "total_operations") for row in seed_rows)
                         for seed_rows in by_seed.values()),
                    mean(mean(number(row, "ndcg") for row in seed_rows)
                         for seed_rows in by_seed.values()),
                ))
            points = sorted(set(points))
            if points:
                present.add(method)
                ax.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.6, markersize=5, markeredgecolor="white", markeredgewidth=0.45,
                )
        ax.set_title(ENVIRONMENT_LABELS[environment], loc="left", fontsize=11)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.margins(x=0.08, y=0.2)
        if not subset:
            ax.text(0.5, 0.5, "No data recorded", transform=ax.transAxes, ha="center")
        style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("Total operations (sampling + searches)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Task nDCG@10")
    handles = [
        Line2D([], [], label=label, color=color, marker=marker, linestyle=linestyle)
        for method, (label, color, marker, linestyle) in METHODS.items() if method in present
    ]
    fig.suptitle("Adaptation quality versus operational cost · 50 future tasks", x=0.08, ha="left", y=0.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.935),
               ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.08, 0.015,
        "Body-term probes; full released corpora; descriptive means across document-sampling seeds.\n"
        "Each point is a recorded budget; lines connect measured configurations. Panel y-axis ranges differ.\n"
        "Operations count target sampling and searches, not wall time, token cost, or source training.",
        fontsize=8.5, color="#555555", va="bottom",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.81, bottom=0.18, hspace=0.30, wspace=0.24)
    save_figure(fig, output, "adaptation_cost")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path, help="Figure directory (default: RESULTS/figures)")
    args = parser.parse_args()
    output = args.output or args.results / "figures"
    # Read and check every input schema before rendering.
    headroom = read_rows(args.results / "headroom.csv", (
        "environment", "n_queries", "original", "source_router", "target_fixed_oracle", "query_oracle",
    ))
    alignment = read_rows(args.results / "alignment.csv", (
        "environment", "family", "seed", "action", "probe_rr_delta", "task_ndcg_delta", "count",
    ))
    adaptation = read_rows(args.results / "adaptation.csv", (
        "environment", "family", "seed", "budget", "method", "ndcg", "workload", "total_operations",
    ))
    output.mkdir(parents=True, exist_ok=True)
    # Use a writable, short-lived font cache when no explicit cache was provided.
    with tempfile.TemporaryDirectory(prefix="pilot-matplotlib-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.titlesize": 11, "axes.labelsize": 10,
            "xtick.labelsize": 9, "ytick.labelsize": 9,
            "figure.titlesize": 14, "svg.fonttype": "none",
            "axes.labelcolor": "#333333", "text.color": "#222222",
        })
        plot_headroom(plt, headroom, output)
        plot_alignment(plt, Line2D, alignment, output)
        plot_adaptation(plt, Line2D, adaptation, output)


if __name__ == "__main__":
    main()
