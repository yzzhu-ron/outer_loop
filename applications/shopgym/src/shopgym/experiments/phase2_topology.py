"""
phase2_topology.py
Archive vs greedy vs random comparison across task categories.
Measures whether archive search outperforms greedy on deceptive landscapes.

Protocol (Section 6.5 / Figure 1):
  - Run GROL (archive), greedy ratchet, and random for 15 steps on each category
  - Compare final completion rate and convergence profile

Outputs: results/tables/phase2_topology.json, results/plots/phase2_*.png
"""

import os

from shopgym.environment import BASELINE_RATES
from shopgym.paths import RESULTS_ROOT
from shopgym.search import run_grol


def run_phase2(task_categories=None, max_steps=15, n_eval=85,
               rng_seed=42, output_dir=None):
    if output_dir is None:
        output_dir = RESULTS_ROOT
    if task_categories is None:
        task_categories = list(BASELINE_RATES.keys())

    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    all_results = {}

    for task in task_categories:
        print(f"\n{'='*60}")
        print(f"Phase 2: Topology for '{task}'")
        print(f"{'='*60}")
        task_results = {}

        for mode in ["archive", "greedy", "random"]:
            run = run_grol(
                task_category=task,
                mode=mode,
                max_steps=max_steps,
                n_eval=n_eval,
                rng_seed=rng_seed,
                use_mock_proposer=True,
            )
            task_results[mode] = {
                "completion_rates": run.completion_rates(),
                "final_rate": run.best_eval.completion_rate if run.best_eval else 0.0,
                "certified": run.certified_improvement,
                "total_episodes": run.total_episodes,
            }

        all_results[task] = task_results
        _save_results(all_results, output_dir)
        _make_plot(task, task_results, output_dir)

    return all_results


def _save_results(results, output_dir):
    import json as _json

    class SafeEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bool):
                return bool(obj)
            if hasattr(obj, 'item'):
                return obj.item()
            return super().default(obj)

    json_path = os.path.join(output_dir, "tables", "phase2_topology.json")
    with open(json_path, "w") as f:
        _json.dump(results, f, indent=2, cls=SafeEncoder)


def _make_plot(task, task_results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"archive": "#6E5FA8", "greedy": "#C96A44", "random": "#8A897F"}
        styles = {"archive": "-", "greedy": "--", "random": ":"}

        for mode, data in task_results.items():
            rates = data["completion_rates"]
            steps = list(range(len(rates)))
            ax.plot(steps, rates, color=colors[mode], linestyle=styles[mode],
                    linewidth=2.2, label=f"{mode} (final={data['final_rate']:.3f})",
                    marker="o", markersize=5)

        # Baseline
        ax.axhline(BASELINE_RATES[task], color="gray", linestyle="-.", alpha=0.5, label="baseline")

        ax.set_xlabel("Outer-loop step", fontsize=12)
        ax.set_ylabel("Task completion rate", fontsize=12)
        ax.set_title(f"Harness search topology: {task}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)

        plot_path = os.path.join(output_dir, "plots", f"phase2_{task}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {plot_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")


if __name__ == "__main__":
    run_phase2(max_steps=15, n_eval=85)
