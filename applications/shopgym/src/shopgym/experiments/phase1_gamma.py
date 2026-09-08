"""
phase1_gamma.py
Measures the trace-conditioning improvement factor gamma for each task category.

Protocol (Section 6.5 / Table 3):
  - For each task category, generate 200 harness candidates
  - Half proposed by trace-conditioned proposer, half by unconditioned (random) proposer
  - Evaluate each on n_eval episodes
  - Compute p_trace, p_0, and gamma = p_trace / p_0

Outputs: results/tables/phase1_gamma.csv
"""

import json
import os

import numpy as np
from shopgym.environment import BASELINE_RATES, MockShopGymEnv
from shopgym.evaluation import evaluate_harness
from shopgym.harness import default_harness
from shopgym.paths import RESULTS_ROOT
from shopgym.proposer import MockLLMProposer, UnconditionedProposer


def run_phase1(task_categories=None, n_candidates=200, n_eval=85,
               rng_seed=42, output_dir=None):
    """
    Measure gamma for each task category.

    Returns dict: {task_category: {p0, p_trace, gamma, gamma_ci_lower, gamma_ci_upper}}
    """
    if output_dir is None:
        output_dir = RESULTS_ROOT / "tables"
    if task_categories is None:
        task_categories = list(BASELINE_RATES.keys())

    os.makedirs(output_dir, exist_ok=True)
    bootstrap_rng = np.random.default_rng(rng_seed)
    results = {}

    for task in task_categories:
        print(f"\n{'='*60}")
        print(f"Phase 1: Gamma measurement for '{task}'")
        print(f"{'='*60}")

        env = MockShopGymEnv(rng_seed=rng_seed)
        baseline_h = default_harness()
        baseline_eval = evaluate_harness(env, baseline_h, task, n_eval)

        # Run trace episodes from baseline harness to build trace buffer
        n_trace_eps = 20
        trace_episodes = env.run_n_episodes(task, baseline_h, n_trace_eps)

        # Proposers
        trace_proposer = MockLLMProposer(rng_seed=rng_seed)
        random_proposer = UnconditionedProposer(rng_seed=rng_seed + 1)

        # Evaluate n_candidates/2 from each proposer
        half = n_candidates // 2
        trace_accepts = 0
        random_accepts = 0

        print(f"  Evaluating {half} trace-conditioned proposals...")
        for i in range(half):
            candidate_h = trace_proposer.propose(trace_episodes, baseline_h)
            candidate_eval = evaluate_harness(env, candidate_h, task, n_eval)
            # Use point-estimate comparison for gamma measurement:
            # p_trace = fraction of trace-conditioned proposals that beat baseline
            if candidate_eval.completion_rate > baseline_eval.completion_rate:
                trace_accepts += 1
            if i % 20 == 0:
                print(f"    [{i}/{half}] accepts so far: {trace_accepts}")

        print(f"  Evaluating {half} unconditioned (random) proposals...")
        for i in range(half):
            candidate_h = random_proposer.propose(trace_episodes, baseline_h)
            candidate_eval = evaluate_harness(env, candidate_h, task, n_eval)
            if candidate_eval.completion_rate > baseline_eval.completion_rate:
                random_accepts += 1
            if i % 20 == 0:
                print(f"    [{i}/{half}] accepts so far: {random_accepts}")

        p_trace = trace_accepts / half
        p_0 = max(random_accepts / half, 1e-6)  # avoid divide-by-zero
        gamma = p_trace / p_0

        # Bootstrap CI for gamma
        gamma_samples = []
        for _ in range(1000):
            t_sample = bootstrap_rng.binomial(half, p_trace) / half
            r_sample = max(bootstrap_rng.binomial(half, p_0) / half, 1e-6)
            gamma_samples.append(t_sample / r_sample)
        gamma_ci_lo = float(np.percentile(gamma_samples, 2.5))
        gamma_ci_hi = float(np.percentile(gamma_samples, 97.5))

        results[task] = {
            "task": task,
            "baseline_rate": BASELINE_RATES[task],
            "p_0": p_0,
            "p_trace": p_trace,
            "gamma": gamma,
            "gamma_ci_lower": gamma_ci_lo,
            "gamma_ci_upper": gamma_ci_hi,
            "n_trace_accepts": trace_accepts,
            "n_random_accepts": random_accepts,
            "n_candidates": half,
        }

        print(f"  p_0={p_0:.4f}, p_trace={p_trace:.4f}, gamma={gamma:.2f} "
              f"[{gamma_ci_lo:.2f}, {gamma_ci_hi:.2f}]")

    # Save results
    import csv
    csv_path = os.path.join(output_dir, "phase1_gamma.csv")
    keys = ["task", "baseline_rate", "p_0", "p_trace", "gamma",
            "gamma_ci_lower", "gamma_ci_upper", "n_trace_accepts",
            "n_random_accepts", "n_candidates"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results.values():
            writer.writerow({k: row[k] for k in keys})

    json_path = os.path.join(output_dir, "phase1_gamma.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {csv_path}")
    return results


if __name__ == "__main__":
    run_phase1(n_candidates=40, n_eval=85)  # reduced for quick demo
