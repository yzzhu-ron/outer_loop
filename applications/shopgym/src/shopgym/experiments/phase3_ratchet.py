"""
phase3_ratchet.py
Ratchet illusion demonstration: compares naive ratchet (N=5, N=20) vs
gated ratchet (N=85) on checkout_with_coupon.

Protocol (Section 6.5 / Table 4):
  - Run each ratchet variant for 20 steps
  - Use oracle evaluation (N=1000) to determine ground-truth improvement
  - Count illusion events (accepted harnesses that are oracle-worse)

Outputs: results/tables/phase3_ratchet.json, results/tables/phase3_ratchet.csv
"""

import json
import os
import csv

from outer_loop.selection import GatedRatchet
from shopgym.environment import MockShopGymEnv
from shopgym.evaluation import evaluate_harness
from shopgym.harness import default_harness
from shopgym.paths import RESULTS_ROOT
from shopgym.proposer import MockLLMProposer


def run_phase3(task_category="checkout_with_coupon", max_steps=20,
               rng_seed=42, oracle_n=1000, output_dir=None):

    if output_dir is None:
        output_dir = RESULTS_ROOT
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    env = MockShopGymEnv(rng_seed=rng_seed)
    proposer = MockLLMProposer(rng_seed=rng_seed)
    baseline_h = default_harness()

    oracle_baseline = evaluate_harness(env, baseline_h, task_category, oracle_n)
    print(f"Oracle baseline: {oracle_baseline.completion_rate:.3f}")

    results = {}

    for n_eval in [5, 20, 85]:
        print(f"\n{'='*50}")
        print(f"Ratchet N={n_eval}")
        print(f"{'='*50}")

        ratchet = GatedRatchet(n_eval=n_eval, alpha=0.05)
        incumbent_h = default_harness()
        incumbent_eval = evaluate_harness(env, incumbent_h, task_category, n_eval)
        ratchet.set_incumbent(incumbent_eval)

        steps_data = []
        illusion_count = 0
        accept_count = 0

        for step in range(1, max_steps + 1):
            # Collect traces and propose
            traces = env.run_n_episodes(task_category, ratchet.incumbent.harness, 10)
            candidate_h = proposer.propose(traces, ratchet.incumbent.harness)

            # Fast evaluation (N=n_eval)
            candidate_eval_fast = evaluate_harness(env, candidate_h, task_category, n_eval)

            # Acceptance decision
            if n_eval == 85:
                accepted = ratchet.accept(candidate_eval_fast)  # gated
            else:
                # Naive: accept if point estimate is higher
                accepted = candidate_eval_fast.completion_rate > ratchet.incumbent.completion_rate
                if accepted:
                    ratchet.incumbent = candidate_eval_fast

            if accepted:
                accept_count += 1
                # Oracle evaluation: is the accepted harness actually better?
                oracle_eval = evaluate_harness(env, candidate_h, task_category, oracle_n)
                is_illusion = oracle_eval.completion_rate < oracle_baseline.completion_rate
                if is_illusion:
                    illusion_count += 1
                oracle_rate = oracle_eval.completion_rate
            else:
                oracle_rate = None
                is_illusion = False

            steps_data.append({
                "step": step,
                "candidate_rate_fast": candidate_eval_fast.completion_rate,
                "accepted": accepted,
                "oracle_rate": oracle_rate,
                "is_illusion": is_illusion,
            })

            status = "✓ accepted" if accepted else "✗ rejected"
            illusion_str = " ← ILLUSION" if is_illusion else ""
            print(f"  Step {step:2d}: rate={candidate_eval_fast.completion_rate:.3f} | "
                  f"{status}{illusion_str}")

        # Final oracle rate of incumbent
        final_oracle = evaluate_harness(env, ratchet.incumbent.harness, task_category, oracle_n)
        illusion_rate = illusion_count / accept_count if accept_count > 0 else 0.0

        results[f"N={n_eval}"] = {
            "n_eval": n_eval,
            "illusion_rate": illusion_rate,
            "illusion_count": illusion_count,
            "accept_count": accept_count,
            "final_oracle_rate": final_oracle.completion_rate,
            "certified": n_eval == 85,
            "steps": steps_data,
        }

        print(f"\n  Summary N={n_eval}:")
        print(f"    Accepts: {accept_count}/{max_steps}")
        print(f"    Illusion rate: {illusion_rate:.3f} ({illusion_count}/{accept_count})")
        print(f"    Final oracle rate: {final_oracle.completion_rate:.3f}")

    # Save results
    json_path = os.path.join(output_dir, "tables", "phase3_ratchet.json")
    with open(json_path, "w") as f:
        # Remove 'steps' for the summary JSON to keep it compact
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "steps"}
                   for k, v in results.items()}
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "tables", "phase3_ratchet.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_eval", "illusion_rate", "final_oracle_rate",
                                                "accept_count", "certified"])
        writer.writeheader()
        for v in results.values():
            writer.writerow({k: v[k] for k in ["n_eval", "illusion_rate", "final_oracle_rate",
                                                "accept_count", "certified"]})

    print(f"\nResults saved to {json_path}")
    return results


if __name__ == "__main__":
    run_phase3(max_steps=20, oracle_n=200)  # reduced oracle_n for speed
