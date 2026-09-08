"""
run_all.py
Master experiment runner.  Runs all four phases and prints a summary report.

Usage:
    python -m shopgym.experiments.run_all              # quick (reduced n)
    python -m shopgym.experiments.run_all --full       # full paper numbers
"""

import argparse
import time

from shopgym.paths import RESULTS_ROOT


def run_all(full=False, include_multi_shop=False):
    n_candidates = 200 if full else 40
    n_eval       = 85  if full else 30
    max_steps    = 15  if full else 8
    oracle_n     = 1000 if full else 100

    print("=" * 70)
    print("GROL Experiment Suite — ShopGym Harness Search")
    print(f"Mode: {'FULL' if full else 'QUICK (reduced n)'}")
    print("=" * 70)

    t0 = time.time()

    # Phase 1: gamma measurement
    print("\n\n▶ PHASE 1: γ Measurement")
    from shopgym.experiments.phase1_gamma import run_phase1
    gamma_results = run_phase1(n_candidates=n_candidates, n_eval=n_eval)

    # Phase 2: topology comparison (subset of tasks for speed)
    print("\n\n▶ PHASE 2: Archive vs Greedy Topology")
    tasks_p2 = (["checkout_with_coupon", "product_search"]
                if not full else None)
    from shopgym.experiments.phase2_topology import run_phase2
    topo_results = run_phase2(task_categories=tasks_p2, max_steps=max_steps, n_eval=n_eval)

    # Phase 3: ratchet illusion
    print("\n\n▶ PHASE 3: Ratchet Illusion")
    from shopgym.experiments.phase3_ratchet import run_phase3
    ratchet_results = run_phase3(max_steps=20 if full else 10, oracle_n=oracle_n)

    if include_multi_shop:
        print("\n\n▶ PHASE 4: Multi-shop transfer")
        from shopgym.experiments.phase4_multi_shop import main as run_phase4
        run_phase4()

    elapsed = time.time() - t0

    # Summary report
    print("\n\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    print("\n── Phase 1: γ (trace-conditioning improvement factor) ──")
    print(f"{'Task':<35} {'γ':>6}  {'95% CI':<18}  {'p_0':>6}  {'p_trace':>8}")
    print("-" * 75)
    for task, r in gamma_results.items():
        print(f"{task:<35} {r['gamma']:>6.2f}  "
              f"[{r['gamma_ci_lower']:.2f}, {r['gamma_ci_upper']:.2f}]  "
              f"{r['p_0']:>6.4f}  {r['p_trace']:>8.4f}")

    print("\n── Phase 2: Final completion rates ──")
    print(f"{'Task':<35} {'Archive':>8}  {'Greedy':>8}  {'Random':>8}  {'Δ(A-G)':>8}")
    print("-" * 75)
    for task, modes in topo_results.items():
        a = modes.get("archive", {}).get("final_rate", float("nan"))
        g = modes.get("greedy",  {}).get("final_rate", float("nan"))
        r = modes.get("random",  {}).get("final_rate", float("nan"))
        print(f"{task:<35} {a:>8.3f}  {g:>8.3f}  {r:>8.3f}  {a-g:>+8.3f}")

    print("\n── Phase 3: Ratchet illusion ──")
    print(f"{'N_eval':>8}  {'Illusion rate':>14}  {'Final oracle rate':>18}  {'Certified':>10}")
    print("-" * 60)
    for label, r in ratchet_results.items():
        cert = "✓" if r["certified"] else "✗"
        print(f"{r['n_eval']:>8}  {r['illusion_rate']:>14.3f}  "
              f"{r['final_oracle_rate']:>18.3f}  {cert:>10}")

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Results saved in: {RESULTS_ROOT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full paper experiments")
    parser.add_argument(
        "--multi-shop", action="store_true", help="Also run phase 4 transfer"
    )
    args = parser.parse_args()
    run_all(full=args.full, include_multi_shop=args.multi_shop)


if __name__ == "__main__":
    main()
