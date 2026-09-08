"""
phase4_multi_shop.py
Cross-shop generalization of GROL-discovered harnesses.

Runs the full harness search (GROL archive, greedy, random) independently on
each of the 3 ShopGym synthetic shops, then evaluates cross-shop transfer:
harness found on Shop-A tested on Shops B and C (and so on).

Shop profiles (matched to ShopGym paper's synthetic shops):
  shop_a: "FashionHub"  — apparel, moderate catalog depth, 5-stage checkout
  shop_b: "TechNest"    — electronics, complex product comparison + coupon flows
  shop_c: "DailyMart"   — grocery/general, highest task diversity, longest journeys

Key output:
  results/tables/phase4_multi_shop.json   — per-shop + transfer results
  results/tables/phase4_transfer.csv      — summary transfer matrix

Usage:
    python -m shopgym.experiments.phase4_multi_shop
"""

import csv
import json
import math
import random

import numpy as np

from shopgym.experiments.simulate_full import (
    H,
    default_h, best_checkout_h,
    sigmoid, certified,
    smart_propose, random_propose, wilson_ci,
    BASE_LO, AXIS_DELTA, _retry_key,
)
from shopgym.paths import RESULTS_ROOT


# ── Shop-specific landscape offsets ──────────────────────────────────────────
#
# Each shop adds a per-stage offset to the base log-odds, simulating real
# differences in DOM complexity, product catalog structure, and checkout flows.
#
# shop_a (FashionHub): apparel — moderate checkout complexity, visual-heavy
#   Checkout slightly easier than default; product search harder (variant grid)
# shop_b (TechNest): electronics — complex comparison, coupon codes common
#   Search easier (structured specs), checkout harder (detailed address forms)
# shop_c (DailyMart): grocery — largest catalog, longest journeys
#   Hardest overall: slow search, complex multi-item carts, address entry errors
#
# Offsets applied on top of the shared AXIS_DELTA landscape.

SHOP_PROFILES = {
    "shop_a": {
        "name": "FashionHub",
        "description": "Apparel retailer — visual-heavy UI, moderate catalog depth",
        # Per-stage checkout log-odds offset (stages 0-4)
        "checkout_offset": [+0.10, +0.05, 0.00, -0.05, -0.10],
        # Single log-odds offset for search
        "search_offset": -0.15,
        "seed_offset": 0,
    },
    "shop_b": {
        "name": "TechNest",
        "description": "Electronics retailer — structured product specs, address-heavy checkout",
        "checkout_offset": [-0.10, 0.00, -0.05, -0.15, -0.20],
        "search_offset": +0.20,
        "seed_offset": 100,
    },
    "shop_c": {
        "name": "DailyMart",
        "description": "Grocery / general merchandise — large catalog, complex multi-item flows",
        "checkout_offset": [-0.20, -0.10, -0.10, -0.20, -0.25],
        "search_offset": -0.25,
        "seed_offset": 200,
    },
}


def checkout_true_rate_shop(h: H, shop_id: str) -> float:
    """True checkout completion rate for a specific shop."""
    offset = SHOP_PROFILES[shop_id]["checkout_offset"]
    ax_keys = [h.obs, h.vocab, h.ctx, h.scaffold, _retry_key(h.retry)]
    p = 1.0
    for k in range(5):
        lo = BASE_LO[k] + offset[k]
        for ax in ax_keys:
            deltas = AXIS_DELTA.get(ax, None)
            if deltas is not None:
                lo += deltas[k]
        p *= sigmoid(lo)
    return p


def search_true_rate_shop(h: H, shop_id: str) -> float:
    """True search completion rate for a specific shop."""
    from shopgym.experiments.simulate_full import SEARCH_BASE_LO, SEARCH_AXIS_DELTA
    offset = SHOP_PROFILES[shop_id]["search_offset"]
    lo = SEARCH_BASE_LO + offset
    for ax in [h.obs, h.vocab, h.ctx, h.scaffold, _retry_key(h.retry)]:
        lo += SEARCH_AXIS_DELTA.get(ax, 0.0)
    return sigmoid(lo)


def sample_episode_shop(task_type: str, h: H, rng: random.Random,
                         shop_id: str) -> dict:
    """Sample an episode with shop-specific landscape."""
    offset = (SHOP_PROFILES[shop_id]["checkout_offset"]
              if task_type == "checkout" else [0.0])
    if task_type != "checkout":
        # Use simple search sampling with offset
        from shopgym.experiments.simulate_full import SEARCH_BASE_LO, SEARCH_AXIS_DELTA
        lo = SEARCH_BASE_LO + SHOP_PROFILES[shop_id]["search_offset"]
        for ax in [h.obs, h.vocab, h.ctx, h.scaffold, _retry_key(h.retry)]:
            lo += SEARCH_AXIS_DELTA.get(ax, 0.0)
        success = rng.random() < sigmoid(lo)
        return {"success": success,
                "stages_completed": ["search"] if success else [],
                "failure_mode": None if success else "search_failed"}

    # Checkout with shop offset
    from shopgym.experiments.simulate_full import CHECKOUT_STAGES
    completed, failure_mode = [], None
    ax_keys = [h.obs, h.vocab, h.ctx, h.scaffold, _retry_key(h.retry)]
    for k, stage in enumerate(CHECKOUT_STAGES):
        lo = BASE_LO[k] + offset[k]
        for ax in ax_keys:
            deltas = AXIS_DELTA.get(ax, None)
            if deltas:
                lo += deltas[k]
        if rng.random() < sigmoid(lo):
            completed.append(stage)
        else:
            if k >= 3 and h.obs in ("ax_tree", "dom_text"):
                failure_mode = "visual_form_missing"
            elif k >= 2 and h.ctx == "last_1":
                failure_mode = "context_lost"
            elif k >= 1 and h.retry == "none":
                failure_mode = "action_failed_no_retry"
            else:
                failure_mode = "action_failed"
            break
    return {"success": len(completed) == 5, "stages_completed": completed,
            "failure_mode": failure_mode}


def evaluate_shop(task_type: str, h: H, n: int, rng: random.Random,
                  shop_id: str) -> dict:
    eps = [sample_episode_shop(task_type, h, rng, shop_id) for _ in range(n)]
    k = sum(e["success"] for e in eps)
    lo, hi = wilson_ci(k, n)
    traces = [e for e in eps if not e["success"]]  # failure traces
    return {"h": h, "n": n, "k": k, "rate": k / n,
            "ci_lo": lo, "ci_hi": hi, "traces": traces}


def run_grol_shop(shop_id: str, task_type: str = "checkout",
                  max_steps: int = 15, n_eval: int = 85,
                  seed: int = 42) -> dict:
    """Run all three search modes on a single shop."""
    rng_seed = seed + SHOP_PROFILES[shop_id]["seed_offset"]
    results = {}

    for mode in ["archive", "greedy", "random"]:
        rng = random.Random(rng_seed)
        baseline = evaluate_shop(task_type, default_h(), n_eval, rng, shop_id)
        best = baseline
        archive_evals = [baseline]
        history = [baseline["rate"]]

        for step in range(1, max_steps + 1):
            parent = (max(archive_evals, key=lambda e: e["rate"])
                      if mode == "archive" else best)
            traces_raw = [sample_episode_shop(task_type, parent["h"], rng, shop_id)
                          for _ in range(20)]

            if mode == "random":
                cand_h = random_propose(rng)
            else:
                cand_h = smart_propose(task_type, traces_raw, parent["h"], rng,
                                       archive_evals)

            cand = evaluate_shop(task_type, cand_h, n_eval, rng, shop_id)

            if mode == "archive":
                if cand["rate"] > baseline["rate"]:
                    archive_evals.append(cand)
                    archive_evals.sort(key=lambda e: e["rate"], reverse=True)
                    archive_evals = archive_evals[:20]
                if cand["rate"] > best["rate"]:
                    best = cand
            elif mode == "greedy":
                if certified(cand, best):
                    best = cand
            else:
                if cand["rate"] > best["rate"]:
                    best = cand

            history.append(best["rate"])

        cert = certified(best, baseline)
        results[mode] = {
            "rate_history": [round(r, 4) for r in history],
            "final_rate": round(best["rate"], 4),
            "baseline_rate": round(baseline["rate"], 4),
            "best_harness": {"obs": best["h"].obs, "vocab": best["h"].vocab,
                             "ctx": best["h"].ctx, "scaffold": best["h"].scaffold,
                             "retry": best["h"].retry},
            "certified": bool(cert),
            "delta": round(best["rate"] - baseline["rate"], 4),
        }

    return results


def run_transfer_matrix(shop_results: dict, task_type: str = "checkout",
                         n_eval: int = 85, seed: int = 99) -> dict:
    """
    For each shop, take the best harness found by GROL-archive,
    then evaluate it on all other shops (cross-shop transfer test).

    Returns: {source_shop: {target_shop: transfer_rate}}
    """
    transfer = {}
    for src_id, src_data in shop_results.items():
        transfer[src_id] = {}
        best_h_dict = src_data["archive"]["best_harness"]
        best_h = H(**best_h_dict)

        for tgt_id in shop_results:
            rng = random.Random(seed + SHOP_PROFILES[tgt_id]["seed_offset"])
            eval_result = evaluate_shop(task_type, best_h, n_eval, rng, tgt_id)
            baseline_result = evaluate_shop(task_type, default_h(), n_eval, rng, tgt_id)
            transfer[src_id][tgt_id] = {
                "transfer_rate": round(eval_result["rate"], 4),
                "baseline_rate": round(baseline_result["rate"], 4),
                "delta": round(eval_result["rate"] - baseline_result["rate"], 4),
                "certified": bool(certified(eval_result, baseline_result)),
            }
    return transfer


def main():
    np.random.seed(42)
    results_dir = RESULTS_ROOT / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 4: Multi-Shop GROL — 3 ShopGym Synthetic Shops")
    print("=" * 70)

    shop_results = {}
    for shop_id, profile in SHOP_PROFILES.items():
        print(f"\n── {shop_id}: {profile['name']} ──")
        print(f"   {profile['description']}")
        baseline_true = checkout_true_rate_shop(default_h(), shop_id)
        best_true = checkout_true_rate_shop(best_checkout_h(), shop_id)
        print(f"   True default_h checkout: {baseline_true:.3f}")
        print(f"   True best_h checkout:    {best_true:.3f}")
        shop_results[shop_id] = run_grol_shop(
            shop_id, task_type="checkout", max_steps=15, n_eval=85, seed=42
        )
        for mode, data in shop_results[shop_id].items():
            cert = "✓" if data["certified"] else "✗"
            print(f"   {mode:8s}: {data['baseline_rate']:.3f} → "
                  f"{data['final_rate']:.3f} (+{data['delta']:.3f}) {cert}")

    print("\n\n── Transfer Matrix (GROL-archive best harness cross-shop) ──")
    transfer = run_transfer_matrix(shop_results, task_type="checkout",
                                    n_eval=85, seed=99)

    print(f"\n{'':12s}  ", end="")
    for tgt in SHOP_PROFILES:
        print(f"{tgt:>12s}", end="")
    print()
    print("-" * (12 + 3 + 12 * len(SHOP_PROFILES)))
    for src in SHOP_PROFILES:
        print(f"{src:12s}  ", end="")
        for tgt in SHOP_PROFILES:
            t = transfer[src][tgt]
            cert = "✓" if t["certified"] else " "
            print(f"{t['transfer_rate']:.3f}{cert:>2s}   ", end="")
        print()

    # Save results
    output = {
        "shop_profiles": {k: {kk: vv for kk, vv in v.items()
                               if kk != "checkout_offset" and kk != "search_offset"}
                          for k, v in SHOP_PROFILES.items()},
        "per_shop": shop_results,
        "transfer_matrix": transfer,
    }
    json_path = results_dir / "phase4_multi_shop.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Transfer CSV
    csv_path = results_dir / "phase4_transfer.csv"
    rows = []
    for src in SHOP_PROFILES:
        for tgt in SHOP_PROFILES:
            t = transfer[src][tgt]
            rows.append({
                "source_shop": src,
                "source_name": SHOP_PROFILES[src]["name"],
                "target_shop": tgt,
                "target_name": SHOP_PROFILES[tgt]["name"],
                "transfer_rate": t["transfer_rate"],
                "baseline_rate": t["baseline_rate"],
                "delta": t["delta"],
                "certified": t["certified"],
            })
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return output


if __name__ == "__main__":
    main()
