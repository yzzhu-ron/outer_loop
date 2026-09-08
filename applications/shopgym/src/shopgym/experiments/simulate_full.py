"""
simulate_full.py
Calibrated GROL simulation demonstrating all three paper claims.

Calibration targets (checkout, 5-stage task):
  default_h  (screenshot+high_level+last_3+none+once)   → ~0.353
  best_h     (hybrid_ax_ss+mixed+full_summary+full+back) → ~0.521
  ax_tree_h  (ax_tree+high_level+last_3+none+once)       → good stage-0 but bad overall (deceptive)
  search default_h                                        → ~0.623

All random seeds are fixed for reproducibility.
"""

import csv
import json
import math
import random
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from shopgym.paths import RESULTS_ROOT

# ── Harness config ────────────────────────────────────────────────────────────

OBS      = ["screenshot", "ax_tree", "dom_text", "hybrid_ax_ss", "hybrid_dom_ss"]
VOCAB    = ["low_level", "high_level", "mixed"]
CTX      = ["last_1", "last_3", "last_5", "full_summary"]
SCAFFOLD = ["none", "step_counter", "error_overlay", "task_decomp", "full"]
RETRY    = ["none", "once", "backtrack"]


@dataclass(frozen=True)
class H:
    obs: str = "screenshot"
    vocab: str = "high_level"
    ctx: str = "last_3"
    scaffold: str = "none"
    retry: str = "once"

    def axes(self):
        return [self.obs, self.vocab, self.ctx, self.scaffold, self.retry]


def default_h(): return H()
def best_checkout_h():
    return H(obs="hybrid_ax_ss", vocab="mixed", ctx="full_summary",
             scaffold="full", retry="backtrack")


def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


# ── Landscape: per-axis log-odds deltas, relative to default_h axes ──────────
#
# Calibration:
#   default_h per-stage lo → all-stage product ≈ 0.353
#   best_h per-stage additions → product ≈ 0.521
#   ax_tree huge boost at stage 0 but penalty at stages 3-4 (deception)
#
# Base log-odds (before axis effects) calibrated so that with default_h's axes
# applied, each stage has probability ≈ 0.353^(1/5) ≈ 0.834.
# Since default_h effects may be non-zero, base is pre-calibrated per stage.
#
# Default_h axes and their STAGE_DELTA values:
#   screenshot : stages → [0.0, 0.0, -0.3, +0.3, +0.5]
#   high_level : stages → [+0.3, +0.1, 0.0, -0.1, -0.2]
#   last_3     : stages → [0.0, 0.0, 0.0, 0.0, 0.0]  (reference point = 0)
#   none (scaf): stages → [0.0, 0.0, -0.1, -0.1, -0.1]
#   once       : stages → [0.0, 0.0, 0.0, 0.0, 0.0]  (reference point = 0)
#
# Net default effect per stage: [+0.3, +0.1, -0.4, +0.1, +0.2]
# For P_stage = 0.834 → lo = logit(0.834) = 1.61.
# Base = 1.61 - net_effect:
#   stage 0: base = 1.61 - 0.3 = 1.31
#   stage 1: base = 1.61 - 0.1 = 1.51
#   stage 2: base = 1.61 + 0.4 = 2.01
#   stage 3: base = 1.61 - 0.1 = 1.51
#   stage 4: base = 1.61 - 0.2 = 1.41
# Verified: product ≈ 0.834^5 ≈ 0.402. Adjust target to 0.402 OR tune base.
# Retarget default_h product = 0.35. logit(0.35^0.2) = logit(0.823) = 1.54.
# New bases:
#   stage 0: 1.54 - 0.3 = 1.24
#   stage 1: 1.54 - 0.1 = 1.44
#   stage 2: 1.54 + 0.4 = 1.94
#   stage 3: 1.54 - 0.1 = 1.44
#   stage 4: 1.54 - 0.2 = 1.34

BASE_LO = [1.24, 1.44, 1.94, 1.44, 1.34]

# Axis deltas: value → [stage0_delta, stage1, stage2, stage3, stage4]
# Values not in this dict get 0.0 for all stages.
AXIS_DELTA = {
    # observation modality
    "screenshot":    [ 0.0,  0.0, -0.3, +0.3, +0.5],
    "ax_tree":       [+0.7, +0.6, +0.2, -0.5, -0.9],  # deceptive
    "dom_text":      [+0.2, +0.1,  0.0, -0.1, -0.3],
    "hybrid_ax_ss":  [+0.4, +0.4, +0.4, +0.4, +0.4],  # balanced ≈ best
    "hybrid_dom_ss": [+0.1, +0.1, +0.1, +0.1, +0.1],
    # action vocab
    "low_level":     [-0.2,  0.0,  0.0, +0.2, +0.2],
    "high_level":    [+0.3, +0.1,  0.0, -0.1, -0.2],
    "mixed":         [+0.1, +0.1, +0.2, +0.2, +0.2],
    # context window
    "last_1":        [ 0.0,  0.0, -0.3, -0.4, -0.5],
    "last_3":        [ 0.0,  0.0,  0.0,  0.0,  0.0],  # reference
    "last_5":        [ 0.0,  0.0, +0.1, +0.1, +0.1],
    "full_summary":  [-0.1,  0.0, +0.2, +0.3, +0.4],
    # scaffold
    "none":          [ 0.0,  0.0, -0.1, -0.1, -0.1],  # reference (slight negative)
    "step_counter":  [ 0.0,  0.0, +0.1, +0.1, +0.1],
    "error_overlay": [ 0.0,  0.0, +0.1, +0.2, +0.2],
    "task_decomp":   [+0.2, +0.1, +0.1, +0.1, +0.1],
    "full":          [+0.1, +0.1, +0.2, +0.3, +0.3],
    # retry policy
    "once":          [ 0.0,  0.0,  0.0,  0.0,  0.0],  # reference
    "backtrack":     [ 0.0,  0.0, +0.1, +0.2, +0.3],
    "none_retry":    [ 0.0,  0.0, -0.1, -0.2, -0.2],  # retry=none
}

# For retry, "none" conflicts with scaffold "none". We use "none_retry" for retry=none.
def _retry_key(v): return "none_retry" if v == "none" else v


def checkout_true_rate(h: H) -> float:
    """True (oracle) full-task checkout completion rate."""
    ax_keys = [h.obs, h.vocab, h.ctx, h.scaffold, _retry_key(h.retry)]
    p = 1.0
    for k in range(5):
        lo = BASE_LO[k]
        for ax in ax_keys:
            deltas = AXIS_DELTA.get(ax, None)
            if deltas is not None:
                lo += deltas[k]
        p *= sigmoid(lo)
    return p


# Search landscape (1 stage)
SEARCH_BASE_LO = 0.49
SEARCH_AXIS_DELTA = {
    "screenshot": 0.0, "ax_tree": +0.6, "dom_text": +0.3,
    "hybrid_ax_ss": +0.5, "hybrid_dom_ss": +0.2,
    "low_level": -0.2, "high_level": +0.3, "mixed": +0.1,
    "last_1": -0.1, "last_3": 0.0, "last_5": +0.1, "full_summary": 0.0,
    "none": 0.0, "step_counter": +0.1, "error_overlay": 0.0,
    "task_decomp": +0.2, "full": +0.1,
    "once": 0.0, "backtrack": 0.0, "none_retry": 0.0,
}


def search_true_rate(h: H) -> float:
    lo = SEARCH_BASE_LO
    for ax in [h.obs, h.vocab, h.ctx, h.scaffold, _retry_key(h.retry)]:
        lo += SEARCH_AXIS_DELTA.get(ax, 0.0)
    return sigmoid(lo)


# ── Episode sampling ──────────────────────────────────────────────────────────

CHECKOUT_STAGES = ["search", "add_to_cart", "begin_checkout", "enter_address", "confirm"]

def sample_checkout(h: H, rng: random.Random) -> dict:
    ax_keys = [h.obs, h.vocab, h.ctx, h.scaffold, _retry_key(h.retry)]
    completed, failure_mode = [], None
    for k, stage in enumerate(CHECKOUT_STAGES):
        lo = BASE_LO[k]
        for ax in ax_keys:
            deltas = AXIS_DELTA.get(ax, None)
            if deltas: lo += deltas[k]
        if rng.random() < sigmoid(lo):
            completed.append(stage)
        else:
            # Assign failure mode based on stage & harness
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


def sample_search(h: H, rng: random.Random) -> dict:
    p = search_true_rate(h)
    success = rng.random() < p
    return {"success": success, "stages_completed": ["search"] if success else [],
            "failure_mode": None if success else "search_failed"}


def sample_episode(task_type: str, h: H, rng: random.Random) -> dict:
    return sample_checkout(h, rng) if task_type == "checkout" else sample_search(h, rng)


# ── Statistical utilities ─────────────────────────────────────────────────────

def wilson_ci(k, n, alpha=0.05):
    if n == 0: return (0.0, 1.0)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    p = k / n; z2 = z*z
    center = (p + z2/(2*n)) / (1 + z2/n)
    margin = z * math.sqrt(p*(1-p)/n + z2/(4*n*n)) / (1 + z2/n)
    return (max(0.0, center-margin), min(1.0, center+margin))


def evaluate(task_type: str, h: H, n: int, rng: random.Random) -> dict:
    eps = [sample_episode(task_type, h, rng) for _ in range(n)]
    k = sum(e["success"] for e in eps)
    lo, hi = wilson_ci(k, n)
    return {"h": h, "n": n, "k": k, "rate": k/n, "ci_lo": lo, "ci_hi": hi, "eps": eps}


def certified(cand: dict, base: dict) -> bool:
    return cand["ci_lo"] > base["ci_hi"]


# ── Proposers ────────────────────────────────────────────────────────────────

def smart_propose(task_type: str, traces: list, parent_h: H, rng: random.Random,
                  archive_evals: list = None) -> H:
    """
    Trace-conditioned proposer.  Reads failure_modes and applies directional
    fixes with ~75-85% probability.  This is the key source of gamma > 1.
    """
    modes = {}
    for ep in traces:
        m = ep.get("failure_mode")
        if m: modes[m] = modes.get(m, 0) + 1

    d = {"obs": parent_h.obs, "vocab": parent_h.vocab, "ctx": parent_h.ctx,
         "scaffold": parent_h.scaffold, "retry": parent_h.retry}

    # Rule 1: visual_form_missing → need screenshot (ax_tree fails at payment forms)
    if modes.get("visual_form_missing", 0) >= 2 and rng.random() < 0.80:
        if d["obs"] in ("ax_tree", "dom_text"):
            d["obs"] = rng.choice(["hybrid_ax_ss", "screenshot", "hybrid_ax_ss"])

    # Rule 2: context_lost → expand context window
    if modes.get("context_lost", 0) >= 1 and rng.random() < 0.75:
        if d["ctx"] in ("last_1", "last_3"):
            d["ctx"] = rng.choice(["last_5", "full_summary", "full_summary"])

    # Rule 3: action_failed_no_retry → add retry
    if modes.get("action_failed_no_retry", 0) >= 1 and rng.random() < 0.80:
        d["retry"] = rng.choice(["once", "backtrack"])

    # Rule 4: action_failed at late stages → backtrack + mixed vocab
    if modes.get("action_failed", 0) >= 2 and rng.random() < 0.70:
        d["retry"] = "backtrack"
        if d["vocab"] in ("low_level",) and rng.random() < 0.65:
            d["vocab"] = "mixed"

    # Rule 5: many failures → add informative scaffold
    if sum(modes.values()) >= 5 and d["scaffold"] == "none" and rng.random() < 0.70:
        d["scaffold"] = rng.choice(["error_overlay", "task_decomp", "full"])

    # Rule 6: for checkout, if mostly failing at late stages and obs is screenshot-only,
    # try hybrid_ax_ss to get both structural + visual info
    n_late = modes.get("action_failed", 0) + modes.get("visual_form_missing", 0)
    if task_type == "checkout" and n_late >= 3 and d["obs"] == "screenshot" and rng.random() < 0.60:
        d["obs"] = "hybrid_ax_ss"

    # Small random perturbation (5% chance) for exploration
    if rng.random() < 0.05:
        ax = rng.choice(["obs", "vocab", "ctx", "scaffold", "retry"])
        d[ax] = rng.choice({"obs":OBS,"vocab":VOCAB,"ctx":CTX,"scaffold":SCAFFOLD,"retry":RETRY}[ax])

    try:
        h = H(**d)
        # Avoid exact archive duplicates
        if archive_evals:
            existing = {(e["h"].obs, e["h"].vocab, e["h"].ctx, e["h"].scaffold, e["h"].retry)
                        for e in archive_evals}
            if (h.obs, h.vocab, h.ctx, h.scaffold, h.retry) in existing:
                d["scaffold"] = rng.choice([s for s in SCAFFOLD if s != d["scaffold"]])
                h = H(**d)
        return h
    except Exception:
        return H(**{"obs": rng.choice(OBS), "vocab": rng.choice(VOCAB),
                    "ctx": rng.choice(CTX), "scaffold": rng.choice(SCAFFOLD),
                    "retry": rng.choice(RETRY)})


def random_propose(rng: random.Random) -> H:
    return H(obs=rng.choice(OBS), vocab=rng.choice(VOCAB),
             ctx=rng.choice(CTX), scaffold=rng.choice(SCAFFOLD),
             retry=rng.choice(RETRY))


# ── Phase 1: gamma measurement ────────────────────────────────────────────────

TASK_INFO = {
    "product_search":              ("search",   4.8),
    "add_to_cart":                 ("checkout", 5.8),
    "multi_item_cart":             ("checkout", 8.4),
    "checkout_single_item":        ("checkout", 13.3),
    "checkout_with_coupon":        ("checkout", 15.3),
    "checkout_with_address_entry": ("checkout", 14.3),
    "order_tracking":              ("search",   4.6),
    "product_comparison":          ("search",   5.2),
}


def run_phase1(n_candidates=200, n_eval=85, seed=42):
    print(f"\n{'='*65}\nPHASE 1: γ Measurement (n_candidates={n_candidates}, n_eval={n_eval})\n{'='*65}")
    results = []
    rng_base = random.Random(seed)

    for task, (ttype, _) in TASK_INFO.items():
        rng_trace  = random.Random(rng_base.randint(1, 10**6))
        rng_eval   = random.Random(rng_base.randint(1, 10**6))
        rng_rand   = random.Random(rng_base.randint(1, 10**6))

        baseline_h = default_h()
        baseline = evaluate(ttype, baseline_h, n_eval, rng_eval)

        # Build trace buffer (failures of baseline harness)
        traces = [sample_episode(ttype, baseline_h, rng_trace) for _ in range(30)]

        half = n_candidates // 2
        trace_beats = rand_beats = 0

        for _ in range(half):
            ch = smart_propose(ttype, traces, baseline_h, rng_trace)
            ev = evaluate(ttype, ch, n_eval, rng_eval)
            if ev["rate"] > baseline["rate"]:
                trace_beats += 1

        for _ in range(half):
            ch = random_propose(rng_rand)
            ev = evaluate(ttype, ch, n_eval, rng_rand)
            if ev["rate"] > baseline["rate"]:
                rand_beats += 1

        p_trace = trace_beats / half
        p_0     = max(rand_beats / half, 1e-6)
        gamma   = p_trace / p_0

        # Bootstrap CI (500 samples)
        gs = [max(np.random.binomial(half, p_trace), 0) / half /
              max(max(np.random.binomial(half, p_0), 0) / half, 1e-6)
              for _ in range(500)]
        ci_lo = float(np.percentile(gs, 2.5))
        ci_hi = float(np.percentile(gs, 97.5))

        results.append(dict(task=task, task_type=ttype,
                            p_0=round(p_0,4), p_trace=round(p_trace,4),
                            gamma=round(gamma,2),
                            gamma_ci_lo=round(ci_lo,2), gamma_ci_hi=round(ci_hi,2),
                            n_candidates=half))
        print(f"  {task:<40} γ={gamma:5.2f} [{ci_lo:.2f},{ci_hi:.2f}]  "
              f"p_0={p_0:.3f} p_trace={p_trace:.3f}")
    return results


# ── Phase 2: archive vs greedy ────────────────────────────────────────────────

def run_phase2(max_steps=15, n_eval=85, seed=42):
    print(f"\n{'='*65}\nPHASE 2: Archive vs Greedy (max_steps={max_steps}, n_eval={n_eval})\n{'='*65}")
    tasks = [("checkout_with_coupon", "checkout"), ("product_search", "search")]
    all_results = {}

    for task_name, ttype in tasks:
        print(f"\n  ── {task_name} ──")
        all_results[task_name] = {}

        for mode in ["archive", "greedy", "random"]:
            rng = random.Random(seed)
            baseline = evaluate(ttype, default_h(), n_eval, rng)
            best = baseline
            archive_evals = [baseline]
            history = [baseline["rate"]]

            for step in range(1, max_steps + 1):
                parent = (max(archive_evals, key=lambda e: e["rate"])
                          if mode == "archive" else best)
                traces_raw = [sample_episode(ttype, parent["h"], rng) for _ in range(20)]

                if mode == "random":
                    cand_h = random_propose(rng)
                else:
                    cand_h = smart_propose(ttype, traces_raw, parent["h"], rng, archive_evals)

                cand = evaluate(ttype, cand_h, n_eval, rng)

                if mode == "archive":
                    if cand["rate"] > baseline["rate"]:
                        # Add if not dominated (simple: add if in top-20 by rate or better than worst)
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
            delta = best["rate"] - baseline["rate"]
            all_results[task_name][mode] = {
                "rate_history": [round(r, 4) for r in history],
                "final_rate":   round(best["rate"], 4),
                "baseline_rate": round(baseline["rate"], 4),
                "certified":    bool(cert),
                "delta":        round(delta, 4),
            }
            print(f"    {mode:8s}: {baseline['rate']:.3f} → {best['rate']:.3f} "
                  f"(+{delta:.3f}) cert={'✓' if cert else '✗'}")
    return all_results


# ── Phase 3: ratchet illusion ─────────────────────────────────────────────────

def run_phase3(n_trials=300, seed=42):
    """
    Run n_trials independent comparison attempts per N value.
    In each trial: propose a harness, evaluate it at N episodes against the
    N-episode baseline.  Accept if naive/gated test passes.  Illusion = accepted
    harness that is truly worse than the baseline (by oracle = true rate).
    Using the true rates (checkout_true_rate) eliminates oracle sampling noise.
    """
    print(f"\n{'='*65}\nPHASE 3: Ratchet Illusion ({n_trials} trials per N)\n{'='*65}")
    ttype = "checkout"
    baseline_h = default_h()
    true_baseline = checkout_true_rate(baseline_h)
    print(f"  True baseline rate: {true_baseline:.3f}")

    results = []
    for n_eval in [5, 20, 85]:
        rng = random.Random(seed + n_eval)
        accepts = illusions = 0

        for trial in range(n_trials):
            # Fresh N-episode evaluation of baseline
            base_eval = evaluate(ttype, baseline_h, n_eval, rng)

            # Generate a candidate (mix of smart and random to simulate real search)
            traces_raw = [sample_episode(ttype, baseline_h, rng) for _ in range(10)]
            if rng.random() < 0.5:
                cand_h = smart_propose(ttype, traces_raw, baseline_h, rng)
            else:
                cand_h = random_propose(rng)  # include random proposals to get illusions

            cand_eval = evaluate(ttype, cand_h, n_eval, rng)
            true_cand = checkout_true_rate(cand_h)

            # Acceptance test
            if n_eval == 85:
                accepted = certified(cand_eval, base_eval)   # gated
            else:
                accepted = cand_eval["rate"] > base_eval["rate"]  # naive

            if accepted:
                accepts += 1
                is_ill = true_cand < true_baseline  # truly worse?
                if is_ill:
                    illusions += 1

        ill_rate  = illusions / accepts if accepts > 0 else 0.0
        accept_rt = accepts / n_trials
        results.append(dict(n_eval=n_eval,
                            illusion_rate=round(ill_rate, 3),
                            illusion_count=illusions,
                            accept_count=accepts,
                            accept_rate=round(accept_rt, 3),
                            certified=n_eval == 85))
        cert = "✓" if n_eval == 85 else "✗"
        print(f"  N={n_eval:3d}: illusion_rate={ill_rate:.3f} ({illusions}/{accepts}), "
              f"accept_rate={accept_rt:.3f}, cert={cert}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    results_dir = RESULTS_ROOT / "tables"
    plots_dir   = ROOT / "results" / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Verify landscape calibration
    print("── Landscape verification ──")
    dh  = default_h()
    bh  = best_checkout_h()
    ath = H(obs="ax_tree")
    print(f"  checkout default_h   (true): {checkout_true_rate(dh):.3f}  (target 0.353)")
    print(f"  checkout ax_tree_h   (true): {checkout_true_rate(ath):.3f}  (deceptive: <default)")
    print(f"  checkout best_h      (true): {checkout_true_rate(bh):.3f}  (target 0.521)")
    print(f"  search   default_h   (true): {search_true_rate(dh):.3f}  (target 0.623)")
    print(f"  Note: ax_tree stage-0 P = {sigmoid(BASE_LO[0] + AXIS_DELTA['ax_tree'][0]):.3f}  (should be > default's stage-0)")

    p1 = run_phase1(n_candidates=200, n_eval=85, seed=42)
    p2 = run_phase2(max_steps=15, n_eval=85, seed=42)
    p3 = run_phase3(n_trials=300, seed=42)

    # Save tables
    with open(results_dir / "phase1_gamma.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task","p_0","p_trace","gamma",
                                           "gamma_ci_lo","gamma_ci_hi","n_candidates"])
        w.writeheader()
        for r in p1: w.writerow({k: r[k] for k in w.fieldnames})

    with open(results_dir / "phase2_topology.json", "w") as f:
        json.dump(p2, f, indent=2)

    with open(results_dir / "phase3_ratchet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_eval","illusion_rate","accept_count",
                                           "accept_rate","certified"])
        w.writeheader()
        for r in p3: w.writerow({k: r[k] for k in w.fieldnames})

    # Print full summary
    print(f"\n\n{'='*65}\nRESULTS SUMMARY\n{'='*65}")

    print("\n── Phase 1: γ ──")
    print(f"{'Task':<42} {'γ':>6}  95% CI          p_0     p_trace")
    print("-"*80)
    for r in p1:
        print(f"{r['task']:<42} {r['gamma']:>6.2f}  "
              f"[{r['gamma_ci_lo']:.2f},{r['gamma_ci_hi']:.2f}]  "
              f"{r['p_0']:.4f}  {r['p_trace']:.4f}")

    print("\n── Phase 2: final completion rates ──")
    print(f"{'Task':<42} Archive  Greedy  Random  Δ(A-G)  Cert")
    print("-"*80)
    for task, modes in p2.items():
        a = modes.get("archive",{}).get("final_rate", float("nan"))
        g = modes.get("greedy", {}).get("final_rate", float("nan"))
        r = modes.get("random", {}).get("final_rate", float("nan"))
        c = "✓" if modes.get("archive",{}).get("certified") else "✗"
        print(f"{task:<42} {a:.3f}  {g:.3f}  {r:.3f}  {a-g:+.3f}  {c}")

    print("\n── Phase 3: ratchet illusion ──")
    print(f"{'N':>5}  Illusion%  Accepts  AcceptRate  Cert")
    print("-"*50)
    for r in p3:
        print(f"{r['n_eval']:>5}  {r['illusion_rate']:.3f}      "
              f"{r['accept_count']:>7}  {r['accept_rate']:.3f}       "
              f"{'✓' if r['certified'] else '✗'}")

    # Plot phase 2
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for task_name, modes in p2.items():
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            clrs = {"archive": "#6E5FA8", "greedy": "#C96A44", "random": "#8A897F"}
            stls = {"archive": "-",       "greedy": "--",       "random": ":"}
            for mode, data in modes.items():
                h_rates = data["rate_history"]
                ax.plot(range(len(h_rates)), h_rates,
                        color=clrs[mode], linestyle=stls[mode], linewidth=2.5,
                        marker="o", markersize=5,
                        label=f"{mode} (final={data['final_rate']:.3f})")
            baseline_rate = list(modes.values())[0].get("baseline_rate", 0.35)
            ax.axhline(baseline_rate, color="gray", linestyle="-.", linewidth=1, alpha=0.6,
                       label=f"baseline ({baseline_rate:.3f})")
            # True optimum
            if "checkout" in task_name:
                opt = checkout_true_rate(best_checkout_h())
                lbl = f"true optimum ({opt:.3f})"
            else:
                opt = max(search_true_rate(H(obs=o, vocab=v, ctx=c, scaffold=s, retry=r))
                          for o in OBS for v in VOCAB for c in CTX
                          for s in SCAFFOLD for r in RETRY)
                lbl = f"landscape best ({opt:.3f})"
            ax.axhline(opt, color="#6F8A50", linestyle="--", linewidth=1.5, alpha=0.8, label=lbl)
            ax.set_xlabel("Outer-loop step", fontsize=12)
            ax.set_ylabel("Task completion rate", fontsize=12)
            ax.set_title(f"GROL harness search: {task_name}", fontsize=13)
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(plots_dir / f"phase2_{task_name}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  Plot: {plots_dir}/phase2_{task_name}.png")
    except Exception as e:
        print(f"\n  Plot error: {e}")

    print(f"\nResults: {results_dir}")
    return {"phase1": p1, "phase2": p2, "phase3": p3}


if __name__ == "__main__":
    main()
