"""Run archive, greedy, or random harness search in ShopGym."""

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from outer_loop.archive import ParetoArchive
from outer_loop.selection import GatedRatchet
from shopgym.environment import MockShopGymEnv, PlaywrightShopGymEnv
from shopgym.evaluation import HarnessEval, evaluate_harness
from shopgym.harness import HarnessConfig, default_harness, random_harness
from shopgym.proposer import LLMHarnessProposer, MockLLMProposer


@dataclass
class GROLStep:
    step: int
    proposed_harness: HarnessConfig
    eval_result: HarnessEval
    accepted: bool
    archive_size: int
    best_rate: float
    elapsed_sec: float


@dataclass
class GROLRun:
    task_category: str
    mode: str  # "archive", "greedy", "random"
    steps: list[GROLStep] = field(default_factory=list)
    best_eval: Optional[HarnessEval] = None
    total_episodes: int = 0
    certified_improvement: bool = False

    def completion_rates(self) -> list[float]:
        return [s.best_rate for s in self.steps]

    def accepted_harnesses(self) -> list[HarnessConfig]:
        return [s.proposed_harness for s in self.steps if s.accepted]


def run_grol(task_category: str,
             initial_harness: Optional[HarnessConfig] = None,
             mode: str = "archive",     # "archive", "greedy", "random"
             n_eval: int = 85,
             alpha: float = 0.05,
             archive_size: int = 20,
             max_steps: int = 15,
             n_trace: int = 10,
             use_mock_proposer: bool = True,
             use_live_env: bool = False,
             rng_seed: Optional[int] = None) -> GROLRun:
    """
    Run GROL (or greedy/random ablation) for one task category.

    Args:
        task_category: one of the 8 ShopGym task categories
        initial_harness: starting harness (default: screenshot+high_level+last_3+none+once)
        mode: "archive" (GROL), "greedy", or "random"
        n_eval: episodes per evaluation
        alpha: significance level for Wilson CI
        archive_size: max Pareto archive size
        max_steps: number of outer-loop iterations
        n_trace: episodes to collect traces from
        use_mock_proposer: use MockLLMProposer instead of live API
        use_live_env: use PlaywrightShopGymEnv instead of MockShopGymEnv
        rng_seed: random seed

    Returns:
        GROLRun with full history
    """
    if mode not in {"archive", "greedy", "random"}:
        raise ValueError(f"unknown search mode: {mode}")
    env = (
        PlaywrightShopGymEnv()
        if use_live_env
        else MockShopGymEnv(rng_seed=rng_seed)
    )
    rng = random.Random(rng_seed)
    if initial_harness is None:
        initial_harness = default_harness()

    if use_mock_proposer:
        proposer = MockLLMProposer(rng_seed=rng_seed)
    else:
        proposer = LLMHarnessProposer()

    ratchet = GatedRatchet(n_eval=n_eval, alpha=alpha)
    archive = ParetoArchive(max_size=archive_size)

    # Phase 0: evaluate baseline
    baseline_eval = evaluate_harness(env, initial_harness, task_category, n_eval, alpha)
    ratchet.set_incumbent(baseline_eval)
    archive.try_add(baseline_eval)

    run = GROLRun(task_category=task_category, mode=mode)
    run.total_episodes = n_eval
    run.best_eval = baseline_eval

    print(f"[{mode}/{task_category}] Baseline: {baseline_eval.completion_rate:.3f} {baseline_eval.ci}")

    for t in range(1, max_steps + 1):
        t_start = time.time()

        # Select parent harness
        if mode == "archive":
            parent_eval = archive.ucb_select(t)
        else:
            parent_eval = ratchet.incumbent

        parent_harness = parent_eval.harness

        # Collect traces
        trace_episodes = env.run_n_episodes(task_category, parent_harness, n_trace)
        run.total_episodes += n_trace

        # Propose new harness
        if mode == "random":
            candidate_harness = random_harness(rng)
        else:
            candidate_harness = proposer.propose(trace_episodes, parent_harness, archive)

        # Skip if we've seen this harness before
        seen = any(s.proposed_harness == candidate_harness for s in run.steps)
        if seen:
            candidate_harness = random_harness(rng)

        # Evaluate candidate
        candidate_eval = evaluate_harness(env, candidate_harness, task_category, n_eval, alpha)
        run.total_episodes += n_eval

        # Acceptance decision
        if mode == "archive":
            # Gated test vs baseline for certification
            certified_over_baseline = (
                candidate_eval.ci.lower > baseline_eval.ci.upper
            )
            # Try to add to archive (Pareto non-dominated)
            added = archive.try_add(candidate_eval) if certified_over_baseline else False
            accepted = added
            # Update greedy incumbent if also greedy-better
            if candidate_eval.completion_rate > ratchet.incumbent.completion_rate:
                ratchet.accept(candidate_eval)
        elif mode == "greedy":
            accepted = ratchet.accept(candidate_eval)  # gated test
        else:  # random
            accepted = candidate_eval.completion_rate > ratchet.incumbent.completion_rate
            if accepted:
                ratchet.incumbent = candidate_eval

        # Track best
        best = archive.best() if mode == "archive" else ratchet.incumbent
        if best and best.completion_rate > run.best_eval.completion_rate:
            run.best_eval = best

        step = GROLStep(
            step=t,
            proposed_harness=candidate_harness,
            eval_result=candidate_eval,
            accepted=accepted,
            archive_size=len(archive.entries) if mode == "archive" else 1,
            best_rate=run.best_eval.completion_rate,
            elapsed_sec=time.time() - t_start,
        )
        run.steps.append(step)

        status = "✓ accepted" if accepted else "✗ rejected"
        print(f"  Step {t:2d}: {candidate_harness.obs_modality:12s} | "
              f"rate={candidate_eval.completion_rate:.3f} {candidate_eval.ci} | "
              f"{status} | best={run.best_eval.completion_rate:.3f}")

    # Check if certified improvement over baseline
    if run.best_eval:
        run.certified_improvement = (
            run.best_eval.ci.lower > baseline_eval.ci.upper
        )

    print(f"[{mode}/{task_category}] Final best: {run.best_eval.completion_rate:.3f} "
          f"| certified={run.certified_improvement} | episodes={run.total_episodes}")

    return run
