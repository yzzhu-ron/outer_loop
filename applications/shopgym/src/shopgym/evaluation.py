"""ShopGym evaluation records and episode aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from outer_loop.evaluation import WilsonCI, wilson_ci
from shopgym.harness import HarnessConfig


class EpisodeEnvironment(Protocol):
    def run_n_episodes(
        self, task_category: str, harness: HarnessConfig, n: int
    ) -> list: ...


@dataclass
class HarnessEval:
    """Evaluation result satisfying the generic `outer_loop.Evaluation` contract."""

    harness: HarnessConfig
    n_episodes: int
    n_success: int
    ci: WilsonCI
    stage_completions: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def artifact(self) -> HarnessConfig:
        return self.harness

    @property
    def completion_rate(self) -> float:
        return self.n_success / self.n_episodes if self.n_episodes else 0.0

    @property
    def score(self) -> float:
        return self.completion_rate

    @property
    def interval(self) -> WilsonCI:
        return self.ci

    @property
    def objectives(self) -> tuple[float, ...]:
        if not self.stage_completions:
            return (self.completion_rate,)
        return tuple(
            self.stage_rate(stage) for stage in sorted(self.stage_completions)
        )

    def stage_rate(self, stage: int) -> float:
        successes, trials = self.stage_completions.get(stage, (0, 0))
        return successes / trials if trials else 0.0


def evaluate_harness(
    environment: EpisodeEnvironment,
    harness: HarnessConfig,
    task_category: str,
    n: int,
    alpha: float = 0.05,
) -> HarnessEval:
    """Run repeated episodes and aggregate task and stage completion."""

    from shopgym.environment import TASK_STAGES

    results = environment.run_n_episodes(task_category, harness, n)
    successes = sum(result.success for result in results)
    stage_completions = {
        index: (
            sum(1 for result in results if len(result.stages_completed) > index),
            n,
        )
        for index, _stage in enumerate(TASK_STAGES[task_category])
    }
    return HarnessEval(
        harness=harness,
        n_episodes=n,
        n_success=successes,
        ci=wilson_ci(successes, n, alpha=alpha),
        stage_completions=stage_completions,
    )
