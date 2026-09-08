"""Selection policies for noisy outer-loop evaluations."""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from outer_loop.evaluation import gated_accept
from outer_loop.protocols import Evaluation

EvaluationT = TypeVar("EvaluationT", bound=Evaluation)


class GatedRatchet(Generic[EvaluationT]):
    """Keep an incumbent only when a candidate clears a confidence gate."""

    def __init__(self, n_eval: int = 85, alpha: float = 0.05):
        self.n_eval = n_eval
        self.alpha = alpha
        self.incumbent: Optional[EvaluationT] = None
        self.history: list[EvaluationT] = []

    def set_incumbent(self, evaluation: EvaluationT) -> None:
        self.incumbent = evaluation
        self.history.append(evaluation)

    def test(self, candidate: EvaluationT) -> bool:
        if self.incumbent is None:
            return True
        return gated_accept(candidate.interval, self.incumbent.interval)

    def accept(self, candidate: EvaluationT) -> bool:
        accepted = self.test(candidate)
        if accepted:
            self.incumbent = candidate
        self.history.append(candidate)
        return accepted

    def naive_test(self, candidate: EvaluationT) -> bool:
        """Point-estimate comparison retained for controlled ablations."""
        if self.incumbent is None:
            return True
        return candidate.score > self.incumbent.score
