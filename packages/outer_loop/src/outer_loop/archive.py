"""Pareto archive and parent selection for outer-loop search."""

from __future__ import annotations

import math
from typing import Generic, Optional, TypeVar

from outer_loop.protocols import Evaluation

EvaluationT = TypeVar("EvaluationT", bound=Evaluation)


class ParetoArchive(Generic[EvaluationT]):
    """Maintain non-dominated evaluations and select parents with UCB1."""

    def __init__(self, max_size: int = 20):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.entries: list[EvaluationT] = []
        self._selection_counts: dict[int, int] = {}

    @staticmethod
    def dominates(a: EvaluationT, b: EvaluationT) -> bool:
        a_vector = a.objectives or (a.score,)
        b_vector = b.objectives or (b.score,)
        if len(a_vector) != len(b_vector):
            return a.score > b.score
        return all(x >= y for x, y in zip(a_vector, b_vector)) and any(
            x > y for x, y in zip(a_vector, b_vector)
        )

    def try_add(self, candidate: EvaluationT) -> bool:
        if any(self.dominates(entry, candidate) for entry in self.entries):
            return False
        self.entries = [
            entry
            for entry in self.entries
            if not self.dominates(candidate, entry)
        ]
        self.entries.append(candidate)
        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda entry: entry.score, reverse=True)
            self.entries = self.entries[: self.max_size]
        return candidate in self.entries

    def best(self) -> Optional[EvaluationT]:
        return max(self.entries, key=lambda entry: entry.score, default=None)

    def ucb_select(self, step: int, exploration: float = 1.4) -> EvaluationT:
        if not self.entries:
            raise ValueError("archive is empty")

        for entry in self.entries:
            key = hash(entry.artifact)
            if self._selection_counts.get(key, 0) == 0:
                self._selection_counts[key] = 1
                return entry

        def ucb(entry: EvaluationT) -> float:
            count = self._selection_counts[hash(entry.artifact)]
            return entry.score + exploration * math.sqrt(
                math.log(max(step, 1) + 1) / count
            )

        selected = max(self.entries, key=ucb)
        key = hash(selected.artifact)
        self._selection_counts[key] += 1
        return selected

    def summary(self) -> str:
        if not self.entries:
            return "Archive: empty"
        lines = [f"Archive ({len(self.entries)} entries):"]
        for entry in sorted(self.entries, key=lambda item: item.score, reverse=True):
            lines.append(f"  {entry.artifact!r} -> {entry.score:.3f}")
        return "\n".join(lines)
