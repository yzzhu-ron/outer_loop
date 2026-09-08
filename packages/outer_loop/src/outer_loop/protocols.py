"""Application-neutral contracts for an outer-loop search."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from outer_loop.evaluation import WilsonCI

ArtifactT = TypeVar("ArtifactT")
TraceT = TypeVar("TraceT")


@runtime_checkable
class Evaluation(Protocol):
    """The information selection policies consume from one evaluation."""

    @property
    def artifact(self) -> Any: ...

    @property
    def score(self) -> float: ...

    @property
    def interval(self) -> WilsonCI: ...

    @property
    def objectives(self) -> tuple[float, ...]: ...


class Evaluator(Protocol[ArtifactT]):
    """Evaluate one artifact under an application-defined task protocol."""

    def evaluate(self, artifact: ArtifactT) -> Evaluation: ...


class ArchiveView(Protocol):
    """Read-only archive surface available to a proposer."""

    entries: Sequence[Evaluation]


class Proposer(Protocol[ArtifactT, TraceT]):
    """Propose a new artifact from traces and prior evaluations."""

    def propose(
        self,
        traces: Sequence[TraceT],
        parent: ArtifactT,
        archive: ArchiveView | None = None,
    ) -> ArtifactT: ...
