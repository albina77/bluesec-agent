from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic


@dataclass
class InvestigationBudget:
    """Separate model proposals, remote calls, wall time, and finish reserve."""

    max_calls: int
    max_proposals: int
    timeout_seconds: float
    finish_reserve_seconds: float = 45.0
    clock: Callable[[], float] = field(default=monotonic, repr=False)
    calls: int = 0
    proposals: int = 0
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.timeout_seconds - (self.clock() - self.started_at))

    @property
    def finish_only(self) -> bool:
        return (self.calls >= self.max_calls - 1 or self.proposals >= self.max_proposals - 5
                or self.remaining_seconds <= self.finish_reserve_seconds)

    @property
    def exhausted(self) -> bool:
        return (self.calls >= self.max_calls or self.proposals >= self.max_proposals
                or self.remaining_seconds <= 0)

    def proposal(self) -> None:
        self.proposals += 1

    def call(self) -> None:
        self.calls += 1

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "calls_remaining": max(0, self.max_calls - self.calls),
            "proposals": self.proposals,
            "proposals_remaining": max(0, self.max_proposals - self.proposals),
            "seconds_remaining": round(self.remaining_seconds, 3),
            "finish_only": self.finish_only,
            "exhausted": self.exhausted,
        }
