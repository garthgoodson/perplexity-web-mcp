"""Smart quota-aware routing data structures.

Provides enums, dataclasses, and helpers for classifying quota levels
and representing routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import Model
from .rate_limits import RateLimits


class QuotaLevel(str, Enum):
    HEALTHY = "healthy"
    LOW = "low"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


class Intent(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DETAILED = "detailed"
    RESEARCH = "research"


def _classify(remaining: int, maximum: int) -> QuotaLevel:
    """Classify pro-style quota: 0=exhausted, <10%=critical, <20%=low, else healthy."""
    if remaining <= 0:
        return QuotaLevel.EXHAUSTED
    if maximum <= 0:
        return QuotaLevel.EXHAUSTED
    pct = remaining / maximum
    if pct < 0.10:
        return QuotaLevel.CRITICAL
    if pct < 0.20:
        return QuotaLevel.LOW
    return QuotaLevel.HEALTHY


def _classify_research(remaining: int, maximum: int) -> QuotaLevel:
    """Classify research quota: 0=exhausted, <20%=critical, <50%=low, else healthy."""
    if remaining <= 0:
        return QuotaLevel.EXHAUSTED
    if maximum <= 0:
        return QuotaLevel.EXHAUSTED
    pct = remaining / maximum
    if pct < 0.20:
        return QuotaLevel.CRITICAL
    if pct < 0.50:
        return QuotaLevel.LOW
    return QuotaLevel.HEALTHY


@dataclass(frozen=True, slots=True)
class QuotaState:
    """Snapshot of current quota levels across all resource types."""

    pro_remaining: int
    pro_level: QuotaLevel
    research_remaining: int
    research_level: QuotaLevel
    labs_remaining: int
    agent_remaining: int

    @classmethod
    def from_rate_limits(
        cls,
        limits: RateLimits,
        pro_max: int = 300,
        research_max: int = 10,
    ) -> QuotaState:
        return cls(
            pro_remaining=limits.remaining_pro,
            pro_level=_classify(limits.remaining_pro, pro_max),
            research_remaining=limits.remaining_research,
            research_level=_classify_research(limits.remaining_research, research_max),
            labs_remaining=limits.remaining_labs,
            agent_remaining=limits.remaining_agentic_research,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pro_remaining": self.pro_remaining,
            "pro_level": self.pro_level.value,
            "research_remaining": self.research_remaining,
            "research_level": self.research_level.value,
            "labs_remaining": self.labs_remaining,
            "agent_remaining": self.agent_remaining,
        }


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Represents the outcome of the smart routing algorithm."""

    model: Model
    model_name: str
    search_type: str
    intent: Intent
    reason: str
    was_downgraded: bool
    quota_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SmartResponse:
    """A query response bundled with routing metadata."""

    answer: str
    citations: list[str]
    routing: RoutingDecision

    def format_metadata_block(self) -> str:
        r = self.routing
        qs = r.quota_snapshot
        lines = [
            f"Routing: {r.model_name} | {r.search_type} | {r.intent.value} intent",
            f"Reason: {r.reason}",
            (
                f"Quota: Pro {qs.get('pro_remaining', '?')}"
                f" | Research {qs.get('research_remaining', '?')}"
                f" | Labs {qs.get('labs_remaining', '?')}"
                f" | Agent {qs.get('agent_remaining', '?')}"
            ),
            f"Downgraded: {'Yes' if r.was_downgraded else 'No'}",
        ]
        return "\n".join(lines)

    def format_response(self) -> str:
        parts = [self.answer]
        if self.citations:
            parts.append("\n".join(self.citations))
        parts.append("---")
        parts.append(self.format_metadata_block())
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "routing": {
                "model": self.routing.model.identifier,
                "model_name": self.routing.model_name,
                "search_type": self.routing.search_type,
                "intent": self.routing.intent.value,
                "reason": self.routing.reason,
                "was_downgraded": self.routing.was_downgraded,
                "quota_snapshot": self.routing.quota_snapshot,
            },
        }
