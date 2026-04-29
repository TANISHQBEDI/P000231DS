from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceRoute:
    decision: str
    confidence: float


class ConfidenceRouter:
    def __init__(self, auto_accept: float = 0.75, review: float = 0.5) -> None:
        self.auto_accept = auto_accept
        self.review = review

    def route(self, confidence: float) -> ConfidenceRoute:
        if confidence >= self.auto_accept:
            return ConfidenceRoute(decision="AUTO_ACCEPT", confidence=confidence)
        if confidence >= self.review:
            return ConfidenceRoute(decision="REVIEW", confidence=confidence)
        return ConfidenceRoute(decision="LOW_CONFIDENCE", confidence=confidence)
