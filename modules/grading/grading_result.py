# modules/grading/grading_result.py
"""Grading result model for Product Grading Core."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ProductGrade:
    """Structure representing the result of product grading evaluation.

    Attributes
    ----------
    grade : str
        The final assigned grade: "PASS", "REVIEW", or "FAIL".
    confidence : float
        Decision confidence value in range [0.0, 1.0].
    reason_codes : List[str]
        Machine-readable keys representing triggered grading rules.
    reasons : List[str]
        Human-readable explanations for the assigned grade.
    evidence : Dict[str, Any]
        Dictionary summarizing the raw metrics and status levels used in decision making.
    """

    grade: str
    confidence: float
    reason_codes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert grading results into a JSON-serialisable dictionary."""
        return {
            "grade": self.grade,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "reasons": self.reasons,
            "evidence": self.evidence,
        }
