"""Minimal exact-match scorer for short-answer evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _as_answer_list(ground_truth: Any) -> list[Any]:
    """Normalize a scalar or sequence of reference answers to a list."""

    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, str | bytes):
        return list(ground_truth)
    return [ground_truth]


def compute_score(solution_str: str, ground_truth: Any) -> float:
    """Return 1.0 iff a predicted short answer exactly matches a reference.

    The caller is responsible for extracting the final answer from any output
    protocol. This scorer intentionally performs no semantic, case, punctuation,
    article, or alias normalization; only surrounding whitespace is ignored.
    """

    if not isinstance(solution_str, str):
        return 0.0
    prediction = solution_str.strip()
    return float(
        any(isinstance(reference, str) and prediction == reference.strip() for reference in _as_answer_list(ground_truth))
    )
