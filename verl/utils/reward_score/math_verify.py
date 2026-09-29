"""Math-Verify based scorer for evaluation or opt-in reward routing.

This module is intentionally not wired into default_compute_score so that
existing training rewards keep their exact-match behaviour unless a caller
explicitly selects this scorer.
"""

from __future__ import annotations

from typing import Any

from .math_reward import last_boxed_only_string

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    _MATH_VERIFY_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    TimeoutException = None
    math_metric = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    _MATH_VERIFY_IMPORT_ERROR = error


def _require_math_verify() -> None:
    if _MATH_VERIFY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "math-verify is required for the math_verify scorer. "
            "Install it with: pip install math-verify."
        ) from _MATH_VERIFY_IMPORT_ERROR


def _score_one(model_output: str, ground_truth: str, timeout_score: float) -> float:
    _require_math_verify()
    prediction_boxed = last_boxed_only_string(model_output)
    if prediction_boxed is None:
        return 0.0

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    try:
        # Gold answers in our parquet files are bare expressions. Restrict the
        # prediction to its final boxed answer to match the output protocol.
        score, _ = verify_func([f"\\boxed{{{ground_truth}}}"], [prediction_boxed])
        return float(score)
    except TimeoutException:
        return float(timeout_score)
    except Exception:
        # Malformed LaTex or an extraction failure is an incorrect answer.
        return 0.0


def compute_score(model_output: str, ground_truth: str | int | float | list[Any], timeout_score: float = 0.0) -> float:
    """Return binary mathematical equivalence of the final answer and gold."""

    if isinstance(ground_truth, list):
        return max(
            (_score_one(model_output, str(answer), timeout_score) for answer in ground_truth),
            default=0.0,
        )
    return _score_one(model_output, str(ground_truth), timeout_score)


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: str | int | float | list[Any],
    extra_info: dict[str, Any] | None = None,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore: Any = None,
) -> float:
    """Standard reward-score interface; data_source and extra fields are unused."""

    del data_source, extra_info, sandbox_fusion_url, concurrent_semaphore
    return compute_score(solution_str, ground_truth)
