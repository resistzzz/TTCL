"""Shared prompt construction for RLCR data-preprocessing scripts."""

from __future__ import annotations

CONFIDENCE_INSTRUCTIONS = {
    # Kept as the default to reproduce every existing parquet and experiment.
    "numeric_example": """Also output your confidence for this answer after <conf> as a floating-point number between 0 (complete uncertainty) and 1 (full confidence).
Example: <conf> Confidence: 0.83
Ensure that the confidence output appears on a single line and strictly follows the specified format.""",
    # Avoid anchoring models such as Llama to a literal numeric example.
    "placeholder_example": """Also output your confidence for this answer after <conf> as a floating-point number between 0 (complete uncertainty) and 1 (full confidence).
Example: <conf> Confidence: <float value>
Ensure that the confidence output appears on a single line and strictly follows the specified format.""",
}

ANSWER_PROMPT_STYLES = ("standard", "strict_ending")


def _strict_ending_instruction(answer_instruction: str, require_confidence: bool) -> str:
    """Build the explicit terminal-output contract used for Llama evaluation."""

    answer_terminal = (
        r"\boxed{<final answer>}"
        if r"\boxed" in answer_instruction
        else "<answer><final answer></answer>"
    )
    if require_confidence:
        return f"""Solve the following problem step by step.

Also output your confidence for the final answer. Confidence must be a
floating-point number between 0 (complete uncertainty) and 1 (full
confidence).

Your response must end with exactly these two lines:
{answer_terminal}
<conf> Confidence: <a floating-point number between 0 and 1>

Do not write anything after the confidence line."""
    return f"""Solve the following problem step by step.

End your response with exactly:
{answer_terminal}

Do not write anything after the final answer."""


def add_confidence_prompt_args(parser) -> None:
    """Add shared CLI switches used by every preprocessing script."""

    import argparse

    parser.add_argument("--require-confidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--confidence-prompt-style",
        choices=tuple(CONFIDENCE_INSTRUCTIONS),
        default="numeric_example",
        help=(
            "'numeric_example' preserves existing parquets; "
            "'placeholder_example' uses '<float value>' to avoid anchoring to 0.83."
        ),
    )
    parser.add_argument(
        "--answer-prompt-style",
        choices=ANSWER_PROMPT_STYLES,
        default="standard",
        help=(
            "'standard' preserves the existing prompt; 'strict_ending' adds "
            "an explicit final-answer/output-order contract for Llama evaluation."
        ),
    )


def build_system_instruction(
    answer_instruction: str,
    require_confidence: bool,
    confidence_prompt_style: str = "numeric_example",
    answer_prompt_style: str = "standard",
) -> str:
    """Append the verbalized-confidence protocol only when explicitly requested."""

    if answer_prompt_style == "strict_ending":
        return _strict_ending_instruction(answer_instruction, require_confidence)
    if answer_prompt_style != "standard":
        raise ValueError(f"Unknown answer prompt style: {answer_prompt_style!r}")
    if not require_confidence:
        return answer_instruction
    return f"{answer_instruction}\n{CONFIDENCE_INSTRUCTIONS[confidence_prompt_style]}"


def confidence_output_suffix(
    require_confidence: bool,
    confidence_prompt_style: str = "numeric_example",
    answer_prompt_style: str = "standard",
) -> str:
    """Return a suffix that prevents alternate prompts overwriting old files."""

    parts = []
    if not require_confidence:
        parts.append("no_conf")
    elif confidence_prompt_style != "numeric_example":
        parts.append("placeholder_conf")
    if answer_prompt_style != "standard":
        parts.append(answer_prompt_style)
    return "" if not parts else "_" + "_".join(parts)
