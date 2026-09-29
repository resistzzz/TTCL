"""Reward manager for Reinforcement Learning with Calibration Rewards (RLCR)."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


# ``Confidence:`` is optional because several instruction-tuned models emit
# ``<conf> 0.83`` even when the requested protocol includes that label.
_CONFIDENCE_PATTERN = re.compile(
    r"<conf>[ \t\r\n]*(?:Confidence:[ \t\r\n]*)?"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def _last_boxed_expression(text: str) -> str | None:
    """Return the last complete ``\\boxed{...}`` expression in *text*."""

    start = text.rfind("\\boxed{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                boxed = text[start : index + 1]
                content = boxed[len("\\boxed{") : -1].strip()
                # An empty answer, or a confidence protocol marker embedded in
                # the answer slot (e.g. ``\\boxed{<conf>}``), is malformed.
                # Treat the final box as invalid rather than falling back to an
                # earlier box: the last box is the model's declared answer.
                if not content or "<conf>" in content.casefold():
                    return None
                return boxed
    return None


def _boxed_text_content(boxed_answer: str) -> str:
    """Extract textual content from a boxed answer without touching math mode."""

    inner = boxed_answer[len("\\boxed{") : -1].strip()
    if not inner.startswith("\\text{"):
        return inner

    text_start = len("\\text")
    depth = 0
    for index in range(text_start, len(inner)):
        if inner[index] == "{":
            depth += 1
        elif inner[index] == "}":
            depth -= 1
            if depth == 0:
                if index == len(inner) - 1:
                    return inner[text_start + 1 : index].strip()
                break
    return inner


def _last_answer_tag_content(text: str) -> tuple[str, int] | None:
    """Return the final non-empty answer-tag content and end offset."""

    matches = list(re.finditer(r"<answer\s*>(.*?)</answer\s*>", text, flags=re.DOTALL))
    if not matches:
        return None
    match = matches[-1]
    answer = match.group(1).strip()
    return (answer, match.end()) if answer else None


def _extract_final_answer(text: str, parse_mode: str = "boxed") -> str | None:
    """Extract the final answer according to the sample output protocol."""

    if parse_mode == "boxed":
        return _last_boxed_expression(text)
    if parse_mode == "boxed_text":
        boxed_answer = _last_boxed_expression(text)
        return _boxed_text_content(boxed_answer) if boxed_answer is not None else None
    if parse_mode == "answer_tag":
        answer_tag = _last_answer_tag_content(text)
        return answer_tag[0] if answer_tag is not None else None
    raise ValueError(
        f"Unsupported parse_mode {parse_mode!r}; expected 'boxed', 'boxed_text', or 'answer_tag'."
    )


def _answer_end_offset(text: str, parse_mode: str) -> int | None:
    """Return the ending character offset of the final answer protocol."""

    if parse_mode in {"boxed", "boxed_text"}:
        answer = _last_boxed_expression(text)
        return text.rfind(answer) + len(answer) if answer is not None else None
    if parse_mode == "answer_tag":
        answer_tag = _last_answer_tag_content(text)
        return answer_tag[1] if answer_tag is not None else None
    raise ValueError(
        f"Unsupported parse_mode {parse_mode!r}; expected 'boxed', 'boxed_text', or 'answer_tag'."
    )


def _parse_confidence(text: str, parse_mode: str = "boxed") -> float | None:
    """Parse numeric confidence appearing after the final parsed answer."""

    answer_end = _answer_end_offset(text, parse_mode)
    if answer_end is None:
        return None

    after_answer = text[answer_end:]
    match = _CONFIDENCE_PATTERN.search(after_answer)
    if match is None:
        return None

    try:
        confidence = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence

def _has_valid_format(text: str, parse_mode: str = "boxed") -> bool:
    """Return whether a parsed final answer is followed by valid confidence."""

    return _parse_confidence(text, parse_mode=parse_mode) is not None

@register("rlcr")
class RLCRRewardManager(AbstractRewardManager):
    """Implement the Brier-score reward from the RLCR paper for math tasks.

    The model must emit a final answer in ``\\boxed{}`` and a confidence line in
    the form ``<conf> Confidence: 0.83`` or ``<conf> 0.83``. Valid responses
    receive ``correctness - brier_weight * (confidence - correctness)^2``.
    Invalid responses receive ``-(brier_weight + invalid_format_margin)``.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        format_reward: float = 0.5,
        brier_weight: float = 1.0,
        invalid_format_margin: float = 1.0,
        separate_correctness_and_calibration: bool = False,
        loss_mask_mode: str = "confidence_or_all",
    ) -> None:
        if not math.isfinite(format_reward) or format_reward < 0.0:
            raise ValueError(f"format_reward must be a finite non-negative value, got {format_reward}.")
        if not math.isfinite(brier_weight) or brier_weight < 0.0:
            raise ValueError(f"brier_weight must be a finite non-negative value, got {brier_weight}.")
        if not math.isfinite(invalid_format_margin) or invalid_format_margin <= 0.0:
            raise ValueError(
                "invalid_format_margin must be a finite positive value, "
                f"got {invalid_format_margin}."
            )
        if loss_mask_mode not in {"confidence", "confidence_or_all"}:
            raise ValueError("rlcr.loss_mask_mode must be 'confidence' or 'confidence_or_all'.")
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # Retained for compatibility with existing launch scripts. Format is
        # monitored separately and does not contribute to the reward.
        self.format_reward = float(format_reward)
        self.brier_weight = float(brier_weight)
        self.invalid_format_reward = -(self.brier_weight + float(invalid_format_margin))
        self.separate_correctness_and_calibration = bool(separate_correctness_and_calibration)
        self.loss_mask_mode = loss_mask_mode
        self.conf_marker_token_patterns = [tokenizer.encode(prefix + "<conf>", add_special_tokens=False) for prefix in ("", " ", "\n", "\n\n", "\t")]

    @staticmethod
    def _correctness(result: Any) -> float:
        """Convert the repository's heterogeneous verifier outputs to {0.0, 1.0}."""

        if isinstance(result, dict):
            value = result.get("acc", result["score"])
        else:
            value = result
        return 1.0 if float(value) > 0.0 else 0.0

    @staticmethod
    def _confidence_char_start(response_text: str, parse_mode: str) -> int | None:
        answer_end = _answer_end_offset(response_text, parse_mode)
        if answer_end is None:
            return None
        match = _CONFIDENCE_PATTERN.search(response_text[answer_end:])
        return answer_end + match.start() if match is not None else None

    def _confidence_token_start(self, response_text: str, response_ids: torch.Tensor, valid_response_length: int, parse_mode: str) -> int | None:
        char_start = self._confidence_char_start(response_text, parse_mode)
        if char_start is None:
            return None
        try:
            encoded = self.tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
            raw_ids = response_ids[:valid_response_length].tolist()
            if raw_ids[:len(encoded["input_ids"])] == encoded["input_ids"]:
                for token_index, (_, end) in enumerate(encoded["offset_mapping"]):
                    if end > char_start:
                        return token_index
        except (NotImplementedError, TypeError, ValueError):
            pass
        raw_ids = response_ids[:valid_response_length].tolist()
        for pattern in self.conf_marker_token_patterns:
            for token_index in range(valid_response_length - len(pattern), -1, -1):
                if pattern and raw_ids[token_index:token_index + len(pattern)] == pattern:
                    return token_index
        return None

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        correctness_reward_tensor = torch.zeros_like(reward_tensor) if self.separate_correctness_and_calibration else None
        calibration_reward_tensor = torch.zeros_like(reward_tensor) if self.separate_correctness_and_calibration else None
        reasoning_loss_mask = torch.zeros_like(reward_tensor) if self.separate_correctness_and_calibration else None
        confidence_loss_mask = torch.zeros_like(reward_tensor) if self.separate_correctness_and_calibration else None
        reward_extra_info = defaultdict(list)
        printed_by_source: dict[str, int] = {}

        for index in range(len(data)):
            data_item = data[index]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch["attention_mask"]

            valid_prompt_length = attention_mask[:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_length = attention_mask[prompt_length:].sum().item()
            if valid_response_length == 0:
                continue
            valid_response_ids = data_item.batch["responses"][:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            parse_mode = extra_info.get("parse_mode", "boxed")
            final_answer = _extract_final_answer(response_str, parse_mode=parse_mode)
            confidence = _parse_confidence(response_str, parse_mode=parse_mode)
            has_boxed_answer = _last_boxed_expression(response_str) is not None
            valid_format = _has_valid_format(response_str, parse_mode=parse_mode)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})

            verifier_result: Any = 0.0
            correctness = 0.0
            if final_answer is not None:
                verifier_result = self.compute_score(
                    data_source=data_source,
                    solution_str=final_answer,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                correctness = self._correctness(verifier_result)

            brier_score = (confidence - correctness) ** 2 if confidence is not None else 1.0
            format_score = float(valid_format)
            reward = (
                correctness - self.brier_weight * brier_score
                if valid_format
                else self.invalid_format_reward
            )
            reward_tensor[index, valid_response_length - 1] = reward
            if self.separate_correctness_and_calibration:
                assert correctness_reward_tensor is not None and calibration_reward_tensor is not None
                assert reasoning_loss_mask is not None and confidence_loss_mask is not None
                correctness_reward_tensor[index, valid_response_length - 1] = correctness if valid_format else 0.0
                calibration_reward_tensor[index, valid_response_length - 1] = -self.brier_weight * brier_score if valid_format else self.invalid_format_reward
                confidence_start = self._confidence_token_start(response_str, data_item.batch["responses"], valid_response_length, parse_mode) if valid_format else None
                if confidence_start is None:
                    reasoning_loss_mask[index, :valid_response_length] = 1.0
                    if self.loss_mask_mode == "confidence_or_all":
                        confidence_loss_mask[index, :valid_response_length] = 1.0
                else:
                    reasoning_loss_mask[index, :confidence_start] = 1.0
                    confidence_loss_mask[index, confidence_start:valid_response_length] = 1.0

            # ``acc`` is retained as the validation core metric; the explicit
            # RLCR names are used by training-time W&B monitoring.
            reward_extra_info["acc"].append(correctness)
            reward_extra_info["correctness"].append(correctness)
            reward_extra_info["has_boxed_answer"].append(has_boxed_answer)
            reward_extra_info["has_final_answer"].append(final_answer is not None)
            reward_extra_info["confidence"].append(confidence if confidence is not None else float("nan"))
            reward_extra_info["confidence_valid"].append(confidence is not None)
            reward_extra_info["brier_score"].append(brier_score)
            reward_extra_info["format_score"].append(format_score)
            if self.separate_correctness_and_calibration:
                reward_extra_info["rlcr_dual_reward"].append(True)

            # Keep reward_extra_info fields batch-aligned. Verifier-specific fields
            # may not exist for samples without a parseable boxed answer, so they are
            # intentionally not forwarded here.

            if printed_by_source.get(data_source, 0) < self.num_examine:
                printed_by_source[data_source] = printed_by_source.get(data_source, 0) + 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[parse_mode]", parse_mode)
                print("[final_answer]", final_answer)
                print("[confidence]", confidence)
                print("[acc]", correctness)
                print("[brier_score]", brier_score)
                print("[format_score]", format_score)
                print("[reward]", reward)

        if self.separate_correctness_and_calibration:
            assert correctness_reward_tensor is not None and calibration_reward_tensor is not None
            assert reasoning_loss_mask is not None and confidence_loss_mask is not None
            data.batch["reasoning_loss_mask"] = reasoning_loss_mask
            data.batch["confidence_loss_mask"] = confidence_loss_mask
            result = {"reward_tensor": reward_tensor, "reward_components": {"correctness": correctness_reward_tensor, "calibration": calibration_reward_tensor}, "reward_extra_info": reward_extra_info}
            return result if return_dict else reward_tensor
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
