"""Test-time calibration reward manager without ground-truth answers.

For each prompt's rollout group, the empirical support for a parsed final
answer is used as that completion's confidence target.  No verifier or
``ground_truth`` field is consulted.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.rlcr import (
    _CONFIDENCE_PATTERN,
    _answer_end_offset,
    _extract_final_answer,
    _parse_confidence,
)


def _answer_key(answer: str | None) -> str:
    """Return an exact-match key, treating unparseable answers as one outcome."""

    if answer is None:
        return "__ttcr_invalid_answer__"
    # Whitespace is not semantically meaningful for a boxed mathematical answer
    # and otherwise artificially splits identical sampled answers.
    return re.sub(r"\s+", "", answer)


_ABSTENTION_ANSWERS = frozenset(
    {
        "unknown",
        "i don't know",
        "i dont know",
        "i do not know",
        "not sure",
        "unsure",
        "cannot determine",
        "can't determine",
        "cannot answer",
        "can't answer",
        "unable to determine",
        "insufficient information",
        "not enough information",
        "n/a",
    }
)


def _is_abstention_answer(answer: str | None) -> bool:
    """Identify explicit no-answer outputs in answerable QA data.

    This deliberately uses exact normalized phrases instead of substring
    matching: a factual answer may legitimately contain words such as
    ``unknown``.  These outputs do not provide evidence for a concrete answer
    under self-consistency training and therefore must not receive a frequency
    confidence target.
    """

    if answer is None:
        return False
    normalized = re.sub(r"\s+", " ", answer).strip().casefold()
    normalized = normalized.strip(" \t\r\n\"'`.,;:!?")
    # FactQA answers are normally plain text, but tolerate a common LaTex
    # wrapper if the same parser is reused with boxed_text data.
    latex_text = re.fullmatch(r"\\text\{\s*(.*?)\s*\}", normalized, flags=re.DOTALL)
    if latex_text is not None:
        normalized = latex_text.group(1).strip().strip(" \t\r\n\"'`.,;:!?")
    return normalized in _ABSTENTION_ANSWERS


def _entropy_from_counts(counts: Counter[str], total: int) -> tuple[float, float]:
    """Return Shannon entropy and entropy normalized by log(total)."""

    entropy = -sum((count / total) * math.log(count / total) for count in counts.values() if count)
    normalized_entropy = entropy / math.log(total) if total > 1 else 0.0
    return entropy, normalized_entropy


@register("ttcr")
class TTCRRewardManager(AbstractRewardManager):
    """Optimize verbalized confidence against within-group answer agreement.

    ``target_mode='frequency'`` uses ``count(answer) / K``.  ``target_mode='loo'``
    uses the leave-one-out estimate ``(count(answer) - 1) / (K - 1)``.
    ``entropy_frequency`` and ``entropy_loo`` multiply their corresponding base
    target by ``1 - entropy_coef * H_norm``, with ``H_norm = H / log(K)``.
    Explicit abstentions (for example ``Unknown``) always use target 0.0 and
    are excluded from pseudo-label selection. They remain in the denominator
    for other answers' frequency targets, because abstaining rollouts are not
    evidence in favor of any concrete answer.
    The scalar outcome reward is the negative Brier score, ``-(confidence-target)^2``.
    Invalid/missing answer or confidence uses Brier score 1.0.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        target_mode: str = "frequency",
        entropy_coef: float = 1.0,
        monitor_correctness: bool = True,
        loss_mask_mode: str = "all",
    ) -> None:
        valid_target_modes = {"frequency", "loo", "entropy_frequency", "entropy_loo"}
        if target_mode not in valid_target_modes:
            raise ValueError(f"target_mode must be one of {sorted(valid_target_modes)}, got {target_mode!r}.")
        if not math.isfinite(entropy_coef) or not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f"entropy_coef must be finite and in [0, 1], got {entropy_coef!r}.")
        if loss_mask_mode not in {"all", "confidence", "confidence_or_all"}:
            raise ValueError(
                "loss_mask_mode must be 'all', 'confidence', or 'confidence_or_all', "
                f"got {loss_mask_mode!r}."
            )
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.target_mode = target_mode
        self.entropy_coef = float(entropy_coef)
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.monitor_correctness = monitor_correctness
        self.loss_mask_mode = loss_mask_mode
        self.conf_marker_token_ids = tokenizer.encode("<conf>", add_special_tokens=False)
        # Fallback patterns cover tokenizers that merge preceding whitespace and
        # ``<conf>`` into one token. The primary path below uses character
        # offsets and does not depend on these patterns.
        self.conf_marker_token_patterns = [
            tokenizer.encode(prefix + "<conf>", add_special_tokens=False)
            for prefix in ("", " ", "\n", "\n\n", "\t")
        ]
        if loss_mask_mode != "all" and not any(self.conf_marker_token_patterns):
            raise ValueError("Tokenizer cannot encode '<conf>', required for confidence-only TTCR loss masking.")

    @staticmethod
    def _correctness(result: Any) -> float:
        """Convert repository verifier outputs to a binary monitoring value."""

        value = result.get("acc", result["score"]) if isinstance(result, dict) else result
        return 1.0 if float(value) > 0.0 else 0.0

    def _monitor_answer_correctness(self, data_item: Any, answer: str | None) -> float:
        """Verify an answer for logging only; never use this value as TTCR reward."""

        if not self.monitor_correctness or answer is None:
            return float("nan")
        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        data_source = data_item.non_tensor_batch.get(self.reward_fn_key)
        if ground_truth is None or data_source is None:
            return float("nan")
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        return self._correctness(
            self.compute_score(
                data_source=data_source,
                solution_str=answer,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        )

    @staticmethod
    def _confidence_char_start(response_text: str, parse_mode: str) -> int | None:
        """Find the exact valid ``<conf>`` matched by the confidence parser."""

        answer_end = _answer_end_offset(response_text, parse_mode)
        if answer_end is None:
            return None
        match = _CONFIDENCE_PATTERN.search(response_text[answer_end:])
        return answer_end + match.start() if match is not None else None

    def _confidence_token_start(
        self, response_text: str, response_ids: torch.Tensor, valid_response_length: int, parse_mode: str
    ) -> int | None:
        """Map the validated ``<conf>`` character offset to a response token.

        A fixed ``encode('<conf>')`` sequence is not reliable: BPE tokenizers
        may split the marker or merge it with a preceding space/newline. Fast
        tokenizer offsets handle both cases. Token-pattern matching remains a
        conservative fallback for tokenizers without offsets or non-invertible
        decode/re-encode sequences.
        """

        char_start = self._confidence_char_start(response_text, parse_mode)
        if char_start is None:
            return None
        try:
            encoded = self.tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
            encoded_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            raw_ids = response_ids[:valid_response_length].tolist()
            # Generated response IDs can include a trailing EOS token; the
            # textual tokenization must otherwise align from position zero.
            if raw_ids[: len(encoded_ids)] == encoded_ids:
                for token_index, (start, end) in enumerate(offsets):
                    if end > char_start:
                        return token_index
        except (NotImplementedError, TypeError, ValueError):
            pass

        raw_ids = response_ids[:valid_response_length].tolist()
        for pattern in self.conf_marker_token_patterns:
            pattern_length = len(pattern)
            if pattern_length == 0:
                continue
            for token_index in range(valid_response_length - pattern_length, -1, -1):
                if raw_ids[token_index : token_index + pattern_length] == pattern:
                    return token_index
        return None

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        confidence_loss_mask = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: defaultdict[str, list[Any]] = defaultdict(list)
        decoded: list[dict[str, Any] | None] = [None] * len(data)
        grouped_indices: defaultdict[Any, list[int]] = defaultdict(list)

        if "uid" not in data.non_tensor_batch:
            raise KeyError("TTCRRewardManager requires non_tensor_batch['uid'] to group K rollouts by question.")

        for index in range(len(data)):
            data_item = data[index]
            prompt_length = data_item.batch["prompts"].shape[-1]
            attention_mask = data_item.batch["attention_mask"]
            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            uid = data_item.non_tensor_batch["uid"]
            grouped_indices[uid].append(index)
            if valid_response_length == 0:
                continue

            response_ids = data_item.batch["responses"][:valid_response_length]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            parse_mode = extra_info.get("parse_mode", "boxed")
            final_answer = _extract_final_answer(response_text, parse_mode=parse_mode)
            confidence = _parse_confidence(response_text, parse_mode=parse_mode)
            decoded[index] = {
                "response_text": response_text,
                "parse_mode": parse_mode,
                "final_answer": final_answer,
                "confidence": confidence,
                "valid_response_length": valid_response_length,
            }

        printed_groups = 0
        for uid, indices in grouped_indices.items():
            answer_counts = Counter(
                _answer_key(decoded[index]["final_answer"]) if decoded[index] is not None else "__ttcr_invalid_answer__"
                for index in indices
            )
            group_size = len(indices)
            answer_entropy, normalized_entropy = _entropy_from_counts(answer_counts, group_size)
            uses_entropy_target = self.target_mode.startswith("entropy_")
            base_target_mode = self.target_mode.removeprefix("entropy_")
            entropy_multiplier = 1.0 - self.entropy_coef * normalized_entropy if uses_entropy_target else 1.0
            # Select one pseudo label deterministically. Invalid responses and
            # explicit abstentions are excluded: neither is a factual target.
            non_abstention_answer_keys = [
                _answer_key(decoded[index]["final_answer"])
                for index in indices
                if decoded[index] is not None
                and decoded[index]["final_answer"] is not None
                and not _is_abstention_answer(decoded[index]["final_answer"])
            ]
            valid_answer_counts = Counter(non_abstention_answer_keys)
            pseudo_label_key: str | None = None
            pseudo_label_count = 0
            if valid_answer_counts:
                pseudo_label_count = max(valid_answer_counts.values())
                # Counter preserves insertion order, so this gives a stable
                # first-rollout tie break while retaining one pseudo label.
                pseudo_label_key = next(
                    answer_key
                    for answer_key in non_abstention_answer_keys
                    if valid_answer_counts[answer_key] == pseudo_label_count
                )
            pseudo_label_support_ratio = pseudo_label_count / group_size
            pseudo_label_answer = next(
                (
                    decoded[index]["final_answer"]
                    for index in indices
                    if decoded[index] is not None
                    and decoded[index]["final_answer"] is not None
                    and not _is_abstention_answer(decoded[index]["final_answer"])
                    and _answer_key(decoded[index]["final_answer"]) == pseudo_label_key
                ),
                None,
            )
            # This is the actual accuracy of the group's consensus answer
            # against hidden ground truth. It is strictly monitoring-only.
            pseudo_label_acc = self._monitor_answer_correctness(data[indices[0]], pseudo_label_answer)

            for index in indices:
                item = decoded[index]
                final_answer = item["final_answer"] if item is not None else None
                confidence = item["confidence"] if item is not None else None
                is_abstention = _is_abstention_answer(final_answer)
                answer_count = answer_counts[_answer_key(final_answer)]
                base_confidence_target = 0.0
                if is_abstention:
                    # A repeated abstention is not a self-consistent factual
                    # answer. Never convert its high occurrence frequency into
                    # high verbal confidence.
                    confidence_target = 0.0
                else:
                    if base_target_mode == "frequency":
                        base_confidence_target = answer_count / group_size
                    else:
                        base_confidence_target = (answer_count - 1) / (group_size - 1) if group_size > 1 else 0.0
                    confidence_target = base_confidence_target * entropy_multiplier

                valid_answer = final_answer is not None
                valid_confidence = confidence is not None
                pseudo_label_agreement = float(valid_answer and _answer_key(final_answer) == pseudo_label_key)
                correctness = self._monitor_answer_correctness(data[index], final_answer)
                brier_score = (confidence - confidence_target) ** 2 if valid_answer and valid_confidence else 1.0
                reward = -brier_score
                if item is not None:
                    reward_tensor[index, item["valid_response_length"] - 1] = reward
                    if self.loss_mask_mode == "all":
                        confidence_loss_mask[index, : item["valid_response_length"]] = 1.0
                    else:
                        confidence_start = (
                            self._confidence_token_start(
                                item["response_text"],
                                data[index].batch["responses"],
                                item["valid_response_length"],
                                item["parse_mode"],
                            )
                            if valid_confidence
                            else None
                        )
                        if confidence_start is not None:
                            confidence_loss_mask[index, confidence_start : item["valid_response_length"]] = 1.0
                        elif self.loss_mask_mode == "confidence_or_all":
                            # A malformed completion has no confidence tokens to
                            # optimize. Fall back to the complete response so its
                            # negative Brier reward still teaches the required format.
                            confidence_loss_mask[index, : item["valid_response_length"]] = 1.0

                reward_extra_info["confidence"].append(confidence if valid_confidence else float("nan"))
                reward_extra_info["confidence_valid"].append(valid_confidence)
                reward_extra_info["final_answer_valid"].append(valid_answer)
                reward_extra_info["is_abstention"].append(is_abstention)
                reward_extra_info["answer_frequency_target"].append(confidence_target)
                reward_extra_info["base_confidence_target"].append(
                    0.0 if is_abstention else base_confidence_target
                )
                reward_extra_info["entropy_multiplier"].append(entropy_multiplier)
                reward_extra_info["answer_support_count"].append(answer_count)
                reward_extra_info["group_size"].append(group_size)
                reward_extra_info["pseudo_label_acc"].append(pseudo_label_acc)
                reward_extra_info["pseudo_label_agreement"].append(pseudo_label_agreement)
                reward_extra_info["pseudo_label_support_ratio"].append(pseudo_label_support_ratio)
                reward_extra_info["correctness"].append(correctness)
                reward_extra_info["brier_score"].append(brier_score)
                response_length = item["valid_response_length"] if item is not None else 0
                masked_tokens = confidence_loss_mask[index, :response_length].sum().item()
                reward_extra_info["loss_mask_token_ratio"].append(masked_tokens / response_length if response_length else 0.0)
                reward_extra_info["answer_entropy"].append(answer_entropy)
                reward_extra_info["answer_entropy_normalized"].append(normalized_entropy)

            if printed_groups < self.num_examine:
                printed_groups += 1
                print("[ttcr uid]", uid)
                print("[ttcr target_mode]", self.target_mode)
                print("[ttcr entropy_multiplier]", entropy_multiplier)
                print("[ttcr loss_mask_mode]", self.loss_mask_mode)
                print("[ttcr answer_counts]", dict(answer_counts))
                print("[ttcr pseudo_label]", pseudo_label_key)
                print("[ttcr pseudo_label_support_ratio]", pseudo_label_support_ratio)
                print("[ttcr answer_entropy]", answer_entropy)

        if self.loss_mask_mode != "all":
            # The synchronous rule-based reward path shares this DataProto with
            # the actor update. ActorDP uses this tensor solely as a loss mask;
            # rewards/advantages themselves remain sequence-level GRPO values.
            data.batch["ttcr_loss_mask"] = confidence_loss_mask
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
