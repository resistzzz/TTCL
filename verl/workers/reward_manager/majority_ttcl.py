"""Joint majority-voting correctness and self-consistency calibration reward.

The manager deliberately returns two *separate* outcome rewards for every
rollout group.  The trainer turns them into independent advantages before the
actor applies them to disjoint reasoning and confidence token masks.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.majority_voting import _pseudo_ground_truth
from verl.workers.reward_manager.rlcr import (
    _CONFIDENCE_PATTERN,
    _answer_end_offset,
    _extract_final_answer,
    _parse_confidence,
)
from verl.workers.reward_manager.ttcr import _answer_key, _entropy_from_counts, _is_abstention_answer


@register("majority_ttcl")
class MajorityTTCLRewardManager(AbstractRewardManager):
    """Train reasoning from a majority pseudo label and confidence from agreement.

    Correctness is a binary score against either the modal non-abstention
    answer in a rollout group or a pseudo label sampled from a frozen historical
    target cache. Calibration is negative Brier score against the answer's
    frequency in its rollout group or epoch-lagged frequency cache. Dataset ground truth is never
    used for either reward, only for monitoring metrics when it is available.

    For a well-formed confidence field, ``reasoning_loss_mask`` covers tokens
    before ``<conf>`` and ``confidence_loss_mask`` covers ``<conf>`` and all
    later tokens.  With malformed confidence output, reasoning remains the
    whole response; ``loss_mask_mode='confidence_or_all'`` (the default)
    additionally gives calibration its whole-response format-learning fallback.
    Set it to ``'confidence'`` to omit calibration gradients for such samples.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        target_cache_mode: str = "online",
        target_cache_ema_rate: float = 0.01,
        pseudo_label_source: str = "policy",
        monitor_correctness: bool = True,
        loss_mask_mode: str = "confidence_or_all",
    ) -> None:
        if loss_mask_mode not in {"confidence", "confidence_or_all"}:
            raise ValueError(
                "majority_ttcl.loss_mask_mode must be 'confidence' or "
                f"'confidence_or_all', got {loss_mask_mode!r}."
            )
        if target_cache_mode not in {"online", "epoch_ema"}:
            raise ValueError(
                "target_cache_mode must be 'online' or 'epoch_ema', got " f"{target_cache_mode!r}."
            )
        if pseudo_label_source not in {"policy", "target_cache_majority"}:
            raise ValueError(
                "pseudo_label_source must be 'policy' or 'target_cache_majority', got "
                f"{pseudo_label_source!r}."
            )
        if pseudo_label_source == "target_cache_majority" and target_cache_mode != "epoch_ema":
            raise ValueError(
                "cache-based pseudo labels require target_cache_mode='epoch_ema'."
            )
        if not math.isfinite(target_cache_ema_rate) or not 0.0 <= target_cache_ema_rate <= 1.0:
            raise ValueError(
                "target_cache_ema_rate must be finite and in [0, 1], got " f"{target_cache_ema_rate!r}."
            )
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # ``epoch_ema`` is a lagged target-policy approximation.  Every
        # question owns a small answer->support cache.  Rewards in epoch e
        # read q_e; current rollout supports are accumulated and only become
        # q_(e+1) at the epoch boundary.
        self.target_cache_mode = target_cache_mode
        self.target_cache_ema_rate = float(target_cache_ema_rate)
        # Cache sampling makes correctness labels lagged too: both objectives
        # read q_e while only the epoch boundary can produce q_(e+1).
        self.pseudo_label_source = pseudo_label_source
        self.target_cache: dict[str, dict[str, float]] = {}
        self.pending_target_cache: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
        self.target_cache_initialized = target_cache_mode == "online"
        self._initializing_target_cache = False
        self.last_target_cache_commit_metrics: dict[str, float] = {}
        self.monitor_correctness = monitor_correctness
        self.loss_mask_mode = loss_mask_mode
        self.conf_marker_token_patterns = [
            tokenizer.encode(prefix + "<conf>", add_special_tokens=False)
            for prefix in ("", " ", "\n", "\n\n", "\t")
        ]

    @staticmethod
    def _correctness(result: Any) -> float:
        value = result.get("acc", result["score"]) if isinstance(result, dict) else result
        return 1.0 if float(value) > 0.0 else 0.0

    def _score(self, data_item: Any, answer: str | None, ground_truth: Any) -> float:
        if answer is None or ground_truth is None:
            return float("nan")
        data_source = data_item.non_tensor_batch.get(self.reward_fn_key)
        if data_source is None:
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

    def _monitor_answer_correctness(self, data_item: Any, answer: str | None) -> float:
        """Use dataset labels only for monitoring; never for either reward."""

        if not self.monitor_correctness:
            return float("nan")
        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        return self._score(data_item, answer, ground_truth)

    def _pseudo_correctness(self, data_item: Any, answer: str | None, pseudo_label: str | None) -> float:
        """Verify against the group pseudo label, including math equivalence."""

        if answer is None or pseudo_label is None or _is_abstention_answer(answer):
            return 0.0
        # Cache keys deliberately collapse whitespace. Preserve exact grouped
        # matches (e.g. "New York" versus "NewYork") before delegating
        # non-identical mathematical forms to math_verify/default scorers.
        if _answer_key(answer) == _answer_key(pseudo_label):
            return 1.0
        result = self._score(data_item, answer, _pseudo_ground_truth(pseudo_label))
        return 0.0 if not math.isfinite(result) else result

    @staticmethod
    def _confidence_char_start(response_text: str, parse_mode: str) -> int | None:
        answer_end = _answer_end_offset(response_text, parse_mode)
        if answer_end is None:
            return None
        match = _CONFIDENCE_PATTERN.search(response_text[answer_end:])
        return answer_end + match.start() if match is not None else None

    def _confidence_token_start(
        self, response_text: str, response_ids: torch.Tensor, valid_response_length: int, parse_mode: str
    ) -> int | None:
        """Locate the parsed marker without assuming a fixed ``<conf>`` token ID."""

        char_start = self._confidence_char_start(response_text, parse_mode)
        if char_start is None:
            return None
        try:
            encoded = self.tokenizer(response_text, add_special_tokens=False, return_offsets_mapping=True)
            encoded_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            raw_ids = response_ids[:valid_response_length].tolist()
            if raw_ids[: len(encoded_ids)] == encoded_ids:
                for token_index, (_, end) in enumerate(offsets):
                    if end > char_start:
                        return token_index
        except (NotImplementedError, TypeError, ValueError):
            pass

        raw_ids = response_ids[:valid_response_length].tolist()
        for pattern in self.conf_marker_token_patterns:
            if not pattern:
                continue
            for token_index in range(valid_response_length - len(pattern), -1, -1):
                if raw_ids[token_index : token_index + len(pattern)] == pattern:
                    return token_index
        return None

    @staticmethod
    def _is_cache_abstention_key(answer_key: str) -> bool:
        """Recognize abstentions after cache key whitespace normalization."""

        if _is_abstention_answer(answer_key):
            return True
        compact = re.sub(r"\s+", "", answer_key).casefold().strip("\"'`.,;:!?")
        return compact in {
            "unknown", "idontknow", "idonotknow", "notsure", "unsure",
            "cannotdetermine", "cantdetermine", "cannotanswer", "cantanswer",
            "unabletodetermine", "insufficientinformation", "notenoughinformation", "n/a",
        }

    def _eligible_cached_pseudo_candidates(self, frequency_map: dict[str, float]) -> list[tuple[str, float]]:
        """Return normalized concrete-answer mass from frozen q_e(y | x)."""

        candidates = [
            (answer_key, float(probability))
            for answer_key, probability in frequency_map.items()
            if answer_key != "__ttcr_invalid_answer__"
            and not self._is_cache_abstention_key(answer_key)
            and math.isfinite(float(probability))
            and float(probability) > 0.0
        ]
        total_mass = sum(probability for _, probability in candidates)
        return [(answer_key, probability / total_mass) for answer_key, probability in candidates] if total_mass else []

    def _majority_cached_pseudo_label(self, frequency_map: dict[str, float]) -> tuple[str | None, float]:
        """Deterministically select the modal concrete answer in frozen q_e."""

        candidates = self._eligible_cached_pseudo_candidates(frequency_map)
        if not candidates:
            return None, 0.0
        # ``max`` preserves insertion order for ties, matching the existing
        # deterministic earliest-answer tie convention of the policy baseline.
        return max(candidates, key=lambda candidate: candidate[1])

    @staticmethod
    def _target_cache_key(data_item: Any) -> str:
        """Return a stable dataset-level key rather than the per-batch UUID."""

        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        if "index" not in extra_info:
            raise KeyError(
                "majority_ttcl target_cache_mode='epoch_ema' requires "
                "extra_info['index'] in every training example."
            )
        data_source = data_item.non_tensor_batch.get("data_source", "")
        return f"{data_source}|{extra_info.get('split', '')}|{extra_info['index']}"

    @staticmethod
    def _support_maps(answer_counts: Counter[str], group_size: int) -> dict[str, float]:
        """Build answer-specific frequency targets for one group."""

        denominator = max(group_size, 1)
        return {answer: count / denominator for answer, count in answer_counts.items()}

    def needs_target_cache_initialization(self) -> bool:
        return self.target_cache_mode == "epoch_ema" and not self.target_cache_initialized

    def initialize_target_cache(self, data: DataProto) -> None:
        """Populate q_0 from a frozen pre-training rollout without updating policy."""

        if self.target_cache_mode != "epoch_ema":
            return
        self._initializing_target_cache = True
        try:
            # Reuse the ordinary parser/grouping path.  The resulting rewards
            # are discarded by the trainer; this pass only fills target_cache.
            self(data, return_dict=True)
        finally:
            self._initializing_target_cache = False

    def finalize_target_cache_initialization(self) -> dict[str, float]:
        """Finish q_0 initialization after the full-dataset rollout pass."""

        if self.target_cache_mode != "epoch_ema":
            return {}
        if not self.target_cache:
            raise RuntimeError("majority_ttcl target-cache initialization produced no question targets.")
        self.target_cache_initialized = True
        metrics = {
            "target_cache/initialized_questions": float(len(self.target_cache)),
        }
        self.last_target_cache_commit_metrics = metrics
        return metrics

    def commit_target_cache_epoch(self) -> dict[str, float]:
        """Apply the accumulated current-policy supports after an epoch ends."""

        if self.target_cache_mode != "epoch_ema" or self._initializing_target_cache:
            return {}

        update_delta_sum = 0.0
        update_value_count = 0
        updated_questions = 0
        for question_key, observations in self.pending_target_cache.items():
            if not observations:
                continue
            old_map = self.target_cache[question_key]
            answer_keys = set().union(*observations, old_map)
            averaged_map = {
                answer: sum(observation.get(answer, 0.0) for observation in observations) / len(observations)
                for answer in answer_keys
            }
            for answer, current_value in averaged_map.items():
                old_value = old_map.get(answer, 0.0)
                new_value = (1.0 - self.target_cache_ema_rate) * old_value + self.target_cache_ema_rate * current_value
                update_delta_sum += abs(new_value - old_value)
                update_value_count += 1
                old_map[answer] = new_value
            updated_questions += 1

        self.pending_target_cache.clear()
        metrics = {
            "target_cache/updated_questions": float(updated_questions),
            "target_cache/size": float(len(self.target_cache)),
            "target_cache/mean_abs_update": update_delta_sum / update_value_count if update_value_count else 0.0,
            "target_cache/ema_rate": self.target_cache_ema_rate,
        }
        self.last_target_cache_commit_metrics = metrics
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_cache": self.target_cache,
            "target_cache_initialized": self.target_cache_initialized,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.target_cache_mode != "epoch_ema":
            return
        # Older checkpoints stored both frequency and LOO maps per question.
        # Keep their frequency map when resuming after the target-mode cleanup.
        self.target_cache = {
            key: value.get("frequency", value) if isinstance(value, dict) else value
            for key, value in state_dict.get("target_cache", {}).items()
        }
        self.target_cache_initialized = bool(state_dict.get("target_cache_initialized", bool(self.target_cache)))

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "uid" not in data.non_tensor_batch:
            raise KeyError("MajorityTTCLRewardManager requires non_tensor_batch['uid'] to group rollouts.")

        correctness_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        calibration_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reasoning_loss_mask = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        confidence_loss_mask = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: defaultdict[str, list[Any]] = defaultdict(list)
        decoded: list[dict[str, Any] | None] = [None] * len(data)
        grouped_indices: defaultdict[Any, list[int]] = defaultdict(list)

        for index in range(len(data)):
            data_item = data[index]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            grouped_indices[data_item.non_tensor_batch["uid"]].append(index)
            if valid_response_length == 0:
                continue

            response_ids = data_item.batch["responses"][:valid_response_length]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            parse_mode = dict(data_item.non_tensor_batch.get("extra_info", {})).get("parse_mode", "boxed")
            decoded[index] = {
                "response_text": response_text,
                "parse_mode": parse_mode,
                "final_answer": _extract_final_answer(response_text, parse_mode=parse_mode),
                "confidence": _parse_confidence(response_text, parse_mode=parse_mode),
                "valid_response_length": valid_response_length,
            }

        # Build group answer frequencies before constructing targets.
        group_stats: dict[Any, dict[str, Any]] = {}
        for uid, indices in grouped_indices.items():
            group_size = len(indices)
            all_answer_counts = Counter(
                _answer_key(decoded[index]["final_answer"]) if decoded[index] is not None else "__ttcr_invalid_answer__"
                for index in indices
            )
            eligible_answer_keys = [
                _answer_key(decoded[index]["final_answer"])
                for index in indices
                if decoded[index] is not None
                and decoded[index]["final_answer"] is not None
                and not _is_abstention_answer(decoded[index]["final_answer"])
            ]
            eligible_counts = Counter(eligible_answer_keys)
            answer_entropy, normalized_entropy = _entropy_from_counts(all_answer_counts, group_size)
            support_map = self._support_maps(all_answer_counts, group_size)
            target_cache_key = self._target_cache_key(data[indices[0]]) if self.target_cache_mode == "epoch_ema" else None
            cache_hit = target_cache_key in self.target_cache if target_cache_key is not None else False
            if target_cache_key is not None and not cache_hit:
                # This is normally populated by the pre-training rollout pass.
                # Keep a safe lazy fallback for old/resumed runs with no cache.
                self.target_cache[target_cache_key] = dict(support_map)
            group_stats[uid] = {
                "group_size": group_size,
                "all_answer_counts": all_answer_counts,
                "support_map": support_map,
                "target_cache_key": target_cache_key,
                "target_cache_hit": cache_hit,
                "eligible_answer_keys": eligible_answer_keys,
                "eligible_counts": eligible_counts,
                "answer_entropy": answer_entropy,
                "normalized_entropy": normalized_entropy,
            }

        printed_groups = 0
        for uid, indices in grouped_indices.items():
            stats = group_stats[uid]
            group_size = stats["group_size"]
            all_answer_counts = stats["all_answer_counts"]
            support_map = stats["support_map"]
            target_cache_key = stats["target_cache_key"]
            target_cache_hit = stats["target_cache_hit"]
            eligible_answer_keys = stats["eligible_answer_keys"]
            eligible_counts = stats["eligible_counts"]
            answer_entropy = stats["answer_entropy"]
            normalized_entropy = stats["normalized_entropy"]

            pseudo_label_key: str | None = None
            pseudo_label_answer: str | None = None
            pseudo_label_support_ratio = 0.0
            if self.pseudo_label_source == "target_cache_majority" and not self._initializing_target_cache:
                assert target_cache_key is not None
                cached_frequency = self.target_cache[target_cache_key]
                pseudo_label_key, pseudo_label_support_ratio = self._majority_cached_pseudo_label(cached_frequency)
                # Cache keys are normalized parsed answers. This is sufficient for
                # the existing scorer, which receives _pseudo_ground_truth(key).
                pseudo_label_answer = pseudo_label_key
            elif eligible_counts:
                # Legacy in-group majority baseline, including deterministic
                # earliest-rollout tie breaking.
                pseudo_label_count = max(eligible_counts.values())
                pseudo_label_key = next(key for key in eligible_answer_keys if eligible_counts[key] == pseudo_label_count)
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
                pseudo_label_support_ratio = pseudo_label_count / group_size
            pseudo_label_acc = self._monitor_answer_correctness(data[indices[0]], pseudo_label_answer)

            for index in indices:
                item = decoded[index]
                final_answer = item["final_answer"] if item is not None else None
                confidence = item["confidence"] if item is not None else None
                valid_answer = final_answer is not None
                valid_confidence = confidence is not None
                is_abstention = _is_abstention_answer(final_answer)
                answer_count = all_answer_counts[_answer_key(final_answer)]

                base_confidence_target = 0.0
                if is_abstention or not valid_answer:
                    confidence_target = 0.0
                else:
                    answer_key = _answer_key(final_answer)
                    if self.target_cache_mode == "epoch_ema":
                        assert target_cache_key is not None
                        base_confidence_target = self.target_cache[target_cache_key].get(answer_key, 0.0)
                    else:
                        base_confidence_target = answer_count / group_size
                    confidence_target = base_confidence_target

                pseudo_correctness = self._pseudo_correctness(data[index], final_answer, pseudo_label_answer)
                actual_correctness = self._monitor_answer_correctness(data[index], final_answer)
                brier_score = (confidence - confidence_target) ** 2 if valid_answer and valid_confidence else 1.0
                calibration_reward = -brier_score
                confidence_start: int | None = None
                if item is not None:
                    response_length = item["valid_response_length"]
                    correctness_reward_tensor[index, response_length - 1] = pseudo_correctness
                    calibration_reward_tensor[index, response_length - 1] = calibration_reward
                    if valid_confidence:
                        confidence_start = self._confidence_token_start(
                            item["response_text"],
                            data[index].batch["responses"],
                            response_length,
                            item["parse_mode"],
                        )
                    if confidence_start is None:
                        # The answer/reasoning objective still has a complete
                        # sequence to optimize.  Calibration may optionally
                        # use the same fallback to teach the missing format.
                        reasoning_loss_mask[index, :response_length] = 1.0
                        if self.loss_mask_mode == "confidence_or_all":
                            confidence_loss_mask[index, :response_length] = 1.0
                    else:
                        reasoning_loss_mask[index, :confidence_start] = 1.0
                        confidence_loss_mask[index, confidence_start:response_length] = 1.0

                response_length = item["valid_response_length"] if item is not None else 0
                reward_extra_info["majority_ttcl_reward"].append(True)
                reward_extra_info["pseudo_correctness"].append(pseudo_correctness)
                reward_extra_info["correctness"].append(actual_correctness)
                reward_extra_info["pseudo_label_acc"].append(pseudo_label_acc)
                reward_extra_info["pseudo_label_support_ratio"].append(pseudo_label_support_ratio)
                reward_extra_info["pseudo_label_valid"].append(pseudo_label_key is not None)
                reward_extra_info["pseudo_label_from_cache"].append(
                    self.pseudo_label_source == "target_cache_majority"
                )
                reward_extra_info["correctness_answer_support_count"].append(
                    eligible_counts[_answer_key(final_answer)] if valid_answer and not is_abstention else 0
                )
                reward_extra_info["group_size"].append(group_size)
                reward_extra_info["confidence"].append(confidence if valid_confidence else float("nan"))
                reward_extra_info["confidence_valid"].append(valid_confidence)
                reward_extra_info["final_answer_valid"].append(valid_answer)
                reward_extra_info["is_abstention"].append(is_abstention)
                reward_extra_info["answer_frequency_target"].append(confidence_target)
                reward_extra_info["base_confidence_target"].append(0.0 if is_abstention else base_confidence_target)
                reward_extra_info["target_cache_hit"].append(target_cache_hit)
                reward_extra_info["target_cache_ema_rate"].append(self.target_cache_ema_rate)
                reward_extra_info["calibration_answer_support_count"].append(answer_count)
                reward_extra_info["brier_score"].append(brier_score)
                reward_extra_info["answer_entropy"].append(answer_entropy)
                reward_extra_info["answer_entropy_normalized"].append(normalized_entropy)
                reward_extra_info["reasoning_loss_mask_token_ratio"].append(
                    reasoning_loss_mask[index, :response_length].sum().item() / response_length if response_length else 0.0
                )
                reward_extra_info["confidence_loss_mask_token_ratio"].append(
                    confidence_loss_mask[index, :response_length].sum().item() / response_length if response_length else 0.0
                )

            if self.target_cache_mode == "epoch_ema" and not self._initializing_target_cache:
                assert target_cache_key is not None
                self.pending_target_cache[target_cache_key].append(support_map)

            if printed_groups < self.num_examine:
                printed_groups += 1
                print("[majority_ttcl uid]", uid)
                print("[majority_ttcl answer_counts]", dict(all_answer_counts))
                print("[majority_ttcl pseudo_label_source]", self.pseudo_label_source)
                print("[majority_ttcl pseudo_label]", pseudo_label_key)

        # These masks and component rewards are consumed synchronously by the
        # trainer/actor.  Keep their sum as the ordinary reward tensor for
        # backwards-compatible reward/critic logging.
        data.batch["reasoning_loss_mask"] = reasoning_loss_mask
        data.batch["confidence_loss_mask"] = confidence_loss_mask
        reward_tensor = correctness_reward_tensor + calibration_reward_tensor
        result = {
            "reward_tensor": reward_tensor,
            "reward_components": {
                "correctness": correctness_reward_tensor,
                "calibration": calibration_reward_tensor,
            },
            "reward_extra_info": reward_extra_info,
        }
        return result if return_dict else reward_tensor
