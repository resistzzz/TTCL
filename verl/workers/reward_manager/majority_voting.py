"""Self-consistency pseudo-label reward manager based on majority voting.

For every prompt, the most frequent non-abstention answer among its rollout
group is used as a pseudo label.  A rollout receives binary reward 1 iff its
answer matches that label.  Ground truth is never used to form the reward;
when present, it is used only for monitoring pseudo-label quality.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.rlcr import _extract_final_answer
from verl.workers.reward_manager.ttcr import _answer_key, _is_abstention_answer


def _pseudo_ground_truth(answer: str) -> str:
    """Convert a parsed boxed answer to the ground-truth form used by scorers.

    ``_extract_final_answer(..., parse_mode='boxed')`` deliberately preserves
    ``\\boxed{...}``, because the math scorer expects it in ``solution_str``.
    The same scorer expects the reference answer itself without that wrapper.
    Answer-tag and boxed-text outputs are already plain answers and pass
    through unchanged.
    """

    if answer.startswith("\\boxed{") and answer.endswith("}"):
        return answer[len("\\boxed{") : -1]
    if answer.startswith("\\boxed "):
        return answer[len("\\boxed ") :]
    return answer


@register("majority_voting")
class MajorityVotingRewardManager(AbstractRewardManager):
    """Provide binary pseudo-correctness rewards from group majority answers.

    The majority is computed from ordinary in-group answer frequencies, not a
    leave-one-out estimate. Explicit abstentions and missing answers cannot
    become pseudo labels and always receive reward 0. Ties are broken by the
    earliest rollout in the group, making the result deterministic.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        monitor_correctness: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.monitor_correctness = monitor_correctness

    @staticmethod
    def _correctness(result: Any) -> float:
        value = result.get("acc", result["score"]) if isinstance(result, dict) else result
        return 1.0 if float(value) > 0.0 else 0.0

    def _monitor_answer_correctness(self, data_item: Any, answer: str | None) -> float:
        """Return verifier correctness for logging only, never for reward."""

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

    def _pseudo_correctness(self, data_item: Any, answer: str | None, pseudo_label: str | None) -> float:
        """Score an answer against the group's pseudo label as ground truth.

        This deliberately invokes the repository scorer rather than comparing
        normalized strings. For mathematical data, equivalent expressions can
        then receive the same binary reward as the selected mode. The original
        dataset ground truth is not read here.
        """

        if answer is None or pseudo_label is None or _is_abstention_answer(answer):
            return 0.0
        data_source = data_item.non_tensor_batch.get(self.reward_fn_key)
        if data_source is None:
            return 0.0
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        return self._correctness(
            self.compute_score(
                data_source=data_source,
                solution_str=answer,
                ground_truth=_pseudo_ground_truth(pseudo_label),
                extra_info=extra_info,
            )
        )

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "uid" not in data.non_tensor_batch:
            raise KeyError("MajorityVotingRewardManager requires non_tensor_batch['uid'] to group rollouts.")

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: defaultdict[str, list[Any]] = defaultdict(list)
        decoded: list[dict[str, Any] | None] = [None] * len(data)
        grouped_indices: defaultdict[Any, list[int]] = defaultdict(list)

        for index in range(len(data)):
            data_item = data[index]
            prompt_length = data_item.batch["prompts"].shape[-1]
            attention_mask = data_item.batch["attention_mask"]
            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            grouped_indices[data_item.non_tensor_batch["uid"]].append(index)
            if valid_response_length == 0:
                continue

            response_ids = data_item.batch["responses"][:valid_response_length]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            parse_mode = extra_info.get("parse_mode", "boxed")
            decoded[index] = {
                "final_answer": _extract_final_answer(response_text, parse_mode=parse_mode),
                "valid_response_length": valid_response_length,
            }

        printed_groups = 0
        for uid, indices in grouped_indices.items():
            eligible_answer_keys = [
                _answer_key(decoded[index]["final_answer"])
                for index in indices
                if decoded[index] is not None
                and decoded[index]["final_answer"] is not None
                and not _is_abstention_answer(decoded[index]["final_answer"])
            ]
            answer_counts = Counter(eligible_answer_keys)
            pseudo_label_key: str | None = None
            pseudo_label_count = 0
            if answer_counts:
                pseudo_label_count = max(answer_counts.values())
                pseudo_label_key = next(
                    answer_key for answer_key in eligible_answer_keys if answer_counts[answer_key] == pseudo_label_count
                )
            group_size = len(indices)
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
            pseudo_label_acc = self._monitor_answer_correctness(data[indices[0]], pseudo_label_answer)

            for index in indices:
                item = decoded[index]
                final_answer = item["final_answer"] if item is not None else None
                is_abstention = _is_abstention_answer(final_answer)
                valid_answer = final_answer is not None
                pseudo_correctness = self._pseudo_correctness(data[index], final_answer, pseudo_label_answer)
                actual_correctness = self._monitor_answer_correctness(data[index], final_answer)
                answer_support_count = answer_counts[_answer_key(final_answer)] if valid_answer and not is_abstention else 0

                if item is not None:
                    reward_tensor[index, item["valid_response_length"] - 1] = pseudo_correctness

                reward_extra_info["majority_voting_reward"].append(True)
                reward_extra_info["pseudo_correctness"].append(pseudo_correctness)
                reward_extra_info["correctness"].append(actual_correctness)
                reward_extra_info["pseudo_label_acc"].append(pseudo_label_acc)
                reward_extra_info["pseudo_label_support_ratio"].append(pseudo_label_support_ratio)
                reward_extra_info["pseudo_label_valid"].append(pseudo_label_key is not None)
                reward_extra_info["answer_support_count"].append(answer_support_count)
                reward_extra_info["group_size"].append(group_size)
                reward_extra_info["final_answer_valid"].append(valid_answer)
                reward_extra_info["is_abstention"].append(is_abstention)

            if printed_groups < self.num_examine:
                printed_groups += 1
                print("[majority_voting uid]", uid)
                print("[majority_voting answer_counts]", dict(answer_counts))
                print("[majority_voting pseudo_label]", pseudo_label_key)
                print("[majority_voting pseudo_label_support_ratio]", pseudo_label_support_ratio)
                print("[majority_voting pseudo_label_acc]", pseudo_label_acc)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
