"""Evaluate a HuggingFace model's accuracy and calibration on RLHF parquet data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.rlcr import (
    _extract_final_answer,
    _has_valid_format,
    _last_boxed_expression,
    _parse_confidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face model id.")
    parser.add_argument("--data-path", required=True, help="RLHF parquet evaluation dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory for results.jsonl and metrics.json.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent rollouts (completions) generated for each question.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N prompts.")
    parser.add_argument("--batch-size", type=int, default=128, help="Prompts submitted to vLLM per generation call.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto", help="vLLM dtype, e.g. auto, bfloat16, float16.")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--correctness-reward",
        choices=("default", "math_verify"),
        default="default",
        help="Scorer for correctness: existing dataset scorer or Math-Verify equivalence.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in apply_chat_template (default: NoThink mode).",
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.ece_bins < 1:
        parser.error("--ece-bins must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0.0 <= args.temperature:
        parser.error("--temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    return args


def score_to_correctness(score: Any) -> float:
    """Normalize repository reward-score conventions to binary correctness."""

    if isinstance(score, dict):
        # Do not use ``score.get("acc", score["score"])`` here: Python
        # evaluates the default expression eagerly, so an ``{"acc": ...}``
        # result without a ``score`` key would incorrectly raise KeyError.
        value = score["acc"] if "acc" in score else score["score"]
    else:
        value = score
    return 1.0 if float(value) > 0.0 else 0.0


def compute_correctness(
    reward_name: str,
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
) -> float:
    """Compute binary correctness with the explicitly selected evaluation scorer."""

    if reward_name == "math_verify":
        from verl.utils.reward_score.math_verify import reward_func

        score = reward_func(data_source, solution_str, ground_truth, extra_info)
    else:
        score = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    return score_to_correctness(score)


def compute_ece(confidences: list[float], correctness: list[float], num_bins: int) -> float | None:
    """Compute equal-width expected calibration error for valid confidences."""

    if not confidences:
        return None

    confidence_array = np.asarray(confidences, dtype=float)
    correctness_array = np.asarray(correctness, dtype=float)
    ece = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == num_bins - 1:
            in_bin = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            in_bin = (confidence_array >= lower) & (confidence_array < upper)
        if not np.any(in_bin):
            continue
        bin_weight = np.mean(in_bin)
        ece += bin_weight * abs(np.mean(correctness_array[in_bin]) - np.mean(confidence_array[in_bin]))
    return float(ece)


def compute_pce(confidences: list[float], correctness: list[float], num_bins: int) -> float | None:
    """Compute Positive Calibration Error (PCE) on valid confidences.

    PCE uses the same equal-width bins as ECE, but includes only bins whose
    mean confidence exceeds their empirical accuracy. It therefore measures
    overconfidence only; its denominator remains the full valid sample count.
    """

    if not confidences:
        return None

    confidence_array = np.asarray(confidences, dtype=float)
    correctness_array = np.asarray(correctness, dtype=float)
    pce = 0.0
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == num_bins - 1:
            in_bin = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            in_bin = (confidence_array >= lower) & (confidence_array < upper)
        if not np.any(in_bin):
            continue

        bin_accuracy = float(np.mean(correctness_array[in_bin]))
        bin_confidence = float(np.mean(confidence_array[in_bin]))
        if bin_confidence > bin_accuracy:
            pce += float(np.mean(in_bin)) * (bin_confidence - bin_accuracy)
    return float(pce)


def compute_auroc(confidences: list[float], correctness: list[float]) -> float | None:
    """Compute binary AUROC with average ranks for tied confidence values."""

    if not confidences:
        return None

    scores = np.asarray(confidences, dtype=float)
    labels = np.asarray(correctness, dtype=int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    position = 0
    while position < len(scores):
        end = position + 1
        while end < len(scores) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end

    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


class MetricAccumulator:
    """Accumulate scalar metrics without retaining prompt or response text in memory."""

    def __init__(self) -> None:
        self.num_completions = 0
        self.correctness_sum = 0.0
        self.brier_sum = 0.0
        self.format_valid_sum = 0.0
        self.answer_parse_failure_count = 0
        self.valid_confidences: list[float] = []
        self.valid_correctness: list[float] = []

    def add(self, record: dict[str, Any]) -> None:
        self.num_completions += 1
        self.correctness_sum += record["correctness"]
        self.brier_sum += record["brier_score"]
        self.format_valid_sum += float(record["format_valid"])
        self.answer_parse_failure_count += int(not record["has_final_answer"])
        if record["confidence_valid"]:
            self.valid_confidences.append(record["confidence"])
            self.valid_correctness.append(record["correctness"])

    def summary(self, ece_bins: int) -> dict[str, Any]:
        if self.num_completions == 0:
            return {
                "num_completions": 0,
                "accuracy": None,
                "brier_score": None,
                "ece": None,
                "pce": None,
                "auroc": None,
                "confidence_valid_rate": None,
                "confidence_valid_mean": None,
                "format_valid_rate": None,
                "invalid_confidence_count": 0,
                "answer_parse_failure_count": 0,
                "answer_parse_failure_rate": None,
                "answer_parse_success_rate": None,
            }

        valid_count = len(self.valid_confidences)
        return {
            "num_completions": self.num_completions,
            "accuracy": self.correctness_sum / self.num_completions,
            "brier_score": self.brier_sum / self.num_completions,
            "ece": compute_ece(self.valid_confidences, self.valid_correctness, ece_bins),
            "pce": compute_pce(self.valid_confidences, self.valid_correctness, ece_bins),
            "auroc": compute_auroc(self.valid_confidences, self.valid_correctness),
            "confidence_valid_rate": valid_count / self.num_completions,
            "confidence_valid_mean": float(np.mean(self.valid_confidences)) if valid_count else None,
            "format_valid_rate": self.format_valid_sum / self.num_completions,
            "invalid_confidence_count": self.num_completions - valid_count,
            # One completion is one case here. This remains meaningful when
            # ``num_samples > 1`` because it counts every rollout separately.
            "answer_parse_failure_count": self.answer_parse_failure_count,
            "answer_parse_failure_rate": self.answer_parse_failure_count / self.num_completions,
            "answer_parse_success_rate": 1.0 - self.answer_parse_failure_count / self.num_completions,
        }


def summarize_records(records: list[dict[str, Any]], ece_bins: int) -> dict[str, Any]:
    """Compute requested calibration metrics for a list of generated completions."""

    accumulator = MetricAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.summary(ece_bins)

def render_prompts(
    dataset: datasets.Dataset, tokenizer: Any, enable_thinking: bool, start_index: int = 0
) -> tuple[list[str], list[dict[str, Any]]]:
    prompts = []
    examples = []
    for sample_index, example in enumerate(dataset, start=start_index):
        prompt_messages = example["prompt"]
        prompts.append(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
        examples.append(
            {
                "sample_index": sample_index,
                "prompt": prompt_messages,
                "data_source": example["data_source"],
                "ground_truth": example["reward_model"]["ground_truth"],
                "extra_info": example.get("extra_info", {}),
                "parse_mode": example.get("extra_info", {}).get("parse_mode", "boxed"),
            }
        )
    return prompts, examples


def build_records(
    outputs: list[Any], examples: list[dict[str, Any]], num_samples: int, correctness_reward: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example, request_output in zip(examples, outputs, strict=True):
        if len(request_output.outputs) != num_samples:
            raise RuntimeError(
                f"Expected {num_samples} completions for sample {example['sample_index']}, "
                f"but vLLM returned {len(request_output.outputs)}."
            )
        for completion_index, completion in enumerate(request_output.outputs):
            response = completion.text
            parse_mode = example["parse_mode"]
            final_answer = _extract_final_answer(response, parse_mode=parse_mode)
            correctness = compute_correctness(
                reward_name=correctness_reward,
                data_source=example["data_source"],
                solution_str=final_answer if final_answer is not None else response,
                ground_truth=example["ground_truth"],
                extra_info=example["extra_info"],
            )
            confidence = _parse_confidence(response, parse_mode=parse_mode)
            has_boxed_answer = _last_boxed_expression(response) is not None
            confidence_valid = confidence is not None
            format_valid = _has_valid_format(response, parse_mode=parse_mode)
            brier_score = (confidence - correctness) ** 2 if confidence_valid else 1.0
            records.append(
                {
                    "record_type": "completion",
                    "sample_index": example["sample_index"],
                    "completion_index": completion_index,
                    "data_source": example["data_source"],
                    "ground_truth": example["ground_truth"],
                    "prompt": example["prompt"],
                    "response": response,
                    "parse_mode": parse_mode,
                    "final_answer": final_answer,
                    "correctness": correctness,
                    "confidence": confidence,
                    "confidence_valid": confidence_valid,
                    "has_boxed_answer": has_boxed_answer,
                    "has_final_answer": final_answer is not None,
                    "format_valid": format_valid,
                    "brier_score": brier_score,
                }
            )
    return records


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_parquet(args.data_path)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.seed is not None:
        llm_kwargs["seed"] = args.seed
    llm = LLM(**llm_kwargs)

    sampling_kwargs = {
        "n": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    overall_metrics = MetricAccumulator()
    source_metrics: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    temporary_records_path = output_dir / ".completion_records.jsonl"
    with temporary_records_path.open("w", encoding="utf-8") as temporary_records, tqdm(
        total=len(dataset), desc="Evaluating prompts"
    ) as progress_bar:
        for start_index in range(0, len(dataset), args.batch_size):
            end_index = min(start_index + args.batch_size, len(dataset))
            batch_dataset = dataset.select(range(start_index, end_index))
            prompts, examples = render_prompts(batch_dataset, tokenizer, args.enable_thinking, start_index)
            outputs = llm.generate(prompts, SamplingParams(**sampling_kwargs), use_tqdm=False)
            for record in build_records(outputs, examples, args.num_samples, args.correctness_reward):
                overall_metrics.add(record)
                source_metrics[record["data_source"]].add(record)
                temporary_records.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
            progress_bar.update(end_index - start_index)

    metrics = {
        "overall": overall_metrics.summary(args.ece_bins),
        "by_data_source": {
            data_source: accumulator.summary(args.ece_bins)
            for data_source, accumulator in sorted(source_metrics.items())
        },
    }
    config = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "vllm_version": "0.11.0",
        "num_prompts": len(dataset),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "ece_bins": args.ece_bins,
        "correctness_reward": args.correctness_reward,
        "batch_size": args.batch_size,
        "enable_thinking": args.enable_thinking,
    }

    metrics_path = output_dir / "metrics.json"
    results_path = output_dir / "results.jsonl"
    metrics_path.write_text(json.dumps({"config": config, "metrics": metrics}, indent=2, ensure_ascii=False) + "\n")
    with results_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"record_type": "summary", **metrics}, ensure_ascii=False, default=json_default) + "\n")
        file.write(json.dumps({"record_type": "config", **config}, ensure_ascii=False, default=json_default) + "\n")
        with temporary_records_path.open("r", encoding="utf-8") as temporary_records:
            shutil.copyfileobj(temporary_records, file)
    temporary_records_path.unlink()

    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved completion records to {results_path}")


if __name__ == "__main__":
    main()
