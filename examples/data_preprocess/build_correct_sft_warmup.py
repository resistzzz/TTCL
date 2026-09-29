"""Build a format-and-confidence warm-up SFT dataset from model rollouts.

The rollout prompt intentionally has no confidence request. Verified-correct
and verified-incorrect responses are paired with the corresponding
confidence-prompt messages and synthetic verbalized confidence labels: high
for correct responses and low for incorrect responses. The resulting parquet files use
verl's MultiTurnSFTDataset schema.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

from verl.workers.reward_manager.rlcr import _extract_final_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-data-path", required=True, help="No-confidence RLHF parquet used for rollout.")
    parser.add_argument(
        "--target-prompt-data-path",
        required=True,
        help="Same examples with the desired confidence system prompt (for example, train_placeholder_conf.parquet).",
    )
    parser.add_argument("--output-dir", required=True, help="Writes train.parquet and val.parquet here.")
    parser.add_argument(
        "--num-correct",
        type=int,
        default=1000,
        help="Number of verified-correct SFT examples to retain.",
    )
    parser.add_argument(
        "--num-incorrect",
        type=int,
        default=1000,
        help="Number of verified-incorrect SFT examples to retain.",
    )
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--correct-confidence-min", type=float, default=0.70)
    parser.add_argument("--correct-confidence-max", type=float, default=0.95)
    parser.add_argument("--incorrect-confidence-min", type=float, default=0.05)
    parser.add_argument("--incorrect-confidence-max", type=float, default=0.35)
    args = parser.parse_args()

    target_size = args.num_correct + args.num_incorrect
    if args.num_correct < 0 or args.num_incorrect < 0 or target_size < 2:
        parser.error("--num-correct and --num-incorrect must sum to at least 2")
    if not 0 < args.val_size < target_size:
        parser.error("--val-size must be positive and smaller than total requested examples")
    if args.rollout_n < 1 or args.batch_size < 1 or args.max_tokens < 1:
        parser.error("--rollout-n, --batch-size, and --max-tokens must be positive")
    if not 0.0 <= args.temperature or not 0.0 < args.top_p <= 1.0:
        parser.error("Invalid sampling parameters")
    for lower_name, upper_name in (
        ("correct_confidence_min", "correct_confidence_max"),
        ("incorrect_confidence_min", "incorrect_confidence_max"),
    ):
        lower, upper = getattr(args, lower_name), getattr(args, upper_name)
        if not 0.0 <= lower <= upper <= 1.0:
            parser.error(f"--{lower_name.replace('_', '-')} and --{upper_name.replace('_', '-')} must be in [0, 1]")
    return args


def as_messages(value: Any) -> list[dict[str, str]]:
    """Normalize Arrow/Pandas nested prompt values into chat-message dicts."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    messages = []
    for message in value:
        if hasattr(message, "item"):
            message = message.item()
        messages.append({"role": str(message["role"]), "content": str(message["content"])})
    return messages


def score_to_correctness(score: Any) -> bool:
    value = score["acc"] if isinstance(score, dict) and "acc" in score else score["score"] if isinstance(score, dict) else score
    return float(value) > 0.0


def is_correct(example: dict[str, Any], response: str) -> tuple[bool, str | None]:
    """Verify a parsed answer with Math-Verify, returning the parsed answer too."""

    parse_mode = example.get("extra_info", {}).get("parse_mode", "boxed")
    answer = _extract_final_answer(response, parse_mode=parse_mode)
    if answer is None:
        return False, None

    from verl.utils.reward_score.math_verify import reward_func

    score = reward_func(
        example["data_source"], answer, example["reward_model"]["ground_truth"], example.get("extra_info", {})
    )
    return score_to_correctness(score), answer


def canonical_response(response: str, final_answer: str, confidence: float) -> str:
    """Keep reasoning through the final answer, then append a canonical confidence line."""

    answer_start = response.rfind(final_answer)
    if answer_start < 0:
        raise ValueError("Parsed final answer is not present in its response")
    reasoning = response[:answer_start].rstrip()
    prefix = f"{reasoning}\n\n" if reasoning else ""
    return f"{prefix}{final_answer}\n<conf> Confidence: {confidence:.2f}"


def sample_cent_confidence(rng: random.Random, lower: float, upper: float) -> float:
    """Sample an inclusive two-decimal confidence within the configured interval."""

    lower_cent = math.ceil(lower * 100 - 1e-9)
    upper_cent = math.floor(upper * 100 + 1e-9)
    return rng.randint(lower_cent, upper_cent) / 100.0


def validate_alignment(source: datasets.Dataset, target: datasets.Dataset) -> None:
    if len(source) != len(target):
        raise ValueError(f"Datasets must have the same row count, got {len(source)} and {len(target)}")
    for index, (source_example, target_example) in enumerate(zip(source, target, strict=True)):
        if source_example["data_source"] != target_example["data_source"]:
            raise ValueError(f"data_source mismatch at row {index}")
        if source_example["reward_model"]["ground_truth"] != target_example["reward_model"]["ground_truth"]:
            raise ValueError(f"ground_truth mismatch at row {index}")


def main() -> None:
    args = parse_args()
    source_dataset = datasets.Dataset.from_parquet(args.source_data_path)
    target_dataset = datasets.Dataset.from_parquet(args.target_prompt_data_path)
    validate_alignment(source_dataset, target_dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=args.rollout_n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    source_indices = list(range(len(source_dataset)))
    rng.shuffle(source_indices)
    selected: list[dict[str, Any]] = []
    correct_count = 0
    incorrect_count = 0
    for offset in tqdm(range(0, len(source_indices), args.batch_size), desc="Generating confidence SFT examples"):
        if correct_count >= args.num_correct and incorrect_count >= args.num_incorrect:
            break
        indices = source_indices[offset : offset + args.batch_size]
        source_examples = [source_dataset[index] for index in indices]
        prompts = [
            tokenizer.apply_chat_template(
                as_messages(example["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            for example in source_examples
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for index, source_example, request_output in zip(indices, source_examples, outputs, strict=True):
            correct_outputs: list[tuple[str, str]] = []
            incorrect_outputs: list[tuple[str, str]] = []
            for output in request_output.outputs:
                correct, final_answer = is_correct(source_example, output.text)
                if final_answer is None:
                    continue
                if correct:
                    correct_outputs.append((output.text, final_answer))
                else:
                    incorrect_outputs.append((output.text, final_answer))

            # Keep only questions for which the model produced both outcomes:
            # the resulting data teaches confidence to distinguish otherwise
            # similarly formatted model-generated solutions.
            if not correct_outputs or not incorrect_outputs:
                continue

            def add_example(
                candidates: list[tuple[str, str]],
                confidence: float,
                is_correct_label: bool,
            ) -> None:
                response, final_answer = rng.choice(candidates)
                messages = as_messages(target_dataset[index]["prompt"])
                messages.append(
                    {
                        "role": "assistant",
                        "content": canonical_response(response, final_answer, confidence),
                    }
                )
                selected.append(
                    {
                        "messages": messages,
                        "enable_thinking": False,
                        "source_index": index,
                        "data_source": source_example["data_source"],
                        "ground_truth": source_example["reward_model"]["ground_truth"],
                        "synthetic_confidence": confidence,
                        "synthetic_correctness": float(is_correct_label),
                    }
                )

            if correct_count < args.num_correct:
                add_example(
                    correct_outputs,
                    sample_cent_confidence(rng, args.correct_confidence_min, args.correct_confidence_max),
                    True,
                )
                correct_count += 1
            if incorrect_count < args.num_incorrect:
                add_example(
                    incorrect_outputs,
                    sample_cent_confidence(rng, args.incorrect_confidence_min, args.incorrect_confidence_max),
                    False,
                )
                incorrect_count += 1
            if correct_count >= args.num_correct and incorrect_count >= args.num_incorrect:
                break

    if correct_count < args.num_correct or incorrect_count < args.num_incorrect:
        raise RuntimeError(
            "Collected insufficient paired examples: "
            f"correct={correct_count}/{args.num_correct}, "
            f"incorrect={incorrect_count}/{args.num_incorrect}. "
            "Increase --rollout-n or use more source prompts."
        )

    rng.shuffle(selected)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_rows = selected[: args.val_size]
    train_rows = selected[args.val_size :]
    datasets.Dataset.from_list(train_rows).to_parquet(output_dir / "train.parquet")
    datasets.Dataset.from_list(val_rows).to_parquet(output_dir / "val.parquet")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"Wrote {len(train_rows)} training and {len(val_rows)} validation examples to {output_dir}")


if __name__ == "__main__":
    main()
