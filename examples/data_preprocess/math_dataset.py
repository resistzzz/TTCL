"""Preprocess the AMC 2023 dataset into an RLCR validation parquet file."""

import argparse
import math
import os
from numbers import Real

import datasets

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
from prompt_utils import add_confidence_prompt_args, build_system_instruction, confidence_output_suffix


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


ANSWER_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def format_answer(answer: str | Real) -> str:
    """Convert numeric answers to canonical strings while preserving LaTex answers."""

    if isinstance(answer, str):
        answer = answer.strip()
        if not answer:
            raise ValueError("contains an empty answer")
        return answer
    if not isinstance(answer, Real):
        raise TypeError(f"Unsupported answer type: {type(answer).__name__}")
    if not math.isfinite(answer):
        raise ValueError(f"contains a non-finite answer: {answer!r}")
    if float(answer).is_integer():
        return str(int(answer))
    return format(answer, ".15g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/math")
    add_confidence_prompt_args(parser)
    args = parser.parse_args()
    instruction_following = build_system_instruction(
        ANSWER_INSTRUCTION, args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style
    )

    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading {data_source} from Hugging Face...", flush=True)
    dataset = datasets.load_dataset(data_source, split="train")

    def process_fn(example, idx):
        return {
            # Reuse the DAPO-Math-17k math_reward branch in default_compute_score.
            "data_source": "lighteval/MATH",
            "prompt": [
                {"role": "system", "content": instruction_following},
                {"role": "user", "content": example["problem"]},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": extract_solution(example["solution"])},
            "extra_info": {
                "split": "train",
                "index": idx,
                "parse_mode": "boxed",
                "requires_confidence": args.require_confidence,
                "confidence_prompt_style": args.confidence_prompt_style,
                "answer_prompt_style": args.answer_prompt_style,
            },
        }

    train_dataset = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    print(train_dataset)

    os.makedirs(args.local_dir, exist_ok=True)
    output_path = os.path.join(
        args.local_dir,
        f"math_train{confidence_output_suffix(args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style)}.parquet",
    )
    print(train_dataset[0])
    train_dataset.to_parquet(output_path)
    print(f"Wrote {len(train_dataset)} validation examples to {output_path}")


if __name__ == "__main__":
    main()
