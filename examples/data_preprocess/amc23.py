"""Preprocess the AMC 2023 dataset into an RLCR validation parquet file."""

import argparse
import math
import os

import datasets
from prompt_utils import add_confidence_prompt_args, build_system_instruction, confidence_output_suffix


ANSWER_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def format_answer(answer: float) -> str:
    """Convert the dataset's float answer column to a math-verifier-friendly string."""

    if not math.isfinite(answer):
        raise ValueError(f"AMC23 contains a non-finite answer: {answer!r}")
    if answer.is_integer():
        return str(int(answer))
    return format(answer, ".15g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/amc23")
    add_confidence_prompt_args(parser)
    args = parser.parse_args()
    instruction_following = build_system_instruction(
        ANSWER_INSTRUCTION, args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style
    )

    data_source = "zwhe99/amc23"
    print(f"Loading {data_source} from Hugging Face...", flush=True)
    dataset = datasets.load_dataset(data_source, split="test")

    def process_fn(example, idx):
        return {
            # Reuse the DAPO-Math-17k math_reward branch in default_compute_score.
            "data_source": "AMC23",
            "prompt": [
                {"role": "system", "content": instruction_following},
                {"role": "user", "content": example["question"]},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": format_answer(example["answer"])},
            "extra_info": {
                "split": "test",
                "index": idx,
                "requires_confidence": args.require_confidence,
                "confidence_prompt_style": args.confidence_prompt_style,
                "answer_prompt_style": args.answer_prompt_style,
            },
        }

    validation_dataset = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    print(validation_dataset)

    os.makedirs(args.local_dir, exist_ok=True)
    output_path = os.path.join(
        args.local_dir, f"amc23{confidence_output_suffix(args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style)}.parquet"
    )
    validation_dataset.to_parquet(output_path)
    print(f"Wrote {len(validation_dataset)} validation examples to {output_path}")


if __name__ == "__main__":
    main()
