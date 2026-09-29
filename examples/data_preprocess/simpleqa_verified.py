"""Preprocess SimpleQA-Verified into an RLCR validation parquet file."""

import argparse
import os

import datasets
from prompt_utils import add_confidence_prompt_args, build_system_instruction, confidence_output_suffix


ANSWER_INSTRUCTION_ANSTAG = "Let's think step by step and output the final answer within <answer> </answer> tags."


ANSWER_INSTRUCTION_BOXED = "Let's think step by step and output the final answer within \\boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/simpleqa_verified")
    add_confidence_prompt_args(parser)
    parser.add_argument(
        "--parse_mode",
        choices=("boxed_text", "answer_tag"),
        default="answer_tag",
        help="Final-answer protocol stored in extra_info and requested in the system prompt.",
    )
    args = parser.parse_args()

    instruction_following = build_system_instruction(
        {"boxed_text": ANSWER_INSTRUCTION_BOXED, "answer_tag": ANSWER_INSTRUCTION_ANSTAG}[args.parse_mode],
        args.require_confidence,
        args.confidence_prompt_style,
        args.answer_prompt_style,
    )

    data_source = "google/simpleqa-verified"
    print(f"Loading {data_source} from Hugging Face...", flush=True)
    dataset = datasets.load_dataset(data_source, split="eval")

    def process_fn(example, idx):
        return {
            "data_source": "SimpleQA",
            "prompt": [
                {"role": "system", "content": instruction_following},
                {"role": "user", "content": example["problem"]},
            ],
            "ability": "fact_qa",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "split": "eval",
                "index": idx,
                "parse_mode": args.parse_mode,
                "requires_confidence": args.require_confidence,
                "confidence_prompt_style": args.confidence_prompt_style,
                "answer_prompt_style": args.answer_prompt_style,
            },
        }

    validation_dataset = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    print(validation_dataset)

    os.makedirs(args.local_dir, exist_ok=True)
    output_path = os.path.join(
        args.local_dir,
        f"simpleqa_{args.parse_mode}{confidence_output_suffix(args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style)}.parquet",
    )
    validation_dataset.to_parquet(output_path)
    print(f"Wrote {len(validation_dataset)} validation examples to {output_path}")


if __name__ == "__main__":
    main()
