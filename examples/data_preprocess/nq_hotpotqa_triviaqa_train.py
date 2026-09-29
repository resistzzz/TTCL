"""Preprocess ChineseSimpleQA into an RLCR validation parquet file."""

import argparse
import os

import datasets
from prompt_utils import add_confidence_prompt_args, build_system_instruction, confidence_output_suffix


ANSWER_INSTRUCTION_ANSTAG = "Let's think step by step and output the final answer within <answer> </answer> tags."


ANSWER_INSTRUCTION_BOXED = "Let's think step by step and output the final answer within \\boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/flashrag")
    parser.add_argument("--subset", default="nq", choices=["nq", "hotpotqa", "triviaqa"])
    add_confidence_prompt_args(parser)
    parser.add_argument(
        "--parse_mode",
        choices=("boxed_text", "answer_tag"),
        default="answer_tag",
        help="Final-answer protocol stored in extra_info and requested in the system prompt.",
    )
    parser.add_argument(
        "--n_limit",
        type=int,
        default=200,
        help="Keep only the first N examples of the selected split; use -1 for the full split.",
    )
    args = parser.parse_args()
    if args.n_limit == 0 or args.n_limit < -1:
        parser.error("--n_limit must be a positive integer or -1 for the full split.")

    instruction_following = build_system_instruction(
        {"boxed_text": ANSWER_INSTRUCTION_BOXED, "answer_tag": ANSWER_INSTRUCTION_ANSTAG}[args.parse_mode],
        args.require_confidence,
        args.confidence_prompt_style,
        args.answer_prompt_style,
    )

    data_source = "RUC-NLPIR/FlashRAG_datasets"
    print(f"Loading {data_source} from Hugging Face...", flush=True)
    split = "train"
    
    dataset = datasets.load_dataset(data_source, args.subset, split=split)
    if args.n_limit != -1:
        selected_count = min(args.n_limit, len(dataset))
        dataset = dataset.select(range(selected_count))
        print(
            f"Selected the first {selected_count} examples from {args.subset}/{split} "
            f"(n_limit={args.n_limit}).",
            flush=True,
        )
    else:
        print(f"Using all {len(dataset)} examples from {args.subset}/{split}.", flush=True)

    def process_fn(example, idx):
        return {
            "data_source": "FlashRAG",
            "prompt": [
                {"role": "system", "content": instruction_following},
                {"role": "user", "content": example["question"]},
            ],
            "ability": "fact_qa",
            "reward_model": {"style": "rule", "ground_truth": example["golden_answers"]},
            "extra_info": {
                "split": "train",
                "index": idx,
                "parse_mode": args.parse_mode,
                "requires_confidence": args.require_confidence,
                "confidence_prompt_style": args.confidence_prompt_style,
                "answer_prompt_style": args.answer_prompt_style,
            },
        }

    validation_dataset = dataset.map(process_fn, with_indices=True, remove_columns=dataset.column_names)
    print(validation_dataset)

    output_path = os.path.join(
        args.local_dir,
        args.subset,
        f"{args.subset}_train{confidence_output_suffix(args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style)}.parquet",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    validation_dataset.to_parquet(output_path)
    print(validation_dataset[0])
    print(f"Wrote {len(validation_dataset)} validation examples to {output_path}")


if __name__ == "__main__":
    main()
