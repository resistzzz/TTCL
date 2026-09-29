
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets
from prompt_utils import add_confidence_prompt_args, build_system_instruction, confidence_output_suffix



ANSWER_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/dapo17k")
    parser.add_argument("--hdfs_dir", default=None)
    add_confidence_prompt_args(parser)

    args = parser.parse_args()
    instruction_following = build_system_instruction(
        ANSWER_INSTRUCTION, args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style
    )

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "open-r1/DAPO-Math-17k-Processed"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, "en", trust_remote_code=True)

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")
            solution = example.pop("solution")
            _ = example.pop("source_prompt")
            data = {
                "data_source": "DAPO-Math-17k",
                "prompt": [
                    {"role": "system", "content": instruction_following},
                    {"role": "user", "content": question},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "requires_confidence": args.require_confidence,
                    "confidence_prompt_style": args.confidence_prompt_style,
                "answer_prompt_style": args.answer_prompt_style,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    print(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(
        os.path.join(
            local_dir, f"train{confidence_output_suffix(args.require_confidence, args.confidence_prompt_style, args.answer_prompt_style)}.parquet"
        )
    )
