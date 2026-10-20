"""Evaluate factual QA inference baselines with the RLCR data and metric protocol.

Methods are direct, cot, self_consistency, bon, and self_refine. BoN uses a
separately served larger open-source judge; SC confidence is the mean
verbal confidence of valid samples in the winning answer cluster. These choices
are recorded in the output so the judge and confidence source remain explicit.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets
import httpx
from openai import BadRequestError, OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluate_rlcr import MetricAccumulator as SharedMetricAccumulator, build_records as shared_build_records, compute_correctness, json_default, render_prompts
from verl.workers.reward_manager.rlcr import _extract_final_answer, _has_valid_format, _last_boxed_expression, _parse_confidence

class MetricAccumulator(SharedMetricAccumulator):
    """Keep baseline valid-confidence statistics independent of shared revisions."""

    def add(self, record):
        normalized = dict(record)
        normalized["brier_score"] = record["brier_score"] if record["confidence_valid"] else 0.0
        super().add(normalized)

    def summary(self, ece_bins):
        result = super().summary(ece_bins)
        valid_count = len(self.valid_confidences)
        result["valid_count"] = valid_count
        result["brier_score"] = self.brier_sum / valid_count if valid_count else None
        return result


def build_records(*args, **kwargs):
    records = shared_build_records(*args, **kwargs)
    for record in records:
        if not record["confidence_valid"]:
            record["brier_score"] = None
    return records


METHODS = ("direct", "cot", "self_consistency", "bon", "self_refine")
DIRECT_SENTENCE = "Let's think step by step and output the final answer within <answer> </answer> tags."
DIRECT_REPLACEMENT = "Answer directly and concisely; output the final answer within <answer> </answer> tags."
REFINE_CONFIDENCE_PATTERN = re.compile(
    r"<conf>\s*(?:Confidence:\s*)?"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:</conf>)?\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--data-path", required=True, help="Same RLHF parquet used by evaluate_rlcr.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--draft-results", type=Path, help="Reuse saved self_refine drafts instead of generating initial rollouts")
    parser.add_argument("--refine-confidence-results", type=Path, help="Reuse saved Self-Refine final answers; rerun only confidence estimation")
    parser.add_argument("--resume-completions", action="store_true", help="Continue from existing .completion_records.jsonl")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--num-samples", type=int, help="K: defaults to 32 for SC and 8 for BoN")
    parser.add_argument("--eval-rollouts", type=int, default=1, help="Independent repeats per question; use 8 for Avg@8")
    parser.add_argument("--bon-judge-base-url", help="OpenAI-compatible endpoint serving the larger judge")
    parser.add_argument("--bon-judge-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--bon-judge-api-key", default="dummy")
    parser.add_argument("--judge-concurrency", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--aux-max-tokens", type=int, default=512, help="Judge/feedback/revision budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-impl", choices=("auto", "vllm", "transformers"), default="auto")
    parser.add_argument("--tokenizer-mode", choices=("auto", "slow", "custom"), default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--correctness-reward", choices=("default", "math_verify"), default="default")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true", help="Default is Qwen NoThink")
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples < 2 and args.method in {"self_consistency", "bon"}:
        parser.error("SC and BoN require --num-samples >= 2")
    if args.method == "bon" and not args.bon_judge_base_url:
        parser.error("BoN requires --bon-judge-base-url for the separate large-model judge")
    if args.draft_results and args.method != "self_refine":
        parser.error("--draft-results is only supported for self_refine")
    if args.refine_confidence_results:
        if args.method != "self_refine" or args.draft_results:
            parser.error("--refine-confidence-results requires self_refine without --draft-results")
        if args.refine_confidence_results.resolve() == (Path(args.output_dir) / "results.jsonl").resolve():
            parser.error("Back up the source results before replacing them")
    if args.draft_results and args.resume_completions:
        parser.error("--resume-completions cannot be combined with --draft-results")
    if args.draft_results and args.draft_results.resolve() == (Path(args.output_dir) / "results.jsonl").resolve():
        parser.error("--draft-results must differ from the output results.jsonl")
    if args.method in {"self_consistency", "bon"} and args.temperature <= 0:
        parser.error("SC and BoN require positive --temperature")
    if args.max_tokens < 1 or args.aux_max_tokens < 1 or args.batch_size < 1 or args.eval_rollouts < 1:
        parser.error("Token budgets, batch size, and eval rollouts must be positive")
    if not 0 < args.top_p <= 1 or args.ece_bins < 1:
        parser.error("--top-p must be in (0, 1] and --ece-bins must be positive")
    if args.judge_concurrency < 1 or any(v is not None and v < 1 for v in (args.max_num_seqs, args.max_num_batched_tokens)):
        parser.error("Concurrency and scheduling limits must be positive")
    return args


def messages_for_method(messages: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    """Keep the dataset's answer/confidence protocol; change only reasoning style."""
    result = [dict(message) for message in messages]
    if method in {"direct", "bon", "self_refine"}:
        for message in result:
            if message["role"] == "system":
                content = message["content"].replace(DIRECT_SENTENCE, DIRECT_REPLACEMENT)
                if content == message["content"]:
                    content = "Answer directly and concisely.\n" + content
                message["content"] = content
                break
        else:
            result.insert(0, {"role": "system", "content": "Answer directly and concisely."})
    return result


def render_messages(messages: list[dict[str, str]], tokenizer: Any, enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen" in tokenizer.__class__.__name__.lower():
        kwargs["enable_thinking"] = enable_thinking
    elif "pangu" in tokenizer.__class__.__name__.lower() and not enable_thinking:
        messages = [dict(message) for message in messages]
        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] += " /no_think"
                break
    return tokenizer.apply_chat_template(messages, **kwargs)


def sampling(args: argparse.Namespace, *, n: int = 1, auxiliary: bool = False, seed: int | None = None) -> SamplingParams:
    return SamplingParams(
        n=n,
        temperature=0.0 if auxiliary else args.temperature,
        top_p=1.0 if auxiliary else args.top_p,
        max_tokens=args.aux_max_tokens if auxiliary else args.max_tokens,
        seed=args.seed if seed is None else seed,
    )


class TransformersLLM:
    """Compatibility backend for custom architectures unsupported by vLLM."""

    def __init__(self, model_path: str, tokenizer: Any, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if dtype in {"auto", "bfloat16"} else dtype,
        ).eval().to("npu:0")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompts: list[str], params: SamplingParams, use_tqdm: bool = False) -> list[Any]:
        torch = self.torch
        torch.manual_seed(params.seed or 0)
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to("npu:0")
        do_sample = params.temperature > 0
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=params.max_tokens,
                do_sample=do_sample,
                temperature=params.temperature if do_sample else None,
                top_p=params.top_p if do_sample else None,
                num_return_sequences=params.n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_length = encoded.input_ids.shape[1]
        texts = self.tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=value) for value in texts[i:i + params.n]])
                for i in range(0, len(texts), params.n)]


def answer_key(answer: str | None) -> str:
    """Use the same whitespace-only normalization as factual QA exact_match."""
    return answer.strip() if answer is not None else ""


def parse_bon_choice(text: str, count: int) -> int | None:
    """Accept one choice, rejecting lists, ranges, and explicit alternatives."""
    text = (text or "").strip()
    if "\n" in text or "\r" in text:
        return None
    if re.search(r"(?:\b(?:and|or)\b|[,;/&、]|和|或)\s*(?:option\s*)?\d+", text, re.IGNORECASE):
        return None
    match = re.fullmatch(r"(?:Option\s*)?(\d+)(?:[.)．。:：]\s*(.*))?", text, flags=re.IGNORECASE)
    if match is None:
        return None
    suffix = match.group(2) or ""
    if re.search(r"\b\d+[.)．。:：]\s*\S", suffix):
        return None
    index = int(match.group(1)) - 1
    return index if 0 <= index < count else None


def parse_sample(response: str, parse_mode: str) -> dict[str, Any]:
    answer = _extract_final_answer(response, parse_mode=parse_mode)
    confidence = _parse_confidence(response, parse_mode=parse_mode)
    return {
        "answer": answer,
        "confidence": confidence,
        "format_valid": _has_valid_format(response, parse_mode=parse_mode),
        "response": response,
    }


def choose_sc(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Vote over all K samples, including invalid outputs in the denominator."""
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        key = answer_key(sample["answer"])
        if key:
            buckets[key].append((index, sample))
    if not buckets:
        return samples[0], {
            "vote_count": 0, "vote_share": 0.0, "valid_confidence_count": 0,
            "confidence_source": "winning_cluster_mean_verbal",
            "selection_fallback": "no_parsed_answer",
            "sample_responses": [sample["response"] for sample in samples],
        }
    # A tie is resolved by the earliest sample, without looking at confidence.
    winner = max(buckets.values(), key=lambda group: (len(group), -group[0][0]))
    valid = [sample["confidence"] for _, sample in winner if sample["format_valid"] and sample["confidence"] is not None]
    representative = next(
        (sample for _, sample in winner if sample["format_valid"] and sample["confidence"] is not None),
        winner[0][1],
    )
    confidence = mean(valid) if valid else None
    selected = dict(representative)
    selected["confidence"] = confidence
    selected["format_valid"] = confidence is not None
    return selected, {
        "vote_count": len(winner),
        "vote_share": len(winner) / len(samples),
        "valid_confidence_count": len(valid),
        "confidence_source": "winning_cluster_mean_verbal",
        "sample_responses": [sample["response"] for sample in samples],
    }


def auxiliary_generate(
    llm: LLM, tokenizer: Any, args: argparse.Namespace, messages: list[list[dict[str, str]]]
) -> list[str]:
    if not messages:
        return []
    rendered = [render_messages(message, tokenizer, args.enable_thinking) for message in messages]
    return [item.outputs[0].text for item in llm.generate(rendered, sampling(args, auxiliary=True), use_tqdm=False)]


def choose_bon(
    judge: OpenAI, args: argparse.Namespace, examples: list[dict[str, Any]], groups: list[list[dict[str, Any]]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    choices = []
    judge_prompts = []
    judge_indices = []
    for group_index, (example, samples) in enumerate(zip(examples, groups, strict=True)):
        unique = []
        seen = set()
        for sample in samples:
            key = answer_key(sample["answer"])
            if key and key not in seen:
                seen.add(key)
                unique.append(sample)
        choices.append(unique)
        if len(unique) > 1:
            options = "\n".join(f"{index + 1}. {sample['answer']}" for index, sample in enumerate(unique))
            judge_prompts.append([
                {"role": "system", "content": "Select the most likely factually correct answer using only the question and candidates. Return only its option number."},
                {"role": "user", "content": f"Question: {example['prompt'][-1]['content']}\n\nCandidates:\n{options}"},
            ])
            judge_indices.append(group_index)
    decisions = []
    context_fallbacks = {}
    def judge_one(item):
        group_index, messages = item
        decision, fallback = request_bon_judgment(
            judge, args, messages, choices[group_index],
            examples[group_index]["prompt"][-1]["content"], args.aux_max_tokens,
        )
        return group_index, decision, fallback

    with ThreadPoolExecutor(max_workers=getattr(args, "judge_concurrency", 1)) as pool:
        judgments = list(pool.map(judge_one, zip(judge_indices, judge_prompts, strict=True)))
    for group_index, decision, fallback in judgments:
        decisions.append(decision)
        context_fallbacks[group_index] = fallback
    judged = dict(zip(judge_indices, decisions, strict=True))
    results = []
    for index, (samples, unique) in enumerate(zip(groups, choices, strict=True)):
        raw_judgment = judged.get(index, "")
        judgment_history = [raw_judgment] if index in judged else []
        position = parse_bon_choice(raw_judgment, len(unique))
        if position is None and len(unique) > 1:
            options = "\n".join(f"{item + 1}. {sample['answer']}" for item, sample in enumerate(unique))
            raw_judgment, fallback = request_bon_judgment(
                judge, args, [
                    {"role": "system", "content": "Choose exactly one listed answer, even if none is fully correct. Return only its number."},
                    {"role": "user", "content": f"Question: {examples[index]['prompt'][-1]['content']}\nCandidates:\n{options}"},
                ],
                unique, examples[index]["prompt"][-1]["content"], 32,
            )
            context_fallbacks.setdefault(index, []).extend(fallback)
            judgment_history.append(raw_judgment)
            position = parse_bon_choice(raw_judgment, len(unique))
        chosen = unique[position] if position is not None else (unique[0] if unique else samples[0])
        results.append((chosen, {
            "selector": "external_large_llm_judge",
            "judge_model": args.bon_judge_model,
            "judge_response": raw_judgment,
            "judge_valid": position is not None or len(unique) == 1,
            "judge_required": len(unique) > 1,
            "judge_responses": judgment_history,
            "selection_fallback": "no_parsed_answer" if not unique else "invalid_judgment" if len(unique) > 1 and position is None else None,
            "unique_candidates": len(unique),
            "judge_context_fallback": context_fallbacks.get(index, []),
            "sample_responses": [sample["response"] for sample in samples],
        }))
    return results


def request_bon_judgment(judge, args, messages, unique, question, max_tokens):
    """Keep full candidates; only oversized requests use pairwise judging.

    The fallback is recorded because a tournament is not identical to a single
    joint ranking. It preserves every candidate and never truncates an answer.
    """
    def request(payload):
        result = judge.chat.completions.create(
            model=args.bon_judge_model, messages=payload, temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return result.choices[0].message.content or ""

    try:
        return request(messages), []
    except BadRequestError as error:
        if "maximum context length" not in str(error).lower():
            raise
    history = []
    winner = 0
    for challenger in range(1, len(unique)):
        payload = [
            {"role": "system", "content": "Select the most likely factually correct answer using only the question and candidates. Return only its option number."},
            {"role": "user", "content": f"Question: {question}\n\nCandidates:\n1. {unique[winner]['answer']}\n2. {unique[challenger]['answer']}"},
        ]
        response = request(payload)
        position = parse_bon_choice(response, 2)
        history.append({"candidates": [winner + 1, challenger + 1], "response": response})
        if position is None:
            return "", history
        if position == 1:
            winner = challenger
    return str(winner + 1), history


def self_refine(
    llm: LLM, tokenizer: Any, args: argparse.Namespace, examples: list[dict[str, Any]], drafts: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    # One call produces both concrete feedback and an answer for every draft.
    # A satisfactory draft may be retained, but never skips the revision pass.
    revision_messages = [[
        {"role": "system", "content": (
            "Check the factual accuracy and requested answer granularity. "
            "Return exactly two blocks: <feedback>one specific issue, or NO_ISSUE</feedback> "
            "and <answer>the best short final answer</answer>. "
            "If there is NO_ISSUE, repeat the draft answer verbatim in <answer>. "
            "Always output an answer. Do not invent a correction when uncertain."
        )},
        {"role": "user", "content": f"Question: {example['prompt'][-1]['content']}\nDraft answer: {draft['answer'] or draft['response']}"},
    ] for example, draft in zip(examples, drafts, strict=True)]
    revision_texts = auxiliary_generate(llm, tokenizer, args, revision_messages)
    answers = []
    histories = []
    for draft, text in zip(drafts, revision_texts, strict=True):
        revised = _extract_final_answer(text, parse_mode="answer_tag")
        feedback_match = re.search(r"<feedback>(.*?)</feedback>", text, flags=re.IGNORECASE | re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else None
        # Small models occasionally truncate the sentinel to NO_ISSU. It is
        # not a concrete issue and must not authorize an answer change.
        no_issue = bool(feedback and re.fullmatch(r"NO[_\s-]*ISSU(?:E)?[.!]?", feedback, flags=re.IGNORECASE))
        # A NO_ISSUE verdict (or missing feedback) cannot justify changing an
        # already parsed draft. If the draft had no answer, retain a parsed
        # revision so that malformed initial output can still be recovered.
        adopt_revision = revised is not None and (draft["answer"] is None or (feedback is not None and not no_issue))
        answers.append(revised if adopt_revision else draft["answer"])
        histories.append([{
            "answer_before": draft["answer"],
            "feedback": feedback,
            "revision_response": text,
            "revision_parse_valid": revised is not None,
            "revision_adopted": adopt_revision,
        }])

    return refined_confidence(llm, tokenizer, args, examples, drafts, answers, histories)


def refined_confidence(llm, tokenizer, args, examples, drafts, answers, histories):
    """Estimate confidence for already fixed answers without rerunning revisions."""
    confidence_messages = [[
        {"role": "system", "content": (
            "Estimate the probability that the fixed answer is factually correct; do not change it. "
            "Repeat it verbatim inside <answer>...</answer>, then output "
            "<conf> Confidence: p, replacing p with one specific decimal number "
            "between 0 and 1. Never output a range, placeholder, or explanation."
        )},
        {"role": "user", "content": f"Question: {example['prompt'][-1]['content']}\nFixed answer: {answer}"},
    ] for example, answer in zip(examples, answers, strict=True)]
    confidence_texts = auxiliary_generate(llm, tokenizer, args, confidence_messages)
    results = []
    for draft, answer, history, raw_conf in zip(drafts, answers, histories, confidence_texts, strict=True):
        parsed = parse_sample(raw_conf, "answer_tag")
        answer_preserved = answer is not None and answer_key(parsed["answer"]) == answer_key(answer)
        suffix_valid = REFINE_CONFIDENCE_PATTERN.search(raw_conf) is not None
        # The revision decides the answer. A confidence-only call that changes
        # that fixed answer is not a valid confidence observation for it.
        valid = parsed["format_valid"] and suffix_valid and answer_preserved
        selected = {
            "answer": answer,
            "confidence": parsed["confidence"] if valid else None,
            "format_valid": parsed["format_valid"],
            "response": raw_conf,
        }
        results.append((selected, {
            "draft_response": draft["response"],
            "refinement_history": history,
            "confidence_source": "same_model_reestimated_after_refinement",
            "confidence_prompt_version": "unclosed_conf_v1",
            "refinement_protocol": "joint_feedback_revision_no_issue_guard_then_fixed_answer_confidence_filter",
            "revision_answer": answer,
            "confidence_response_format_valid": parsed["format_valid"],
            "confidence_response_suffix_valid": suffix_valid,
            "confidence_answer_preserved": answer_preserved,
            "confidence_selection_valid": valid,
            "revision_answer_changed": answer_key(answer) != answer_key(draft["answer"]),
            "answer_changed": answer_key(answer) != answer_key(draft["answer"]),
        }))
    return results


def make_record(example: dict[str, Any], selected: dict[str, Any], args: argparse.Namespace, extra: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    if args.method == "self_refine":
        # Self-Refine scores the revision answer, not a possibly changed answer
        # echoed by the separate confidence-estimation call.
        answer = selected["answer"]
        correctness = compute_correctness(
            args.correctness_reward,
            example["data_source"],
            answer if answer is not None else selected["response"],
            example["ground_truth"],
            example["extra_info"],
        )
        confidence = selected["confidence"]
        return {
            "record_type": "completion",
            "sample_index": example["sample_index"],
            "data_source": example["data_source"],
            "ground_truth": example["ground_truth"],
            "prompt": example["prompt"],
            "response": selected["response"],
            "parse_mode": example["parse_mode"],
            "final_answer": answer,
            "correctness": correctness,
            "confidence": confidence,
            "confidence_valid": confidence is not None,
            "has_boxed_answer": _last_boxed_expression(selected["response"]) is not None,
            "has_final_answer": answer is not None,
            "format_valid": bool(selected["format_valid"]),
            "brier_score": (confidence - correctness) ** 2 if confidence is not None else None,
            "method": args.method,
            "completion_index": repeat_index,
            **extra,
        }
    output = SimpleNamespace(outputs=[SimpleNamespace(text=selected["response"])])
    record = build_records([output], [example], 1, args.correctness_reward)[0]
    if args.method == "self_consistency":
        # The only method-specific metric input: the winning cluster's mean
        # verbal confidence rather than its representative's confidence.
        confidence = selected["confidence"] if selected["format_valid"] else None
        record.update(
            confidence=confidence,
            confidence_valid=confidence is not None,
            format_valid=bool(selected["format_valid"]),
            brier_score=(confidence - record["correctness"]) ** 2 if confidence is not None else None,
        )
    return {**record, "method": args.method, "completion_index": repeat_index, **extra}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_parquet(args.data_path)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if not dataset:
        raise ValueError("Evaluation dataset is empty")
    confidence_sources = {}
    if args.refine_confidence_results:
        with args.refine_confidence_results.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("record_type") != "completion":
                    continue
                key = (record["sample_index"], record["completion_index"])
                if record.get("method") != "self_refine" or key in confidence_sources:
                    raise ValueError("Invalid or duplicate source Self-Refine completion")
                confidence_sources[key] = record
        expected = {(i, j) for i in range(len(dataset)) for j in range(args.eval_rollouts)}
        if set(confidence_sources) != expected:
            raise ValueError("Source completions must exactly cover this dataset and rollouts")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "model_impl": args.model_impl,
        "tokenizer_mode": args.tokenizer_mode,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = TransformersLLM(args.model_path, tokenizer, args.dtype) if args.backend == "transformers" else LLM(**llm_kwargs)
    overall = MetricAccumulator()
    by_source: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    records_path = output_dir / ".completion_records.jsonl"
    completed_pairs: set[tuple[int, int]] = set()
    if args.resume_completions and records_path.exists():
        with records_path.open(encoding="utf-8") as saved_records:
            for line in saved_records:
                record = json.loads(line)
                key = (record["sample_index"], record["completion_index"])
                if record.get("method") != args.method or key in completed_pairs:
                    raise RuntimeError(f"Invalid or duplicate saved completion: {key}")
                if not 0 <= key[0] < len(dataset) or not 0 <= key[1] < args.eval_rollouts:
                    raise RuntimeError(f"Saved completion is outside this evaluation: {key}")
                completed_pairs.add(key)
                overall.add(record)
                by_source[record["data_source"]].add(record)
    n = (args.num_samples or 32) if args.method == "self_consistency" else (args.num_samples or 8) if args.method == "bon" else 1
    bon_judge = (
        OpenAI(
            base_url=args.bon_judge_base_url,
            api_key=args.bon_judge_api_key,
            http_client=httpx.Client(trust_env=False),
        )
        if args.method == "bon"
        else None
    )
    draft_file = args.draft_results.open(encoding="utf-8") if args.draft_results else None
    if draft_file is not None:
        next(draft_file)  # summary
        next(draft_file)  # config
    with records_path.open("a" if args.resume_completions else "w", encoding="utf-8") as file:
        for start in tqdm(range(0, len(dataset), args.batch_size), desc=args.method):
            batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
            _, examples = render_prompts(batch, tokenizer, args.enable_thinking, start)
            method_messages = [messages_for_method(example["prompt"], args.method) for example in examples]
            prompts = [render_messages(messages, tokenizer, args.enable_thinking) for messages in method_messages]
            batch_examples, batch_prompts, batch_messages = examples, prompts, method_messages
            for repeat_index in range(args.eval_rollouts):
                pending = [i for i, example in enumerate(batch_examples) if (example["sample_index"], repeat_index) not in completed_pairs]
                if not pending:
                    continue
                examples = [batch_examples[i] for i in pending]
                prompts = [batch_prompts[i] for i in pending]
                method_messages = [batch_messages[i] for i in pending]
                groups = []
                if args.refine_confidence_results:
                    saved = [confidence_sources[(example["sample_index"], repeat_index)] for example in examples]
                    for example, record in zip(examples, saved, strict=True):
                        if example["ground_truth"] != record["ground_truth"] or example["prompt"][-1] != record["prompt"][-1]:
                            raise ValueError("Source answer does not match the current evaluation question")
                    selected = refined_confidence(
                        llm, tokenizer, args, examples,
                        [parse_sample(record["draft_response"], record["parse_mode"]) for record in saved],
                        [record["final_answer"] for record in saved],
                        [record["refinement_history"] for record in saved],
                    )
                elif draft_file is not None:
                    for example in examples:
                        saved = json.loads(next(draft_file))
                        if saved["sample_index"] != example["sample_index"] or saved["completion_index"] != repeat_index:
                            raise RuntimeError("Saved draft order does not match the current dataset and eval rollouts")
                        groups.append([parse_sample(saved["draft_response"], example["parse_mode"])])
                else:
                    outputs = llm.generate(prompts, sampling(args, n=n, seed=args.seed + repeat_index), use_tqdm=False)
                    for example, output in zip(examples, outputs, strict=True):
                        if len(output.outputs) != n:
                            raise RuntimeError(f"Expected {n} samples for question {example['sample_index']}")
                        groups.append([parse_sample(completion.text, example["parse_mode"]) for completion in output.outputs])
                if args.refine_confidence_results:
                    pass  # Fixed-answer confidence outputs were built above.
                elif args.method == "self_consistency":
                    selected = [choose_sc(group) for group in groups]
                elif args.method == "bon":
                    selected = choose_bon(bon_judge, args, examples, groups)
                elif args.method == "self_refine":
                    selected = self_refine(llm, tokenizer, args, examples, [group[0] for group in groups])
                else:
                    selected = [(group[0], {}) for group in groups]
                for example, messages, (choice, extra) in zip(examples, method_messages, selected, strict=True):
                    key = (example["sample_index"], repeat_index)
                    if key in completed_pairs:
                        continue
                    example["prompt"] = messages
                    record = make_record(example, choice, args, extra, repeat_index)
                    overall.add(record)
                    by_source[record["data_source"]].add(record)
                    file.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
                    file.flush()
                    completed_pairs.add(key)
    if draft_file is not None:
        if args.limit is None and draft_file.readline():
            raise RuntimeError("Saved draft file contains more records than requested")
        draft_file.close()
    metrics = {
        "overall": overall.summary(args.ece_bins),
        "by_data_source": {source: accumulator.summary(args.ece_bins) for source, accumulator in sorted(by_source.items())},
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "bon_judge_api_key"} | {
        "num_prompts": len(dataset),
        "effective_num_samples": n,
        "inference_backend": args.backend,
        "vllm_version": importlib.metadata.version("vllm") if args.backend == "vllm" else None,
        "bon_selector": "external_large_llm_judge" if args.method == "bon" else None,
        "sc_confidence": "winning_cluster_mean_verbal" if args.method == "self_consistency" else None,
        "answer_grouping": "exact_strip_v2",
        "evaluation_protocol": (
            "self_refine_no_issue_guard_fixed_answer_confidence_filter" if args.method == "self_refine"
            else "evaluate_rlcr.build_records"
        ),
        "invalid_confidence_policy": "excluded_from_brier_ece_pce_auroc",
        "confidence_suffix_policy": "optional_closing_conf_tag" if args.method == "self_refine" else None,
        "confidence_prompt_version": "unclosed_conf_v1" if args.method == "self_refine" else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps({"config": config, "metrics": metrics}, indent=2, ensure_ascii=False, default=json_default) + "\n")
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as result_file:
        result_file.write(json.dumps({"record_type": "summary", **metrics}, ensure_ascii=False, default=json_default) + "\n")
        result_file.write(json.dumps({"record_type": "config", **config}, ensure_ascii=False, default=json_default) + "\n")
        with records_path.open("r", encoding="utf-8") as records_file:
            shutil.copyfileobj(records_file, result_file)
    records_path.unlink()
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
