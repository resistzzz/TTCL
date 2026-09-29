"""Project-local Lighteval tasks.

``mmlu_pro_fixed`` corrects a bug in Lighteval 0.13's built-in ``mmlu_pro``
task: that task puts the complete prompt in both ``query`` and ``instruction``.
When chat templating is enabled, PromptManager strips every few-shot query and
leaves only its answer. This task intentionally leaves ``instruction`` unset,
so each demonstration remains a normal user/assistant turn.
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


MMLU_PRO_TEMPLATE = """
Answer the following multiple choice question. Think step by step, but keep the
reasoning concise. The last line of your response must be exactly `Answer: X`,
where X is one of {letters}.

{question}

{choices}
""".strip()


def mmlu_pro_fixed_prompt(line, task_name: str | None = None) -> Doc:
    """Create a chat-template-safe MMLU-Pro document with its real A--J set."""
    options = line["options"]
    letters = ascii_uppercase[: len(options)]
    choices = "\n".join(f"{letter}: {option}" for letter, option in zip(letters, options))
    query = MMLU_PRO_TEMPLATE.format(
        letters=letters,
        question=line["question"],
        choices=choices,
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=list(letters),
        gold_index=line["answer_index"],
        instruction=None,
    )


def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=ascii_uppercase[record["answer_index"]],
        choices=record["options"],
    )


mmlu_pro_fixed = LightevalTaskConfig(
    name="mmlu_pro_fixed",
    prompt_function=mmlu_pro_fixed_prompt,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="TIGER-Lab/MMLU-Pro",
    hf_subset="default",
    hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
    evaluation_splits=("test",),
    few_shots_split="validation",
    generation_size=1024,
    metrics=[Metrics.gpqa_instruct_metric],
    version=1,
)


TASKS_TABLE = [mmlu_pro_fixed]
