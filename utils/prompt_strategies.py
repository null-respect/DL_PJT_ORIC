"""Prompt strategies for ORIC contextual-incongruity ablation experiments."""

from __future__ import annotations

from typing import Literal

PromptStrategy = Literal[
    "basic",
    "evidence_focused",
    "context_warning",
    "abstention_allowed",
    "localization_first",
]

PROMPT_STRATEGIES: tuple[PromptStrategy, ...] = (
    "basic",
    "evidence_focused",
    "context_warning",
    "abstention_allowed",
    "localization_first",
)

PROMPT_STRATEGY_DESCRIPTIONS: dict[PromptStrategy, str] = {
    "basic": "Standard yes/no question without extra instructions.",
    "evidence_focused": "Answer only based on visible evidence, not scene expectations.",
    "context_warning": "Warn that scene context may be misleading.",
    "abstention_allowed": "Allow yes/no/uncertain responses.",
    "localization_first": "Localize the object first, then answer yes/no.",
}

_INSTRUCTION_SUFFIXES: dict[PromptStrategy, str] = {
    "basic": "",
    "evidence_focused": " Answer only based on visible evidence, not scene expectations.",
    "context_warning": " The scene context may be misleading. Carefully inspect the image.",
    "abstention_allowed": " Answer yes, no, or uncertain.",
    "localization_first": " First point out where the object is, then answer yes or no.",
}


def build_prompt(base_question: str, strategy: PromptStrategy) -> str:
    """Compose a single prompt from the base ORIC question and a strategy."""
    suffix = _INSTRUCTION_SUFFIXES[strategy]
    if not suffix:
        return base_question
    return f"{base_question.rstrip('?')}?{suffix}"


def get_questions_for_example(ex: dict, strategy: PromptStrategy) -> list[str]:
    """Return prompt(s) for one benchmark item under the given strategy."""
    problems = ex.get("problem") or []
    if not problems:
        target = ex.get("target_object", "object")
        article = "an" if target[:1].lower() in "aeiou" else "a"
        base = f"Is there {article} {target} in the image?"
    else:
        base = problems[0]
    return [build_prompt(base, strategy)]


def default_max_new_tokens(strategy: PromptStrategy, requested: int) -> int:
    """Use longer generations for localization-first unless explicitly overridden."""
    if strategy == "localization_first" and requested <= 32:
        return 128
    if strategy == "abstention_allowed" and requested <= 32:
        return 64
    return requested
