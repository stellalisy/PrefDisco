"""Helpers for normalizing PrefDisco benchmark records across source datasets."""

from __future__ import annotations

from typing import Any, Dict


def _as_options(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        labels = value.get("label")
        texts = value.get("text")
        if isinstance(labels, list) and isinstance(texts, list):
            return dict(zip(labels, texts))
        return value
    if isinstance(value, list):
        return {chr(65 + index): option for index, option in enumerate(value)}
    return {}


def normalize_problem(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the evaluator's common problem representation.

    Released records contain the untouched source row in ``original_problem``.
    Older internal evaluation files also contain ``parsed_problem``. This
    function supports both formats so downloaded Hub data can be evaluated
    directly.
    """

    parsed = record.get("parsed_problem")
    if isinstance(parsed, dict) and parsed.get("question"):
        return parsed

    source = record.get("original_problem", record)
    if not isinstance(source, dict):
        return {
            "question": str(source),
            "options": {},
            "answer": "",
            "solution": None,
            "image": None,
        }

    context = source.get("context")
    question = (
        source.get("query")
        or source.get("question")
        or source.get("problem")
        or source.get("Question")
        or source.get("prompt")
        or ""
    )
    if context:
        question = f"{context}\n\n{question}" if question else str(context)

    options = _as_options(source.get("choices") or source.get("options"))
    if not options:
        social_options = {
            key[-1]: source[key]
            for key in ("answerA", "answerB", "answerC", "answerD")
            if source.get(key) is not None
        }
        options = social_options

    answer_field = next(
        (
            key
            for key in (
                "correct_answer",
                "answerKey",
                "correct_option",
                "answer_idx",
                "Answer",
                "answer",
                "label",
            )
            if source.get(key) is not None
        ),
        None,
    )
    answer = source.get(answer_field, "") if answer_field else ""
    if isinstance(answer, int) and options:
        answer = chr(65 + answer)
    elif answer_field == "label" and isinstance(answer, str) and answer.isdigit():
        # SocialIQA labels are one-indexed.
        answer = chr(64 + int(answer))

    solution_parts = [
        str(source[key])
        for key in ("lecture", "solution")
        if source.get(key) not in (None, "")
    ]
    return {
        "question": str(question),
        "options": options,
        "answer": answer,
        "solution": "\n\n".join(solution_parts) if solution_parts else None,
        "image": source.get("image"),
    }
