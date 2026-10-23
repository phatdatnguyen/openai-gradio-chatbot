"""Shared validation for source-grounded Flashcards and Quizzes data.

These validators perform no generation, rendering, or persistence. Citation
validation establishes source membership, not factual support for an answer.
"""

import re


MAX_FLASHCARDS = 60
MAX_QUIZ_QUESTIONS = 40
DEFAULT_LEARNING_COUNT = 20
MAX_PROMPT_CHARACTERS = 1200
MAX_ANSWER_CHARACTERS = 3000
MAX_OPTION_CHARACTERS = 1000
QUIZ_OPTION_COUNT = 4
_INVALID_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_CARD_KEYS = {"front", "back", "source_ids"}
_QUESTION_KEYS = {"question", "options", "correct_option", "explanation", "source_ids"}


def validate_item_count(tool_id, value):
    """Normalize a requested count without changing validation of saved outputs."""
    if tool_id == "flashcards":
        name, maximum = "Flashcards", MAX_FLASHCARDS
    elif tool_id == "quizzes":
        name, maximum = "Quizzes", MAX_QUIZ_QUESTIONS
    else:
        raise ValueError("Choose Flashcards or Quizzes to set an item count.")
    # Gradio Number may pass integral floats. Reject booleans, text, NaN,
    # infinities, and fractions before conversion; never round a user's count.
    if (type(value) not in (int, float) or not 1 <= value <= maximum
            or value != int(value)):
        raise ValueError(f"{name} count must be a whole number from 1 to {maximum}.")
    return int(value)


def _text(value, label, maximum):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or _INVALID_TEXT.search(value)):
        raise ValueError(f"{label} must contain 1–{maximum} characters of valid text.")
    return value


def _comparison_key(value):
    return " ".join(value.split()).casefold()


def validate_learning_content(tool_id, content, *, allowed_source_ids=None):
    """Validate bounded study content without changing its text or citations."""
    if tool_id == "flashcards":
        name, key, maximum, item_keys, prompt_key = (
            "Flashcards", "cards", MAX_FLASHCARDS, _CARD_KEYS, "front")
    elif tool_id == "quizzes":
        name, key, maximum, item_keys, prompt_key = (
            "Quizzes", "questions", MAX_QUIZ_QUESTIONS, _QUESTION_KEYS, "question")
    else:
        raise ValueError("Choose Flashcards or Quizzes to validate study content.")
    if not isinstance(content, dict) or set(content) != {key}:
        raise ValueError(f"{name} content must contain only a {key} list.")
    items = content[key]
    if not isinstance(items, list) or not 1 <= len(items) <= maximum:
        raise ValueError(f"{name} must contain 1–{maximum} {key}.")
    known_sources = None if allowed_source_ids is None else set(allowed_source_ids)
    prompts = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != item_keys:
            raise ValueError(f"A {name} item has missing or unexpected fields.")
        prompt = _text(item[prompt_key], f"{name} prompts", MAX_PROMPT_CHARACTERS)
        comparison = _comparison_key(prompt)
        if comparison in prompts:
            raise ValueError(f"{name} prompts must be unique.")
        prompts.add(comparison)
        if tool_id == "flashcards":
            _text(item["back"], "Flashcard answers", MAX_ANSWER_CHARACTERS)
        else:
            _text(item["explanation"], "Quiz explanations", MAX_ANSWER_CHARACTERS)
            options = item["options"]
            if not isinstance(options, list) or len(options) != QUIZ_OPTION_COUNT:
                raise ValueError(f"Every quiz question must have exactly {QUIZ_OPTION_COUNT} options.")
            for option in options:
                _text(option, "Quiz options", MAX_OPTION_CHARACTERS)
            if len({_comparison_key(option) for option in options}) != QUIZ_OPTION_COUNT:
                raise ValueError("Quiz options must be distinct within each question.")
            answer = item["correct_option"]
            if type(answer) is not int or not 0 <= answer < QUIZ_OPTION_COUNT:
                raise ValueError("Each quiz correct_option must be an integer from 0 to 3.")
        citations = item["source_ids"]
        if (not isinstance(citations, list) or not citations
                or any(not isinstance(value, str) or not value.strip() for value in citations)
                or len(citations) != len(set(citations))):
            raise ValueError(f"Each {name} item needs unique, nonempty source citations.")
        if known_sources is not None and not set(citations).issubset(known_sources):
            raise ValueError(f"{name} contains unknown source citations.")
    return content
