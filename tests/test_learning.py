"""Flashcard and quiz format validation is shared by generation and persistence."""

from copy import deepcopy

import pytest

from rag_chatbot.tools.learning import (
    DEFAULT_LEARNING_COUNT, MAX_ANSWER_CHARACTERS, MAX_FLASHCARDS, MAX_OPTION_CHARACTERS,
    MAX_PROMPT_CHARACTERS, MAX_QUIZ_QUESTIONS, validate_item_count, validate_learning_content,
)


def study_content(tool_id):
    if tool_id == "flashcards":
        return {"cards": [{"front": "What was measured?", "back": "10 nm", "source_ids": ["chunk-1"]}]}
    return {"questions": [{"question": "What was measured?", "options": ["10 nm", "20 nm", "30 nm", "40 nm"],
                           "correct_option": 0, "explanation": "The paper reports 10 nm.", "source_ids": ["chunk-1"]}]}


@pytest.mark.parametrize("tool_id,maximum", [("flashcards", MAX_FLASHCARDS), ("quizzes", MAX_QUIZ_QUESTIONS)])
def test_requested_item_count_accepts_default_boundaries_and_integral_float(tool_id, maximum):
    assert DEFAULT_LEARNING_COUNT == 20
    for value in (1, 1.0, DEFAULT_LEARNING_COUNT, float(DEFAULT_LEARNING_COUNT), maximum, float(maximum)):
        normalized = validate_item_count(tool_id, value)
        assert normalized == value and type(normalized) is int


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
@pytest.mark.parametrize("value", [None, True, False, "20", [], {}, 0, -1, 1.5,
                                 float("nan"), float("inf"), float("-inf"), 10 ** 100])
def test_requested_item_count_rejects_invalid_values(tool_id, value):
    with pytest.raises(ValueError, match="count must be a whole number"):
        validate_item_count(tool_id, value)


@pytest.mark.parametrize("tool_id,maximum", [("flashcards", MAX_FLASHCARDS), ("quizzes", MAX_QUIZ_QUESTIONS)])
def test_requested_item_count_rejects_tool_specific_overflow(tool_id, maximum):
    with pytest.raises(ValueError, match=f"1 to {maximum}"):
        validate_item_count(tool_id, maximum + 1)


def test_requested_item_count_rejects_unrelated_tool():
    with pytest.raises(ValueError, match="Choose Flashcards or Quizzes"):
        validate_item_count("summary", 20)


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
def test_learning_validates_without_mutating_content(tool_id, key):
    content = study_content(tool_id)
    before = deepcopy(content)
    assert validate_learning_content(tool_id, content, allowed_source_ids={"chunk-1"}) is content
    assert content == before
    assert content[key][0]["source_ids"] == ["chunk-1"]


@pytest.mark.parametrize("tool_id,key,maximum", [
    ("flashcards", "cards", MAX_FLASHCARDS), ("quizzes", "questions", MAX_QUIZ_QUESTIONS)])
def test_learning_item_limits(tool_id, key, maximum):
    content = study_content(tool_id)
    field = "front" if tool_id == "flashcards" else "question"
    content[key] = [{**deepcopy(content[key][0]), field: f"Question {index}"} for index in range(maximum)]
    validate_learning_content(tool_id, content)
    content[key].append({**deepcopy(content[key][0]), field: "Too many"})
    with pytest.raises(ValueError, match="must contain"):
        validate_learning_content(tool_id, content)
    with pytest.raises(ValueError, match="must contain"):
        validate_learning_content(tool_id, {key: []})


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
@pytest.mark.parametrize("invalid", [None, [], "content", {}, {"unexpected": []}])
def test_learning_rejects_wrong_top_level(tool_id, key, invalid):
    with pytest.raises(ValueError):
        validate_learning_content(tool_id, invalid)


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
@pytest.mark.parametrize("invalid", [None, [], "item", {}, {"extra": True}])
def test_learning_rejects_wrong_item_structure(tool_id, key, invalid):
    with pytest.raises(ValueError):
        validate_learning_content(tool_id, {key: [invalid]})


@pytest.mark.parametrize("tool_id,key,field,maximum", [
    ("flashcards", "cards", "front", MAX_PROMPT_CHARACTERS),
    ("flashcards", "cards", "back", MAX_ANSWER_CHARACTERS),
    ("quizzes", "questions", "question", MAX_PROMPT_CHARACTERS),
    ("quizzes", "questions", "explanation", MAX_ANSWER_CHARACTERS),
])
def test_learning_text_boundaries(tool_id, key, field, maximum):
    content = study_content(tool_id)
    content[key][0][field] = "x" * maximum
    validate_learning_content(tool_id, content)
    content[key][0][field] += "x"
    with pytest.raises(ValueError, match="characters"):
        validate_learning_content(tool_id, content)


@pytest.mark.parametrize("tool_id,key,field", [
    ("flashcards", "cards", "front"), ("flashcards", "cards", "back"),
    ("quizzes", "questions", "question"), ("quizzes", "questions", "explanation")])
@pytest.mark.parametrize("invalid", ["", " \n\t", None, 4, [], "Bad\x00text", "Lone\ud800surrogate"])
def test_learning_rejects_empty_or_invalid_text(tool_id, key, field, invalid):
    content = study_content(tool_id)
    content[key][0][field] = invalid
    with pytest.raises(ValueError, match="characters"):
        validate_learning_content(tool_id, content)


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
@pytest.mark.parametrize("invalid", [[], None, "chunk-1", [""], [" "], [None], [[]], ["chunk-1", "chunk-1"], ["unknown"]])
def test_learning_rejects_bad_citations(tool_id, key, invalid):
    content = study_content(tool_id)
    content[key][0]["source_ids"] = invalid
    with pytest.raises(ValueError, match="citations"):
        validate_learning_content(tool_id, content, allowed_source_ids={"chunk-1"})


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
def test_learning_rejects_duplicate_prompts_with_case_or_whitespace_variants(tool_id, key):
    content = study_content(tool_id)
    field = "front" if tool_id == "flashcards" else "question"
    content[key].append({**deepcopy(content[key][0]), field: "  WHAT   was\nmeasured? "})
    with pytest.raises(ValueError, match="unique"):
        validate_learning_content(tool_id, content)


@pytest.mark.parametrize("answer", [True, False, -1, 4, 1.0, "1", None, []])
def test_quizzes_reject_invalid_answer_indices(answer):
    content = study_content("quizzes")
    content["questions"][0]["correct_option"] = answer
    with pytest.raises(ValueError, match="integer from 0 to 3"):
        validate_learning_content("quizzes", content)


@pytest.mark.parametrize("answer", [0, 1, 2, 3])
def test_quizzes_accept_each_correct_option(answer):
    content = study_content("quizzes")
    content["questions"][0]["correct_option"] = answer
    validate_learning_content("quizzes", content)


@pytest.mark.parametrize("options", [None, "ABCD", [], ["A", "B", "C"], ["A", "B", "C", "D", "E"],
                                    ["A", "B", "C", ""], ["A", "B", "C", " a "],
                                    ["A", "B", "C", None], ["A", "B", "C", "x" * (MAX_OPTION_CHARACTERS + 1)]])
def test_quizzes_reject_invalid_options(options):
    content = study_content("quizzes")
    content["questions"][0]["options"] = options
    with pytest.raises(ValueError):
        validate_learning_content("quizzes", content)


def test_quizzes_accept_long_option_boundary():
    content = study_content("quizzes")
    content["questions"][0]["options"][0] = "x" * MAX_OPTION_CHARACTERS
    validate_learning_content("quizzes", content)


def test_learning_validator_rejects_unrelated_tool():
    with pytest.raises(ValueError, match="Choose Flashcards or Quizzes"):
        validate_learning_content("summary", {"sections": []})
