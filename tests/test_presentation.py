"""Presentation content/count validation, independent of model calls and exports."""

from copy import deepcopy

import pytest

from rag_chatbot.tools.presentation import (
    DEFAULT_SLIDE_COUNT, MAX_BULLET_CHARACTERS, MAX_BULLET_TOTAL_CHARACTERS,
    MAX_BULLETS, MAX_NOTES_CHARACTERS, MAX_SLIDES, MAX_TITLE_CHARACTERS,
    validate_presentation, validate_slide_count,
)


def slide_content():
    return {"slides": [{"title": "Measurements", "bullets": ["Measured value: 10 nm."],
                        "notes": "The sample was measured at room temperature.", "source_ids": ["chunk-1"]}]}


def test_presentation_validates_without_mutating_content():
    content = slide_content()
    original = deepcopy(content)
    assert validate_presentation(content, allowed_source_ids={"chunk-1"}) is content
    assert content == original


def test_slide_count_accepts_default_bounds_and_integral_float():
    assert DEFAULT_SLIDE_COUNT == 10 and MAX_SLIDES == 20
    for value in (1, 1.0, DEFAULT_SLIDE_COUNT, float(DEFAULT_SLIDE_COUNT), MAX_SLIDES, float(MAX_SLIDES)):
        result = validate_slide_count(value)
        assert result == value and type(result) is int


@pytest.mark.parametrize("value", [None, True, False, "10", [], {}, 0, -1, MAX_SLIDES + 1,
                                 1.5, float("nan"), float("inf"), float("-inf"), 10 ** 100])
def test_slide_count_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="slide count must be a whole number"):
        validate_slide_count(value)


def test_presentation_accepts_slide_limit_and_rejects_overflow():
    template = slide_content()["slides"][0]
    content = {"slides": [{**deepcopy(template), "title": f"Slide {index}"} for index in range(MAX_SLIDES)]}
    validate_presentation(content)
    content["slides"].append({**deepcopy(template), "title": "One extra slide"})
    with pytest.raises(ValueError, match="must contain"):
        validate_presentation(content)


@pytest.mark.parametrize("content", [None, [], "slides", {}, {"slides": None}, {"slides": []},
                                   {"slides": "not-a-list"}, {"slides": [], "extra": True}])
def test_presentation_rejects_wrong_content_structure(content):
    with pytest.raises(ValueError):
        validate_presentation(content)


@pytest.mark.parametrize("slide", [None, [], "slide", {}, {"title": "Missing fields"}])
def test_presentation_rejects_wrong_slide_structure(slide):
    with pytest.raises(ValueError, match="missing or unexpected"):
        validate_presentation({"slides": [slide]})


def test_presentation_rejects_unknown_slide_fields():
    content = slide_content()
    content["slides"][0]["html"] = "<p>Ignored?</p>"
    with pytest.raises(ValueError, match="missing or unexpected"):
        validate_presentation(content)


@pytest.mark.parametrize("field,maximum", [("title", MAX_TITLE_CHARACTERS), ("notes", MAX_NOTES_CHARACTERS)])
def test_presentation_text_limits(field, maximum):
    content = slide_content()
    content["slides"][0][field] = "x" * maximum
    validate_presentation(content)
    content["slides"][0][field] += "x"
    with pytest.raises(ValueError, match="characters"):
        validate_presentation(content)


@pytest.mark.parametrize("notes", ["", " \n\t"])
def test_speaker_notes_may_be_empty(notes):
    content = slide_content()
    content["slides"][0]["notes"] = notes
    validate_presentation(content)


@pytest.mark.parametrize("field", ["title", "bullets", "notes"])
@pytest.mark.parametrize("invalid", [None, 5, [], "Bad\x00text", "Invalid\x0btext", "Lone\ud800surrogate",
                                   "Invalid\ufffecharacter", "Invalid\uffffcharacter"])
def test_presentation_rejects_invalid_text_and_xml_characters(field, invalid):
    content = slide_content()
    content["slides"][0][field] = [invalid] if field == "bullets" else invalid
    with pytest.raises(ValueError, match="characters"):
        validate_presentation(content)


@pytest.mark.parametrize("field", ["title", "bullets"])
@pytest.mark.parametrize("value", ["", " \t\n"])
def test_titles_and_bullets_must_be_nonempty(field, value):
    content = slide_content()
    content["slides"][0][field] = [value] if field == "bullets" else value
    with pytest.raises(ValueError, match="characters"):
        validate_presentation(content)


@pytest.mark.parametrize("bullets", [None, "One bullet", [], ["Bullet"] * (MAX_BULLETS + 1)])
def test_slide_bullet_count_and_structure(bullets):
    content = slide_content()
    content["slides"][0]["bullets"] = bullets
    with pytest.raises(ValueError, match="bullets"):
        validate_presentation(content)


def test_individual_and_total_bullet_length_limits():
    content = slide_content()
    content["slides"][0]["bullets"] = ["x" * MAX_BULLET_CHARACTERS]
    validate_presentation(content)
    content["slides"][0]["bullets"][0] += "x"
    with pytest.raises(ValueError, match="characters"):
        validate_presentation(content)
    content["slides"][0]["bullets"] = ["x" * 120] * MAX_BULLETS
    assert sum(map(len, content["slides"][0]["bullets"])) == MAX_BULLET_TOTAL_CHARACTERS
    validate_presentation(content)
    content["slides"][0]["bullets"][-1] += "x"
    with pytest.raises(ValueError, match="total at most"):
        validate_presentation(content)


def test_slide_titles_are_unique_ignoring_case_and_whitespace():
    content = slide_content()
    content["slides"][0]["title"] = "First Slide"
    content["slides"].append({**deepcopy(content["slides"][0]), "title": " first   SLIDE "})
    with pytest.raises(ValueError, match="titles must be unique"):
        validate_presentation(content)


@pytest.mark.parametrize("citations", [[], None, "chunk-1", [""], [" "], [None], [[]],
                                     ["chunk-1", "chunk-1"], ["unknown"]])
def test_presentation_rejects_missing_duplicate_or_unknown_citations(citations):
    content = slide_content()
    content["slides"][0]["source_ids"] = citations
    with pytest.raises(ValueError, match="citations"):
        validate_presentation(content, allowed_source_ids={"chunk-1"})


def test_presentation_preserves_unicode_math_and_plain_text_markup():
    content = slide_content()
    content["slides"][0].update(title="Kết quả: α < β", bullets=[r"Result: \(x^2\) & comparison <tag>"],
                                notes="Unicode 😀 with\nnew lines and\ttabs")
    original = deepcopy(content)
    assert validate_presentation(content) == original
