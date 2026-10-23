"""Bounded, plain-text slide data shared by generation, storage, and export.

Citation checks establish source membership, not factual support. Rendering and
exporting use trusted templates; model output never supplies executable markup.
"""

import re


DEFAULT_SLIDE_COUNT = 10
MAX_SLIDES = 20
MAX_TITLE_CHARACTERS = 100
MAX_BULLETS = 5
MAX_BULLET_CHARACTERS = 180
MAX_BULLET_TOTAL_CHARACTERS = 600
MAX_NOTES_CHARACTERS = 1500
_INVALID_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_SLIDE_KEYS = {"title", "bullets", "notes", "source_ids"}


def validate_slide_count(value):
    """Accept an integral UI value without rounding or coercing strings/bools."""
    if (type(value) not in (int, float) or not 1 <= value <= MAX_SLIDES
            or value != int(value)):
        raise ValueError(f"Presentation slide count must be a whole number from 1 to {MAX_SLIDES}.")
    return int(value)


def _text(value, label, maximum, *, allow_empty=False):
    if (not isinstance(value, str) or (not allow_empty and not value.strip())
            or len(value) > maximum or _INVALID_TEXT.search(value)):
        minimum = 0 if allow_empty else 1
        raise ValueError(f"{label} must contain {minimum}-{maximum} characters of valid text.")
    return value


def validate_presentation(content, *, allowed_source_ids=None):
    """Validate slide content without rewriting text, order, or source IDs."""
    if not isinstance(content, dict) or set(content) != {"slides"}:
        raise ValueError("Presentation content must contain only a slides list.")
    slides = content["slides"]
    if not isinstance(slides, list) or not 1 <= len(slides) <= MAX_SLIDES:
        raise ValueError(f"Presentation must contain 1-{MAX_SLIDES} slides.")
    known_sources = None if allowed_source_ids is None else set(allowed_source_ids)
    titles = set()
    for slide in slides:
        if not isinstance(slide, dict) or set(slide) != _SLIDE_KEYS:
            raise ValueError("A Presentation slide has missing or unexpected fields.")
        title = _text(slide["title"], "Slide titles", MAX_TITLE_CHARACTERS)
        comparison = " ".join(title.split()).casefold()
        if comparison in titles:
            raise ValueError("Presentation slide titles must be unique.")
        titles.add(comparison)
        bullets = slide["bullets"]
        if not isinstance(bullets, list) or not 1 <= len(bullets) <= MAX_BULLETS:
            raise ValueError(f"Each Presentation slide must contain 1-{MAX_BULLETS} bullets.")
        for bullet in bullets:
            _text(bullet, "Slide bullets", MAX_BULLET_CHARACTERS)
        if sum(len(bullet) for bullet in bullets) > MAX_BULLET_TOTAL_CHARACTERS:
            raise ValueError(f"Slide bullets must total at most {MAX_BULLET_TOTAL_CHARACTERS} characters.")
        _text(slide["notes"], "Slide notes", MAX_NOTES_CHARACTERS, allow_empty=True)
        citations = slide["source_ids"]
        if (not isinstance(citations, list) or not citations
                or any(not isinstance(value, str) or not value.strip() for value in citations)
                or len(citations) != len(set(citations))):
            raise ValueError("Each Presentation slide needs unique, nonempty source citations.")
        if known_sources is not None and not set(citations).issubset(known_sources):
            raise ValueError("Presentation contains unknown source citations.")
    return content
