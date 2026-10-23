"""Bounded, source-checked numeric chart data; no generated plotting code."""

from decimal import Decimal, InvalidOperation
import math
import re
import unicodedata

from .evidence import require_object, validate_source_ids


CHART_TYPES = ("bar", "line", "scatter")
DEFAULT_CHART_TYPE = "auto"
MAX_SERIES = 6
MAX_POINTS_PER_SERIES = 10_000
MAX_POINTS = 10_000
MAX_LIMITATIONS = 40
MAX_EVIDENCE_CHARACTERS = 600
MAX_ABSOLUTE_VALUE = 1e100
_BAD_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_NUMBER = re.compile(
    r"(?<![\w.,])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?!\w|[.,]\d)"
)


def validate_chart_type(value):
    if not isinstance(value, str) or value not in (DEFAULT_CHART_TYPE, *CHART_TYPES):
        raise ValueError("Choose Auto, Bar, Line, or Scatter for the Plot chart type.")
    return value


def _text(value, label, limit, *, empty=False):
    if (not isinstance(value, str) or len(value) > limit or _BAD_TEXT.search(value)
            or (not empty and not value.strip())):
        raise ValueError(f"Plot {label} must be plain text of at most {limit} characters.")
    return value


def _number(value):
    # bool is deliberately excluded despite being an int subclass.
    if (type(value) not in (int, float) or abs(value) > MAX_ABSOLUTE_VALUE
            or not math.isfinite(value)):
        raise ValueError("Plot coordinates must be finite numbers with magnitude at most 1e100.")
    return value


def _excerpt_text(value):
    # Compatibility normalization would change 2³ into 23. Keep numeric and
    # punctuation semantics intact while accepting whitespace extraction noise.
    return " ".join(unicodedata.normalize("NFC", value).split())


def explicit_numbers(text):
    """Parse explicit numbers only, with no evaluation or inferred conversion.

    Accepted scientific forms include 1.2e-3 and 1.2 × 10⁻³. Commas must be
    thousands separators. Ambiguous decimal commas and number-word compounds
    are deliberately not interpreted as unrelated smaller numbers.
    """
    text = re.sub(r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
                  lambda match: "10^" + match[1].translate(_SUPERSCRIPTS), text)
    text = unicodedata.normalize("NFC", text).replace("−", "-")
    text = re.sub(r"\s*[×·]\s*10\s*\^\s*([-+]?\d+)", r"e\1", text)
    text = re.sub(r"(?<![\w.])10\s*\^\s*([-+]?\d+)", r"1e\1", text)
    # An unsupported mathematical expression is not a reported measurement.
    # In particular, ×100 must never be mistaken for ×10^0, or 2³ for 23.
    text = re.sub(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[×·*^]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+))+", " ", text)
    numbers = set()
    for match in _NUMBER.finditer(text):
        try:
            value = Decimal(match[0].replace(",", ""))
        except InvalidOperation:
            continue
        if value.is_finite():
            numbers.add(value)
    return numbers


def validate_plot(content, *, allowed_source_ids=None, sources=None):
    """Check chart data and optionally verify quoted numeric source occurrences.

    Occurrence checks cannot establish that a quantity has the right meaning,
    label, or unit. Source references and the data table remain reviewable.
    The input is never mutated and no plotting dependency is imported here.
    """
    require_object(content, {"chart_type", "x_label", "y_label", "x_unit", "y_unit", "series", "limitations"})
    kind = content["chart_type"]
    if not isinstance(kind, str) or kind not in CHART_TYPES:
        raise ValueError("Plot results must use Bar, Line, or Scatter.")
    for key in ("x_label", "y_label"):
        _text(content[key], key.replace("_", " "), 160)
    for key in ("x_unit", "y_unit"):
        _text(content[key], key.replace("_", " "), 80, empty=True)
    series = content["series"]
    if not isinstance(series, list) or not 1 <= len(series) <= MAX_SERIES:
        raise ValueError(f"Plot must contain 1–{MAX_SERIES} data series and at least one reported numeric point.")
    original = None
    if sources is not None:
        if (not isinstance(sources, list) or any(not isinstance(source, dict)
                or not isinstance(source.get("id"), str) or not isinstance(source.get("text"), str)
                for source in sources)):
            raise ValueError("Plot source excerpts require a valid source snapshot.")
        original = {source["id"]: _excerpt_text(source["text"]) for source in sources}
        if len(original) != len(sources):
            raise ValueError("Plot source identifiers must be unique.")
    if allowed_source_ids is None and original is not None:
        allowed_source_ids = set(original)

    def citations(value):
        if allowed_source_ids is not None:
            return validate_source_ids(value, allowed_source_ids)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("Plot points and limitations need source citations.")
        return validate_source_ids(value, set(value))

    total, names = 0, set()
    for item in series:
        require_object(item, {"name", "points"})
        name = _text(item["name"], "series name", 120).strip().casefold()
        if name in names:
            raise ValueError("Plot series names must be unique.")
        names.add(name)
        points = item["points"]
        if not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS_PER_SERIES:
            raise ValueError(f"Each Plot series must contain 1–{MAX_POINTS_PER_SERIES} reported numeric points.")
        total += len(points)
        if total > MAX_POINTS:
            raise ValueError(f"Plot supports at most {MAX_POINTS} points in total.")
        categories, coordinates = set(), set()
        for point in points:
            require_object(point, {"label", "x", "y", "source_ids", "evidence"})
            label = _text(point["label"], "point label", 160, empty=kind != "bar")
            y = _number(point["y"])
            if kind == "bar":
                if point["x"] is not None:
                    raise ValueError("Bar Plot points use category labels and a null x coordinate.")
                category = label.strip().casefold()
                if category in categories:
                    raise ValueError("Bar Plot categories must be unique within each series.")
                categories.add(category)
            else:
                _number(point["x"])
                if kind == "line" and point["x"] in coordinates:
                    raise ValueError("Line Plot x coordinates must be unique within each series.")
                coordinates.add(point["x"])
            ids = citations(point["source_ids"])
            evidence = _text(point["evidence"], "point evidence", MAX_EVIDENCE_CHARACTERS)
            normalized_evidence = _excerpt_text(evidence)
            if original is not None and not any(
                    identifier in original and normalized_evidence in original[identifier] for identifier in ids):
                raise ValueError("Plot point evidence must quote an original cited source. Try narrower instructions or sources.")
            values = explicit_numbers(evidence)
            required = [y] if kind == "bar" else [point["x"], y]
            if any(Decimal(str(value)) not in values for value in required):
                raise ValueError("Plot coordinates must appear explicitly in their quoted evidence; no inferred or converted values.")
    limitations = content["limitations"]
    if not isinstance(limitations, list) or len(limitations) > MAX_LIMITATIONS:
        raise ValueError(f"Plot supports at most {MAX_LIMITATIONS} cited limitations.")
    for item in limitations:
        require_object(item, {"text", "source_ids"})
        _text(item["text"], "limitation", 1000)
        citations(item["source_ids"])
    return content
