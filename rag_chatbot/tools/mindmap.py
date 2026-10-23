"""Validated Mind Map data and a deterministic, dependency-free SVG renderer."""

from html import escape
import math
import re
import unicodedata


MAX_NODES = 60
MAX_LEVELS = 5
_NODE_KEYS = {"id", "parent_id", "label", "description", "source_ids"}
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,39}\Z")
_INVALID_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_COLORS = ("#2563eb", "#0f766e", "#9333ea", "#c2410c", "#be185d", "#0369a1", "#4d7c0f")


def _safe_text(value, label, minimum, maximum):
    if (not isinstance(value, str) or not minimum <= len(value) <= maximum
            or minimum and not value.strip() or isinstance(value, str) and _INVALID_XML.search(value)):
        raise ValueError(f"Mind Map {label} must contain {minimum}–{maximum} characters of valid text.")
    return value


def normalize_mind_map_ids(content):
    """Replace opaque model IDs with safe app-owned IDs, preserving exact edges.

    Model identifiers are lookup keys, never SVG markup. Renumber every node
    together so punctuation, Unicode, long IDs, or IDs resembling our generated
    names cannot collide. Do not trim or otherwise guess ambiguous references.
    Saved maps and the renderer still use the strict validator below.
    """
    if not isinstance(content, dict) or set(content) != {"nodes"}:
        raise ValueError("Mind Map content must contain only a nodes list.")
    nodes = content["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError(f"Mind Maps must contain 1–{MAX_NODES} nodes.")
    identifiers = {}
    for index, node in enumerate(nodes, 1):
        if not isinstance(node, dict) or set(node) != _NODE_KEYS:
            raise ValueError("Each Mind Map node must contain id, parent_id, label, description, and source_ids.")
        identifier = node["id"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("The model returned a missing or invalid Mind Map node identifier.")
        if identifier in identifiers:
            raise ValueError("Mind Map node identifiers must be unique.")
        identifiers[identifier] = f"n{index}"
    normalized = []
    for node in nodes:
        parent = node["parent_id"]
        if not isinstance(parent, str) or (parent and parent not in identifiers):
            raise ValueError("Every Mind Map parent must exist in the same map.")
        normalized.append({**node, "id": identifiers[node["id"]],
                           "parent_id": identifiers[parent] if parent else ""})
    return validate_mind_map({"nodes": normalized})


def validate_mind_map(content, allowed_source_ids=None):
    """Require one cited, connected tree; model-provided markup remains plain text."""
    if not isinstance(content, dict) or set(content) != {"nodes"}:
        raise ValueError("Mind Map content must contain only a nodes list.")
    nodes = content["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError(f"Mind Maps must contain 1–{MAX_NODES} nodes.")
    known_sources = None if allowed_source_ids is None else set(allowed_source_ids)
    indexed = {}
    roots = []
    for node in nodes:
        if not isinstance(node, dict) or set(node) != _NODE_KEYS:
            raise ValueError("Each Mind Map node must contain id, parent_id, label, description, and source_ids.")
        identifier, parent = node["id"], node["parent_id"]
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
            raise ValueError("Mind Map node identifiers must start with a letter and use at most 40 letters, digits, underscores, or hyphens.")
        if identifier in indexed:
            raise ValueError("Mind Map node identifiers must be unique.")
        if not isinstance(parent, str) or parent and not _IDENTIFIER.fullmatch(parent):
            raise ValueError("Invalid Mind Map parent identifier.")
        _safe_text(node["label"], "labels", 1, 80)
        _safe_text(node["description"], "descriptions", 0, 500)
        citations = node["source_ids"]
        if (not isinstance(citations, list) or any(not isinstance(value, str) or not value for value in citations)
                or len(citations) != len(set(citations))):
            raise ValueError("Mind Map citations must be a list of unique source identifiers.")
        if parent and not citations:
            raise ValueError("Every non-root Mind Map node needs at least one source citation.")
        if known_sources is not None and not set(citations).issubset(known_sources):
            raise ValueError("The Mind Map cites an unknown source.")
        indexed[identifier] = node
        if not parent:
            roots.append(identifier)
    if len(roots) != 1:
        raise ValueError("A Mind Map must have exactly one root.")
    for node in nodes:
        seen = set()
        current = node
        while True:
            identifier = current["id"]
            if identifier in seen:
                raise ValueError("A Mind Map cannot contain cycles.")
            seen.add(identifier)
            if len(seen) > MAX_LEVELS:
                raise ValueError(f"Mind Maps can have at most {MAX_LEVELS} levels including the root.")
            parent = current["parent_id"]
            if not parent:
                break
            if parent not in indexed:
                raise ValueError("Every Mind Map parent must exist in the same map.")
            current = indexed[parent]
    return content


def _tree(payload):
    if not isinstance(payload, dict):
        raise ValueError("A Mind Map output must be an object.")
    _safe_text(payload.get("title"), "titles", 1, 512)
    content = validate_mind_map(payload.get("content"))
    nodes = {node["id"]: node for node in content["nodes"]}
    children = {identifier: [] for identifier in nodes}
    root = None
    for node in content["nodes"]:
        if node["parent_id"]:
            children[node["parent_id"]].append(node["id"])
        else:
            root = node["id"]
    return nodes, children, root


def _clusters(text):
    """Keep combining marks and common joined emoji together without extra packages."""
    result = []
    for character in text:
        modifier = (unicodedata.category(character).startswith("M") or character == "\u200d"
                    or "\U0001f3fb" <= character <= "\U0001f3ff")
        regional_pair = (result and len(result[-1]) == 1
                         and "\U0001f1e6" <= result[-1] <= "\U0001f1ff"
                         and "\U0001f1e6" <= character <= "\U0001f1ff")
        if result and (modifier or result[-1].endswith("\u200d") or regional_pair):
            result[-1] += character
        else:
            result.append(character)
    return result


def _glyph_units(cluster):
    # Reserve extra width for capitals, wide Latin glyphs, and non-Latin text.
    # Combining marks do not consume a separate slot. This deliberately leaves
    # breathing room with the proportional UI font rather than clipping labels.
    first = cluster[0]
    return 2 if first.isupper() or first in "mw" or ord(first) > 127 else 1


def _wrap(text, width):
    """Prefer word boundaries and retain every character, including long words."""
    lines, current, units = [], [], 0
    for cluster in _clusters(text):
        if cluster == "\n":
            lines.append("".join(current))
            current, units = [], 0
            continue
        size = _glyph_units(cluster)
        while units + size > width and current:
            boundary = next((index + 1 for index in range(len(current) - 1, -1, -1)
                             if current[index].isspace()), len(current))
            lines.append("".join(current[:boundary]))
            current = current[boundary:]
            units = sum(_glyph_units(value) for value in current)
        current.append(cluster)
        units += size
    if current or not lines:
        lines.append("".join(current))
    return lines


def _render_svg(payload, zoom=1.0):
    nodes, children, root = _tree(payload)
    card_width, column_gap, sibling_gap = 250, 86, 26
    layouts = {}

    def measure(identifier, depth, color):
        node = nodes[identifier]
        label = _wrap(node["label"], 26)
        description = _wrap(node["description"], 32) if node["description"] else []
        height = 30 + len(label) * 19 + (10 + len(description) * 15 if description else 0)
        layout = {"depth": depth, "height": height, "label": label, "description": description, "color": color}
        layouts[identifier] = layout
        child_heights = []
        for index, child in enumerate(children[identifier]):
            child_color = _COLORS[index % len(_COLORS)] if identifier == root else color
            child_heights.append(measure(child, depth + 1, child_color))
        layout["subtree_height"] = max(height, sum(child_heights) + max(0, len(child_heights) - 1) * sibling_gap)
        return layout["subtree_height"]

    total_height = measure(root, 0, "#172554")
    width = max(640, 48 + (max(item["depth"] for item in layouts.values()) + 1) * card_width
                + max(item["depth"] for item in layouts.values()) * column_gap)
    title_lines = _wrap(payload["title"], max(30, int((width - 48) / 12)))
    header_height = 46 + 25 * len(title_lines)
    height = header_height + total_height + 28

    def position(identifier, top):
        layout = layouts[identifier]
        # Top-align each parent with its first branch so the root is visible
        # immediately even when a large map needs vertical scrolling.
        layout.update(x=24 + layout["depth"] * (card_width + column_gap), y=top)
        next_top = top
        for child in children[identifier]:
            position(child, next_top)
            next_top += layouts[child]["subtree_height"] + sibling_gap

    position(root, header_height)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * zoom:g}" height="{height * zoom:g}" '
             f'viewBox="0 0 {width} {height}" style="max-width:none;max-height:none;display:block" '
             'role="img" aria-labelledby="mind-map-title">',
             f'<title id="mind-map-title">{escape(payload["title"])}</title>',
             '<rect width="100%" height="100%" fill="#f5f7fb"/>',
             '<g font-family="Segoe UI, Noto Sans, Arial, sans-serif" xml:space="preserve">']
    for index, line in enumerate(title_lines):
        parts.append(f'<text x="24" y="{31 + index * 25}" font-size="20" font-weight="700" fill="#0f172a">{escape(line)}</text>')
    parts.append(f'<text x="24" y="{header_height - 17}" font-size="11" fill="#64748b">{len(nodes)} concepts · Source-grounded Mind Map</text>')
    for node in nodes.values():
        if not node["parent_id"]:
            continue
        parent, child = layouts[node["parent_id"]], layouts[node["id"]]
        x1, y1 = parent["x"] + card_width, parent["y"] + parent["height"] / 2
        x2, y2 = child["x"], child["y"] + child["height"] / 2
        middle = (x1 + x2) / 2
        parts.append(f'<path d="M {x1:g},{y1:g} C {middle:g},{y1:g} {middle:g},{y2:g} {x2:g},{y2:g}" '
                     f'fill="none" stroke="{child["color"]}" stroke-width="2" opacity="0.7"/>')
    for identifier, node in nodes.items():
        layout = layouts[identifier]
        x, y, h, color = layout["x"], layout["y"], layout["height"], layout["color"]
        is_root = identifier == root
        background, text_color = ("#172554", "#ffffff") if is_root else ("#ffffff", "#0f172a")
        description_color = "#dbeafe" if is_root else "#475569"
        parts.extend([f'<g id="node-{identifier}">',
                      f'<title>{escape(node["label"])}\n{escape(node["description"])}</title>',
                      f'<rect x="{x}" y="{y + 3}" width="{card_width}" height="{h}" rx="12" fill="#0f172a" opacity="0.06"/>',
                      f'<rect x="{x}" y="{y}" width="{card_width}" height="{h}" rx="12" fill="{background}" stroke="{color}" stroke-width="1.2"/>'])
        if not is_root:
            parts.append(f'<rect x="{x + 1}" y="{y + 13}" width="4" height="{h - 26}" rx="2" fill="{color}"/>')
        text_y = y + 26
        for line in layout["label"]:
            parts.append(f'<text x="{x + 16}" y="{text_y}" font-size="14" font-weight="600" fill="{text_color}">{escape(line)}</text>')
            text_y += 19
        if layout["description"]:
            text_y += 6
            for line in layout["description"]:
                parts.append(f'<text x="{x + 16}" y="{text_y}" font-size="11.5" fill="{description_color}">{escape(line)}</text>')
                text_y += 15
        parts.append('</g>')
    parts.append('</g></svg>')
    return "".join(parts)


def render_mind_map_svg(payload):
    """Return a self-contained SVG made only from trusted shapes and escaped text."""
    return _render_svg(payload)


def render_mind_map_html(payload, zoom=1.0):
    try:
        zoom = float(zoom)
    except (ValueError, TypeError) as exc:
        raise ValueError("Mind Map zoom must be a finite number.") from exc
    if not math.isfinite(zoom):
        raise ValueError("Mind Map zoom must be a finite number.")
    zoom = max(0.5, min(2.0, zoom))
    return ('<div class="studio-mind-map" style="height:480px;max-height:480px;overflow:auto;'
            'border:1px solid #dbe3ef;border-radius:12px;background:#f5f7fb;" tabindex="0" '
            'aria-label="Mind Map. Scroll to explore all branches.">'
            + _render_svg(payload, zoom) + '</div>')


def mind_map_outline(payload):
    """Yield nodes with their depth in stable tree order for text exports."""
    nodes, children, root = _tree(payload)

    def visit(identifier, depth):
        yield depth, nodes[identifier]
        for child in children[identifier]:
            yield from visit(child, depth + 1)

    yield from visit(root, 0)
