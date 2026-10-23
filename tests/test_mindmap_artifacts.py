"""Mind Map validation, trusted rendering, and one-current-map persistence."""

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from rag_chatbot.tools import artifacts, mindmap
from rag_chatbot.workspaces import create_workspace, workspace_path


def node(identifier, parent="", label="Concept", description="", citations=None):
    return {"id": identifier, "parent_id": parent, "label": label, "description": description,
            "source_ids": citations if citations is not None else ["chunk-one"] if parent else []}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    return "Research"


@pytest.fixture
def payload(workspace):
    return {"tool_id": "mind_maps", "title": "Research map", "model": "gpt-6-luna",
            "content": {"nodes": [node("root", label="Research"),
                                  node("method", "root", "Method", "Measure the sample."),
                                  node("finding", "root", "Finding", "A peak was measured.")]},
            "sources": [{"id": "chunk-one", "document_id": "doc-one", "filename": "paper.pdf",
                         "locator": "Page 2", "text": "A peak was measured.",
                         "source_path": str(workspace_path(workspace) / "sources" / "paper.pdf")}],
            "documents": [{"id": "doc-one", "name": "paper.pdf", "fingerprint": "v1", "generation": "g1"}],
            "coverage": {"documents": 1, "chunks": 1, "processed_chunks": 1}, "settings": {}}


def test_validator_accepts_tree_and_preserves_order(payload):
    content = payload["content"]
    assert mindmap.validate_mind_map(content, {"chunk-one"}) is content
    assert [(depth, item["id"]) for depth, item in mindmap.mind_map_outline(payload)] == [
        (0, "root"), (1, "method"), (1, "finding")]


def test_generated_ids_remap_forward_references_without_merging_or_changing_sources():
    content = {"nodes": [node("n2", " root "), node(" root "), node("root", "n2")]}
    original = deepcopy(content)
    normalized = mindmap.normalize_mind_map_ids(content)
    assert [item["id"] for item in normalized["nodes"]] == ["n1", "n2", "n3"]
    assert [item["parent_id"] for item in normalized["nodes"]] == ["n2", "", "n1"]
    assert [item["source_ids"] for item in normalized["nodes"]] == [
        item["source_ids"] for item in content["nodes"]]
    assert content == original
    assert mindmap.validate_mind_map(normalized, {"chunk-one"}) is normalized


@pytest.mark.parametrize("nodes", [
    [node("")], [node("  ")], [node(None)], [node(1)], [node([])],
    [node("root"), node("root", "root")],
    [node("1"), node("child", "n1")],  # New IDs cannot turn a missing parent into a valid link.
    [node(" root "), node("child", "root")],
    [node("root"), node("child", None)], [node("root"), node("child", 1)],
    [node("1"), node("2")],
    [node("1"), node("2", "3"), node("3", "2")],
    [node("1"), node("2", "2")],
])
def test_id_normalization_still_rejects_ambiguous_or_invalid_trees(nodes):
    with pytest.raises(ValueError):
        mindmap.normalize_mind_map_ids({"nodes": nodes})


@pytest.mark.parametrize("nodes", [[], [node("root"), node("root")], [node("one"), node("two")],
                                  [node("root"), node("orphan", "missing")],
                                  [node("root"), node("a", "b"), node("b", "a")],
                                  [node("root"), node("a", "a")],
                                  [node("bad id")], [node("root"), node("child", "root", citations=[])],
                                  [node("root", label=" ")], [node("root", label="x" * 81)],
                                  [node("root", description="x" * 501)], [node("root", label="bad\x00text")],
                                  [node("root"), node("child", "root", citations=[["chunk-one"]])]])
def test_validator_rejects_malformed_trees(nodes):
    with pytest.raises(ValueError):
        mindmap.validate_mind_map({"nodes": nodes}, {"chunk-one"})


def test_validator_rejects_unknown_keys_and_unknown_citations():
    with pytest.raises(ValueError, match="only"):
        mindmap.validate_mind_map({"nodes": [node("root")], "html": "<script>"})
    with pytest.raises(ValueError, match="Each"):
        mindmap.validate_mind_map({"nodes": [{**node("root"), "onclick": "alert(1)"}]})
    with pytest.raises(ValueError, match="unknown source"):
        mindmap.validate_mind_map({"nodes": [node("root"), node("child", "root")]}, {"different"})


def test_validator_enforces_node_count_and_depth_boundaries():
    nodes = [node("root"), *[node(f"child{number}", "root") for number in range(59)]]
    assert mindmap.validate_mind_map({"nodes": nodes})
    with pytest.raises(ValueError, match="60"):
        mindmap.validate_mind_map({"nodes": [*nodes, node("too_many", "root")]})
    chain = [node("root")]
    for number in range(1, 5):
        chain.append(node(f"level{number}", chain[-1]["id"]))
    assert mindmap.validate_mind_map({"nodes": chain})
    with pytest.raises(ValueError, match="5 levels"):
        mindmap.validate_mind_map({"nodes": [*chain, node("too_deep", chain[-1]["id"])]})


def test_renderer_escapes_markup_and_never_emits_model_code(payload):
    attack = '<script>alert("x")</script>'
    payload["title"] = attack
    payload["content"]["nodes"][1].update(label=attack, description='<img src=x onerror="alert(1)"> & text')
    svg = mindmap.render_mind_map_svg(payload)
    root = ET.fromstring(svg)
    allowed = {"svg", "title", "rect", "g", "text", "path"}
    assert {element.tag.rsplit("}", 1)[-1] for element in root.iter()} <= allowed
    assert not any(key.startswith("on") for element in root.iter() for key in element.attrib)
    assert attack in "".join(root.itertext())
    assert "<script>" not in svg and "<img " not in svg and "&lt;script&gt;" in svg
    assert "max-width:none;max-height:none;display:block" in svg


@pytest.mark.parametrize("text", ["W" * 80, "M" * 80, "晶体结构" * 20,
                                 " " + "W" * 12 + "bW",
                                 "Cafe\u0301 " * 12, "👩\u200d🔬 " * 12,
                                 "A reasonably long scientific concept with several separate words"])
def test_wrapping_preserves_unicode_and_respects_conservative_width(text):
    lines = mindmap._wrap(text, 26)
    assert "".join(lines) == text
    assert all(sum(mindmap._glyph_units(cluster) for cluster in mindmap._clusters(line)) <= 26 for line in lines)
    assert not any(line.startswith("\u0301") or line.endswith("\u200d") for line in lines)
    assert mindmap._wrap("Alpha Beta Gamma", 13)[0] == "Alpha Beta "


def test_svg_card_bounds_include_long_labels_descriptions_and_branches(payload):
    payload["content"]["nodes"] = [node("root", label="W" * 80, description="晶体结构 " * 90)]
    for index in range(12):
        payload["content"]["nodes"].append(node(f"branch{index}", "root", "M" * 80, "A complete description " * 20))
    svg = ET.fromstring(mindmap.render_mind_map_svg(payload))
    _, _, width, height = map(float, svg.attrib["viewBox"].split())
    namespace = {"s": "http://www.w3.org/2000/svg"}
    cards = []
    for group in svg.findall(".//s:g", namespace):
        if not group.attrib.get("id", "").startswith("node-"):
            continue
        box = group.findall("s:rect", namespace)[1]
        x, y, w, h = (float(box.attrib[name]) for name in ("x", "y", "width", "height"))
        assert x >= 0 and y >= 0 and x + w <= width and y + h + 3 <= height
        for text in group.findall("s:text", namespace):
            assert x <= float(text.attrib["x"]) <= x + w
            assert y <= float(text.attrib["y"]) <= y + h
        cards.append((x, y, w, h))
    for index, first in enumerate(cards):
        for second in cards[index + 1:]:
            x1, y1, w1, h1 = first
            x2, y2, w2, h2 = second
            assert x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1
    assert len(cards) == 13
    assert height < 10000


def test_html_is_scrollable_and_zoom_scales_without_changing_viewbox(payload):
    normal = mindmap.render_mind_map_html(payload)
    small = mindmap.render_mind_map_html(payload, 0.01)
    large = mindmap.render_mind_map_html(payload, 100)
    assert "height:480px;max-height:480px;overflow:auto" in normal
    for html, factor in ((normal, 1), (small, 0.5), (large, 2)):
        svg = ET.fromstring(html[html.index("<svg"):html.index("</svg>") + 6])
        width = float(svg.attrib["viewBox"].split()[2])
        assert float(svg.attrib["width"]) == width * factor
    with pytest.raises(ValueError, match="finite"):
        mindmap.render_mind_map_html(payload, float("nan"))


def test_current_map_replaces_atomically_without_saved_versions(workspace, payload):
    assert artifacts.load_current_mind_map(workspace) is None
    assert not (workspace_path(workspace) / "artifacts").exists()
    first = artifacts.save_current_mind_map(workspace, payload)
    changed = deepcopy(payload)
    changed["content"]["nodes"][1]["label"] = "Revised method"
    second = artifacts.save_current_mind_map(workspace, changed)
    assert second["id"] == first["id"] and second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert second["sources"][0]["source_path"] == "sources/paper.pdf"
    assert artifacts.load_current_mind_map(workspace) == second
    assert artifacts.list_artifacts(workspace) == []
    directory = workspace_path(workspace) / "artifacts"
    assert [path.name for path in directory.iterdir()] == ["mind_maps"]
    assert [path.name for path in (directory / "mind_maps").iterdir()] == ["current.json"]
    create_workspace("Other")
    assert artifacts.load_current_mind_map("Other") is None


def test_canceled_map_save_preserves_previous_map(workspace, payload):
    first = artifacts.save_current_mind_map(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / "mind_maps" / "current.json"
    before = path.read_bytes()
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) == 2

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_mind_map(workspace, {**first, "title": "Late map"}, is_cancelled=cancelled)
    assert path.read_bytes() == before
    assert [member.name for member in path.parent.iterdir()] == ["current.json"]


def test_map_exports_preserve_sources_without_saving_or_changing_current(workspace, payload):
    output = artifacts.export_mind_map(workspace, payload, "svg")
    assert output.name == "Mind Map.svg"
    assert ET.fromstring(output.read_text(encoding="utf-8")).tag.endswith("svg")
    assert artifacts.load_current_mind_map(workspace) is None
    current = artifacts.save_current_mind_map(workspace, payload)
    path = output.parent.parent / "current.json"
    before = path.read_bytes()
    markdown = artifacts.export_mind_map(workspace, current, "markdown", "Kết quả.svg")
    assert markdown.name == "Kết quả.md"
    text = markdown.read_text(encoding="utf-8")
    assert "**Method**" in text and "[1]" in text and "Page 2" in text and "gpt-6-luna" in text
    export = artifacts.export_mind_map(workspace, current, "json", "Map.md")
    assert export.name == "Map.json"
    assert json.loads(export.read_text(encoding="utf-8"))["sources"][0]["source_path"] == "sources/paper.pdf"
    assert path.read_bytes() == before and artifacts.list_artifacts(workspace) == []


@pytest.mark.parametrize("name", ["../outside", "sub/map", "sub\\map", "CON.svg", "NUL", "name.", ".hidden", "a" * 101, "界" * 85])
def test_map_export_validates_portable_filenames(workspace, payload, name):
    with pytest.raises(ValueError, match="filename"):
        artifacts.export_mind_map(workspace, payload, "svg", name)
    assert not (workspace_path(workspace) / "artifacts").exists()


def test_map_storage_rejects_bad_citations_corrupt_files_and_link_escape(workspace, payload, tmp_path):
    invalid = deepcopy(payload)
    invalid["content"]["nodes"][1]["source_ids"] = ["invented"]
    with pytest.raises(ValueError, match="unknown source"):
        artifacts.save_current_mind_map(workspace, invalid)
    artifacts.save_current_mind_map(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / "mind_maps" / "current.json"
    path.write_text('{"tool_id":[]}', encoding="utf-8")
    with pytest.raises(ValueError):
        artifacts.load_current_mind_map(workspace)
    outside = tmp_path / "outside.json"
    outside.write_text("untouched", encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    with pytest.raises(ValueError, match="linked"):
        artifacts.load_current_mind_map(workspace)
    with pytest.raises(ValueError, match="linked"):
        artifacts.save_current_mind_map(workspace, payload)
    assert outside.read_text(encoding="utf-8") == "untouched"
