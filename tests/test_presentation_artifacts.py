"""Single-deck storage, trusted previews, and portable editable exports."""

import builtins
from copy import deepcopy
from io import BytesIO
import json
import zipfile
from xml.etree import ElementTree

import pytest

from rag_chatbot.tools import artifacts, presentation_render as render
from rag_chatbot.tools.exports import export_bytes, presentation_export_filename
from rag_chatbot.workspaces import create_workspace, workspace_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    return "Research"


@pytest.fixture
def payload(workspace):
    return {"tool_id": "presentation", "title": "Research presentation", "model": "gpt-6-luna",
            "content": {"slides": [
                {"title": "Methods", "bullets": ["Measure the sample.", "Retain uncertainty."],
                 "notes": "Explain the measurement setup.", "source_ids": ["chunk-one"]},
                {"title": "Findings", "bullets": ["A peak was measured."],
                 "notes": "Discuss the limitations.", "source_ids": ["chunk-one"]}]},
            "sources": [{"id": "chunk-one", "document_id": "doc-one", "filename": "paper.pdf",
                         "locator": "Page 2", "text": "A peak was measured.",
                         "source_path": str(workspace_path(workspace) / "sources" / "paper.pdf")}],
            "documents": [{"id": "doc-one", "name": "paper.pdf", "fingerprint": "v1", "generation": "g1"}],
            "coverage": {"documents": 1, "chunks": 1, "processed_chunks": 1}, "settings": {"slide_count": 2}}


def test_current_presentation_stable_identity_isolation_and_no_history(workspace, payload):
    assert artifacts.load_current_presentation(workspace) is None
    assert not (workspace_path(workspace) / "artifacts").exists()
    before = deepcopy(payload)
    first = artifacts.save_current_presentation(workspace, payload)
    second = artifacts.save_current_presentation(workspace, {**payload, "title": "Revised deck"})
    assert payload == before
    assert first["id"] == second["id"] and first["created_at"] == second["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert artifacts.load_current_presentation(workspace) == second
    assert second["sources"][0]["source_path"] == "sources/paper.pdf"
    assert artifacts.list_artifacts(workspace) == []
    assert [path.name for path in (workspace_path(workspace) / "artifacts" / "presentation").iterdir()] == ["current.json"]
    create_workspace("Other")
    assert artifacts.load_current_presentation("Other") is None
    assert not (workspace_path("Other") / "artifacts").exists()


def test_cancel_before_save_and_before_replace_preserves_previous(workspace, payload):
    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_presentation(workspace, payload, is_cancelled=lambda: True)
    assert not (workspace_path(workspace) / "artifacts").exists()
    artifacts.save_current_presentation(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / "presentation" / "current.json"
    before = path.read_bytes()
    calls = []

    def cancelled():
        calls.append(True)
        return len(calls) == 2

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_presentation(workspace, {**payload, "title": "Late deck"}, is_cancelled=cancelled)
    assert path.read_bytes() == before
    assert [child.name for child in path.parent.iterdir()] == ["current.json"]
    with pytest.raises(ValueError, match="callable"):
        artifacts.save_current_presentation(workspace, payload, is_cancelled=True)


@pytest.mark.parametrize("damage", ["empty", "citations", "bullets", "title", "extra", "control"])
def test_invalid_deck_cannot_replace_saved_result(workspace, payload, damage):
    artifacts.save_current_presentation(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / "presentation" / "current.json"
    before = path.read_bytes()
    slide = payload["content"]["slides"][0]
    if damage == "empty":
        payload["content"]["slides"] = []
    elif damage == "citations":
        slide["source_ids"] = ["unknown"]
    elif damage == "bullets":
        slide["bullets"] = ["x" * 180] * 5
    elif damage == "title":
        slide["title"] = " "
    elif damage == "extra":
        slide["html"] = "<script>bad()</script>"
    else:
        slide["notes"] = "Null\x00character"
    with pytest.raises(ValueError):
        artifacts.save_current_presentation(workspace, payload)
    assert path.read_bytes() == before


def test_current_read_rejects_corruption_and_wrong_tool(workspace, payload):
    saved = artifacts.save_current_presentation(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / "presentation" / "current.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        artifacts.load_current_presentation(workspace)
    saved.update(tool_id="summary", content={"sections": []})
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="different tool"):
        artifacts.load_current_presentation(workspace)


@pytest.mark.parametrize("tool_id", [None, [], "summary", "../presentation"])
def test_rejects_other_tools_without_storage(workspace, payload, tool_id):
    payload["tool_id"] = tool_id
    with pytest.raises(ValueError, match="Only a Presentation"):
        artifacts.save_current_presentation(workspace, payload)
    with pytest.raises(ValueError, match="Only a Presentation"):
        artifacts.export_presentation(workspace, payload, "json")
    assert not (workspace_path(workspace) / "artifacts").exists()


@pytest.mark.parametrize("component", ["tool", "manifest", "exports", "export_file"])
def test_rejects_linked_paths(workspace, payload, tmp_path, component):
    directory = workspace_path(workspace) / "artifacts"
    directory.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("untouched", encoding="utf-8")
    tool_directory = directory / "presentation"
    if component == "tool":
        link, target, is_dir = tool_directory, outside, True
    else:
        tool_directory.mkdir()
        if component == "manifest":
            link, target, is_dir = tool_directory / "current.json", sentinel, False
        elif component == "exports":
            link, target, is_dir = tool_directory / "exports", outside, True
        else:
            (tool_directory / "exports").mkdir()
            link, target, is_dir = tool_directory / "exports" / "Presentation.json", sentinel, False
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    if component in ("tool", "manifest"):
        with pytest.raises(ValueError, match="linked"):
            artifacts.load_current_presentation(workspace)
        with pytest.raises(ValueError, match="linked"):
            artifacts.save_current_presentation(workspace, payload)
    if component != "manifest":
        with pytest.raises(ValueError, match="linked"):
            artifacts.export_presentation(workspace, payload, "json")
    assert sentinel.read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("name", ["../outside", "sub/deck", "sub\\deck", "CON.pptx", "NUL", ".hidden", "name.",
                                  "a" * 101, "界" * 85, " Deck", "Deck ", "a\x00b", "\ud800"])
def test_export_rejects_unsafe_filenames_without_storage(workspace, payload, name):
    with pytest.raises(ValueError, match="filename"):
        artifacts.export_presentation(workspace, payload, "json", name)
    assert not (workspace_path(workspace) / "artifacts").exists()


def test_exports_do_not_replace_current_and_preserve_sources_provenance(workspace, payload):
    current = artifacts.save_current_presentation(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / "presentation" / "current.json"
    before = path.read_bytes()
    output = artifacts.export_presentation(workspace, current, "markdown", "Bài trình bày.PPTX")
    assert output.name == "Bài trình bày.md" and output.parent == path.parent / "exports"
    text = output.read_text(encoding="utf-8")
    assert "## Slide 1: Methods" in text and "## Slide 2: Findings" in text
    assert "**References:** [1]" in text and "### Speaker notes" in text
    assert "Page 2" in text and "A peak was measured" in text
    assert "## Provenance" in text and '"slide\\_count": 2' in text
    output = artifacts.export_presentation(workspace, current, "json", "Deck.markdown")
    assert json.loads(output.read_text(encoding="utf-8")) == current
    assert path.read_bytes() == before and artifacts.list_artifacts(workspace) == []
    assert presentation_export_filename("Deck.json", "pptx") == "Deck.pptx"
    with pytest.raises(ValueError, match="available export"):
        artifacts.export_presentation(workspace, current, "html")
    with pytest.raises(ValueError, match="Unsupported"):
        presentation_export_filename("Deck", "html")


def test_optional_missing_pptx_does_not_break_preview_markdown_json(workspace, payload, monkeypatch):
    original_import = builtins.__import__

    def without_pptx(name, *args, **kwargs):
        if name == "pptx" or name.startswith("pptx."):
            raise ImportError("simulated missing optional package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pptx)
    with pytest.raises(RuntimeError, match="python-pptx"):
        artifacts.export_presentation(workspace, payload)
    assert not (workspace_path(workspace) / "artifacts").exists()
    assert "Methods" in render.render_slide_html(payload, 0)
    assert export_bytes(payload, "markdown")[0] == "md"
    assert export_bytes(payload, "json")[0] == "json"


@pytest.mark.parametrize("index", [-1, 2, "0", True, None])
def test_preview_rejects_invalid_slide_index(payload, index):
    with pytest.raises(ValueError, match="index"):
        render.render_slide_html(payload, index)


def test_preview_and_markdown_escape_untrusted_text(payload):
    attack = '<img src="https://example.invalid" onerror="bad()"> & [x](javascript:bad)'
    payload["content"]["slides"][0].update(title=attack, bullets=[attack], notes=attack)
    payload["sources"][0].update(filename=attack, locator=attack, text=attack)
    before = deepcopy(payload)
    preview = render.render_slide_html(payload, 0)
    assert "<img" not in preview and "&lt;img" in preview
    assert 'style="max-height:580px;overflow:auto' in preview and "aspect-ratio:16/9" in preview
    assert "Sources: [1]" in preview and "SLIDE 1 / 2" in preview
    markdown = export_bytes(payload, "markdown")[1].decode("utf-8")
    assert "<img" not in markdown and "[x](javascript:" not in markdown
    assert payload == before


@pytest.mark.parametrize("text", ["W" * 100, "晶" * 100, "Cafe\u0301 " * 15, "👩\u200d🔬 " * 15,
                                 "A long token " + "x" * 120, "Alpha\r\nBeta\tGamma\u2028Delta"])
def test_wrap_retains_normalized_text_and_bounded_line_width(text):
    for prefer_words in (True, False):
        lines = render._wrap_text(text, 30, prefer_words=prefer_words)
        assert "".join(lines) == " ".join(text.split())
        assert all(sum(render._units(cluster) for cluster in render._clusters(line)) <= 30 for line in lines)
        assert not any(line.startswith("\u0301") or line.endswith("\u200d") for line in lines)


@pytest.mark.parametrize("character", ["W", "晶", "a", "e", "m"])
def test_dense_slide_layout_uses_readable_sizes_and_fits_all_text(character):
    for sizes in ([120] * 5, [180, 180, 144, 48, 48], [150] * 4):
        slide = {"title": character * 100, "bullets": [character * size for size in sizes]}
        layout = render.slide_layout(slide)
        assert layout["title_size"] >= 28 and layout["body_size"] >= 18
        assert "".join(layout["title_lines"]) == slide["title"]
        assert ["".join(lines) for lines in layout["body_lines"]] == slide["bullets"]
        assert layout["body_top"] + sum(layout["body_heights"]) + 3 * (len(sizes) - 1) <= render._BODY_BOTTOM * 72


def test_pathological_joined_grapheme_fails_instead_of_overflowing(payload):
    payload["content"]["slides"][0]["bullets"] = ["\u200d".join(["👩"] * 80)]
    with pytest.raises(ValueError, match="readable size"):
        export_bytes(payload, "pptx")


def test_many_source_citations_keep_footer_bounded_and_all_details_in_notes(payload):
    source = payload["sources"][0]
    payload["sources"] = [{**source, "id": f"chunk-{index}", "filename": f"source-{index}.pdf"}
                          for index in range(512)]
    slide = payload["content"]["slides"][0]
    slide["source_ids"] = [source["id"] for source in payload["sources"]]
    payload["content"]["slides"] = [slide]
    footer = render._footer(payload, slide)
    assert len(footer) < 100 and "[8] +504 more" in footer and "[9]" not in footer
    notes = render._speaker_notes(payload, slide)
    assert "[1] source-0.pdf" in notes and "[512] source-511.pdf" in notes
    assert "Details in References" in render.render_slide_html(payload)


def test_pptx_roundtrip_native_text_notes_references_and_no_external_relationships(workspace, payload):
    pptx = pytest.importorskip("pptx")
    before = deepcopy(payload)
    output = artifacts.export_presentation(workspace, payload)
    assert output.suffix == ".pptx"
    presentation = pptx.Presentation(output)
    assert abs(presentation.slide_width / presentation.slide_height - 16 / 9) < 0.00001
    assert len(presentation.slides) == len(payload["content"]["slides"])
    for index, slide in enumerate(presentation.slides):
        original = payload["content"]["slides"][index]
        by_name = {shape.name: shape for shape in slide.shapes}
        title = by_name["Slide title"]
        assert title.text.replace("\v", "") == original["title"]
        assert title.text_frame.paragraphs[0].font.size.pt >= 28
        for number, bullet in enumerate(original["bullets"], 1):
            shape = by_name[f"Bullet {number}"]
            assert shape.text.replace("\v", "") == bullet
            assert shape.text_frame.paragraphs[0].font.size.pt >= 18
        assert "Sources: [1]" in by_name["References"].text
        notes = slide.notes_slide.notes_text_frame.text
        assert original["notes"] in notes and "[1] paper.pdf — Page 2" in notes
        assert "A peak was measured." in notes and "Model: gpt-6-luna" in notes
        assert '"slide_count": 2' in notes
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= presentation.slide_width
            assert shape.top + shape.height <= presentation.slide_height
    with zipfile.ZipFile(output) as archive:
        assert not any("vbaProject" in name for name in archive.namelist())
        for name in archive.namelist():
            if name.endswith(".rels"):
                assert not any(node.attrib.get("TargetMode") == "External" for node in ElementTree.fromstring(archive.read(name)))
    assert payload == before


def test_pptx_dense_unicode_and_markup_are_text_only_with_bounded_shapes(payload):
    pptx = pytest.importorskip("pptx")
    payload["content"]["slides"] = [{"title": "晶" * 100,
        "bullets": ["W" * 180, "晶" * 180, "m" * 144, "界" * 48, "W" * 48],
        "notes": '<script>bad()</script> & \\(λ = 2d sin θ\\)', "source_ids": ["chunk-one"]}]
    payload["sources"][0]["filename"] = '<svg onload="bad()">'
    _, data = export_bytes(payload, "pptx")
    presentation = pptx.Presentation(BytesIO(data))
    slide = presentation.slides[0]
    notes = slide.notes_slide.notes_text_frame.text
    assert payload["content"]["slides"][0]["notes"] in notes
    assert '<svg onload="bad()">' in notes
    for shape in slide.shapes:
        assert shape.top + shape.height <= presentation.slide_height
        assert shape.left + shape.width <= presentation.slide_width
    with zipfile.ZipFile(BytesIO(data)) as archive:
        for name in archive.namelist():
            if name.endswith(".xml"):
                root = ElementTree.fromstring(archive.read(name))
                assert not root.findall(".//script") and not root.findall(".//svg")
