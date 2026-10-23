"""Current Compare/Evidence Check results and complete, safe local exports."""

from copy import deepcopy
import csv
from io import StringIO
import json

import pytest

from chatbot import config
from rag_chatbot.tools import artifacts
from rag_chatbot.tools.exports import export_bytes, render_review_markdown, review_export_filename
from rag_chatbot.workspaces import create_workspace, workspace_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Review")
    return "Review"


def make_payload(workspace, tool_id):
    sources = [
        {"id": "chunk-1", "document_id": "doc-1", "filename": "Study A.txt", "locator": "Page 1",
         "text": "Sample A is red. The trial used ten participants.",
         "source_path": str(workspace_path(workspace) / "sources" / "study-a.txt")},
        {"id": "chunk-2", "document_id": "doc-2", "filename": "Study B.txt", "locator": "Page 2",
         "text": "Sample A is blue. The trial used twenty participants.",
         "source_path": str(workspace_path(workspace) / "sources" / "study-b.txt")},
    ]
    settings = {"instruction": "Review these documents", "model_requests": 2}
    if tool_id == "compare":
        content = {"aspects": [{"aspect": "Sample color", "findings": [
            {"document_id": "doc-1", "status": "found", "text": "Red", "source_ids": ["chunk-1"]},
            {"document_id": "doc-2", "status": "found", "text": "Blue", "source_ids": ["chunk-2"]}],
            "synthesis": {"text": "The sources describe different colors.", "source_ids": ["chunk-1", "chunk-2"]}}]}
    else:
        settings["claims"] = [{"id": "c1", "text": "Sample A is red."},
                              {"id": "c2", "text": "The trial lasted three years."}]
        content = {"claims": [
            {"id": "c1", "text": settings["claims"][0]["text"], "verdict": "mixed",
             "rationale": "The sources disagree about the color.", "evidence": [
                 {"source_id": "chunk-1", "quote": "Sample A is red.", "stance": "supports"},
                 {"source_id": "chunk-2", "quote": "Sample A is blue.", "stance": "contradicts"}]},
            {"id": "c2", "text": settings["claims"][1]["text"], "verdict": "not_found",
             "rationale": "Neither selected source reports the study duration.", "evidence": []}]}
    return {"tool_id": tool_id, "title": "Source review", "model": config.DEFAULT_MODEL,
            "content": content, "sources": sources,
            "documents": [{"id": "doc-1", "name": "Study A.txt", "fingerprint": "v1", "generation": "g1"},
                          {"id": "doc-2", "name": "Study B.txt", "fingerprint": "v1", "generation": "g2"}],
            "coverage": {"documents": 2, "chunks": 2, "processed_chunks": 2}, "settings": settings}


@pytest.fixture(params=["compare", "evidence_check"])
def payload(workspace, request):
    return make_payload(workspace, request.param)


def test_current_review_is_one_scoped_atomic_output(workspace, payload):
    tool_id = payload["tool_id"]
    assert artifacts.load_current_review(workspace, tool_id) is None
    assert not (workspace_path(workspace) / "artifacts").exists()
    before = deepcopy(payload)
    first = artifacts.save_current_review(workspace, payload)
    second = artifacts.save_current_review(workspace, {**payload, "title": "Replacement"})
    assert payload == before
    assert first["id"] == second["id"] and first["created_at"] == second["created_at"]
    assert artifacts.load_current_review(workspace, tool_id) == second
    assert second["sources"][0]["source_path"] == "sources/study-a.txt"
    assert artifacts.list_artifacts(workspace) == []
    assert list((workspace_path(workspace) / "artifacts" / tool_id).iterdir()) == [
        workspace_path(workspace) / "artifacts" / tool_id / "current.json"]
    create_workspace("Other")
    assert artifacts.load_current_review("Other", tool_id) is None
    other_tool = "compare" if tool_id == "evidence_check" else "evidence_check"
    assert artifacts.load_current_review(workspace, other_tool) is None


@pytest.mark.parametrize("stop_at", [1, 2])
def test_cancelled_save_preserves_previous_result(workspace, payload, stop_at):
    first = artifacts.save_current_review(workspace, payload)
    current = workspace_path(workspace) / "artifacts" / payload["tool_id"] / "current.json"
    before, calls = current.read_bytes(), []

    def cancelled():
        calls.append(True)
        return len(calls) == stop_at

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_review(workspace, {**payload, "title": "Stale"}, is_cancelled=cancelled)
    assert current.read_bytes() == before
    assert artifacts.load_current_review(workspace, payload["tool_id"]) == first
    assert not list(current.parent.glob(".pending-*"))


def test_invalid_citations_do_not_replace_previous_result(workspace, payload):
    first = artifacts.save_current_review(workspace, payload)
    if payload["tool_id"] == "compare":
        payload["content"]["aspects"][0]["findings"][0]["source_ids"] = ["chunk-2"]
    else:
        payload["content"]["claims"][0]["evidence"][0]["quote"] = "An invented quotation."
    with pytest.raises(ValueError):
        artifacts.save_current_review(workspace, payload)
    assert artifacts.load_current_review(workspace, payload["tool_id"]) == first


def test_saved_evidence_claims_cannot_drift_from_original_claims(workspace):
    payload = make_payload(workspace, "evidence_check")
    payload["content"]["claims"][0]["text"] = "Rewritten claim."
    with pytest.raises(ValueError):
        artifacts.save_current_review(workspace, payload)


@pytest.mark.parametrize("format", ["md", "markdown", "csv", "json"])
def test_exports_preserve_current_result_and_complete_snapshots(workspace, payload, format):
    saved = artifacts.save_current_review(workspace, payload)
    current = workspace_path(workspace) / "artifacts" / payload["tool_id"] / "current.json"
    original = current.read_bytes()
    output = artifacts.export_review(workspace, saved, format, "Review.json")
    assert output.parent == current.parent / "exports"
    assert output.name == "Review." + ("md" if format == "markdown" else format)
    text = output.read_text(encoding="utf-8-sig")
    if format == "json":
        assert json.loads(text) == saved
    else:
        markdown = format in {"md", "markdown"}
        for source in saved["sources"]:
            # Markdown escapes punctuation, but neither format drops source text.
            expected = source["text"].replace(".", "\\.") if markdown else source["text"]
            assert expected in text
            expected_path = source["source_path"].replace("-", "\\-").replace(".", "\\.") if markdown else source["source_path"]
            assert expected_path in text
        assert "fingerprint" in text and "generation" in text and "Provenance" in text
        assert ("model\\_requests" if markdown else "model_requests") in text
    assert current.read_bytes() == original and artifacts.load_current_review(workspace, payload["tool_id"]) == saved


@pytest.mark.parametrize("stop_at", [1, 2, 3])
def test_cancelled_export_does_not_overwrite_previous_download(workspace, payload, stop_at):
    saved = artifacts.save_current_review(workspace, payload)
    path = artifacts.export_review(workspace, saved, "json")
    before, calls = path.read_bytes(), []

    def cancelled():
        calls.append(True)
        return len(calls) == stop_at

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.export_review(workspace, {**saved, "title": "Stale"}, "json", is_cancelled=cancelled)
    assert path.read_bytes() == before and not list(path.parent.glob(".pending-*"))


@pytest.mark.parametrize("filename", ["../Review", "sub\\Review", "NUL", "CON.json", " Review", "Review.",
                                      "x" * 101, "界" * 85, "\ud800"])
def test_export_filenames_are_portable(filename):
    with pytest.raises(ValueError, match="filename"):
        review_export_filename(filename, "md")


@pytest.mark.parametrize("format", ["html", "svg", "xlsx", "aiken", ""])
def test_unavailable_export_formats_make_no_files(workspace, payload, format):
    with pytest.raises(ValueError, match="format"):
        artifacts.export_review(workspace, payload, format)
    assert not (workspace_path(workspace) / "artifacts").exists()


def test_review_markdown_uses_escaped_data_and_optional_source_appendix(workspace, payload):
    payload["title"] = '<script>alert("x")</script> [link](https://example.test)'
    payload["sources"][0]["text"] += "\n<img src=x onerror=bad> &lt;script&gt;"
    text = render_review_markdown(payload)
    assert "<script>" not in text and "<img" not in text
    assert "&lt;script&gt;" in text and "&amp;lt;script&amp;gt;" in text
    assert "\\[link\\]\\(https://example\\.test\\)" in text
    preview = render_review_markdown(payload, include_sources=False)
    assert "## Sources" not in preview and "## Provenance" not in preview
    if payload["tool_id"] == "evidence_check":
        assert "does not mean the claim is false" in preview
        assert "Sample A is red\\." in preview and "Sample A is blue\\." in preview


def test_compare_csv_escapes_formula_like_values(workspace):
    payload = make_payload(workspace, "compare")
    payload["content"]["aspects"][0]["aspect"] = "=SUM(A1)"
    payload["content"]["aspects"][0]["findings"][0]["text"] = " @HYPERLINK(bad)"
    payload["sources"][0]["filename"] = "+danger.txt"
    payload["sources"][0]["text"] = "\t=bad"
    extension, data = export_bytes(payload, "csv")
    rows = list(csv.reader(StringIO(data.decode("utf-8-sig"))))
    assert extension == "csv"
    assert rows[1][0] == "'=SUM(A1)" and rows[1][2] == "'+danger.txt"
    assert rows[1][4] == "' @HYPERLINK(bad)"
    source_row = rows[rows.index(["Sources"]) + 2]
    assert source_row[5] == "'\t=bad"


def test_evidence_csv_keeps_every_quote_and_not_found_claim(workspace):
    payload = make_payload(workspace, "evidence_check")
    extension, data = export_bytes(payload, "csv")
    rows = list(csv.reader(StringIO(data.decode("utf-8-sig"))))
    assert extension == "csv"
    assert rows[1][0] == rows[2][0] == "c1"
    assert rows[1][4:8] == ["supports", "Sample A is red.", "[1]", "chunk-1"]
    assert rows[2][4:8] == ["contradicts", "Sample A is blue.", "[2]", "chunk-2"]
    assert rows[3][0] == "c2" and rows[3][2] == "not_found" and rows[3][4:] == ["", "", "", ""]


@pytest.mark.parametrize("invalid", ["summary", "../compare", None, [], {}])
def test_review_storage_only_accepts_trusted_tool_ids(workspace, invalid):
    with pytest.raises(ValueError):
        artifacts.load_current_review(workspace, invalid)
    with pytest.raises(ValueError):
        artifacts.save_current_review(workspace, {"tool_id": invalid})
    with pytest.raises(ValueError):
        artifacts.export_review(workspace, {"tool_id": invalid})


def test_manifest_for_other_tool_is_not_loaded(workspace):
    compare = artifacts.save_current_review(workspace, make_payload(workspace, "compare"))
    artifacts.save_current_review(workspace, make_payload(workspace, "evidence_check"))
    target = workspace_path(workspace) / "artifacts/evidence_check/current.json"
    target.write_text(json.dumps(compare), encoding="utf-8")
    with pytest.raises(ValueError, match="different tool"):
        artifacts.load_current_review(workspace, "evidence_check")


def test_linked_manifest_is_rejected(workspace, payload, tmp_path):
    artifacts.save_current_review(workspace, payload)
    current = workspace_path(workspace) / "artifacts" / payload["tool_id"] / "current.json"
    external = tmp_path / "outside.json"
    external.write_bytes(current.read_bytes())
    current.unlink()
    try:
        current.symlink_to(external)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    with pytest.raises(ValueError, match="linked"):
        artifacts.load_current_review(workspace, payload["tool_id"])


def test_absolute_source_outside_workspace_is_rejected(workspace, payload, tmp_path):
    payload["sources"][0]["source_path"] = str(tmp_path / "outside.txt")
    with pytest.raises(ValueError, match="belong to their workspace"):
        artifacts.save_current_review(workspace, payload)


@pytest.mark.parametrize("operation", ["save", "export"])
def test_review_cancellation_guard_must_be_callable(workspace, payload, operation):
    callback = artifacts.save_current_review if operation == "save" else artifacts.export_review
    with pytest.raises(ValueError, match="guard must be callable"):
        callback(workspace, payload, is_cancelled=True)
