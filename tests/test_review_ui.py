"""Compare and Evidence Check controls, source review, and stale session guards."""

import pytest

from rag_chatbot.tools import review_ui, ui


@pytest.fixture(params=["compare", "evidence_check"])
def payload(request):
    content = ({"aspects": [{"aspect": "Outcome", "findings": [
        {"document_id": "doc1", "status": "found", "text": "A reported 12 participants.", "source_ids": ["c1"]},
        {"document_id": "doc2", "status": "not_found", "text": "No participant count found.", "source_ids": []}],
        "synthesis": {"text": "Only A reports a count.", "source_ids": ["c1"]}}]}
        if request.param == "compare" else {"claims": [
            {"id": "c1", "text": "A had 12 participants.", "verdict": "supported",
             "rationale": "A explicitly reports this count.",
             "evidence": [{"source_id": "c1", "quote": "A: 12 participants.", "stance": "supports"}]},
            {"id": "c2", "text": "B had 20 participants.", "verdict": "not_found",
             "rationale": "No count found in the selected documents.", "evidence": []}]})
    return {"tool_id": request.param, "title": "Study review", "content": content,
            "model": "gpt-6-luna", "settings": {"claims": [
                {"id": "c1", "text": "A had 12 participants."}, {"id": "c2", "text": "B had 20 participants."}]},
            "sources": [{"id": "c1", "document_id": "doc1", "filename": "A.pdf", "locator": "Page 1",
                         "text": "A: 12 participants.", "source_path": "sources/doc1.pdf"},
                        {"id": "c2", "document_id": "doc2", "filename": "B.pdf", "locator": "Page 1",
                         "text": "B: a pilot study.", "source_path": "sources/doc2.pdf"}],
            "documents": [{"id": "doc1", "filename": "A.pdf"}, {"id": "doc2", "filename": "B.pdf"}],
            "coverage": {"documents": 2, "chunks": 2, "processed_chunks": 2}}


@pytest.fixture
def operation():
    return {"revision": 0, "answer_revision": 0, "switching": False, "selected_ids": ["doc1", "doc2"]}


@pytest.fixture(autouse=True)
def fake_preview(monkeypatch):
    monkeypatch.setattr(review_ui, "_preview", lambda payload: "A source-based review.")


def start(payload, operation):
    result = review_ui.render_output("Alpha", payload, "Ready")
    review_ui.publish(result[0], operation)
    return result


def test_render_retains_absence_distinction_and_evidence(payload, operation):
    result = start(payload, operation)
    assert len(result) == 10 and result[2] == "A source-based review."
    assert result[5]["choices"][0][1] == "0"
    assert "**2 / 2 chunks processed**" in result[4]
    rows = result[3]["value"]
    if payload["tool_id"] == "compare":
        assert result[3]["headers"] == ["Aspect", "A.pdf", "B.pdf", "Comparison"]
        assert rows == [["Outcome", "Found: A reported 12 participants. [1]",
                         "Not found: No participant count found.", "Only A reports a count. [1]"]]
    else:
        assert result[3]["headers"] == review_ui.HEADERS["evidence_check"]
        assert len(rows) == 2 and rows[0][-1] == "[1]" and rows[1][-1] == ""
        assert rows[0][1] == "Supported" and rows[0][3] == "Supports: A: 12 participants."
        assert rows[1][1] == "Not found in selected sources" and rows[1][3] == ""


def test_compare_matrix_prefers_snapshot_names_and_disambiguates_duplicates():
    payload = {"tool_id": "compare", "documents": [
        {"id": "a", "name": "Paper.pdf", "filename": "old.pdf"}, {"id": "b", "name": "Paper.pdf"},
        {"id": "c", "filename": "Third.pdf"}, {"id": "d"}],
        "sources": [{"id": "s1", "document_id": "a", "filename": "old.pdf"},
                    {"id": "s2", "document_id": "b", "filename": "second.pdf"},
                    {"id": "s3", "document_id": "c", "filename": "oldthird.pdf"},
                    {"id": "s4", "document_id": "d", "filename": "Fourth.pdf"}],
        "content": {"aspects": [{"aspect": "Design", "findings": [
            {"document_id": identifier, "status": "found", "text": identifier, "source_ids": [f"s{index}"]}
            for index, identifier in enumerate("abcd", 1)],
            "synthesis": {"text": "Different designs.", "source_ids": ["s1", "s2", "s3", "s4"]}}]}}
    result = review_ui._table(payload)
    assert result["headers"] == ["Aspect", "Paper.pdf (document 1)", "Paper.pdf (document 2)",
                                  "Third.pdf", "Fourth.pdf", "Comparison"]
    assert result["value"][0] == ["Design", "Found: a [1]", "Found: b [2]", "Found: c [3]", "Found: d [4]",
                                   "Different designs. [1] [2] [3] [4]"]


def test_actual_preview_renderer_preserves_literal_text_and_omits_full_sources(payload, monkeypatch):
    from rag_chatbot.tools.exports import render_review_markdown

    payload["title"] = '<img src="https://example.invalid/tracker"> $literal$'
    monkeypatch.setattr(review_ui, "_preview", lambda value: render_review_markdown(value, include_sources=False))
    result = review_ui.render_output("Alpha", payload, "Ready")
    assert "<img" not in result[2] and "&lt;img" in result[2] and "$literal$" in result[2]
    assert "## Sources" not in result[2] and "## Provenance" not in result[2]
    assert "[1]" in result[2]


def test_empty_clear_and_guarded_publish_preserve_other_tools(payload, operation):
    tool_id = payload["tool_id"]
    old = start(payload, operation)[0]
    operation["other_tool"] = {"token": "other"}
    replacement = review_ui.render_output("Alpha", payload, "Ready")[0]
    assert not review_ui.publish(replacement, operation, is_cancelled=lambda: True)
    assert operation[tool_id]["token"] == old["token"]
    empty = review_ui.empty_output(tool_id)
    assert len(empty) == 10 and empty[0] is None and empty[-1] == review_ui.LABELS[tool_id]
    assert empty[3]["value"] == [] and empty[5]["choices"] == []
    review_ui.clear(tool_id, operation)
    assert tool_id not in operation and operation["other_tool"] == {"token": "other"}
    assert not review_ui._matches(tool_id, old, "Alpha", operation)


def test_restored_state_matches_after_workspace_switch_commit(payload, operation):
    operation.update(revision=3, switching=True)
    envelope = start(payload, operation)[0]
    assert not review_ui._matches(payload["tool_id"], envelope, "Alpha", operation)
    operation.update(revision=4, switching=False)
    assert review_ui._matches(payload["tool_id"], envelope, "Alpha", operation)
    assert not review_ui._matches(payload["tool_id"], envelope, "Beta", operation)


@pytest.mark.parametrize("tool_id", ["compare", "evidence_check"])
def test_generation_forwards_only_relevant_options(monkeypatch, operation, tool_id):
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        yield "progress"

    monkeypatch.setattr(ui, "on_generate", generate)
    assert list(review_ui.on_generate(tool_id, "Focus on outcomes", "A has 12 participants.",
                "Alpha", ["doc1", "doc2"], "Research", "gpt-6-luna", "medium", operation)) == ["progress"]
    args, options = calls[0]
    assert args == (tool_id, "Focus on outcomes", "", "Alpha", ["doc1", "doc2"], "Research",
                    "gpt-6-luna", "medium", operation)
    assert options == ({"claims_text": "A has 12 participants."} if tool_id == "evidence_check" else {})


@pytest.mark.parametrize("change", [None, "workspace", "output", "options"])
def test_export_download_rejects_late_output(payload, operation, monkeypatch, tmp_path, change):
    tool_id = payload["tool_id"]
    envelope = start(payload, operation)[0]

    def export(workspace, value, output_format, *, filename, is_cancelled):
        assert (workspace, value, output_format, filename) == ("Alpha", payload, "md", "Review")
        assert not is_cancelled()
        if change == "workspace":
            operation["revision"] += 1
        elif change == "output":
            start(payload, operation)
        elif change == "options":
            review_ui.on_export_options(tool_id, operation)
        assert is_cancelled() == bool(change)
        return tmp_path / "Review.md"

    monkeypatch.setattr(ui._artifacts(), "export_review", export, raising=False)
    result = review_ui.on_export(tool_id, envelope, "Review", "md", "Alpha", operation)
    assert result == ((ui.gr.skip(), None) if change else
                      (f"{review_ui.LABELS[tool_id]} exported.", str(tmp_path / "Review.md")))


def test_export_error_keeps_saved_result(payload, operation, monkeypatch):
    tool_id = payload["tool_id"]
    envelope = start(payload, operation)[0]

    def export(*args, **kwargs):
        raise ValueError("Choose a portable filename.")

    monkeypatch.setattr(ui._artifacts(), "export_review", export, raising=False)
    message, download = review_ui.on_export(tool_id, envelope, "../bad", "md", "Alpha", operation)
    assert "Choose a portable filename" in message and download is None
    assert review_ui._matches(tool_id, envelope, "Alpha", operation)


@pytest.mark.parametrize("change", ["workspace", "session", "tool", "output"])
def test_stale_output_cannot_read_or_export(payload, operation, monkeypatch, change):
    tool_id = payload["tool_id"]
    envelope = start(payload, operation)[0]
    workspace = "Alpha"
    if change == "workspace":
        workspace = "Beta"
    elif change == "session":
        operation = {"revision": 0}
        start(payload, operation)
    elif change == "tool":
        tool_id = "evidence_check" if tool_id == "compare" else "compare"
    else:
        start(payload, operation)
    monkeypatch.setattr(ui, "on_reference", lambda *args: pytest.fail("Stale read"))
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("Stale export"))
    assert review_ui.on_reference(tool_id, "0", envelope, workspace, operation) == (ui.gr.skip(), ui.gr.skip())
    assert review_ui.on_export(tool_id, envelope, "Review", "json", workspace, operation) == (ui.gr.skip(), None)


def test_reference_rejects_out_of_order_selection(payload, operation, monkeypatch):
    tool_id = payload["tool_id"]
    envelope = start(payload, operation)[0]
    latest = []

    def reference(selection, *args):
        if selection == "0":
            latest.append(review_ui.on_reference(tool_id, "1", envelope, "Alpha", operation))
            return "Old excerpt", "old.pdf"
        return "Latest excerpt", "new.pdf"

    monkeypatch.setattr(ui, "on_reference", reference)
    assert review_ui.on_reference(tool_id, "0", envelope, "Alpha", operation) == (ui.gr.skip(), ui.gr.skip())
    assert latest == [("Latest excerpt", "new.pdf")]


def test_other_tool_export_settings_do_not_invalidate_this_download(payload, operation):
    tool_id = payload["tool_id"]
    other = "evidence_check" if tool_id == "compare" else "compare"
    envelope = start(payload, operation)[0]
    revision = operation[f"{tool_id}_export_revision"]
    review_ui.on_export_options(other, operation)
    assert operation[f"{tool_id}_export_revision"] == revision
    assert review_ui._matches(tool_id, envelope, "Alpha", operation)


@pytest.mark.parametrize("tool_id", ["compare", "evidence_check"])
def test_layout_is_lazy_bounded_readonly_and_vertically_stacked(monkeypatch, tool_id):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("Build must not read storage"))
    monkeypatch.setattr(ui, "_enabled", lambda _: pytest.fail("Build must not read settings"))
    monkeypatch.setattr(review_ui, "_preview", lambda _: pytest.fail("Build must not render results"))
    with ui.gr.Blocks() as app:
        workspace, selected, collection = ui.gr.State("Alpha"), ui.gr.State(["doc1", "doc2"]), ui.gr.State("Research")
        model, effort, operation = ui.gr.State("gpt-6-luna"), ui.gr.State("medium"), ui.gr.State({"revision": 0})
        tool = review_ui.build(tool_id, workspace, selected, collection, model, effort, operation)
    assert tool.instructions.value == review_ui.DEFAULT_INSTRUCTIONS[tool_id]
    assert tool.claims.value == "" and tool.claims.visible == (tool_id == "evidence_check")
    assert tool.preview.height == tool.preview.max_height == 480 and tool.preview.latex_delimiters == []
    assert not tool.table.interactive and tool.table.max_height == 320
    assert tool.excerpt.lines == tool.excerpt.max_lines == 6
    assert tool.main_column.scale == 3 and tool.sidebar.scale == 1
    assert tool.output_format.value == "md" and tool.filename.value == review_ui.LABELS[tool_id]
    assert [value for _, value in tool.output_format.choices] == ["md", "csv", "json"]
    assert len(tool.outputs) == 10 and not hasattr(tool, "save_button") and not hasattr(tool, "saved")
    event = app.fns[tool.generation["id"]]
    assert event.fn.func is review_ui.on_generate and event.fn.args == (tool_id,)
    assert event.inputs == [tool.instructions, tool.claims, workspace, selected, collection, model, effort, operation]
    assert event.outputs == tool.outputs
    assert tool.download_event["trigger_only_on_success"]
    assert tool.download_event["trigger_after"] == tool.export_event["id"]
    children = {}

    def walk(node):
        children[node["id"]] = [child["id"] for child in node.get("children", [])]
        for child in node.get("children", []):
            walk(child)

    walk(app.config["layout"])
    assert children[tool.body._id] == [tool.main_column._id, tool.sidebar._id]
    assert children[tool.sidebar._id] == [tool.references_group._id, tool.export_group._id]
    # Gradio wraps the textbox/dropdown in a Form, but neither export control is in a Row.
    assert not any(component.get("type") == "row" and component["id"] in children[tool.export_group._id]
                   for component in app.config["components"])


def test_unknown_tool_rejected():
    with pytest.raises(ValueError, match="Unknown review tool"):
        review_ui.empty_output("unknown")


def _generate_review(payload, operation):
    return list(review_ui.on_generate(payload["tool_id"], "Focus on outcomes", "A had 12 participants.",
                "Alpha", ["doc1", "doc2"], "Research", "gpt-6-luna", "medium", operation))


def test_generation_saves_current_result_and_publishes(payload, operation, monkeypatch):
    tool_id = payload["tool_id"]
    old = start(payload, operation)[0]
    saved = []
    monkeypatch.setattr(ui, "_enabled", lambda _: True)

    def backend(*args, **kwargs):
        assert args == ("Alpha", tool_id, ["doc1", "doc2"])
        assert kwargs["instructions"] == "Focus on outcomes" and kwargs["collection"] == "Research"
        assert kwargs["model"] == "gpt-6-luna" and kwargs["reasoning_effort"] == "medium"
        if tool_id == "evidence_check":
            assert kwargs["claims_text"] == "A had 12 participants."
        else:
            assert "claims_text" not in kwargs
        assert not kwargs["is_cancelled"]()
        yield {"result": payload}

    def save(workspace, result, *, is_cancelled):
        assert workspace == "Alpha" and result == payload and not is_cancelled()
        saved.append(result)
        return dict(payload, id="saved")

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui._artifacts(), "save_current_review", save)
    result = _generate_review(payload, operation)[-1]
    assert len(result) == ui.OUTPUT_COUNTS[tool_id] == 10 and len(saved) == 1
    assert result[0]["payload"]["id"] == "saved" and result[0]["token"] != old["token"]
    assert "saved for this workspace" in result[1]
    assert review_ui._matches(tool_id, result[0], "Alpha", operation) and "tool_job" not in operation


@pytest.mark.parametrize("change", ["workspace", "selection", "stop"])
def test_cancelled_generation_cannot_replace_saved_review(payload, operation, monkeypatch, change):
    old = start(payload, operation)[0]
    monkeypatch.setattr(ui, "_enabled", lambda _: True)

    def backend(*args, **kwargs):
        if change == "workspace":
            operation["revision"] += 1
        elif change == "selection":
            operation["selected_ids"] = []
        else:
            ui.on_stop(operation)
        assert kwargs["is_cancelled"]()
        yield {"result": payload}

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui._artifacts(), "save_current_review", lambda *args, **kwargs: pytest.fail("Stale save"))
    assert len(_generate_review(payload, operation)) == 1
    assert operation[payload["tool_id"]]["token"] == old["token"] and "tool_job" not in operation


def test_workspace_restore_reopens_current_review_and_clears_empty_workspace(payload, operation, monkeypatch):
    tool_id = payload["tool_id"]
    old = start(payload, operation)[0]
    store = ui._artifacts()
    for name in ("summary", "spreadsheet", "mind_map", "presentation", "graphics", "plot"):
        monkeypatch.setattr(store, "load_current_" + name, lambda workspace: None)
    monkeypatch.setattr(store, "load_current_learning", lambda workspace, tool: None)
    monkeypatch.setattr(store, "load_current_review", lambda workspace, tool:
                        payload if workspace == "Alpha" and tool == tool_id else None)
    operation.update(revision=2, switching=True)
    result = ui.on_workspace_reset("Alpha", operation)
    offset = sum(ui.OUTPUT_COUNTS[key] for key in list(ui.TOOL_LABELS)[:list(ui.TOOL_LABELS).index(tool_id)])
    envelope = result[offset]
    assert envelope["payload"] == payload and envelope["token"] != old["token"]
    assert result[offset + 9] == review_ui.LABELS[tool_id]
    operation.update(revision=3, switching=False)
    assert review_ui._matches(tool_id, envelope, "Alpha", operation)
    operation["revision"] += 1
    result = ui.on_reset("Beta", operation)
    assert result[offset] is None and tool_id not in operation
    assert review_ui.on_export(tool_id, envelope, "Review", "md", "Beta", operation) == (ui.gr.skip(), None)
