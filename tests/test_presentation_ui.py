"""Presentation session isolation, cancellation, and lazy UI wiring."""

from copy import deepcopy

import pytest

from rag_chatbot.tools import presentation_ui, ui


@pytest.fixture
def operation():
    return {"revision": 0, "answer_revision": 0, "switching": False, "selected_ids": ["doc1"]}


@pytest.fixture
def payload():
    return {"tool_id": "presentation", "title": "Research", "content": {"slides": [
        {"title": "First slide", "bullets": ["Evidence one"], "notes": "First speaker notes", "source_ids": ["c1"]},
        {"title": "Second slide", "bullets": ["Evidence two"], "notes": "Second speaker notes", "source_ids": ["c1"]}]},
        "sources": [{"id": "c1", "document_id": "doc1", "filename": "Paper.pdf", "locator": "Page 2",
                     "text": "Evidence here", "source_path": "sources/doc1.pdf"}],
        "documents": [{"id": "doc1"}], "coverage": {"documents": 1, "chunks": 3, "processed_chunks": 3}}


@pytest.fixture(autouse=True)
def fake_services(monkeypatch):
    monkeypatch.setattr(ui, "_enabled", lambda _: True)
    store = ui._artifacts()
    monkeypatch.setattr(store, "save_current_presentation", lambda workspace, payload, **kwargs: deepcopy(payload))
    for name in ("summary", "spreadsheet", "mind_map", "presentation", "graphics", "plot"):
        monkeypatch.setattr(store, "load_current_" + name, lambda workspace: None)
    monkeypatch.setattr(store, "load_current_learning", lambda workspace, tool_id: None)


def start(payload, operation):
    result = presentation_ui.render_output("Alpha", payload, "Ready")
    presentation_ui.publish(result[0], operation)
    return result


def generate(operation, slide_count=10):
    return list(presentation_ui.on_generate("Focus on findings", slide_count, "Alpha", ["doc1"],
                "Research", "gpt-6-luna", "medium", operation))


def test_navigation_and_notes_are_local(payload, operation, monkeypatch):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Navigation must stay local"))
    first = start(payload, operation)
    assert "First slide" in first[2] and first[3] == "First speaker notes"
    assert first[4] == "Slide 1 of 2" and first[5]["value"] == "0"
    assert not first[6]["interactive"] and first[7]["interactive"]
    following = presentation_ui.on_navigate("next", None, first[0], "Alpha", operation)
    assert "Second slide" in following[2] and following[3] == "Second speaker notes"
    assert following[4] == "Slide 2 of 2" and following[5]["value"] == "1"
    assert following[6]["interactive"] and not following[7]["interactive"]
    previous = presentation_ui.on_navigate("previous", None, following[0], "Alpha", operation)
    selected = presentation_ui.on_navigate("select", "1", previous[0], "Alpha", operation)
    assert selected[4] == "Slide 2 of 2" and selected[0]["payload"] == payload
    assert selected[8:] == (ui.gr.skip(),) * 6


@pytest.mark.parametrize("selection", [None, "-1", "2", "0.0", 0, True, "bad"])
def test_invalid_slide_selection_preserves_state(payload, operation, selection):
    envelope = start(payload, operation)[0]
    before = deepcopy(operation)
    result = presentation_ui.on_navigate("select", selection, envelope, "Alpha", operation)
    assert "Choose an available slide" in result[1] and operation == before


@pytest.mark.parametrize("change", ["workspace", "switching", "output", "navigation"])
def test_stale_navigation_is_discarded(payload, operation, change):
    envelope = start(payload, operation)[0]
    if change == "workspace":
        operation["revision"] += 1
    elif change == "switching":
        operation["switching"] = True
    elif change == "output":
        start(payload, operation)
    else:
        presentation_ui.on_navigate("next", None, envelope, "Alpha", operation)
    before = deepcopy(operation)
    assert presentation_ui.on_navigate("next", None, envelope, "Alpha", operation) == (ui.gr.skip(),) * 14
    assert operation == before


def test_navigation_rechecks_after_rendering(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]
    render = presentation_ui._view

    def delayed(*args, **kwargs):
        result = render(*args, **kwargs)
        operation["revision"] += 1
        return result

    monkeypatch.setattr(presentation_ui, "_view", delayed)
    assert presentation_ui.on_navigate("next", None, envelope, "Alpha", operation) == (ui.gr.skip(),) * 14
    assert operation["presentation"]["index"] == 0


@pytest.mark.parametrize("count", [1, 10, 20.0])
def test_generation_passes_count_and_saves_current(payload, operation, monkeypatch, count):
    old = start(payload, operation)[0]
    operation["presentation"]["index"] = 1
    calls = []

    def backend(*args, **kwargs):
        calls.append((args, kwargs))
        yield {"result": payload}

    def save(workspace, result, *, is_cancelled):
        assert workspace == "Alpha" and result == payload and not is_cancelled()
        return dict(result, id="current")

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui._artifacts(), "save_current_presentation", save)
    result = generate(operation, count)[-1]
    assert len(result) == ui.OUTPUT_COUNTS["presentation"] == 14
    assert calls[0][0] == ("Alpha", "presentation", ["doc1"])
    assert calls[0][1]["slide_count"] == count and type(calls[0][1]["slide_count"]) is int
    assert calls[0][1]["instructions"] == "Focus on findings"
    assert "item_count" not in calls[0][1]
    assert result[0]["payload"]["id"] == "current" and result[0]["token"] != old["token"]
    assert "saved for this workspace" in result[1] and operation["presentation"]["index"] == 0


@pytest.mark.parametrize("count", [0, 21, True, "10", 1.5, None])
def test_invalid_count_prevents_generation(operation, monkeypatch, count):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("No paid work for invalid count"))
    result = generate(operation, count)
    assert len(result) == 1 and "slide count" in result[0][1]
    assert "tool_job" not in operation


def test_disabled_plugin_prevents_generation(operation, monkeypatch):
    monkeypatch.setattr(ui, "_enabled", lambda _: False)
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Disabled tools cannot generate"))
    result = generate(operation)
    assert len(result) == 1 and "disabled" in result[0][1]


def test_cancellation_after_render_prevents_saving(payload, operation, monkeypatch):
    old = start(payload, operation)[0]
    render = ui._render

    def cancelled(*args):
        result = render(*args)
        operation["revision"] += 1
        return result

    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"result": payload}]))
    monkeypatch.setattr(ui, "_render", cancelled)
    monkeypatch.setattr(ui._artifacts(), "save_current_presentation", lambda *args, **kwargs: pytest.fail("Cancelled result cannot save"))
    assert len(generate(operation)) == 1
    assert operation["presentation"]["token"] == old["token"] and "tool_job" not in operation


def test_cancelled_publication_cannot_replace_navigation(payload, operation):
    old = start(payload, operation)[0]
    replacement = presentation_ui.render_output("Alpha", payload, "Ready")[0]
    assert not presentation_ui.publish(replacement, operation, is_cancelled=lambda: True)
    assert operation["presentation"]["token"] == old["token"]


def test_workspace_restore_survives_switch_commit_and_empty_reset(payload, operation, monkeypatch):
    from rag_chatbot.chatbot import on_finish_workspace_switch
    old = start(payload, operation)[0]
    monkeypatch.setattr(ui._artifacts(), "load_current_presentation", lambda workspace: payload if workspace == "Alpha" else None)
    operation.update(revision=2, switching=True, workspace_open_pending=False)
    restored = ui.on_workspace_reset("Alpha", operation)
    offset = ui.OUTPUT_COUNTS["summary"] + ui.OUTPUT_COUNTS["spreadsheet"]
    envelope = restored[offset]
    assert envelope["token"] != old["token"] and restored[offset + 13] == "Presentation"
    assert not presentation_ui._matches(envelope, "Alpha", operation)
    on_finish_workspace_switch(operation)
    assert presentation_ui._matches(envelope, "Alpha", operation)
    assert presentation_ui.on_navigate("next", None, envelope, "Alpha", operation)[4] == "Slide 2 of 2"
    operation["revision"] += 1
    empty = ui.on_reset("Beta", operation)
    assert empty[offset] is None and "presentation" not in operation
    assert presentation_ui.on_export(envelope, "Deck", "json", "Beta", operation) == (ui.gr.skip(), None)


def test_late_workspace_restore_cannot_publish(payload, operation, monkeypatch):
    def load(workspace):
        operation["revision"] += 1
        return payload

    monkeypatch.setattr(ui._artifacts(), "load_current_presentation", load)
    assert ui.on_reset("Alpha", operation) == (ui.gr.skip(),) * ui.TOTAL_OUTPUT_COUNT
    assert "presentation" not in operation


@pytest.mark.parametrize("change", [None, "workspace", "output", "filename", "format"])
def test_export_complete_deck_with_stale_download_guard(payload, operation, monkeypatch, tmp_path, change):
    envelope = start(payload, operation)[0]
    calls = []

    def export(workspace, value, output_format, filename, *, is_cancelled):
        assert not is_cancelled()
        calls.append((workspace, value, output_format, filename))
        if change == "workspace":
            operation["revision"] += 1
        elif change == "output":
            start(payload, operation)
        elif change:
            presentation_ui.on_export_options(operation)
        assert is_cancelled() == bool(change)
        return tmp_path / "Deck.pptx"

    monkeypatch.setattr(ui._artifacts(), "export_presentation", export)
    result = presentation_ui.on_export(envelope, "Deck", "pptx", "Alpha", operation)
    assert calls == [("Alpha", payload, "pptx", "Deck")]
    assert result == ((ui.gr.skip(), None) if change else ("Presentation exported.", str(tmp_path / "Deck.pptx")))


def test_reference_result_from_replaced_output_is_discarded(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]

    def reference(*args):
        start(payload, operation)
        return "Old source", None

    monkeypatch.setattr(ui, "on_reference", reference)
    assert presentation_ui.on_reference("0", envelope, "Alpha", operation) == (ui.gr.skip(), ui.gr.skip())


def test_older_reference_request_cannot_overwrite_newer_selection(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]
    latest = []

    def reference(selection, *args):
        if selection == "0":
            latest.append(presentation_ui.on_reference("1", envelope, "Alpha", operation))
            return "Older source", "older.pdf"
        return "Latest source", "latest.pdf"

    monkeypatch.setattr(ui, "on_reference", reference)
    assert presentation_ui.on_reference("0", envelope, "Alpha", operation) == (ui.gr.skip(), ui.gr.skip())
    assert latest == [("Latest source", "latest.pdf")]


def test_reference_revision_is_invalidated_by_publish_and_clear(payload, operation):
    start(payload, operation)
    initial = operation["presentation_reference_revision"]
    start(payload, operation)
    assert operation["presentation_reference_revision"] == initial + 1
    presentation_ui.clear(operation)
    assert operation["presentation_reference_revision"] == initial + 2


def test_session_navigation_is_independent(payload, operation):
    other = deepcopy(operation)
    first = start(payload, operation)[0]
    second = start(payload, other)[0]
    presentation_ui.on_navigate("next", None, first, "Alpha", operation)
    assert other["presentation"]["index"] == 0
    assert presentation_ui.on_navigate("next", None, first, "Alpha", other) == (ui.gr.skip(),) * 14
    assert presentation_ui.on_navigate("next", None, second, "Alpha", other)[4] == "Slide 2 of 2"


def test_lazy_ui_layout_controls_and_event_wiring(monkeypatch):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("UI construction must not access storage"))
    monkeypatch.setattr(ui, "_enabled", lambda _: pytest.fail("UI construction must not access settings"))
    with ui.gr.Blocks() as app:
        workspace, selected, collection = ui.gr.State("Alpha"), ui.gr.State(["doc1"]), ui.gr.State("Research")
        model, reasoning, operation = ui.gr.State("gpt-6-luna"), ui.gr.State("medium"), ui.gr.State({"revision": 0})
        tool = ui.build_tools(workspace, selected, collection, model, reasoning, operation).controls["presentation"]
    assert tool.instructions.value == presentation_ui.DEFAULT_INSTRUCTIONS
    assert tool.slide_count.value == 10 and tool.slide_count.minimum == 1 and tool.slide_count.maximum == 20
    assert tool.slide_count.step == 1 and tool.instructions.scale == 3 and tool.slide_count.scale == 1
    assert tool.preview.max_height == 520 and tool.notes.lines == tool.notes.max_lines == 6
    assert tool.excerpt.lines == tool.excerpt.max_lines == 6
    assert tool.main_column.scale == 3 and tool.sidebar.scale == 1
    assert tool.output_format.value == "pptx" and tool.filename.value == "Presentation"
    assert [value for _, value in tool.output_format.choices] == ["pptx", "markdown", "json"]
    assert len(tool.outputs) == ui.OUTPUT_COUNTS["presentation"]
    assert not hasattr(tool, "saved") and not hasattr(tool, "save_button")
    assert app.fns[tool.generation["id"]].fn is presentation_ui.on_generate
    assert app.fns[tool.generation["id"]].inputs == [tool.instructions, tool.slide_count, workspace, selected,
                                                     collection, model, reasoning, operation]
    assert app.fns[tool.generation["id"]].outputs == tool.outputs
    for event, action in ((tool.previous_event, "previous"), (tool.next_event, "next"), (tool.select_event, "select")):
        assert app.fns[event["id"]].fn.func is presentation_ui.on_navigate
        assert app.fns[event["id"]].fn.args == (action,)
        assert app.fns[event["id"]].inputs == [tool.slide_selection, tool.payload_state, workspace, operation]
    assert tool.select_event["targets"] == [(tool.slide_selection._id, "input")]
    assert app.fns[tool.export_event["id"]].inputs == [tool.payload_state, tool.filename, tool.output_format, workspace, operation]
    assert tool.download_event["trigger_only_on_success"] and tool.download_event["trigger_after"] == tool.export_event["id"]
    children = {}

    def walk(node):
        children[node["id"]] = [child["id"] for child in node.get("children", [])]
        for child in node.get("children", []):
            walk(child)

    walk(app.config["layout"])

    def descendants(identifier):
        for child in children.get(identifier, []):
            yield child
            yield from descendants(child)

    instruction_ids = list(descendants(tool.instruction_row._id))
    assert instruction_ids.index(tool.instructions._id) < instruction_ids.index(tool.slide_count._id)
    types = {item["id"]: item["type"] for item in app.config["components"]}
    export_ids = list(descendants(tool.export_group._id))
    assert not any(types[identifier] == "row" for identifier in export_ids)
    assert export_ids.index(tool.filename._id) < export_ids.index(tool.output_format._id) < export_ids.index(tool.export_button._id)
