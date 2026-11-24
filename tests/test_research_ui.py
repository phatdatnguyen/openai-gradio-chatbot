"""Research UI wiring and workspace/selection guards without model calls."""

from pathlib import Path

import gradio as gr
import pytest

from rag_chatbot import research, research_ui as ui
from rag_chatbot.workspaces import create_workspace, workspace_path


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    create_workspace("Other")
    state = {"revision": 1, "selected_ids": ["d1"]}
    outputs = ui.on_reset("Research", state)
    return state, outputs[ui.SEARCH_OUTPUT_COUNT]


@pytest.fixture
def source(session):
    path = workspace_path("Research") / "sources" / "report.txt"
    path.write_text("Measurement: 40 participants.", encoding="utf-8")
    return {"number": 1, "document_id": "d1", "filename": "report.txt", "locator": "Paragraph 1",
            "text": path.read_text(encoding="utf-8"), "source_path": str(path), "image_path": None}


def test_notes_save_edit_refresh_delete_and_restore(session):
    state, envelope = session
    saved = ui.on_save("My note", "**Finding**", envelope, "Research", state)
    assert saved[6] == "Saved note."
    assert saved[4] == "**Finding**"
    payload = saved[0]["note"]
    assert payload["origin"] == "human"
    unchanged_blur = ui.on_open(payload["id"], saved[0], "Research", state)
    assert unchanged_blur == (gr.skip(),) * ui.NOTE_OUTPUT_COUNT
    edited = ui.on_save("My note", "Revised finding", saved[0], "Research", state)
    assert edited[0]["note"]["revision"] == 2
    fresh = ui.on_reset("Research", state)
    assert fresh[ui.SEARCH_OUTPUT_COUNT + 1]["choices"] == [("My note — Human", payload["id"])]
    opened = ui.on_open(payload["id"], fresh[ui.SEARCH_OUTPUT_COUNT], "Research", state)
    assert opened[3] == "Revised finding"
    deleted = ui.on_delete(opened[0], "Research", state)
    assert deleted[6] == "Deleted note."
    assert research.list_notes("Research") == []


def test_conflicting_editor_does_not_overwrite_and_reload_recovers(session):
    state, envelope = session
    saved = ui.on_save("A note", "Original", envelope, "Research", state)
    note = saved[0]["note"]
    research.save_note("Research", "A note", "Other browser", identifier=note["id"], expected_revision=1)
    conflict = ui.on_save("A note", "Old browser", saved[0], "Research", state)
    assert "another session" in conflict[6]
    assert conflict[3] == gr.skip()
    reloaded = ui.on_reload(saved[0], "Research", state)
    assert reloaded[3] == "Other browser"


def test_filter_only_updates_choices_and_preserves_editor(session):
    state, envelope = session
    saved = ui.on_save("Sample", "A long finding", envelope, "Research", state)
    filtered = ui.on_filter("finding", saved[0], "Research", state)
    assert len(filtered[0]["choices"]) == 1
    assert len(filtered) == 2 and "Unsaved editor text is unchanged" in filtered[1]


def test_capture_answer_preserves_ai_origin_web_flag_and_reference_snapshot(session, source):
    state, envelope = session
    history = [{"role": "user", "content": "How many?"},
               {"role": "assistant", "content": "40 participants [1].", "sources": [source], "web_search": True}]
    result = ui.on_capture_answer(history, envelope, "Research", state)
    note = result[0]["note"]
    assert note["origin"] == "ai" and note["web_search"]
    assert note["sources"][0]["text"] == source["text"]
    assert "model interpretation" in result[5] and "web search" in result[5]
    assert ui.on_note_reference("0", result[0], "Research", state) == (source["text"], source["source_path"])


def test_capture_incomplete_answer_never_saves(session):
    state, envelope = session
    result = ui.on_capture_answer([{"role": "assistant", "content": "partial"}], envelope,
                                  "Research", state, pending={"question": "Still working"})
    assert "finish" in result[6]
    assert research.list_notes("Research") == []


def test_search_matches_can_be_saved_as_an_excerpt_note(session, source, monkeypatch):
    state, envelope = session
    calls = []
    monkeypatch.setattr(research, "search_sources", lambda *args, **kwargs: calls.append((args, kwargs)) or [source])
    result = ui.on_search("Measurement", "keyword", "Research", "Science", ["d1"], state)
    assert result[0]["sources"] == [source]
    assert "no model request" in result[1]
    assert calls[0][0][2] == ["d1"]
    assert ui.on_search_reference("0", result[0], "Research", state) == (source["text"], source["source_path"])
    captured = ui.on_capture_excerpt("0", result[0], envelope, "Research", state)
    assert captured[0]["note"]["origin"] == "excerpt"
    assert captured[3] == source["text"]


def test_search_result_is_discarded_when_selection_changes_during_read(session, monkeypatch):
    state, _ = session

    def changed(*args, **kwargs):
        state["selected_ids"] = []
        assert kwargs["is_cancelled"]()
        return []

    monkeypatch.setattr(research, "search_sources", changed)
    assert ui.on_search("Test", "hybrid", "Research", "Science", ["d1"], state) == (gr.skip(),) * ui.SEARCH_OUTPUT_COUNT


def test_changed_query_invalidates_prior_result_and_capture(session, source, monkeypatch):
    state, envelope = session
    monkeypatch.setattr(research, "search_sources", lambda *args, **kwargs: [source])
    result = ui.on_search("Measurement", "keyword", "Research", "Science", ["d1"], state)
    ui.on_search_options(state)
    assert ui.on_search_reference("0", result[0], "Research", state) == (gr.skip(), gr.skip())
    assert "select a result" in ui.on_capture_excerpt("0", result[0], envelope, "Research", state)[6]
    assert research.list_notes("Research") == []


@pytest.mark.parametrize("action", ["save", "delete", "capture", "preview", "reference", "open"])
def test_stale_note_callbacks_never_read_or_write_new_workspace(session, monkeypatch, action):
    state, envelope = session
    state["revision"] += 1
    monkeypatch.setattr(research, "save_note", lambda *a, **k: pytest.fail("No stale writes"))
    monkeypatch.setattr(research, "load_note", lambda *a, **k: pytest.fail("No stale reads"))
    monkeypatch.setattr(research, "delete_note", lambda *a, **k: pytest.fail("No stale deletion"))
    callbacks = {
        "save": lambda: ui.on_save("Title", "Body", envelope, "Other", state),
        "delete": lambda: ui.on_delete(envelope, "Other", state),
        "capture": lambda: ui.on_capture_answer([], envelope, "Other", state),
        "preview": lambda: ui.on_preview("Old editor", envelope, "Other", state),
        "reference": lambda: ui.on_note_reference("0", envelope, "Other", state),
        "open": lambda: ui.on_open("stale", envelope, "Other", state),
    }
    result = callbacks[action]()
    assert all(item == gr.skip() for item in result) if isinstance(result, tuple) else result == gr.skip()


def test_reference_error_after_workspace_switch_never_publishes(session, source, monkeypatch):
    state, envelope = session
    monkeypatch.setattr(research, "search_sources", lambda *args, **kwargs: [source])
    result = ui.on_search("Measurement", "keyword", "Research", "Science", ["d1"], state)

    def changed(*args):
        state["revision"] += 1
        raise ValueError("Late error")

    monkeypatch.setattr(research, "reference_path", changed)
    assert ui.on_search_reference("0", result[0], "Research", state) == (gr.skip(), gr.skip())


def test_reset_handles_unavailable_workspace_and_switch_revision():
    state = {"revision": 5, "switching": True}
    result = ui.on_reset(None, state)
    assert len(result) == ui.OUTPUT_COUNT
    assert result[ui.SEARCH_OUTPUT_COUNT]["revision"] == 6
    assert not ui._matches(result[ui.SEARCH_OUTPUT_COUNT], None, state)
    state.update(revision=6, switching=False)
    assert ui._matches(result[ui.SEARCH_OUTPUT_COUNT], None, state)


def test_build_is_lazy_and_dynamic_selectors_bypass_stale_choices(monkeypatch):
    monkeypatch.setattr(research, "list_notes", lambda *a, **k: pytest.fail("No storage at build"))
    monkeypatch.setattr(research, "search_sources", lambda *a, **k: pytest.fail("No search at build"))
    with gr.Blocks() as app:
        controls = ui.build(gr.State("Research"), gr.Dropdown(["Science"]), gr.CheckboxGroup(["d1"]),
                            gr.State([]), gr.State([]), gr.State({"revision": 0}), pending_state=gr.State(None))
    assert len(controls.outputs) == ui.OUTPUT_COUNT
    for fn in (ui.on_search_reference, ui.on_open, ui.on_note_reference):
        block = next(value for value in app.fns.values() if value.fn is fn)
        assert block.preprocess is False
    capture = next(value for value in app.fns.values() if value.fn is ui.on_capture_answer)
    assert len(capture.inputs) == 6


@pytest.mark.parametrize("text", ["\ud800", "before\udfffafter"])
def test_invalid_unicode_note_reports_validation_error(session, text):
    state, envelope = session
    result = ui.on_save("Title", text, envelope, "Research", state)
    assert "Could not save note" in result[6]
    assert research.list_notes("Research") == []


@pytest.mark.parametrize("switch_workspace", [False, True])
def test_queued_browser_save_cannot_write_into_new_editor_or_workspace(session, monkeypatch, switch_workspace):
    """Real RPC reads current State, but must reject the browser's old editor token."""
    import asyncio
    from gradio.state_holder import SessionState

    operation, first = session
    old_client_token = first["token"]
    if switch_workspace:
        operation["revision"] += 1
    workspace = "Other" if switch_workspace else "Research"
    new_editor = ui.on_reset(workspace, operation)[ui.SEARCH_OUTPUT_COUNT]
    with gr.Blocks() as app:
        controls = ui.build(gr.State(workspace), gr.Dropdown(["Science"]), gr.CheckboxGroup(["d1"]),
                            gr.State([]), gr.State([]), gr.State(operation))
    state = SessionState(app)
    state[controls.workspace._id] = workspace
    state[controls.operation_state._id] = operation
    state[controls.note_state._id] = new_editor
    monkeypatch.setattr(research, "save_note", lambda *a, **k: pytest.fail("Stale browser writes must not reach storage"))
    function = next(item for item in app.fns.values() if item.fn is ui.on_save)
    values = {controls.note_title._id: "Old title", controls.note_body._id: "Old unsaved text",
              controls.note_epoch._id: old_client_token}

    async def invoke():
        return await app.process_api(function, [values.get(component._id) for component in function.inputs], state=state)

    result = asyncio.run(invoke())
    assert all(value is None if component.stateful else value in (
        gr.skip(), {"__type__": "update", "props": gr.skip()})
        for component, value in zip(function.outputs, result["data"]))
    assert research.list_notes(workspace) == []


def test_queued_search_rejects_old_workspace_epoch_before_checkbox_validation(session, monkeypatch):
    import asyncio
    from gradio.state_holder import SessionState

    operation, _ = session
    old_client_token = ui._search_epoch(operation)
    operation.update(revision=2, selected_ids=["new-source"])
    ui.on_reset("Other", operation)
    with gr.Blocks() as app:
        controls = ui.build(gr.State("Other"), gr.Dropdown(["New collection"]), gr.CheckboxGroup(["new-source"]),
                            gr.State([]), gr.State([]), gr.State(operation))
    state = SessionState(app)
    state[controls.workspace._id] = "Other"
    state[controls.operation_state._id] = operation
    monkeypatch.setattr(research, "search_sources", lambda *a, **k: pytest.fail("No stale search or embedding"))
    function = next(item for item in app.fns.values() if item.fn is ui.on_search)
    values = {controls.search_query._id: "old question", controls.search_mode._id: "hybrid",
              controls.collection._id: "Old collection", controls.selected_ids._id: ["File"],
              controls.workspace_epoch._id: old_client_token}

    async def invoke():
        return await app.process_api(function, [values.get(component._id) for component in function.inputs], state=state)

    result = asyncio.run(invoke())
    assert all(value is None if component.stateful else value in (
        gr.skip(), {"__type__": "update", "props": gr.skip()})
        for component, value in zip(function.outputs, result["data"]))


def test_old_result_selection_cannot_open_a_new_result_with_same_number(session, source, monkeypatch):
    state, _ = session
    monkeypatch.setattr(research, "search_sources", lambda *args, **kwargs: [source])
    first = ui.on_search("first", "keyword", "Research", "Science", ["d1"], state)
    second = ui.on_search("second", "keyword", "Research", "Science", ["d1"], state)
    monkeypatch.setattr(research, "reference_path", lambda *a, **k: pytest.fail("No stale file reads"))
    assert ui.on_search_reference("0", second[0], "Research", state, first[-1]) == (gr.skip(), gr.skip())


@pytest.mark.parametrize("action", ["new", "filter"])
def test_stale_note_list_error_cannot_replace_new_workspace_status(session, monkeypatch, action):
    state, envelope = session

    def changed(*args):
        state["revision"] += 1
        raise ValueError("Old list failed")

    monkeypatch.setattr(research, "list_notes", changed)
    result = ui.on_new("Research", state) if action == "new" else ui.on_filter("term", envelope, "Research", state)
    assert all(item == gr.skip() for item in result)


def test_failed_search_can_retry_with_current_browser_epoch(session, source, monkeypatch):
    state, _ = session
    browser_epoch = ui._search_epoch(state)

    def failed(*args, **kwargs):
        raise ValueError("Temporary failure")

    monkeypatch.setattr(research, "search_sources", failed)
    result = ui.on_search("term", "keyword", "Research", "Science", ["d1"], state, browser_epoch)
    assert "Temporary failure" in result[1]
    monkeypatch.setattr(research, "search_sources", lambda *a, **k: [source])
    retry = ui.on_search("term", "keyword", "Research", "Science", ["d1"], state, browser_epoch)
    assert retry[0]["sources"] == [source]
    assert retry[-1] == ui._search_epoch(state)
