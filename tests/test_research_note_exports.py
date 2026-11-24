"""Saved-note exports retain provenance and cannot publish stale editor actions."""

import asyncio
import json
from pathlib import Path

import gradio as gr
import pytest

from rag_chatbot import research, research_ui as ui
from rag_chatbot.workspaces import create_workspace, workspace_path


@pytest.fixture
def saved_note():
    create_workspace("Research")
    original = workspace_path("Research") / "sources" / "report.txt"
    original.write_text("The measured value is 40.", encoding="utf-8")
    note = research.save_note("Research", "Measurement", "The value is **40** [1].", origin="ai", web_search=True,
        sources=[{"number": 1, "document_id": "d1", "filename": "report.txt", "locator": "Section 1",
                  "text": original.read_text(encoding="utf-8"), "source_path": str(original)}])
    return research.save_note("Research", note["title"], note["body"] + " Added analysis.",
                              identifier=note["id"], expected_revision=note["revision"])


@pytest.mark.parametrize("output_format", ["md", "json"])
def test_export_saved_note_retains_text_references_and_provenance(saved_note, output_format):
    before = research.load_note("Research", saved_note["id"])
    path = research.export_note("Research", saved_note["id"], saved_note["revision"], output_format, "Findings.json")
    assert path == workspace_path("Research") / "notes" / "exports" / f"Findings.{output_format}"
    text = path.read_text(encoding="utf-8")
    assert saved_note["body"] in text
    assert "The measured value is 40" in text
    assert str(workspace_path("Research")) not in text
    if output_format == "json":
        value = json.loads(text)
        assert value == saved_note
        assert value["sources"][0]["source_path"] == "sources/report.txt"
    else:
        assert "Origin: AI-derived" in text and "User edited: Yes" in text
        assert "Original answer used web search: Yes" in text
        assert "not independent evidence" in text and "may not support" in text
    assert research.load_note("Research", saved_note["id"]) == before
    assert len(research.list_notes("Research")) == 1


@pytest.mark.parametrize("filename", ["../escape", "x/y", "x\\y", ".hidden", "CON", "LPT1", "name.", "bad?", "💡" * 100])
def test_export_rejects_nonportable_filenames_before_writing(saved_note, filename):
    with pytest.raises(ValueError):
        research.export_note("Research", saved_note["id"], saved_note["revision"], filename=filename)
    assert not (workspace_path("Research") / "notes" / "exports").exists()


@pytest.mark.parametrize("stop_at", [1, 2, 3])
def test_cancelled_export_preserves_previous_download(saved_note, stop_at):
    path = research.export_note("Research", saved_note["id"], saved_note["revision"])
    path.write_text("Existing download", encoding="utf-8")
    calls = []

    def cancelled():
        calls.append(True)
        return len(calls) == stop_at

    with pytest.raises(RuntimeError, match="cancelled"):
        research.export_note("Research", saved_note["id"], saved_note["revision"], is_cancelled=cancelled)
    assert path.read_text(encoding="utf-8") == "Existing download"
    assert not list(path.parent.glob(".pending-*"))


def test_changed_saved_note_during_serialization_preserves_previous_download(saved_note, monkeypatch):
    path = research.export_note("Research", saved_note["id"], saved_note["revision"])
    old = path.read_bytes()
    serialize = research._note_export_bytes

    def changed(note, output_format):
        encoded = serialize(note, output_format)
        research.save_note("Research", "Changed in another session", "New text", identifier=note["id"],
                           expected_revision=note["revision"])
        return encoded

    monkeypatch.setattr(research, "_note_export_bytes", changed)
    with pytest.raises(ValueError, match="changed in another session"):
        research.export_note("Research", saved_note["id"], saved_note["revision"])
    assert path.read_bytes() == old


def test_export_rejects_stale_revision_or_invalid_format(saved_note):
    with pytest.raises(ValueError, match="changed in another session"):
        research.export_note("Research", saved_note["id"], 1)
    with pytest.raises(ValueError, match="Markdown or JSON"):
        research.export_note("Research", saved_note["id"], saved_note["revision"], "html")


def test_export_size_limit_and_failed_replace_preserve_previous_download(saved_note, monkeypatch):
    path = research.export_note("Research", saved_note["id"], saved_note["revision"])
    old = path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(research, "MAX_NOTE_EXPORT_BYTES", 1)
        with pytest.raises(ValueError, match="supported size"):
            research.export_note("Research", saved_note["id"], saved_note["revision"])
    assert path.read_bytes() == old

    def failed_replace(*args):
        raise OSError("Disk unavailable")

    monkeypatch.setattr(research.os, "replace", failed_replace)
    with pytest.raises(OSError, match="Disk unavailable"):
        research.export_note("Research", saved_note["id"], saved_note["revision"])
    assert path.read_bytes() == old
    assert not list(path.parent.glob(".pending-*"))


@pytest.mark.parametrize("linked_part", ["exports", "Note.md"])
def test_export_rejects_linked_output_paths(saved_note, monkeypatch, linked_part):
    path = research.export_note("Research", saved_note["id"], saved_note["revision"])
    old = path.read_bytes()
    linked = research._linked
    monkeypatch.setattr(research, "_linked", lambda value: value.name == linked_part or linked(value))
    with pytest.raises(ValueError, match="linked paths"):
        research.export_note("Research", saved_note["id"], saved_note["revision"])
    assert path.read_bytes() == old


def _view(saved_note):
    operation = {"revision": 1}
    output = ui._note_view("Research", operation, research.list_notes("Research"), saved_note)
    return operation, output[0], output[-1]


def _export(operation, envelope, epoch, **overrides):
    values = {"title": envelope["note"]["title"], "body": envelope["note"]["body"],
        "filename": "Note", "output_format": "md", "envelope": envelope,
        "workspace": "Research", "operation_state": operation,
        "client_epoch": envelope["token"], "export_epoch": epoch}
    values.update(overrides)
    return ui.on_note_export(**values)


def test_ui_export_requires_saved_edits_and_keeps_stored_note(saved_note):
    operation, envelope, epoch = _view(saved_note)
    result = _export(operation, envelope, epoch)
    assert result[0] == "Saved note exported." and Path(result[1]).exists()
    _, epoch = ui.on_note_export_options(saved_note["title"], "Unsaved", "Note", "md",
                                        envelope, "Research", operation, envelope["token"])
    result = _export(operation, envelope, epoch, body="Unsaved")
    assert "Save note edits before exporting" in result[0] and result[1] is None
    assert research.load_note("Research", saved_note["id"]) == saved_note


def test_old_filename_or_format_click_does_not_use_current_server_export_state(saved_note, monkeypatch):
    operation, envelope, old_epoch = _view(saved_note)
    ui.on_note_export_options(saved_note["title"], saved_note["body"], "Renamed", "json",
                              envelope, "Research", operation, envelope["token"])
    monkeypatch.setattr(research, "export_note", lambda *a, **k: pytest.fail("Old options must not reach disk"))
    assert _export(operation, envelope, old_epoch) == (gr.skip(), None)


@pytest.mark.parametrize("change", ["workspace", "editor", "filename", "format"])
def test_ui_export_cancellation_preserves_previous_file(saved_note, monkeypatch, change):
    operation, envelope, epoch = _view(saved_note)
    path = Path(_export(operation, envelope, epoch)[1])
    old = path.read_bytes()
    serialize = research._note_export_bytes

    def interrupted(note, output_format):
        encoded = serialize(note, output_format)
        if change == "workspace":
            operation["revision"] += 1
        elif change == "editor":
            ui._note_view("Research", operation, [], None)
        else:
            ui.on_note_export_options(saved_note["title"], saved_note["body"],
                "Other" if change == "filename" else "Note", "json" if change == "format" else "md",
                envelope, "Research", operation, envelope["token"])
        return encoded

    monkeypatch.setattr(research, "_note_export_bytes", interrupted)
    assert _export(operation, envelope, epoch) == (gr.skip(), None)
    assert path.read_bytes() == old
    assert not list(path.parent.glob(".pending-*"))


def test_queued_export_with_old_client_epoch_and_new_states_never_reads_saved_note(saved_note, monkeypatch):
    from gradio.state_holder import SessionState

    operation, old_envelope, old_export_epoch = _view(saved_note)
    new_view = ui._note_view("Research", operation, research.list_notes("Research"), saved_note)
    with gr.Blocks() as app:
        controls = ui.build(gr.State("Research"), gr.Dropdown(["Science"]), gr.CheckboxGroup(["d1"]),
                            gr.State([]), gr.State([]), gr.State(operation))
    state = SessionState(app)
    state[controls.workspace._id] = "Research"
    state[controls.operation_state._id] = operation
    state[controls.note_state._id] = new_view[0]
    monkeypatch.setattr(research, "export_note", lambda *a, **k: pytest.fail("Stale queued exports must not touch storage"))
    function = next(item for item in app.fns.values() if item.fn is ui.on_note_export)
    values = {controls.note_title._id: saved_note["title"], controls.note_body._id: saved_note["body"],
              controls.note_filename._id: "Note", controls.note_format._id: "md",
              controls.note_epoch._id: old_envelope["token"], controls.note_export_epoch._id: old_export_epoch}

    async def invoke():
        return await app.process_api(function, [values.get(component._id) for component in function.inputs], state=state)

    result = asyncio.run(invoke())
    assert result["data"][0] in (gr.skip(), {"__type__": "update", "props": gr.skip()})
    assert result["data"][1] is None
    assert isinstance(controls.note_export_button, gr.DownloadButton)
    assert controls.note_export_button.value is None
    assert controls.note_download_event is not None
    assert len(controls.outputs) == ui.OUTPUT_COUNT

