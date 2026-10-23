"""Tool UI isolation, cancellation and edit/save behavior without API calls."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_chatbot.tools import ui


@pytest.fixture
def payload():
    return {"tool_id": "summary", "title": "Study", "content": {"sections": [
        {"heading": "Findings", "paragraphs": [{"text": "A supported finding.", "source_ids": ["c1"]}]}]},
        "sources": [{"id": "c1", "document_id": "doc1", "filename": "Paper.pdf", "locator": "Page 2",
                     "text": "Evidence here", "source_path": "sources/doc1.pdf"}],
        "documents": [{"id": "doc1"}], "coverage": {"documents": 1, "chunks": 3, "processed_chunks": 3}}


@pytest.fixture
def operation():
    return {"revision": 0, "switching": False, "answer_revision": 0}


@pytest.fixture
def spreadsheet_payload(payload):
    value = deepcopy(payload)
    value.update(tool_id="spreadsheet", content={"columns": ["Findings"], "rows": [
        {"document_id": "doc1", "filename": "Paper.pdf", "cells": [
            {"column": "Findings", "value": "Original", "source_ids": ["c1"]}]}]})
    return value


@pytest.fixture
def mind_map_payload(payload):
    value = deepcopy(payload)
    value.update(tool_id="mind_maps", title="Research map", content={"nodes": [
        {"id": "root", "parent_id": "", "label": "Research", "description": "", "source_ids": []},
        {"id": "finding", "parent_id": "root", "label": "Finding", "description": "Supported by evidence.",
         "source_ids": ["c1"]}]})
    return value


@pytest.fixture(autouse=True)
def enable(monkeypatch):
    monkeypatch.setattr(ui, "_enabled", lambda tool_id: True)
    store = ui._artifacts()
    monkeypatch.setattr(store, "save_current_summary", lambda workspace, payload, **kwargs: deepcopy(payload))
    monkeypatch.setattr(store, "load_current_summary", lambda workspace: None)
    monkeypatch.setattr(store, "save_current_spreadsheet", lambda workspace, payload, **kwargs: deepcopy(payload))
    monkeypatch.setattr(store, "load_current_spreadsheet", lambda workspace: None)
    monkeypatch.setattr(store, "save_current_mind_map", lambda workspace, payload, **kwargs: deepcopy(payload))
    monkeypatch.setattr(store, "load_current_mind_map", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_presentation", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_graphics", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_plot", lambda workspace: None)
    monkeypatch.setattr(store, "save_current_learning", lambda workspace, payload, **kwargs: deepcopy(payload))
    monkeypatch.setattr(store, "load_current_learning", lambda workspace, tool_id: None)


def generate(operation, *, selected=None, model="gpt-6-luna", tool="summary", columns="Findings"):
    return ui.on_generate(tool, "Be concise", columns, "Alpha", ["doc1"] if selected is None else selected,
                          "Research", model, "medium", operation)


def test_generate_preserves_scope_and_exhaustive_coverage(monkeypatch, payload, operation):
    calls = []

    def backend(*args, **kwargs):
        calls.append((args, kwargs))
        yield {"status": "Reading 3 chunks", "result": None}
        yield {"status": "Ready", "result": payload}

    monkeypatch.setattr(ui, "_generate", backend)
    updates = list(generate(operation))
    assert len(updates) == 3
    assert all(len(update) == ui.OUTPUT_COUNTS["summary"] for update in updates)
    assert calls[0][0] == ("Alpha", "summary", ["doc1"])
    assert calls[0][1]["collection"] == "Research"
    assert calls[0][1]["instructions"] == "Be concise"
    assert updates[-1][0] == {"workspace": "Alpha", "payload": payload}
    assert "3 / 3 chunks" in updates[-1][4]
    assert updates[-1][5]["choices"][0][0].startswith("[1] Paper.pdf")
    assert "tool_job" not in operation


@pytest.mark.parametrize("condition,expected", [("empty", "No sources selected"),
    ("disabled", "Plugins"), ("model", "structured tool outputs"), ("switching", "Wait for")])
def test_preflight_blocks_paid_work(monkeypatch, operation, condition, expected):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Must not start paid work"))
    if condition == "disabled":
        monkeypatch.setattr(ui, "_enabled", lambda _: False)
    if condition == "switching":
        operation["switching"] = True
    result = list(generate(operation, selected=[] if condition == "empty" else None,
                           model="gpt-3.5-turbo" if condition == "model" else "gpt-6-luna"))
    assert len(result) == 1 and expected in result[0][1]
    assert "tool_job" not in operation


@pytest.mark.parametrize("action", ["workspace", "sources", "stop"])
def test_late_results_never_replace_new_workspace_or_selection(monkeypatch, payload, operation, action):
    closed = []

    def backend(*args, **kwargs):
        try:
            yield {"status": "Reading", "result": None}
            if action == "workspace":
                operation["revision"] += 1
            elif action == "sources":
                operation["answer_revision"] += 1
            else:
                ui.on_stop(operation)
            assert kwargs["is_cancelled"]()
            yield {"status": "Must be discarded", "result": payload}
        finally:
            closed.append(True)

    monkeypatch.setattr(ui, "_generate", backend)
    result = list(generate(operation))
    assert len(result) == 2
    assert all(update[0] == ui.gr.skip() for update in result)
    assert closed == [True] and "tool_job" not in operation


@pytest.mark.parametrize("changed", ["revision", "answer_revision", "selected_ids"])
def test_generation_rechecks_scope_after_rendering(monkeypatch, payload, operation, changed):
    original = ui._render
    operation["selected_ids"] = ["doc1"]

    def render(*args, **kwargs):
        result = original(*args, **kwargs)
        if changed == "selected_ids":
            operation[changed] = ["doc2"]
        else:
            operation[changed] += 1
        return result

    monkeypatch.setattr(ui, "_render", render)
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"status": "Done", "result": payload}]))
    updates = list(generate(operation))
    assert len(updates) == 1 and updates[0][0] == ui.gr.skip()
    assert "tool_job" not in operation


def test_session_allows_one_tool_until_current_request_finishes(monkeypatch, payload, operation):
    def backend(*args, **kwargs):
        yield {"status": "Working", "result": None}
        yield {"status": "Done", "result": payload}

    monkeypatch.setattr(ui, "_generate", backend)
    running = generate(operation)
    next(running)
    next(running)
    assert "Another tool request" in list(generate(operation))[0][1]
    assert "Stopping after" in ui.on_stop(operation)
    assert "Another tool request" in list(generate(operation))[0][1]
    assert list(running) == []
    assert "tool_job" not in operation
    assert list(generate(operation))[-1][0]["payload"] == payload


def test_closing_generator_releases_session_and_backend(monkeypatch, operation):
    closed = []

    def backend(*args, **kwargs):
        try:
            yield {"status": "Working", "result": None}
            yield {"status": "More", "result": None}
        finally:
            closed.append(True)

    monkeypatch.setattr(ui, "_generate", backend)
    stream = generate(operation)
    next(stream)
    next(stream)
    stream.close()
    assert closed == [True] and "tool_job" not in operation


def test_stop_is_session_local():
    alpha = {"tool_job": {"id": "a", "cancelled": False}}
    beta = {"tool_job": {"id": "b", "cancelled": False}}
    ui.on_stop(alpha)
    assert alpha["tool_job"]["cancelled"] and not beta["tool_job"]["cancelled"]


def test_spreadsheet_export_rejects_previous_workspace_output(monkeypatch, spreadsheet_payload, operation):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("Must not access another workspace's output"))
    result = ui.on_spreadsheet_export({"workspace": "Beta", "payload": spreadsheet_payload},
                                      [["Original"]], "Spreadsheet", "csv", "Alpha", operation)
    assert "this workspace" in result[0] and result[1] is None


def test_summary_edits_are_separate_and_keep_original_evidence(payload):
    envelope = {"workspace": "Alpha", "payload": payload}
    updated = ui._edited(envelope, "Alpha", "summary", "Changed by user [1].")
    assert updated["edited_text"] == "Changed by user [1]."
    assert updated["sources"] == payload["sources"]
    assert "edited_text" not in payload


def test_summary_generation_saves_one_current_output(monkeypatch, payload, operation):
    saved = []
    store = ui._artifacts()

    def save(workspace, value, *, is_cancelled):
        assert not is_cancelled()
        saved.append((workspace, value))
        return dict(value, id="current-summary")

    monkeypatch.setattr(store, "save_current_summary", save)
    monkeypatch.setattr(store, "save_artifact", lambda *args: pytest.fail("Summary must not create history versions"))

    def backend(*args, **kwargs):
        yield {"status": "Completed", "result": payload}

    monkeypatch.setattr(ui, "_generate", backend)
    result = list(generate(operation))[-1]
    assert saved == [("Alpha", payload)]
    assert result[0]["payload"]["id"] == "current-summary"
    assert "saved for this workspace" in result[1]
    assert result[2]["visible"] is False and result[3]["visible"] is True


def test_summary_edit_and_save_toggle_preview_without_new_history(monkeypatch, payload, operation):
    envelope = {"workspace": "Alpha", "payload": payload}
    operation["tool_job"] = {"id": "pending", "cancelled": False}
    editing = ui.on_summary_edit(envelope, "Alpha", operation)
    assert editing[2]["visible"] and not editing[3]["visible"]
    assert not editing[8]["visible"] and editing[9]["visible"]
    assert operation["summary_editing"] and operation["tool_job"]["cancelled"]
    operation.pop("tool_job")
    store = ui._artifacts()
    calls = []

    def save(workspace, value, **kwargs):
        calls.append(value)
        return value

    monkeypatch.setattr(store, "save_current_summary", save)
    monkeypatch.setattr(store, "save_artifact", lambda *args: pytest.fail("Edit Save must not create history"))
    result = ui.on_summary_save(envelope, "Edited Summary [1]", "Alpha", operation)
    assert calls[0]["edited_text"] == "Edited Summary [1]"
    assert not result[2]["visible"] and result[3]["visible"]
    assert result[3]["value"] == "Edited Summary [1]"
    assert result[8]["visible"] and not result[9]["visible"]
    assert not operation["summary_editing"]


def test_summary_save_does_not_publish_after_rendering_changes_workspace(monkeypatch, payload, operation):
    original = ui._render

    def render(*args, **kwargs):
        result = original(*args, **kwargs)
        operation["revision"] += 1
        return result

    monkeypatch.setattr(ui, "_render", render)
    monkeypatch.setattr(ui._artifacts(), "save_current_summary", lambda *args, **kwargs: pytest.fail("Stale edit must not publish"))
    result = ui.on_summary_save({"workspace": "Alpha", "payload": payload}, "Edited", "Alpha", operation)
    assert result == (ui.gr.skip(),) * ui.OUTPUT_COUNTS["summary"]


def test_summary_edit_rechecks_workspace_after_rendering(monkeypatch, payload, operation):
    original = ui._render

    def render(*args, **kwargs):
        result = original(*args, **kwargs)
        operation["revision"] += 1
        return result

    monkeypatch.setattr(ui, "_render", render)
    result = ui.on_summary_edit({"workspace": "Alpha", "payload": payload}, "Alpha", operation)
    assert result == (ui.gr.skip(),) * ui.OUTPUT_COUNTS["summary"]
    assert not operation.get("summary_editing")


def test_summary_save_rechecks_workspace_after_disk_write(monkeypatch, payload, operation):
    def save(workspace, value, *, is_cancelled):
        assert not is_cancelled()
        operation["revision"] += 1
        assert is_cancelled()
        return value

    monkeypatch.setattr(ui._artifacts(), "save_current_summary", save)
    result = ui.on_summary_save({"workspace": "Alpha", "payload": payload}, "Edited", "Alpha", operation)
    assert result == (ui.gr.skip(),) * ui.OUTPUT_COUNTS["summary"]


def test_summary_save_failure_keeps_edit_mode(monkeypatch, payload, operation):
    operation["summary_editing"] = True

    def fail(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(ui._artifacts(), "save_current_summary", fail)
    result = ui.on_summary_save({"workspace": "Alpha", "payload": payload}, "Edited", "Alpha", operation)
    assert "Disk full" in result[1]
    assert result[2] == ui.gr.skip() and operation["summary_editing"]


def test_summary_export_uses_committed_output_and_named_file(monkeypatch, payload, operation, tmp_path):
    store = ui._artifacts()
    calls = []

    def export(workspace, value, output_format, *, filename, is_cancelled):
        assert not is_cancelled()
        calls.append((workspace, value, output_format, filename))
        return tmp_path / "Research.docx"

    monkeypatch.setattr(store, "export_summary", export)
    monkeypatch.setattr(store, "save_current_summary", lambda *args, **kwargs: pytest.fail("Export must not save"))
    result = ui.on_summary_export({"workspace": "Alpha", "payload": payload}, "Research", "docx", "Alpha", operation)
    assert calls == [("Alpha", payload, "docx", "Research")]
    assert result == ("Summary exported.", str(tmp_path / "Research.docx"))


def test_summary_export_requires_saving_edits_first(monkeypatch, payload, operation):
    operation["summary_editing"] = True
    monkeypatch.setattr(ui._artifacts(), "export_summary", lambda *args, **kwargs: pytest.fail("Do not export unsaved edits"))
    result = ui.on_summary_export({"workspace": "Alpha", "payload": payload}, "Summary", "markdown", "Alpha", operation)
    assert "Save your Summary edits" in result[0] and result[1] is None


@pytest.mark.parametrize("change", ["workspace", "filename"])
def test_summary_export_clears_stale_download(monkeypatch, payload, operation, change, tmp_path):
    def export(*args, **kwargs):
        assert not kwargs["is_cancelled"]()
        if change == "workspace":
            operation["revision"] += 1
        else:
            assert ui.on_summary_export_options(operation) is None
        assert kwargs["is_cancelled"]()
        return tmp_path / "old.md"

    monkeypatch.setattr(ui._artifacts(), "export_summary", export)
    result = ui.on_summary_export({"workspace": "Alpha", "payload": payload}, "Summary", "markdown", "Alpha", operation)
    assert result == (ui.gr.skip(), None)


def test_spreadsheet_named_export_includes_visible_edits_without_saving(monkeypatch, spreadsheet_payload, operation, tmp_path):
    store = ui._artifacts()
    captured = []

    def export(workspace, payload, output_format, *, filename, is_cancelled):
        assert not is_cancelled()
        captured.append((workspace, payload, output_format, filename))
        return tmp_path / "Measurements.xlsx"

    monkeypatch.setattr(store, "export_spreadsheet", export)
    monkeypatch.setattr(store, "save_artifact", lambda *args: pytest.fail("Export must not create a history version"))
    monkeypatch.setattr(store, "save_current_spreadsheet", lambda *args, **kwargs: pytest.fail("Export must not replace current"))
    before = deepcopy(spreadsheet_payload)
    editor = pd.DataFrame([["User edit"]], columns=["Findings"])
    result = ui.on_spreadsheet_export({"workspace": "Alpha", "payload": spreadsheet_payload},
                                      editor, "Measurements", "xlsx", "Alpha", operation)
    assert result[1] == str(tmp_path / "Measurements.xlsx")
    assert captured == [("Alpha", dict(before, edited_rows=[["User edit"]]), "xlsx", "Measurements")]
    assert spreadsheet_payload == before


@pytest.mark.parametrize("change", ["workspace", "filename_or_format", "cell_edits"])
def test_spreadsheet_export_clears_download_when_inputs_change(monkeypatch, spreadsheet_payload, operation, change, tmp_path):
    def export(*args, **kwargs):
        assert not kwargs["is_cancelled"]()
        if change == "workspace":
            operation["revision"] += 1
        else:
            assert ui.on_spreadsheet_export_options(operation) is None
        assert kwargs["is_cancelled"]()
        return tmp_path / "previous.csv"

    monkeypatch.setattr(ui._artifacts(), "export_spreadsheet", export)
    result = ui.on_spreadsheet_export({"workspace": "Alpha", "payload": spreadsheet_payload},
                                      [["Original"]], "Spreadsheet", "csv", "Alpha", operation)
    assert result == (ui.gr.skip(), None)


def test_spreadsheet_export_rejects_a_different_tool(monkeypatch, payload, operation):
    monkeypatch.setattr(ui._artifacts(), "export_spreadsheet", lambda *args, **kwargs: pytest.fail("Wrong tool cannot export"))
    result = ui.on_spreadsheet_export({"workspace": "Alpha", "payload": payload},
                                      [["Original"]], "Spreadsheet", "csv", "Alpha", operation)
    assert "different tool" in result[0] and result[1] is None


def test_spreadsheet_export_rechecks_scope_after_preparing_edits(monkeypatch, spreadsheet_payload, operation):
    validate = ui._edited

    def changed(*args, **kwargs):
        payload = validate(*args, **kwargs)
        operation["revision"] += 1
        return payload

    monkeypatch.setattr(ui, "_edited", changed)
    monkeypatch.setattr(ui._artifacts(), "export_spreadsheet", lambda *args, **kwargs: pytest.fail("Stale output cannot export"))
    result = ui.on_spreadsheet_export({"workspace": "Alpha", "payload": spreadsheet_payload},
                                      [["Original"]], "Spreadsheet", "csv", "Alpha", operation)
    assert result == (ui.gr.skip(), None)


def test_workspace_open_restores_the_current_summary(monkeypatch, payload, operation):
    monkeypatch.setattr(ui._artifacts(), "load_current_summary", lambda workspace: payload)
    operation["switching"] = True
    operation["summary_editing"] = True
    result = ui.on_workspace_reset("Alpha", operation)
    assert result[0] == {"workspace": "Alpha", "payload": payload}
    assert "Restored" in result[1]
    assert result[3]["visible"] and "supported finding" in result[3]["value"]
    assert result[11] == "Summary"
    assert not operation["summary_editing"]


def test_spreadsheet_generation_saves_current_output_without_creating_versions(monkeypatch, spreadsheet_payload, operation):
    saved = []

    def save(workspace, value, *, is_cancelled):
        assert not is_cancelled()
        saved.append((workspace, value))
        return dict(value, id="current-sheet")

    monkeypatch.setattr(ui._artifacts(), "save_current_spreadsheet", save)
    monkeypatch.setattr(ui._artifacts(), "save_artifact", lambda *args: pytest.fail("Generation must not create a version"))
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"result": spreadsheet_payload}]))
    result = list(generate(operation, tool="spreadsheet"))[-1]
    assert saved == [("Alpha", spreadsheet_payload)]
    assert result[0]["payload"]["id"] == "current-sheet"
    assert result[1] == "Completed. Spreadsheet saved for this workspace."
    assert result[2]["value"] == [["Original"]]
    assert len(result) == ui.OUTPUT_COUNTS["spreadsheet"] == 10
    assert result[8] is None
    assert operation["spreadsheet_export_revision"] == 1


@pytest.mark.parametrize("action", ["workspace", "sources", "stop"])
def test_spreadsheet_cancelled_during_render_is_not_saved(monkeypatch, spreadsheet_payload, operation, action):
    render = ui._render

    def changed(*args, **kwargs):
        result = render(*args, **kwargs)
        if action == "stop":
            ui.on_stop(operation)
        else:
            operation["revision" if action == "workspace" else "answer_revision"] += 1
        return result

    monkeypatch.setattr(ui, "_render", changed)
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"result": spreadsheet_payload}]))
    monkeypatch.setattr(ui._artifacts(), "save_current_spreadsheet",
                        lambda *args, **kwargs: pytest.fail("Stale output must not be saved"))
    updates = list(generate(operation, tool="spreadsheet"))
    assert len(updates) == 1 and updates[0][0] == ui.gr.skip()


def test_workspace_restore_preserves_legacy_spreadsheet_edits_and_original_references(monkeypatch, spreadsheet_payload, operation):
    spreadsheet_payload["edited_rows"] = [["Reviewed"]]
    monkeypatch.setattr(ui._artifacts(), "load_current_spreadsheet", lambda workspace: spreadsheet_payload)
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Restore must not regenerate"))
    operation["switching"] = True
    result = ui.on_workspace_reset("Alpha", operation)[ui.OUTPUT_COUNTS["summary"]:]
    assert result[0] == {"workspace": "Alpha", "payload": spreadsheet_payload}
    assert "Restored" in result[1]
    assert result[2]["headers"] == ["Findings"] and result[2]["value"] == [["Reviewed"]]
    assert "Findings: [1]" in result[3]
    assert result[5]["choices"][0][0].startswith("[1] Paper.pdf")
    assert result[8] is None and result[9] == "Spreadsheet"


@pytest.mark.parametrize("condition", ["empty", "corrupt", "switched"])
def test_spreadsheet_restore_cannot_leak_previous_output(monkeypatch, spreadsheet_payload, operation, condition):
    def load(workspace):
        assert workspace == "Beta"
        if condition == "corrupt":
            raise ValueError("Invalid saved spreadsheet")
        if condition == "switched":
            operation["revision"] += 1
            return spreadsheet_payload
        return None

    monkeypatch.setattr(ui._artifacts(), "load_current_spreadsheet", load)
    result = ui.on_workspace_reset("Beta", operation)
    if condition == "switched":
        assert result == (ui.gr.skip(),) * ui.TOTAL_OUTPUT_COUNT
    else:
        offset = ui.OUTPUT_COUNTS["summary"]
        assert result[offset] is None and result[offset + 2]["value"] == []
        assert result[offset + 5]["choices"] == []
        assert ("Could not restore Spreadsheet" in result[offset + 1]) == (condition == "corrupt")


def test_mind_map_generation_saves_current_map_and_resets_zoom(monkeypatch, mind_map_payload, operation):
    saved = []
    store = ui._artifacts()

    def save(workspace, payload, *, is_cancelled):
        assert not is_cancelled()
        saved.append((workspace, payload))
        return dict(payload, id="current-map")

    def backend(*args, **kwargs):
        assert args == ("Alpha", "mind_maps", ["doc1"])
        assert kwargs["columns"] is None
        yield {"status": "Completed", "result": mind_map_payload}

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui, "_render_mind_map", lambda payload, zoom=1: f"<svg>zoom={zoom}</svg>")
    monkeypatch.setattr(store, "save_current_mind_map", save)
    monkeypatch.setattr(store, "save_artifact", lambda *args: pytest.fail("Mind Map must not create history versions"))
    result = list(generate(operation, tool="mind_maps"))[-1]
    assert saved == [("Alpha", mind_map_payload)]
    assert len(result) == ui.OUTPUT_COUNTS["mind_maps"]
    assert result[0]["payload"]["id"] == "current-map"
    assert result[1] == "Completed. Mind Map saved for this workspace."
    assert "<svg>" in result[2] and "References below" not in result[3]
    assert result[4] == 1.0 and result[5] is None
    assert operation["mind_map_revision"] == 1


def test_mind_map_output_uses_bounded_svg_renderer(mind_map_payload):
    result = ui._render("Alpha", mind_map_payload, "Ready")
    assert len(result) == ui.OUTPUT_COUNTS["mind_maps"]
    assert "<svg" in result[2] and "overflow:auto" in result[2]
    assert "Research map" in result[2] and "Supported by evidence." in result[2]
    assert "References below" not in result[3]


def test_mind_map_cancelled_while_rendering_is_not_saved(monkeypatch, mind_map_payload, operation):
    def render(payload, zoom=1):
        operation["answer_revision"] += 1
        return "<svg></svg>"

    def backend(*args, **kwargs):
        yield {"result": mind_map_payload}

    monkeypatch.setattr(ui, "_render_mind_map", render)
    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui._artifacts(), "save_current_mind_map", lambda *args, **kwargs: pytest.fail("Cancelled map cannot save"))
    result = list(generate(operation, tool="mind_maps"))
    assert len(result) == 1 and result[0][0] == ui.gr.skip()
    assert "tool_job" not in operation


def test_workspace_restore_includes_current_mind_map(monkeypatch, mind_map_payload, operation):
    monkeypatch.setattr(ui._artifacts(), "load_current_mind_map", lambda workspace: mind_map_payload)
    monkeypatch.setattr(ui, "_render_mind_map", lambda payload, zoom=1: "<svg>restored map</svg>")
    operation["switching"] = True
    result = ui.on_workspace_reset("Alpha", operation)
    offset = sum(ui.OUTPUT_COUNTS[name] for name in ("summary", "spreadsheet", "presentation", "graphics", "plot"))
    assert result[offset]["payload"] == mind_map_payload
    assert "Restored" in result[offset + 1]
    assert result[offset + 2] == "<svg>restored map</svg>"
    assert result[offset + 4] == 1.0 and result[offset + 6] == "Mind Map"


def test_mind_map_zoom_is_local_and_preserves_content(monkeypatch, mind_map_payload, operation):
    calls = []

    def render(payload, zoom=1):
        calls.append((payload, zoom))
        return "<svg>zoomed</svg>"

    monkeypatch.setattr(ui, "_render_mind_map", render)
    result = ui.on_mind_map_zoom(1.5, {"workspace": "Alpha", "payload": mind_map_payload}, "Alpha", operation)
    assert calls == [(mind_map_payload, 1.5)]
    assert result == ("<svg>zoomed</svg>", ui.gr.skip())


@pytest.mark.parametrize("changed", ["revision", "mind_map_revision", "mind_map_view_revision"])
def test_late_mind_map_zoom_cannot_replace_new_map(monkeypatch, mind_map_payload, operation, changed):
    def render(payload, zoom=1):
        operation[changed] = operation.get(changed, 0) + 1
        return "<svg>old map</svg>"

    monkeypatch.setattr(ui, "_render_mind_map", render)
    assert ui.on_mind_map_zoom(1.5, {"workspace": "Alpha", "payload": mind_map_payload}, "Alpha", operation) == (
        ui.gr.skip(), ui.gr.skip())


def test_mind_map_export_uses_filename_without_saving(monkeypatch, mind_map_payload, operation, tmp_path):
    calls = []

    def export(workspace, payload, output_format, *, filename, is_cancelled):
        assert not is_cancelled()
        calls.append((workspace, payload, output_format, filename))
        return tmp_path / "Study.svg"

    monkeypatch.setattr(ui._artifacts(), "export_mind_map", export)
    monkeypatch.setattr(ui._artifacts(), "save_current_mind_map", lambda *args, **kwargs: pytest.fail("Export cannot rewrite map"))
    result = ui.on_mind_map_export({"workspace": "Alpha", "payload": mind_map_payload}, "Study", "svg", "Alpha", operation)
    assert result == ("Mind Map exported.", str(tmp_path / "Study.svg"))
    assert calls == [("Alpha", mind_map_payload, "svg", "Study")]


@pytest.mark.parametrize("changed", ["workspace", "export_options"])
def test_mind_map_export_discards_stale_download(monkeypatch, mind_map_payload, operation, tmp_path, changed):
    def export(*args, **kwargs):
        assert not kwargs["is_cancelled"]()
        if changed == "workspace":
            operation["revision"] += 1
        else:
            ui.on_mind_map_export_options(operation)
        assert kwargs["is_cancelled"]()
        return tmp_path / "old.svg"

    monkeypatch.setattr(ui._artifacts(), "export_mind_map", export)
    assert ui.on_mind_map_export({"workspace": "Alpha", "payload": mind_map_payload}, "Map", "svg", "Alpha", operation) == (
        ui.gr.skip(), None)


def test_spreadsheet_editor_cannot_change_source_row_mapping(payload):
    sheet = deepcopy(payload)
    sheet.update(tool_id="spreadsheet", content={"columns": ["Findings"], "rows": [
        {"document_id": "doc1", "filename": "Paper.pdf", "cells": [
            {"column": "Findings", "value": "Original", "source_ids": ["c1"]}]}]})
    envelope = {"workspace": "Alpha", "payload": sheet}
    edited = ui._edited(envelope, "Alpha", "spreadsheet", pd.DataFrame([["Changed"]], columns=["Findings"]))
    assert edited["edited_rows"] == [["Changed"]]
    assert edited["content"] == sheet["content"]
    with pytest.raises(ValueError, match="rows and columns"):
        ui._edited(envelope, "Alpha", "spreadsheet", [["A"], ["B"]])
    with pytest.raises(ValueError, match="columns"):
        ui._edited(envelope, "Alpha", "spreadsheet", pd.DataFrame([["Changed"]], columns=["Other"]))
    rendered = ui._render("Alpha", sheet, "Ready")
    assert rendered[2]["row_count"] == (1, "fixed")
    assert "Findings: [1]" in rendered[3]


def test_spreadsheet_save_updates_current_edits_and_references_without_versions(monkeypatch, spreadsheet_payload, operation):
    calls = []
    operation["tool_job"] = {"id": "pending", "cancelled": False}

    def save(workspace, payload, *, is_cancelled):
        assert not is_cancelled()
        calls.append((workspace, deepcopy(payload)))
        return dict(payload, id="current-sheet")

    monkeypatch.setattr(ui._artifacts(), "save_current_spreadsheet", save)
    monkeypatch.setattr(ui._artifacts(), "save_artifact", lambda *args: pytest.fail("Save must not create a version"))
    result = ui.on_spreadsheet_save({"workspace": "Alpha", "payload": spreadsheet_payload},
                                    pd.DataFrame([["Reviewed"]], columns=["Findings"]), "Alpha", operation)
    expected = dict(spreadsheet_payload, edited_rows=[["Reviewed"]])
    assert calls == [("Alpha", expected)]
    assert len(result) == ui.OUTPUT_COUNTS["spreadsheet"]
    assert result[0] == {"workspace": "Alpha", "payload": dict(expected, id="current-sheet")}
    assert result[2]["value"] == [["Reviewed"]]
    assert "Findings: [1]" in result[3]
    assert result[8] is None
    assert "edited_rows" not in spreadsheet_payload
    assert operation["tool_job"]["cancelled"]


@pytest.mark.parametrize("stage", ["prepare", "render", "write"])
def test_spreadsheet_save_never_publishes_after_workspace_switch(monkeypatch, spreadsheet_payload, operation, stage):
    def save(workspace, payload, *, is_cancelled):
        if stage != "write":
            pytest.fail("Stale edits must not reach storage")
        assert not is_cancelled()
        operation["revision"] += 1
        assert is_cancelled()
        return payload

    monkeypatch.setattr(ui._artifacts(), "save_current_spreadsheet", save)
    if stage != "write":
        target = "_edited" if stage == "prepare" else "_render"
        original = getattr(ui, target)

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            operation["revision"] += 1
            operation["tool_job"] = {"id": "new-workspace-job", "cancelled": False}
            return result

        monkeypatch.setattr(ui, target, changed)
    result = ui.on_spreadsheet_save({"workspace": "Alpha", "payload": spreadsheet_payload},
                                    [["Reviewed"]], "Alpha", operation)
    assert result == (ui.gr.skip(),) * ui.OUTPUT_COUNTS["spreadsheet"]
    if stage != "write":
        assert not operation["tool_job"]["cancelled"]


def test_spreadsheet_save_rejects_a_different_workspace(monkeypatch, spreadsheet_payload, operation):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("Do not access a different workspace"))
    result = ui.on_spreadsheet_save({"workspace": "Beta", "payload": spreadsheet_payload},
                                    [["Reviewed"]], "Alpha", operation)
    assert "this workspace" in result[1]


@pytest.mark.parametrize("action", ["save", "export"])
def test_spreadsheet_storage_rejects_changed_grid_shape(monkeypatch, spreadsheet_payload, operation, action):
    monkeypatch.setattr(ui._artifacts(), "save_current_spreadsheet", lambda *args, **kwargs: pytest.fail("Invalid grid cannot save"))
    monkeypatch.setattr(ui._artifacts(), "export_spreadsheet", lambda *args, **kwargs: pytest.fail("Invalid grid cannot export"))
    envelope = {"workspace": "Alpha", "payload": spreadsheet_payload}
    if action == "save":
        result = ui.on_spreadsheet_save(envelope, [["A"], ["B"]], "Alpha", operation)
        assert "rows and columns" in result[1]
    else:
        result = ui.on_spreadsheet_export(envelope, [["A"], ["B"]], "Spreadsheet", "csv", "Alpha", operation)
        assert "rows and columns" in result[0] and result[1] is None


def test_spreadsheet_save_failure_keeps_visible_edits(monkeypatch, spreadsheet_payload, operation):
    def failed(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(ui._artifacts(), "save_current_spreadsheet", failed)
    result = ui.on_spreadsheet_save({"workspace": "Alpha", "payload": spreadsheet_payload},
                                    [["Reviewed"]], "Alpha", operation)
    assert "Disk full" in result[1]
    assert result[0] == result[2] == ui.gr.skip()


def test_spreadsheet_metadata_is_escaped_for_preview():
    assert ui._literal_markdown("![image](https://example.test)\n# Heading") == (
        r"\!\[image\]\(https://example\.test\) \# Heading")


def test_reference_rejects_a_different_workspace(payload, operation):
    assert ui.on_reference("0", {"workspace": "Beta", "payload": payload}, "Alpha", operation) == ("", None)


def test_reference_error_does_not_overwrite_a_new_workspace(monkeypatch, payload, operation):
    from rag_chatbot import workspaces

    def unavailable(*args):
        operation["revision"] += 1
        raise ValueError("Old workspace unavailable")

    monkeypatch.setattr(workspaces, "workspace_path", unavailable)
    assert ui.on_reference("0", {"workspace": "Alpha", "payload": payload}, "Alpha", operation) == (
        ui.gr.skip(), ui.gr.skip())


def test_reference_rechecks_workspace_after_file_lookup(monkeypatch, payload, operation, tmp_path):
    from pathlib import Path
    from rag_chatbot import workspaces
    monkeypatch.setattr(workspaces, "workspace_path", lambda *args: tmp_path)

    def is_file(path):
        operation["revision"] += 1
        return True

    monkeypatch.setattr(Path, "is_file", is_file)
    assert ui.on_reference("0", {"workspace": "Alpha", "payload": payload}, "Alpha", operation) == (
        ui.gr.skip(), ui.gr.skip())


def test_source_change_cancels_tools_as_well_as_chat(operation):
    from rag_chatbot import chatbot
    operation["tool_job"] = {"id": "job", "cancelled": False}
    chatbot.on_source_selection([], operation)
    assert operation["tool_job"]["cancelled"]
    assert operation["answer_revision"] == 1


def test_workspace_switch_marks_pending_tool_cancelled(monkeypatch, operation):
    from rag_chatbot import chatbot
    monkeypatch.setattr(chatbot, "get_service", lambda _: SimpleNamespace(cancel_ingestion=lambda: None))
    operation["tool_job"] = {"id": "job", "cancelled": False}
    chatbot.on_prepare_workspace_switch("Alpha", operation)
    assert operation["tool_job"]["cancelled"] and operation["switching"]


def test_output_reset_does_not_clear_a_newly_started_job(operation):
    operation["tool_job"] = {"id": "new", "workspace": "Alpha", "cancelled": False}
    assert ui.on_reset("Alpha", operation) == (ui.gr.skip(),) * ui.TOTAL_OUTPUT_COUNT


def test_output_reset_rechecks_new_jobs_after_loading(monkeypatch, operation):
    def load(*args):
        operation["tool_job"] = {"id": "new", "workspace": "Alpha", "cancelled": False}
        return None

    monkeypatch.setattr(ui._artifacts(), "load_current_spreadsheet", load)
    assert ui.on_reset("Alpha", operation) == (ui.gr.skip(),) * ui.TOTAL_OUTPUT_COUNT


def test_workspace_reset_clears_old_outputs_while_generation_is_blocked(monkeypatch, operation):
    operation["switching"] = True
    result = ui.on_workspace_reset("Beta", operation)
    assert result[0] is None and result[ui.OUTPUT_COUNTS["summary"]] is None
    assert operation["switching"]
    assert "Wait for" in list(generate(operation))[0][1]


def test_programmatic_selection_update_cancels_active_tool(monkeypatch, operation):
    from rag_chatbot import chatbot
    documents = [{"id": identifier, "name": identifier, "status": "ready", "chunk_count": 1}
                 for identifier in ("doc1", "doc2")]
    service = SimpleNamespace(settings=SimpleNamespace(fingerprint=lambda: "current"),
        store=SimpleNamespace(collections=lambda: ["Research"], list_documents=lambda *args: documents))
    monkeypatch.setattr(chatbot, "get_service", lambda *args: service)
    operation.update(selected_ids=["doc1"], known_ready_ids=["doc1"], tool_job={"id": "job", "cancelled": False})
    chatbot._library_updates("Alpha", "Research", operation_state=operation)
    assert operation["selected_ids"] == ["doc1", "doc2"]
    assert operation["tool_job"]["cancelled"]


def test_tool_ui_build_is_lazy_and_uses_explicit_workspace(monkeypatch):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("No storage access while building UI"))
    monkeypatch.setattr(ui, "_enabled", lambda _: pytest.fail("No plugin access while building UI"))
    with ui.gr.Blocks() as app:
        workspace = ui.gr.State(None)
        operation = ui.gr.State({"revision": 0})
        sources = ui.gr.CheckboxGroup(choices=[])
        collection = ui.gr.Textbox()
        model = ui.gr.Dropdown(choices=["gpt-6-luna"], value="gpt-6-luna")
        reasoning = ui.gr.Dropdown(choices=["medium"], value="medium")
        controls = ui.build_tools(workspace, sources, collection, model, reasoning, operation)
    assert list(controls.controls) == ["summary", "spreadsheet", "presentation", "graphics", "plot", "mind_maps", "flashcards", "quizzes", "compare", "evidence_check"]
    assert len(controls.outputs) == ui.TOTAL_OUTPUT_COUNT
    for tool in controls.controls.values():
        fn = app.fns[tool.generation["id"]]
        assert workspace in fn.inputs and sources in fn.inputs and operation in fn.inputs
        assert tool.payload_state in fn.outputs
    assert controls.controls["summary"].payload_state is not controls.controls["spreadsheet"].payload_state
    assert controls.controls["summary"].preview.latex_delimiters == ui.LATEX_DELIMITERS
    summary = controls.controls["summary"]
    assert summary.instructions.value == ui.DEFAULT_SUMMARY_INSTRUCTIONS
    assert summary.preview.height == summary.preview.max_height == 480
    assert summary.filename.value == "Summary"
    assert summary.main_column.scale == 3 and summary.sidebar.scale == 1
    assert not summary.editor.visible and not summary.save_button.visible
    assert summary.edit_button.visible and not summary.download.visible
    assert not hasattr(summary, "saved") and not hasattr(summary, "load_button")
    assert summary.download_event["trigger_after"] == summary.export_event["id"]
    assert summary.download_event["trigger_only_on_success"] is True
    download_fn = app.fns[summary.download_event["id"]]
    assert download_fn.js == ui.SUMMARY_DOWNLOAD_JS and download_fn.inputs == [summary.download]
    labels = [component["props"].get("label") for component in app.config["components"]]
    assert "Saved Summary outputs" not in labels and "Saved Spreadsheet outputs" not in labels
    spreadsheet = controls.controls["spreadsheet"]
    assert spreadsheet.instructions.value == "Please help me to generate a spreadsheet for the documents"
    assert spreadsheet.main_column.scale == 3 and spreadsheet.sidebar.scale == 1
    assert spreadsheet.filename.value == "Spreadsheet" and spreadsheet.export_button.value == "Export"
    assert spreadsheet.editor.interactive and spreadsheet.save_button.visible
    assert spreadsheet.save_button.value == "Save"
    assert not hasattr(spreadsheet, "saved") and not hasattr(spreadsheet, "load_button")
    save_fn = app.fns[spreadsheet.save_event["id"]]
    assert save_fn.fn == ui.on_spreadsheet_save
    assert save_fn.inputs == [spreadsheet.payload_state, spreadsheet.editor, workspace, operation]
    assert save_fn.outputs == spreadsheet.outputs
    export_fn = app.fns[spreadsheet.export_event["id"]]
    assert export_fn.inputs == [spreadsheet.payload_state, spreadsheet.editor, spreadsheet.filename,
                                spreadsheet.output_format, workspace, operation]
    assert not spreadsheet.download.visible
    assert spreadsheet.download_event["trigger_after"] == spreadsheet.export_event["id"]
    assert spreadsheet.download_event["trigger_only_on_success"] is True
    spreadsheet_download = app.fns[spreadsheet.download_event["id"]]
    assert spreadsheet_download.js == ui.SPREADSHEET_DOWNLOAD_JS
    assert spreadsheet_download.inputs == [spreadsheet.download]

    children = {}
    def walk(node):
        children[node["id"]] = [child["id"] for child in node.get("children", [])]
        for child in node.get("children", []):
            walk(child)
    walk(app.config["layout"])
    def descendants(component_id):
        for child in children.get(component_id, []):
            yield child
            yield from descendants(child)
    types = {component["id"]: component["type"] for component in app.config["components"]}
    for tool in (summary, spreadsheet):
        assert tool.excerpt.lines == tool.excerpt.max_lines == 6
        sidebar_ids = list(descendants(tool.sidebar._id))
        assert tool.references_group._id in sidebar_ids and tool.export_group._id in sidebar_ids
        export_ids = list(descendants(tool.export_group._id))
        assert not any(types[component_id] == "row" for component_id in export_ids)
        assert export_ids.index(tool.filename._id) < export_ids.index(tool.output_format._id) < export_ids.index(tool.export_button._id)
        assert tool.editor._id in list(descendants(tool.main_column._id))
    spreadsheet_main = list(descendants(spreadsheet.main_column._id))
    assert spreadsheet_main.index(spreadsheet.editor._id) < spreadsheet_main.index(spreadsheet.save_button._id)
    mind_map = controls.controls["mind_maps"]
    assert mind_map.instructions.value == "Please help me create a mind map of the documents"
    assert mind_map.filename.value == "Mind Map"
    assert mind_map.output_format.choices == [("svg", "svg"), ("markdown", "markdown"), ("json", "json")]
    assert mind_map.main_column.scale == 3 and mind_map.sidebar.scale == 1
    assert mind_map.preview.max_height == 520
    assert mind_map.zoom.minimum == 0.5 and mind_map.zoom.maximum == 2.0
    assert not hasattr(mind_map, "reference") and not hasattr(mind_map, "saved")
    assert not any(types[item] == "accordion" for item in descendants(mind_map.body._id))
    assert mind_map.download_event["trigger_after"] == mind_map.export_event["id"]
    assert mind_map.download_event["trigger_only_on_success"] is True
    assert app.fns[mind_map.download_event["id"]].js == ui.MIND_MAP_DOWNLOAD_JS
    tabs = [component["props"]["label"] for component in app.config["components"] if component["type"] == "tabitem"]
    assert tabs == ["Summary", "Spreadsheet", "Presentation", "Graphics", "Plot", "Mind Map", "Flashcards", "Quizzes", "Compare", "Evidence Check"]
