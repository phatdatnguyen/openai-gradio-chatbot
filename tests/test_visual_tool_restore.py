"""Reopening visual tools restores only their durable result, without paid work."""

from copy import deepcopy
from functools import partial

import pytest

from chatbot import config
from rag_chatbot.tools import graphics_ui, plot_ui, ui


@pytest.fixture(params=["graphics", "plot"])
def visual(request, monkeypatch, tmp_path):
    tool_id = request.param
    view = graphics_ui if tool_id == "graphics" else plot_ui
    payload = {
        "tool_id": tool_id, "title": "Saved result", "id": "saved-result",
        "sources": [{"id": "c1", "document_id": "doc1", "filename": "Dataset.txt",
                     "locator": "Section 1", "text": "A: 12.5 Pa", "source_path": "sources/doc1.txt"}],
        "documents": [{"id": "doc1"}],
        "coverage": {"documents": 1, "chunks": 1, "processed_chunks": 1},
    }
    if tool_id == "graphics":
        payload["content"] = {
            "prompt": "Illustrate the evidence", "caption": "Saved graphic caption",
            "source_ids": ["c1"], "asset_path": "artifacts/graphics/assets/saved.png",
            "output_format": "png",
        }
    else:
        payload["content"] = {
            "chart_type": "bar", "x_label": "Sample", "y_label": "Pressure", "x_unit": "", "y_unit": "Pa",
            "series": [{"name": "Measured", "points": [{"label": "A", "x": None, "y": 12.5,
                        "source_ids": ["c1"], "evidence": "A: 12.5 Pa"}]}], "limitations": [],
        }
    store = ui._artifacts()
    image_path = tmp_path / "saved.png"
    monkeypatch.setattr(store, "graphics_image_path", lambda workspace, value: image_path)
    monkeypatch.setattr(plot_ui, "_preview", lambda value: "<div>Saved plot preview</div>")
    monkeypatch.setattr(store, "load_current_" + tool_id, lambda workspace: deepcopy(payload))
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Restoration must not generate"))
    monkeypatch.setattr(config, "get_client", lambda: pytest.fail("Restoration must not access OpenAI"))
    return tool_id, view, payload, image_path


@pytest.fixture
def operation():
    return {"revision": 3, "switching": False, "selected_ids": ["doc1"],
            "summary": {"token": "unrelated-summary"}}


def _start(visual, operation):
    tool_id, view, payload, _ = visual
    result = view.render_output("Alpha", payload, "Previous result")
    assert view.publish(result[0], operation)
    return result


def _skipped(tool_id):
    return (ui.gr.skip(),) * ui.OUTPUT_COUNTS[tool_id]


def test_open_restores_saved_image_or_plot_with_new_session_token(visual, operation):
    tool_id, view, payload, image_path = visual
    old = _start(visual, operation)
    summary = deepcopy(operation["summary"])
    result = ui.on_visual_tool_open(tool_id, "Alpha", operation)
    assert len(result) == ui.OUTPUT_COUNTS[tool_id]
    assert result[0]["payload"] == payload and result[0]["token"] != old[0]["token"]
    assert result[0]["workspace"] == "Alpha" and "Restored" in result[1]
    assert result[2] == (str(image_path) if tool_id == "graphics" else "<div>Saved plot preview</div>")
    assert result[-1] == ui.gr.skip()  # Keep the user's chosen export filename.
    assert view._matches(result[0], "Alpha", operation)
    assert operation["summary"] == summary
    if tool_id == "graphics":
        assert result[3] == payload["content"]["caption"]
    else:
        assert result[3]["value"][0][3] == 12.5


@pytest.mark.parametrize("job", [
    {"token": "active", "workspace": "Alpha", "cancelled": False},
    {"token": "stopping", "workspace": "Alpha", "cancelled": True},
    {"token": "other-workspace", "workspace": "Beta", "cancelled": True},
])
def test_open_during_running_or_stopping_job_does_not_read_or_publish(visual, operation, monkeypatch, job):
    tool_id = visual[0]
    operation["tool_job"] = job
    before = deepcopy(operation)
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("A pending job must block restoration"))
    assert ui.on_visual_tool_open(tool_id, "Alpha", operation) == _skipped(tool_id)
    assert operation == before


@pytest.mark.parametrize("blocked", ["deselected", "switching"])
def test_inactive_selection_never_reads_storage(visual, operation, monkeypatch, blocked):
    tool_id = visual[0]
    event = ui.gr.SelectData(None, {"index": 0, "value": ui.TOOL_LABELS[tool_id], "selected": False})
    if blocked == "switching":
        operation["switching"] = True
        event = None
    before = deepcopy(operation)
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("Inactive selection must not read storage"))
    assert ui.on_visual_tool_open(tool_id, "Alpha", operation, event) == _skipped(tool_id)
    assert operation == before


@pytest.mark.parametrize("workspace", [None, "Empty"])
def test_empty_workspace_clears_only_this_tool(visual, operation, monkeypatch, workspace):
    tool_id, view, _, _ = visual
    _start(visual, operation)
    other = "plot" if tool_id == "graphics" else "graphics"
    operation[other] = {"token": "other-output", "workspace": "Alpha", "workspace_revision": 3}
    unrelated = deepcopy(operation[other])
    monkeypatch.setattr(ui._artifacts(), "load_current_" + tool_id, lambda workspace: None)
    result = ui.on_visual_tool_open(tool_id, workspace, operation)
    assert result == view.empty_output()
    assert tool_id not in operation and operation[other] == unrelated
    assert operation["summary"] == {"token": "unrelated-summary"}


@pytest.mark.parametrize("stage", ["load", "render"])
@pytest.mark.parametrize("change", ["revision", "switching", "new_output", "new_job", "finished_job"])
def test_late_restore_cannot_replace_switch_or_generation(visual, operation, monkeypatch, stage, change):
    tool_id, view, payload, _ = visual
    _start(visual, operation)

    def mutate():
        if change == "revision":
            operation["revision"] += 1
        elif change == "switching":
            operation["switching"] = True
        elif change == "new_output":
            operation[tool_id] = dict(operation[tool_id], token="newer-generation")
        elif change == "finished_job":
            operation["tool_job_revision"] = operation.get("tool_job_revision", 0) + 1
        else:
            operation["tool_job"] = {"token": "new-generation", "workspace": "Alpha"}

    if stage == "load":
        def load(workspace):
            mutate()
            return payload
        monkeypatch.setattr(ui._artifacts(), "load_current_" + tool_id, load)
    else:
        render = view.render_output

        def delayed_render(*args):
            result = render(*args)
            mutate()
            return result
        monkeypatch.setattr(view, "render_output", delayed_render)
    before = operation[tool_id]["token"]
    assert ui.on_visual_tool_open(tool_id, "Alpha", operation) == _skipped(tool_id)
    assert operation[tool_id]["token"] == ("newer-generation" if change == "new_output" else before)


def test_newer_open_wins_over_older_disk_read(visual, operation, monkeypatch):
    tool_id, view, payload, _ = visual
    _start(visual, operation)
    calls, latest = [], []

    def load(workspace):
        calls.append(workspace)
        if len(calls) == 1:
            latest.append(ui.on_visual_tool_open(tool_id, workspace, operation))
        return payload

    monkeypatch.setattr(ui._artifacts(), "load_current_" + tool_id, load)
    assert ui.on_visual_tool_open(tool_id, "Alpha", operation) == _skipped(tool_id)
    assert calls == ["Alpha", "Alpha"]
    assert view._matches(latest[0][0], "Alpha", operation)


@pytest.mark.parametrize("stale", [False, True])
def test_read_error_preserves_prior_preview_and_does_not_publish(visual, operation, monkeypatch, stale):
    tool_id = visual[0]
    _start(visual, operation)
    previous = deepcopy(operation[tool_id])

    def load(workspace):
        if stale:
            operation["revision"] += 1
        raise OSError("Cannot read saved output")

    monkeypatch.setattr(ui._artifacts(), "load_current_" + tool_id, load)
    result = ui.on_visual_tool_open(tool_id, "Alpha", operation)
    assert operation[tool_id] == previous
    assert all(value == ui.gr.skip() for index, value in enumerate(result) if index != 1)
    if stale:
        assert result[1] == ui.gr.skip()
    else:
        assert "Cannot read saved output" in result[1]


def test_visual_restore_wiring_is_per_tab_and_lazy(monkeypatch):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("Building tabs must not read saved outputs"))
    with ui.gr.Blocks() as app:
        workspace, selected, collection = ui.gr.State("Alpha"), ui.gr.State(["doc1"]), ui.gr.State("Default")
        model, effort, operation = ui.gr.State("gpt-6-luna"), ui.gr.State("medium"), ui.gr.State({"revision": 0})
        tools = ui.build_tools(workspace, selected, collection, model, effort, operation)
    for tool_id, tool in tools.controls.items():
        if tool_id not in {"graphics", "plot"}:
            assert not hasattr(tool, "restore_event")
            continue
        event = tool.restore_event
        callback = app.fns[event["id"]]
        assert isinstance(callback.fn, partial) and callback.fn.func is ui.on_visual_tool_open
        assert callback.fn.args == (tool_id,)
        assert callback.inputs == [workspace, operation] and callback.outputs == tool.outputs
        assert event["targets"] == [(tool.tab._id, "select")]
        assert not event["queue"] and event["trigger_mode"] == "always_last"


def test_unrelated_tools_reject_visual_restore():
    with pytest.raises(ValueError, match="does not support"):
        ui.on_visual_tool_open("summary", "Alpha", {})
