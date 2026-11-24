"""Workspace descriptions remain local, scoped, and safe under concurrent edits."""

import asyncio
import json

import pytest

from rag_chatbot import workspace_metadata as metadata, workspace_ui, workspaces


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    workspaces.create_workspace("Alpha")
    workspaces.create_workspace("Beta")
    return workspaces.workspace_path("Alpha")


def test_description_roundtrip_isolated_and_read_does_not_create_metadata(workspace):
    assert metadata.load("Alpha") == {"version": 1, "revision": 0, "description": ""}
    assert not (workspace / "workspace.json").exists()
    first = metadata.save("Alpha", "XRD study\nCompare samples α and β.", 0)
    assert metadata.load("Alpha") == first and first["revision"] == 1
    assert metadata.load("Beta")["description"] == ""
    with pytest.raises(ValueError, match="another session"):
        metadata.save("Alpha", "Stale edit", 0)
    assert metadata.load("Alpha") == first


@pytest.mark.parametrize("bad", [None, 5, "x" * 2001, "bad\x00text", "\ud800"])
def test_description_validation_never_creates_bad_files(workspace, bad):
    with pytest.raises(ValueError, match="description"):
        metadata.save("Alpha", bad, 0)
    assert not (workspace / "workspace.json").exists()


@pytest.mark.parametrize("when", [1, 2])
def test_cancelled_save_preserves_existing_description_and_cleans_temp(workspace, when):
    before = metadata.save("Alpha", "Keep", 0)
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) == when

    with pytest.raises(RuntimeError, match="cancelled"):
        metadata.save("Alpha", "Obsolete", 1, is_cancelled=cancelled)
    assert metadata.load("Alpha") == before
    assert not list(workspace.glob(".description-*"))


def test_linked_settings_rejected_even_inside_workspace(workspace, monkeypatch):
    metadata.save("Alpha", "Keep", 0)
    monkeypatch.setattr(metadata, "_linked", lambda path: True)
    with pytest.raises(ValueError, match="ordinary file"):
        metadata.load("Alpha")
    with pytest.raises(ValueError, match="ordinary file"):
        metadata.save("Alpha", "Replacement", 1)


@pytest.mark.parametrize("raw", [b"[", b"[0]", b"x" * (metadata.MAX_BYTES + 1),
    json.dumps({"version": True, "revision": 1, "description": "Bad"}).encode()])
def test_corrupt_settings_are_reported_without_overwrite(workspace, raw):
    path = workspace / "workspace.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        metadata.load("Alpha")
    with pytest.raises(ValueError):
        metadata.save("Alpha", "Replacement", 1)
    assert path.read_bytes() == raw


def test_ui_switch_restore_and_stale_save_guards(workspace):
    operation = {"revision": 3, "switching": True}
    _, _, state, epoch = workspace_ui.on_reset("Alpha", operation)
    assert state["workspace_revision"] == 4
    assert workspace_ui.on_save("Too early", state, "Alpha", operation, epoch) == (workspace_ui.gr.skip(),) * 4
    operation.update(revision=4, switching=False)
    saved = workspace_ui.on_save("Project goals", state, "Alpha", operation, epoch)
    assert "saved" in saved[1] and saved[2]["revision"] == 1
    workspace_ui.on_reset("Beta", operation)
    assert workspace_ui.on_save("Old editor", saved[2], "Alpha", operation, epoch) == (workspace_ui.gr.skip(),) * 4
    assert metadata.load("Alpha")["description"] == "Project goals"
    assert metadata.load("Beta")["description"] == ""


def test_ui_late_reset_discards_old_result(workspace, monkeypatch):
    operation = {"revision": 1}

    def load(_):
        operation["revision"] += 1
        return {"revision": 0, "description": "Old"}

    monkeypatch.setattr(metadata, "load", load)
    assert workspace_ui.on_reset("Alpha", operation) == (workspace_ui.gr.skip(),) * 4


@pytest.mark.parametrize("epoch", [None, "", "old-epoch", 12, {}])
def test_save_rejects_missing_or_stale_client_epoch_even_with_current_state(workspace, epoch):
    operation = {"revision": 1}
    _description, _message, state, _epoch = workspace_ui.on_reset("Alpha", operation)
    result = workspace_ui.on_save("Old form text", state, "Alpha", operation, epoch)
    assert result == (workspace_ui.gr.skip(),) * workspace_ui.OUTPUT_COUNT
    assert metadata.load("Alpha")["revision"] == 0


@pytest.mark.parametrize("callback", [workspace_ui.on_save, workspace_ui.on_reload])
def test_queued_description_event_with_old_browser_epoch_and_new_server_states_is_ignored(workspace, monkeypatch, callback):
    from gradio.state_holder import SessionState

    gr = workspace_ui.gr
    with gr.Blocks() as app:
        active_workspace = gr.State("Alpha")
        operation_state = gr.State({"revision": 1})
        controls = workspace_ui.build(active_workspace, operation_state)
    state = SessionState(app)
    operation = {"revision": 1}
    _description, _message, _old_snapshot, old_epoch = workspace_ui.on_reset("Alpha", operation)
    operation["revision"] += 1
    _description, _message, new_snapshot, new_epoch = workspace_ui.on_reset("Beta", operation)
    state[active_workspace._id] = "Beta"
    state[operation_state._id] = operation
    state[controls.state._id] = new_snapshot
    assert old_epoch != new_epoch and controls.workspace_epoch.elem_id == "rag-workspace-description-epoch"

    def unexpected(*args, **kwargs):
        pytest.fail("The stale queued event must be rejected before accessing workspace metadata")

    monkeypatch.setattr(metadata, "save", unexpected)
    monkeypatch.setattr(metadata, "load", unexpected)
    function = next(item for item in app.fns.values() if item.fn is callback)
    assert function.inputs[-1] is controls.workspace_epoch and not controls.workspace_epoch.stateful
    values = {controls.description._id: "Text typed in Alpha", controls.workspace_epoch._id: old_epoch}

    async def invoke():
        return await app.process_api(function, [values.get(component._id) for component in function.inputs], state=state)

    result = asyncio.run(invoke())
    assert all(value is None if component.stateful else value in (
        gr.skip(), {"__type__": "update", "props": gr.skip()})
        for component, value in zip(function.outputs, result["data"]))
    assert state[controls.state._id] == new_snapshot
    assert operation["description_token"] == new_epoch


def test_reload_can_retry_failed_read_but_rejects_old_epoch(workspace, monkeypatch):
    operation = {"revision": 1}
    original = metadata.load

    def fail(_):
        raise OSError("Temporary failure")

    monkeypatch.setattr(metadata, "load", fail)
    _description, message, state, epoch = workspace_ui.on_reset("Alpha", operation)
    assert "Temporary failure" in message and state is None
    monkeypatch.setattr(metadata, "load", original)
    result = workspace_ui.on_reload(state, "Alpha", operation, epoch)
    assert result[2]["workspace"] == "Alpha" and result[3] != epoch
    assert workspace_ui.on_reload(result[2], "Alpha", operation, epoch) == (workspace_ui.gr.skip(),) * 4


def test_save_honors_size_limit_before_creating_a_temporary_file(workspace, monkeypatch):
    monkeypatch.setattr(metadata, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="supported size"):
        metadata.save("Alpha", "Project", 0)
    assert not (workspace / "workspace.json").exists()
    assert not list(workspace.glob(".description-*"))
