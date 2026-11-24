"""Instruction presets are reusable plain text, isolated from generated outputs."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

from rag_chatbot.tools import presets, presets_ui


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    path = tmp_path / ".cache" / "studio-presets.json"
    monkeypatch.setattr(presets, "SETTINGS_PATH", path)
    return path


@pytest.fixture
def operation():
    return {"revision": 3, "switching": False}


def _state(tool="summary", workspace="Research", revision=3, records=None, selected=None):
    return presets_ui._view(records or [], tool, workspace, revision, selected)[-1]


def _invoke(callback, *args, **kwargs):
    state = args[-1]
    if isinstance(state, dict) and "epoch" in state:
        kwargs.setdefault("client_epoch", state["epoch"])
    return callback(*args, **kwargs)


def test_empty_read_is_lazy(storage):
    assert presets.list_presets("summary") == []
    assert not storage.parent.exists()


def test_create_update_rename_delete_preserves_other_tools_and_sessions(storage):
    first = presets.save_preset("summary", "Methods", "Keep experimental detail.")
    second = presets.save_preset("plot", "Methods", "Plot all reported data.")
    third = presets.save_preset("summary", "Limitations", "Describe limitations.")
    changed = presets.save_preset("summary", "Detailed methods", "Include instruments.",
        preset_id=first["id"], expected_revision=1)
    assert changed["revision"] == 2 and changed["id"] == first["id"]
    assert [item["name"] for item in presets.list_presets("summary")] == ["Detailed methods", "Limitations"]
    assert presets.get_preset("plot", second["id"]) == second
    assert presets.get_preset("summary", second["id"]) is None
    presets.delete_preset("summary", changed["id"], expected_revision=2)
    assert presets.list_presets("summary") == [third]
    assert json.loads(storage.read_text(encoding="utf-8"))["version"] == 1
    assert not list(storage.parent.glob(".pending-*"))


def test_presets_are_data_and_copy_results_are_independent():
    instructions = "Return `<script>alert('x')</script>` as quoted source text.\nUse \\( x^2 \\)."
    saved = presets.save_preset("summary", "日本語 / methods", instructions)
    saved["instructions"] = "mutated"
    assert presets.get_preset("summary", saved["id"])["instructions"] == instructions


@pytest.mark.parametrize("name", ["", " ", "x" * 81, "x\nname", "bad\x00name", "\ud800", "😀" * 61])
def test_invalid_names_are_rejected_before_writing(storage, name):
    with pytest.raises(ValueError, match="name"):
        presets.save_preset("summary", name, "Keep it concise")
    assert not storage.exists()


@pytest.mark.parametrize("instructions", ["", "  ", "x" * 8001, None, "bad\x00instruction", "\ud800"])
def test_invalid_instruction_text_is_rejected(storage, instructions):
    with pytest.raises(ValueError, match="instructions"):
        presets.save_preset("summary", "Test", instructions)
    assert not storage.exists()


def test_names_are_normalized_and_case_insensitive_per_tool():
    saved = presets.save_preset("summary", " Cafe\u0301 ", "One")
    assert saved["name"] == "Café"
    with pytest.raises(ValueError, match="already exists"):
        presets.save_preset("summary", "CAFÉ", "Two")
    assert presets.save_preset("plot", "CAFÉ", "Three")


@pytest.mark.parametrize("action", ["update", "delete", "cross_tool"])
def test_stale_revision_or_cross_tool_mutation_does_not_overwrite_current(storage, action):
    first = presets.save_preset("summary", "Methods", "Original")
    latest = presets.save_preset("summary", "Methods", "Other session", preset_id=first["id"], expected_revision=1)
    before = storage.read_bytes()
    with pytest.raises(ValueError, match="changed in another session"):
        if action == "update":
            presets.save_preset("summary", "Methods", "Old tab", preset_id=first["id"], expected_revision=1)
        else:
            presets.delete_preset("plot" if action == "cross_tool" else "summary", first["id"],
                                  expected_revision=2 if action == "cross_tool" else 1)
    assert storage.read_bytes() == before
    assert presets.get_preset("summary", first["id"]) == latest


@pytest.mark.parametrize("tool", ["unknown", "../summary", None, []])
def test_only_trusted_tools_allowed(tool, storage):
    with pytest.raises(ValueError, match="available Studio"):
        presets.save_preset(tool, "Preset", "Instructions")
    assert not storage.exists()


@pytest.mark.parametrize("identifier", ["../other", "", "a" * 31, "b" * 33, None])
def test_preset_ids_are_not_paths(identifier):
    with pytest.raises(ValueError, match="identifier"):
        presets.get_preset("summary", identifier)


def test_count_and_byte_budgets_fail_without_replacing_existing(storage, monkeypatch):
    first = presets.save_preset("summary", "First", "One")
    before = storage.read_bytes()
    monkeypatch.setattr(presets, "MAX_PRESETS", 1)
    with pytest.raises(ValueError, match="at most"):
        presets.save_preset("summary", "Second", "Two")
    assert storage.read_bytes() == before
    monkeypatch.setattr(presets, "MAX_SETTINGS_BYTES", len(before) + 1)
    with pytest.raises(ValueError, match="4 MiB"):
        presets.save_preset("summary", "First", "x" * 7000, preset_id=first["id"], expected_revision=1)
    assert storage.read_bytes() == before


@pytest.mark.parametrize("malformed", ["{", "null", '{"version":true,"presets":[]}',
    '{"version":2,"presets":[]}', '{"version":1,"presets":{},"extra":0}'])
def test_invalid_storage_never_silently_resets(storage, malformed):
    storage.parent.mkdir()
    storage.write_text(malformed, encoding="utf-8")
    with pytest.raises(ValueError):
        presets.save_preset("summary", "Test", "Instructions")
    assert storage.read_text(encoding="utf-8") == malformed


def test_duplicate_or_invalid_stored_records_rejected(storage):
    presets.save_preset("summary", "Test", "Instructions")
    data = json.loads(storage.read_text(encoding="utf-8"))
    data["presets"].append(deepcopy(data["presets"][0]))
    storage.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        presets.list_presets("summary")


@pytest.mark.parametrize("action", ["save", "delete"])
@pytest.mark.parametrize("when", ["before_work", "before_replace"])
def test_cancelled_mutation_preserves_previous_file_and_cleans_temporary(storage, monkeypatch, action, when):
    first = presets.save_preset("summary", "Test", "Original")
    before = storage.read_bytes()
    stopped = when == "before_work"
    original_fsync = presets.os.fsync

    def cancel_after_sync(descriptor):
        nonlocal stopped
        original_fsync(descriptor)
        stopped = True

    if when == "before_replace":
        monkeypatch.setattr(presets.os, "fsync", cancel_after_sync)
    with pytest.raises(presets.PresetCancelled):
        if action == "save":
            presets.save_preset("summary", "Test", "Stale", preset_id=first["id"], expected_revision=1,
                                 is_cancelled=lambda: stopped)
        else:
            presets.delete_preset("summary", first["id"], expected_revision=1, is_cancelled=lambda: stopped)
    assert storage.read_bytes() == before
    assert not list(storage.parent.glob(".pending-*"))


def test_linked_storage_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / "private.json"
    target.write_text("private", encoding="utf-8")
    link = tmp_path / "presets.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    monkeypatch.setattr(presets, "SETTINGS_PATH", link)
    with pytest.raises(ValueError, match="linked paths"):
        presets.save_preset("summary", "Test", "Instructions")
    assert target.read_text(encoding="utf-8") == "private"


def test_reset_scopes_all_tool_states_to_workspace_final_revision(operation):
    saved = presets.save_preset("summary", "Methods", "Extract methods")
    operation["switching"] = True
    result = presets_ui.reset_all("New workspace", operation)
    assert len(result) == presets_ui.RESET_OUTPUT_COUNT == 50
    assert result[0]["choices"] == [("Methods", saved["id"])]
    assert result[4]["workspace"] == "New workspace" and result[4]["revision"] == 4
    assert all(result[index]["selected"] is None for index in range(4, 50, 5))


def test_save_apply_and_delete_do_not_generate_or_touch_saved_outputs(operation):
    initial = _state()
    saved = _invoke(presets_ui.on_save, "summary", "Methods", "Extract methods", "Research", operation, initial)
    state = saved[-1]
    assert saved[0]["value"] == state["selected"]
    assert _invoke(presets_ui.on_apply, "summary", "Research", operation, state)[0] == "Extract methods"
    reset = _invoke(presets_ui.on_new, "summary", "Research", operation, state)
    assert reset[-1]["selected"] is None
    assert len(reset) == 5  # No instruction/result/chat output is part of these mutations.
    result = _invoke(presets_ui.on_delete, "summary", "Research", operation, state)
    assert result[-1]["selected"] is None and presets.list_presets("summary") == []


def test_stale_and_wrong_workspace_ui_actions_are_ignored_before_storage(operation, monkeypatch):
    state = _state(workspace="Old workspace")
    monkeypatch.setattr(presets, "list_presets", lambda *a: pytest.fail("Stale action must not read presets"))
    monkeypatch.setattr(presets, "save_preset", lambda *a, **kw: pytest.fail("Stale action must not save presets"))
    assert _invoke(presets_ui.on_save, "summary", "Test", "Text", "Research", operation, state) == (gr.skip(),) * 5
    assert _invoke(presets_ui.on_select, "summary", "File", "Research", operation, _state()) == (gr.skip(),) * 5
    assert _invoke(presets_ui.on_apply, "summary", "Research", operation, state) == (gr.skip(),) * 2
    assert _invoke(presets_ui.on_delete, "summary", "Research", operation, state) == (gr.skip(),) * 5


def test_external_changes_block_apply_and_save_until_refreshed(operation):
    first = presets.save_preset("summary", "Test", "First")
    state = _state(records=[first], selected=first["id"])
    presets.save_preset("summary", "Test", "Second", preset_id=first["id"], expected_revision=1)
    applied = _invoke(presets_ui.on_apply, "summary", "Research", operation, state)
    assert applied[0] == gr.skip() and "changed in another session" in applied[1]
    saved = _invoke(presets_ui.on_save, "summary", "Test", "Stale", "Research", operation, state)
    assert "changed in another session" in saved[2]
    refreshed = _invoke(presets_ui.on_refresh, "summary", "Research", operation, state)
    assert refreshed[-1]["selected"] is None
    selected = _invoke(presets_ui.on_select, "summary", first["id"], "Research", operation, refreshed[-1])
    assert _invoke(presets_ui.on_apply, "summary", "Research", operation, selected[-1])[0] == "Second"


@pytest.mark.parametrize("action", ["refresh", "select", "apply", "reset"])
def test_workspace_change_during_read_discards_late_result(operation, monkeypatch, action):
    first = presets.save_preset("summary", "Test", "First")
    state = _state(records=[first], selected=None if action == "select" else first["id"])
    read = presets.list_presets

    def switched(tool):
        result = read(tool)
        operation["revision"] += 1
        return result

    monkeypatch.setattr(presets, "list_presets", switched)
    if action == "refresh":
        result = _invoke(presets_ui.on_refresh, "summary", "Research", operation, state)
    elif action == "select":
        result = _invoke(presets_ui.on_select, "summary", first["id"], "Research", operation, state)
    elif action == "apply":
        result = _invoke(presets_ui.on_apply, "summary", "Research", operation, state)
    else:
        result = presets_ui.reset_all("Research", operation)
    assert all(item == gr.skip() for item in result)


def test_tab_deselection_does_not_read_and_unchanged_blur_preserves_name(operation, monkeypatch):
    first = presets.save_preset("summary", "Test", "First")
    state = _state(records=[first], selected=first["id"])
    monkeypatch.setattr(presets, "list_presets", lambda *a: pytest.fail("Do not refresh on blur/deselection"))
    assert _invoke(presets_ui.on_refresh, "summary", "Research", operation, state, evt=SimpleNamespace(selected=False)) == (gr.skip(),) * 5
    assert _invoke(presets_ui.on_select, "summary", first["id"], "Research", operation, state) == (gr.skip(),) * 5


@pytest.mark.parametrize("action", ["refresh", "select", "apply", "save", "delete", "reset"])
def test_error_after_workspace_switch_cannot_publish_old_status(operation, monkeypatch, action):
    first = presets.save_preset("summary", "Test", "First")
    state = _state(records=[first], selected=None if action == "select" else first["id"])

    def fail_after_switch(*args, **kwargs):
        operation["revision"] += 1
        raise OSError("Old workspace read failed")

    monkeypatch.setattr(presets, "list_presets", fail_after_switch)
    monkeypatch.setattr(presets, "save_preset", fail_after_switch)
    monkeypatch.setattr(presets, "delete_preset", fail_after_switch)
    if action == "refresh":
        result = _invoke(presets_ui.on_refresh, "summary", "Research", operation, state)
    elif action == "select":
        result = _invoke(presets_ui.on_select, "summary", first["id"], "Research", operation, state)
    elif action == "apply":
        result = _invoke(presets_ui.on_apply, "summary", "Research", operation, state)
    elif action == "save":
        result = _invoke(presets_ui.on_save, "summary", "Test", "Stale", "Research", operation, state)
    elif action == "delete":
        result = _invoke(presets_ui.on_delete, "summary", "Research", operation, state)
    else:
        result = presets_ui.reset_all("Research", operation)
    assert all(item == gr.skip() for item in result)


def test_all_tools_have_presets_without_changing_existing_result_outputs(monkeypatch):
    from rag_chatbot.tools import ui

    monkeypatch.setattr(presets, "list_presets", lambda *a: pytest.fail("UI building must not read storage"))
    with gr.Blocks() as app:
        controls = ui.build_tools(gr.State("Research"), gr.CheckboxGroup(choices=[]), gr.Dropdown(choices=["Default"]),
                                  gr.Dropdown(choices=["gpt-6-luna"]), gr.Dropdown(choices=["medium"]), gr.State({"revision": 0}))
    assert len(controls.outputs) == ui.TOTAL_OUTPUT_COUNT
    assert len(controls.preset_outputs) == presets_ui.RESET_OUTPUT_COUNT
    assert controls.preset_reset_callback is presets_ui.reset_all
    assert set(controls.controls) == set(ui.TOOL_LABELS)
    for tool in controls.controls.values():
        selection = next(event for event in app.fns.values()
                         if getattr(event.fn, "func", None) is presets_ui.on_select and tool.presets.selector in event.inputs)
        assert selection.preprocess is False
        apply = next(event for event in app.fns.values()
                     if getattr(event.fn, "func", None) is presets_ui.on_apply and tool.presets.state in event.inputs)
        assert apply.outputs == [tool.instructions, tool.presets.status]


def test_gradio_preset_save_apply_roundtrip_and_stale_choice(operation, monkeypatch):
    import asyncio
    from gradio.state_holder import SessionState

    with gr.Blocks() as app:
        workspace = gr.State("Research")
        operation_component = gr.State(operation)
        instructions = gr.Textbox(value="Original instructions")
        panel = presets_ui.build("summary", instructions, workspace, operation_component)
    session = SessionState(app)
    session[panel.state._id] = _state()
    client_epoch = session[panel.state._id]["epoch"]

    async def invoke(callback, values=None):
        nonlocal client_epoch
        values = dict(values or {})
        values.setdefault(panel.epoch._id, client_epoch)
        function = next(item for item in app.fns.values() if getattr(item.fn, "func", None) is callback)
        result = await app.process_api(function, [values.get(component._id) for component in function.inputs], state=session)
        if panel.epoch in function.outputs:
            returned = result["data"][function.outputs.index(panel.epoch)]
            if isinstance(returned, str):
                client_epoch = returned
        return result

    asyncio.run(invoke(presets_ui.on_refresh))
    asyncio.run(invoke(presets_ui.on_save, {panel.name._id: "Methods", instructions._id: "Keep method details."}))
    result = asyncio.run(invoke(presets_ui.on_apply))
    assert result["data"][0] == "Keep method details."
    selected = session[panel.state._id]["selected"]
    assert presets.get_preset("summary", selected)["instructions"] == "Keep method details."

    monkeypatch.setattr(presets, "list_presets", lambda *a: pytest.fail("Invalid selection must not read storage"))
    result = asyncio.run(invoke(presets_ui.on_select, {panel.selector._id: "File"}))
    assert all(value is None if component.stateful else value == gr.skip()
               for component, value in zip(panel.outputs, result["data"]))
    assert session[panel.state._id]["selected"] == selected


@pytest.mark.parametrize("action", [presets_ui.on_save, presets_ui.on_delete, presets_ui.on_apply])
@pytest.mark.parametrize("change", ["workspace", "selected_preset"])
def test_queued_client_epoch_cannot_act_on_new_gradio_state(storage, operation, action, change):
    import asyncio
    from gradio.state_holder import SessionState

    first = presets.save_preset("summary", "First", "Original first instructions")
    second = presets.save_preset("summary", "Second", "Original second instructions")
    before = storage.read_bytes()
    records = [first, second]
    old_state = _state(records=records, selected=first["id"])
    with gr.Blocks() as app:
        workspace = gr.State("Research")
        operation_component = gr.State(operation)
        instructions = gr.Textbox(value="Default instructions")
        panel = presets_ui.build("summary", instructions, workspace, operation_component)
    session = SessionState(app)
    current_workspace = "Other workspace" if change == "workspace" else "Research"
    current_revision = 4 if change == "workspace" else 3
    session[workspace._id] = current_workspace
    session[operation_component._id] = {"revision": current_revision, "switching": False}
    session[panel.state._id] = _state(workspace=current_workspace, revision=current_revision,
                                    records=records, selected=second["id"])
    function = next(item for item in app.fns.values() if getattr(item.fn, "func", None) is action)
    # This represents a click captured before the reset but executed against
    # the new server-side gr.State. Only the client textbox epoch remains old.
    values = {panel.epoch._id: old_state["epoch"], panel.name._id: "Stale replacement",
              instructions._id: "Old browser instruction text"}

    async def execute():
        return await app.process_api(function, [values.get(component._id) for component in function.inputs], state=session)

    result = asyncio.run(execute())
    assert all(value is None if component.stateful else value == gr.skip()
               for component, value in zip(function.outputs, result["data"]))
    assert storage.read_bytes() == before
    assert session[panel.state._id]["selected"] == second["id"]


@pytest.mark.parametrize("epoch", [None, ""])
def test_actual_missing_client_epoch_never_saves(operation, epoch, storage):
    state = _state()
    assert presets_ui.on_save("summary", "Name", "Instructions", "Research", operation, state, epoch) == (gr.skip(),) * 5
    assert not storage.exists()
