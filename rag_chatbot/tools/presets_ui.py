"""Instruction preset controls shared by all Studio tools, without generation."""

from functools import partial
import json
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from chatbot.log import logger
from . import presets


OUTPUTS_PER_TOOL = 5
RESET_OUTPUT_COUNT = 50


def _tool_ids():
    from .ui import TOOL_LABELS
    return tuple(TOOL_LABELS)


def _current(state, tool_id, workspace, operation_state, client_epoch):
    # Gradio resolves State when a queued callback executes, while textbox
    # values were captured when the browser submitted it. The client epoch
    # prevents those old values from acting on a new workspace/preset view.
    return (bool(workspace) and isinstance(state, dict) and state.get("tool_id") == tool_id
            and state.get("workspace") == workspace
            and isinstance(client_epoch, str) and bool(client_epoch)
            and state.get("epoch") == client_epoch
            and state.get("revision") == operation_state.get("revision", 0)
            and not operation_state.get("switching", False))


def _view(records, tool_id, workspace, revision, selected=None, message=""):
    record = next((item for item in records if item["id"] == selected), None)
    state = {"tool_id": tool_id, "workspace": workspace, "revision": revision,
             "records": {item["id"]: item["revision"] for item in records},
             "selected": record["id"] if record else None,
             "selected_revision": record["revision"] if record else None,
             "epoch": json.dumps([workspace, revision, uuid4().hex], ensure_ascii=False)}
    return (gr.update(choices=[(item["name"], item["id"]) for item in records], value=state["selected"]),
            record["name"] if record else "", message, state["epoch"], state)


def _error(message):
    return gr.skip(), gr.skip(), message, gr.skip(), gr.skip()


def reset_all(workspace, operation_state):
    """Workspace reset outputs are independent of saved Studio result outputs."""
    revision = operation_state.get("revision", 0)
    switching = bool(operation_state.get("switching", False))
    expected_revision = revision + switching
    outputs = []
    for tool_id in _tool_ids():
        try:
            records = presets.list_presets(tool_id) if workspace else []
            outputs.extend(_view(records, tool_id, workspace, expected_revision))
        except (ValueError, OSError) as exc:
            logger.warning("Could not load instruction presets for %s: %s", tool_id, exc)
            outputs.extend(_view([], tool_id, workspace, expected_revision, message=f"Could not read presets: {exc}"))
    if (operation_state.get("revision", 0) != revision
            or bool(operation_state.get("switching", False)) != switching):
        return (gr.skip(),) * len(outputs)
    return tuple(outputs)


def on_refresh(tool_id, workspace, operation_state, state, client_epoch=None, evt: gr.SelectData = None):
    if evt is not None and not evt.selected:
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    if not _current(state, tool_id, workspace, operation_state, client_epoch):
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    revision = operation_state.get("revision", 0)
    try:
        records = presets.list_presets(tool_id)
        if not _current(state, tool_id, workspace, operation_state, client_epoch):
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        selected = state.get("selected") if _current(state, tool_id, workspace, operation_state, client_epoch) else None
        record = next((item for item in records if item["id"] == selected), None)
        unchanged = record is not None and record["revision"] == state.get("selected_revision")
        result = list(_view(records, tool_id, workspace, revision, selected if unchanged else None))
        if _current(state, tool_id, workspace, operation_state, client_epoch) and (not selected or unchanged):
            result[1] = gr.skip()  # Preserve an unfinished name edit on tab reopen.
        return tuple(result)
    except (ValueError, OSError) as exc:
        logger.warning("Could not refresh instruction presets: %s", exc)
        if not _current(state, tool_id, workspace, operation_state, client_epoch):
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        return _error(f"Could not read presets: {exc}")


def on_select(tool_id, selected, workspace, operation_state, state, client_epoch=None):
    if not _current(state, tool_id, workspace, operation_state, client_epoch):
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    if selected is None:
        return on_new(tool_id, workspace, operation_state, state, client_epoch)
    if not isinstance(selected, str) or selected not in state.get("records", {}):
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    if selected == state.get("selected"):
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    try:
        records = presets.list_presets(tool_id)
        if not _current(state, tool_id, workspace, operation_state, client_epoch):
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        if not any(item["id"] == selected for item in records):
            return _error("This preset was deleted. Refresh the list to continue.")
        return _view(records, tool_id, workspace, state["revision"], selected,
                     "Click Apply preset to use its instructions. Save preset updates the selected preset.")
    except (ValueError, OSError) as exc:
        if not _current(state, tool_id, workspace, operation_state, client_epoch):
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        return _error(f"Could not select preset: {exc}")


def on_new(tool_id, workspace, operation_state, state, client_epoch=None):
    if not _current(state, tool_id, workspace, operation_state, client_epoch):
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    state = {**state, "selected": None, "selected_revision": None,
             "epoch": json.dumps([workspace, state["revision"], uuid4().hex], ensure_ascii=False)}
    return (gr.update(value=None), "", "Enter a name, then save the current instruction text as a new preset.",
            state["epoch"], state)


def on_apply(tool_id, workspace, operation_state, state, client_epoch=None):
    if not _current(state, tool_id, workspace, operation_state, client_epoch):
        return gr.skip(), gr.skip()
    selected = state.get("selected")
    if selected not in state.get("records", {}):
        return gr.skip(), "Select an instruction preset first."
    try:
        record = presets.get_preset(tool_id, selected)
        if not _current(state, tool_id, workspace, operation_state, client_epoch):
            return gr.skip(), gr.skip()
        if record is None or record["revision"] != state.get("selected_revision"):
            return gr.skip(), "This preset changed in another session. Refresh and select it again before applying."
        return record["instructions"], "Preset instructions applied. Generate when you are ready."
    except (ValueError, OSError) as exc:
        if not _current(state, tool_id, workspace, operation_state, client_epoch):
            return gr.skip(), gr.skip()
        return gr.skip(), f"Could not apply preset: {exc}"


def on_save(tool_id, name, instructions, workspace, operation_state, state, client_epoch=None):
    if not _current(state, tool_id, workspace, operation_state, client_epoch):
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    current = lambda: _current(state, tool_id, workspace, operation_state, client_epoch)
    try:
        record = presets.save_preset(tool_id, name, instructions, preset_id=state.get("selected"),
                                     expected_revision=state.get("selected_revision"),
                                     is_cancelled=lambda: not current())
        records = presets.list_presets(tool_id)
        if not current():
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        return _view(records, tool_id, workspace, state["revision"], record["id"],
                     "Preset saved. It is available for this tool in every workspace.")
    except presets.PresetCancelled:
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    except (ValueError, OSError) as exc:
        logger.warning("Could not save instruction preset: %s", exc)
        if not current():
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        return _error(f"Could not save preset: {exc}")


def on_delete(tool_id, workspace, operation_state, state, client_epoch=None):
    if not _current(state, tool_id, workspace, operation_state, client_epoch):
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    if state.get("selected") not in state.get("records", {}):
        return _error("Select an instruction preset to delete.")
    current = lambda: _current(state, tool_id, workspace, operation_state, client_epoch)
    try:
        presets.delete_preset(tool_id, state["selected"], expected_revision=state.get("selected_revision"),
                              is_cancelled=lambda: not current())
        records = presets.list_presets(tool_id)
        if not current():
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        return _view(records, tool_id, workspace, state["revision"], message="Preset deleted. Current tool instructions are unchanged.")
    except presets.PresetCancelled:
        return (gr.skip(),) * OUTPUTS_PER_TOOL
    except (ValueError, OSError) as exc:
        if not current():
            return (gr.skip(),) * OUTPUTS_PER_TOOL
        return _error(f"Could not delete preset: {exc}")


def build(tool_id, instructions, workspace, operation_state):
    """Build with no storage access; parents register lazy tab refresh events."""
    state = gr.State(None)
    epoch = gr.Textbox(value="", visible=False, interactive=False, elem_id=f"rag-preset-epoch-{tool_id}")
    with gr.Accordion("Instruction presets", open=False) as accordion:
        gr.Markdown("Reuse instruction text across workspaces. Presets do not change sources, model settings, or saved outputs.")
        with gr.Row():
            selector = gr.Dropdown(label="Saved instruction preset", choices=[], scale=3)
            apply_button = gr.Button("Apply preset", scale=1)
            refresh_button = gr.Button("Refresh presets", scale=1)
        name = gr.Textbox(label="Preset name", max_lines=1, placeholder="For example: Experimental methods")
        with gr.Row():
            new_button = gr.Button("New preset")
            save_button = gr.Button("Save preset")
            delete_button = gr.Button("Delete preset")
        status = gr.Markdown()
    outputs = [selector, name, status, epoch, state]
    scope = [workspace, operation_state, state, epoch]
    queue = {"concurrency_id": "studio-instruction-presets", "concurrency_limit": 1}
    refresh_callback = partial(on_refresh, tool_id)
    refresh_button.click(refresh_callback, scope, outputs, **queue)
    # A delayed dropdown blur can carry a removed choice; membership checks
    # belong in the handler so it can discard that value before any file read.
    selector.input(partial(on_select, tool_id), [selector, *scope], outputs,
                   preprocess=False, trigger_mode="always_last", **queue)
    new_button.click(partial(on_new, tool_id), scope, outputs, **queue)
    apply_button.click(partial(on_apply, tool_id), scope, [instructions, status], **queue)
    save_button.click(partial(on_save, tool_id), [name, instructions, *scope], outputs, **queue)
    delete_button.click(partial(on_delete, tool_id), scope, outputs, **queue)
    return SimpleNamespace(selector=selector, name=name, status=status, state=state, epoch=epoch, accordion=accordion,
                           apply_button=apply_button, refresh_button=refresh_button, new_button=new_button,
                           save_button=save_button, delete_button=delete_button, outputs=outputs,
                           refresh_callback=refresh_callback, inputs=scope)
