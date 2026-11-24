"""Workspace description controls, independent of source/chat persistence."""

from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from chatbot.log import logger
from . import workspace_metadata


OUTPUT_COUNT = 4


def _current(state, workspace, operation, workspace_epoch):
    return (isinstance(state, dict) and state.get("workspace") == workspace
            and state.get("workspace_revision") == operation.get("revision", 0)
            and isinstance(workspace_epoch, str) and bool(workspace_epoch)
            and state.get("token") == workspace_epoch
            and state.get("token") == operation.get("description_token")
            and not operation.get("switching"))


def on_reset(workspace, operation):
    revision = operation.get("revision", 0)
    token = uuid4().hex
    operation["description_token"] = token
    operation["description_workspace"] = workspace
    try:
        payload = workspace_metadata.load(workspace) if workspace else {"description": "", "revision": 0}
        if operation.get("revision", 0) != revision or operation.get("description_token") != token:
            return (gr.skip(),) * OUTPUT_COUNT
        state = {"workspace": workspace, "workspace_revision": revision + bool(operation.get("switching")),
                 "token": token, "revision": payload["revision"]}
        return payload["description"], "", state, token
    except (ValueError, OSError, RuntimeError) as exc:
        logger.warning("Could not load workspace description: %s", exc)
        if operation.get("revision", 0) != revision or operation.get("description_token") != token:
            return (gr.skip(),) * OUTPUT_COUNT
        return "", f"Could not load description: {exc}", None, token


def on_save(description, state, workspace, operation, workspace_epoch):
    # State inputs are resolved by Gradio when a queued job executes. The
    # hidden client value travels with the click, so old editor text cannot
    # be paired with the new workspace's freshly reset server-side State.
    if not _current(state, workspace, operation, workspace_epoch):
        return (gr.skip(),) * OUTPUT_COUNT
    try:
        value = workspace_metadata.save(workspace, description, state["revision"],
                                        is_cancelled=lambda: not _current(state, workspace, operation, workspace_epoch))
        if not _current(state, workspace, operation, workspace_epoch):
            return (gr.skip(),) * OUTPUT_COUNT
        return gr.skip(), "Description saved.", {**state, "revision": value["revision"]}, gr.skip()
    except (ValueError, OSError, RuntimeError) as exc:
        if not _current(state, workspace, operation, workspace_epoch):
            return (gr.skip(),) * OUTPUT_COUNT
        return gr.skip(), f"Could not save description: {exc}", gr.skip(), gr.skip()


def on_reload(state, workspace, operation, workspace_epoch):
    if not _current(state, workspace, operation, workspace_epoch):
        # A failed read has no editable snapshot but remains explicitly
        # retryable from that same visible workspace/epoch.
        if not (state is None and isinstance(workspace_epoch, str) and workspace_epoch
                and workspace_epoch == operation.get("description_token")
                and workspace == operation.get("description_workspace")
                and not operation.get("switching")):
            return (gr.skip(),) * OUTPUT_COUNT
    return on_reset(workspace, operation)


def build(workspace, operation):
    with gr.Accordion("Workspace description", open=False):
        description = gr.Textbox(label="Description", lines=2, max_lines=4,
            info="A personal description of this project; it is not sent to the model.")
        with gr.Row():
            save = gr.Button("Save description", size="sm")
            refresh = gr.Button("Reload description", size="sm")
        status = gr.Markdown()
        state = gr.State(None)
        workspace_epoch = gr.Textbox(value="", visible=False, elem_id="rag-workspace-description-epoch")
    outputs = [description, status, state, workspace_epoch]
    queue = {"concurrency_id": "rag-workspace-storage", "concurrency_limit": 1}
    save.click(on_save, [description, state, workspace, operation, workspace_epoch], outputs, **queue)
    refresh.click(on_reload, [state, workspace, operation, workspace_epoch], outputs, **queue)
    return SimpleNamespace(description=description, status=status, state=state, save=save,
                           refresh=refresh, outputs=outputs, workspace_epoch=workspace_epoch,
                           reset_callback=on_reset)
