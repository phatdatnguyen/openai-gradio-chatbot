"""Document library and grounded-chat UI, with lazy service initialization."""

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
from gradio.context import get_blocks_context
import pandas as pd

from chatbot import config as model_config
from chatbot.log import logger

LIBRARY_COLUMNS = ["Document", "Status", "Chunks", "Error"]
SOURCE_COLUMNS = ["Reference", "Document", "Location"]
DEFAULT_COLLECTION = "Default"
VIEWER_OUTPUT_COUNT = 10
REVIEW_OUTPUT_COUNT = VIEWER_OUTPUT_COUNT + 1
UPLOAD_OUTPUT_COUNT = 4


def get_service(workspace):
    if not workspace:
        raise ValueError("Create or open a workspace first.")
    from .service import get_service as service_factory
    return service_factory(workspace)


def _workspaces():
    from . import workspaces
    return workspaces


def _history_store():
    from . import history
    return history


def _display(history):
    return [{"role": entry["role"], "content": entry.get("content", "")}
            for entry in history or []]


def _source_updates(sources):
    sources = list(sources or [])
    frame = pd.DataFrame([
        [f"[{item['number']}]", item.get("filename", ""), item.get("locator", "")]
        for item in sources
    ], columns=SOURCE_COLUMNS)
    choices = [(f"[{item['number']}] {item.get('filename', '')} — {item.get('locator', '')}",
                str(index)) for index, item in enumerate(sources)]
    return frame, gr.update(choices=choices, value=None)


def _library_updates(workspace, collection=None, selected_ids=None, operation_state=None, *, reset=False):
    service = get_service(workspace)
    collections = service.store.collections()
    current = collection if collection in collections else (
        collections[0] if collections else DEFAULT_COLLECTION
    )
    collection_choices = collections or [DEFAULT_COLLECTION]
    documents = service.store.list_documents(current)
    fingerprint = service.settings.fingerprint()
    statuses = {
        item["id"]: ("reindex needed" if item["status"] == "ready"
                     and item.get("fingerprint", fingerprint) != fingerprint else item["status"])
        for item in documents
    }
    frame = pd.DataFrame([
        [item["name"], statuses[item["id"]], item.get("chunk_count", 0),
         item.get("error", "")]
        for item in documents
    ], columns=LIBRARY_COLUMNS)
    choices = [(item["name"] if statuses[item["id"]] == "ready" else
                f"{item['name']} ({statuses[item['id']]})", item["id"]) for item in documents]
    allowed = {value for _, value in choices}
    ready = {document_id for document_id, status in statuses.items() if status == "ready"}
    if reset or selected_ids is None and (operation_state is None or "selected_ids" not in operation_state):
        selected = ready
    else:
        selected = set(operation_state.get("selected_ids", selected_ids or [])
                       if operation_state is not None else selected_ids or [])
        if operation_state is not None:
            selected |= ready - set(operation_state.get("known_ready_ids", ready))
    selected = [value for _, value in choices if value in selected and value in allowed]
    if operation_state is not None:
        operation_state["selected_ids"] = selected
        operation_state["known_ready_ids"] = list(ready)
    return (frame, gr.update(choices=collection_choices, value=current),
            gr.update(choices=choices, value=selected))


def on_source_selection(selected_ids, operation_state, history=None, pending=None):
    if operation_state.get("switching", False):
        return (gr.skip(),) * 11
    operation_state["selected_ids"] = list(dict.fromkeys(selected_ids or []))
    operation_state["answer_revision"] = operation_state.get("answer_revision", 0) + 1
    return on_stop_chat(history, pending) if pending else (gr.skip(),) * 11


def on_select_all_sources(workspace, collection, operation_state, history=None, pending=None):
    if operation_state.get("switching", False):
        return (gr.skip(),) * 12
    service = get_service(workspace)
    fingerprint = service.settings.fingerprint()
    selected = [item["id"] for item in service.store.list_documents(collection)
                if item["status"] == "ready" and item.get("fingerprint", fingerprint) == fingerprint]
    return gr.update(value=selected), *on_source_selection(selected, operation_state, history, pending)


def on_clear_sources(operation_state, history=None, pending=None):
    if operation_state.get("switching", False):
        return (gr.skip(),) * 12
    return gr.update(value=[]), *on_source_selection([], operation_state, history, pending)


def _save_upload_edit(filename, selected, names, *, dirty=False):
    if selected in names:
        entry = names[selected]
        value = filename if filename is not None else entry["name"]
        if dirty or value != entry["name"]:
            entry["dirty"] = True
        entry["name"] = value


def on_upload_change(paths, filename, selected, editor_path, upload_names):
    paths = list(paths or [])
    names = upload_names.setdefault("names", {})
    _save_upload_edit(filename, editor_path, names)
    names = {path: names.get(path, {"name": Path(path).name, "dirty": False}) for path in paths}
    selected = selected if selected in names else next(iter(names), None)
    upload_names.update(names=names, selected=selected)
    return (upload_names, gr.update(value=names[selected]["name"] if selected else "", interactive=bool(paths)),
            gr.update(choices=[(f"{index + 1}. {Path(path).name}", path) for index, path in enumerate(paths)],
                      value=selected, visible=len(paths) > 1), selected or "")


def on_upload_name(filename, editor_path, upload_names):
    _save_upload_edit(filename, editor_path, upload_names.get("names", {}), dirty=True)
    return upload_names


def on_upload_selection(selected, filename, editor_path, upload_names):
    names = upload_names.get("names", {})
    _save_upload_edit(filename, editor_path, names)
    upload_names["selected"] = selected if selected in names else None
    return upload_names, names[selected]["name"] if selected in names else "", selected if selected in names else ""


def _upload_display_names(paths, filename, selected, upload_names, editor_path):
    # The click carries the current textbox even if its input callback has not
    # run yet. Full temporary paths distinguish uploads with the same basename.
    if editor_path != selected or selected and selected not in paths:
        raise ValueError("Wait for the selected upload's filename to appear, then index again.")
    names = {path: dict(entry) for path, entry in (upload_names or {}).get("names", {}).items()
             if path in paths}
    if selected in paths:
        names.setdefault(selected, {"name": Path(selected).name, "dirty": False})
        _save_upload_edit(filename, selected, names)
    return [names[path]["name"] if names.get(path, {}).get("dirty") else None for path in paths]


def _empty_viewer(status=""):
    return (None, None, gr.update(value=None, visible=False),
            gr.update(value="", visible=False), gr.update(value=[], visible=False),
            gr.update(value="", visible=False), gr.update(value=1, label="Page", maximum=None), "",
            gr.update(choices=[], value=None, visible=False), status)


def _reset_viewer(operation_state=None, status=""):
    if operation_state is not None:
        operation_state["viewer_revision"] = operation_state.get("viewer_revision", 0) + 1
    return _empty_viewer(status)


def _review_updates(workspace, document_id=None, operation_state=None, *, reset=False):
    if operation_state is not None:
        document_id = None if reset else operation_state.get("review_document_id", document_id)
    documents = get_service(workspace).store.list_documents()
    choices = [(document["name"], document["id"]) for document in documents
               if document["status"] != "deleted"]
    selected = document_id if document_id in {value for _, value in choices} else None
    if operation_state is not None:
        operation_state["review_document_id"] = selected
    viewer = (gr.skip(),) * VIEWER_OUTPUT_COUNT if selected else _reset_viewer(operation_state)
    return (gr.update(choices=choices, value=selected), *viewer)


def on_refresh(workspace, collection=None, selected_ids=None, review_id=None, operation_state=None):
    try:
        updates = _library_updates(workspace, collection, selected_ids, operation_state)
        return (*updates, "Library refreshed.", *_review_updates(workspace, review_id, operation_state))
    except Exception as exc:
        logger.exception("Refreshing the RAG library failed")
        return (gr.skip(), gr.skip(), gr.skip(), f"Could not refresh library: {exc}",
                *((gr.skip(),) * REVIEW_OUTPUT_COUNT))


def on_initial_load(operation_state=None):
    try:
        return on_open_workspace(_workspaces().default_workspace(), None, operation_state=operation_state)
    except Exception as exc:
        logger.exception("Opening the initial RAG workspace failed")
        return _workspace_error(f"Could not open workspace: {exc}")


def _workspace_error(message, answer=None):
    return ((gr.skip(), gr.skip(), message) + (gr.skip(),) * 10
            + ((gr.skip(),) * 11 if answer is None else answer)
            + (gr.skip(),) * (REVIEW_OUTPUT_COUNT + UPLOAD_OUTPUT_COUNT))


def on_prepare_workspace_switch(workspace, operation_state):
    # Gradio passes this session's State object by reference. Invalidate any
    # generator still inside a blocking request before changing visible state.
    operation_state["revision"] += 1
    operation_state["switching"] = True
    if workspace:
        try:
            get_service(workspace).cancel_ingestion()
        except Exception:
            logger.exception("Canceling indexing before a workspace switch failed")
    return operation_state, "Opening workspace...", gr.update(interactive=False)


def on_finish_workspace_switch(operation_state):
    # This runs only after Gradio has applied the open callback's workspace,
    # conversation and reference outputs, so new jobs cannot capture old state.
    operation_state["revision"] += 1
    operation_state["switching"] = False
    return operation_state, gr.update(interactive=True)


def on_open_workspace(name, current_workspace, current_history=None, pending=None, operation_state=None):
    try:
        canonical = _workspaces().workspace_path(name, create=True).name
        service = get_service(canonical)
        selection = {}
        library = _library_updates(canonical, operation_state=selection, reset=True)
        saved = _history_store().list_chats(data_dir=service.settings.data_dir)
        if operation_state is not None:
            operation_state.update(selection)
        return (canonical, gr.update(choices=_workspaces().list_workspaces(), value=canonical),
                f"Workspace: {canonical}", *library, "Library refreshed.",
                gr.update(choices=saved, value=None), None, DEFAULT_COLLECTION, "", "RAG chat", "",
                *_answer_update([], [], question=""),
                *_review_updates(canonical, operation_state=operation_state, reset=True),
                {}, gr.update(value="", interactive=False), gr.update(choices=[], value=None, visible=False), "")
    except Exception as exc:
        logger.exception("Opening a RAG workspace failed")
        stopped = on_stop_chat(current_history, pending) if pending is not None else None
        return _workspace_error(f"Could not open workspace: {exc}", stopped)
    finally:
        if operation_state is not None:
            # Also invalidate work submitted between the switch request and
            # this serialized commit; it must never render into the new chat.
            operation_state["revision"] += 1


def _current_operation(operation_state, revision):
    return (operation_state["revision"] == revision
            and not operation_state.get("switching", False))


def on_index(paths, collection, use_vision, workspace, operation_state, review_id=None,
             filename=None, upload_names=None, selected_upload=None, editor_path=None):
    revision = operation_state["revision"]
    unchanged = (gr.skip(),) * (3 + REVIEW_OUTPUT_COUNT)
    if operation_state.get("switching", False):
        yield "Wait for the workspace to open before indexing.", *unchanged
        return
    if not paths:
        yield "Upload at least one document before indexing.", *unchanged
        return
    collection = (collection or "").strip()
    if not collection:
        yield "Open a workspace before indexing.", *unchanged
        return
    try:
        display_names = _upload_display_names(paths, filename, selected_upload, upload_names, editor_path)
    except ValueError as exc:
        yield str(exc), *unchanged
        return
    try:
        service = get_service(workspace)
        with closing(service.ingest(paths, collection, use_vision=use_vision, display_names=display_names)) as progress:
            for message in progress:
                if not _current_operation(operation_state, revision):
                    return
                yield str(message), *unchanged
        if _current_operation(operation_state, revision):
            yield ("Indexing finished. Check each document's status below.",
                   *_library_updates(workspace, collection, operation_state=operation_state),
                   *_review_updates(workspace, review_id, operation_state))
    except Exception as exc:
        logger.exception("RAG document indexing failed")
        if _current_operation(operation_state, revision):
            updates = on_refresh(workspace, collection, review_id=review_id, operation_state=operation_state)
            yield f"Indexing failed: {exc}", *updates[:3], *updates[4:]


def on_stop_index(workspace):
    try:
        get_service(workspace).cancel_ingestion()
        return "Stopping after the current request…"
    except Exception as exc:
        logger.exception("Stopping RAG indexing failed")
        return f"Could not stop indexing: {exc}"


def on_delete_documents(selected_ids, collection, workspace, review_id=None, operation_state=None):
    if not selected_ids:
        return (gr.skip(), gr.skip(), gr.skip(), "Select documents before removing them.",
                *((gr.skip(),) * REVIEW_OUTPUT_COUNT))
    try:
        service = get_service(workspace)
        available = {item["id"] for item in service.store.list_documents(collection)}
        if any(item not in available for item in selected_ids):
            raise ValueError("The selected documents do not belong to this workspace. Refresh the library.")
        for document_id in selected_ids:
            service.store.delete_document(document_id)
        return (*_library_updates(workspace, collection, operation_state=operation_state),
                "Selected documents removed from search. Original files remain available for saved citations.",
                *_review_updates(workspace, review_id, operation_state))
    except Exception as exc:
        logger.exception("Removing RAG documents failed")
        return (gr.skip(), gr.skip(), gr.skip(), f"Could not remove documents: {exc}",
                *((gr.skip(),) * REVIEW_OUTPUT_COUNT))


def on_model_change(model):
    choices = model_config.reasoning_effort_choices(model)
    return gr.update(choices=choices or [model_config.NO_REASONING_EFFORT],
                     value="medium" if "medium" in choices else model_config.NO_REASONING_EFFORT,
                     interactive=bool(choices))


def _answer_update(history, sources, question=None, status="", display=None, pending=None):
    table, selector = _source_updates(sources)
    return (history, _display(history) if display is None else display, sources, table, selector,
            gr.skip() if question is None else question, status, None, None, "", pending)


def on_answer(question, history, collection, selected_ids, model, reasoning_effort, workspace, operation_state):
    revision = operation_state["revision"]
    answer_revision = operation_state.get("answer_revision", 0)

    def still_current():
        return (_current_operation(operation_state, revision)
                and operation_state.get("answer_revision", 0) == answer_revision)

    if operation_state.get("switching", False):
        yield (gr.skip(),) * 6 + ("Wait for the workspace to open before sending a question.",) + (gr.skip(),) * 4
        return
    question = (question or "").strip()
    previous = list(history or [])
    if not question:
        yield (gr.skip(),) * 6 + ("Enter a question about your documents.",) + (gr.skip(),) * 4
        return
    if not selected_ids:
        yield (gr.skip(),) * 6 + ("No sources selected. Check at least one ready source before sending a question.",) + (gr.skip(),) * 4
        return
    pending_turn = {"history": previous, "question": question, "workspace": workspace}
    pending = _display(previous) + [{"role": "user", "content": question}]
    yield _answer_update(previous, [], question="", status="Searching your documents…", display=pending,
                         pending=pending_turn)
    try:
        if not still_current():
            return
        latest = None
        service = get_service(workspace)
        with closing(service.answer(
            question, previous, collection, document_ids=list(selected_ids),
            model=model, reasoning_effort=reasoning_effort,
        )) as replies:
            for updated, sources in replies:
                if not still_current():
                    return
                latest = (updated, sources)
                yield _answer_update(updated, sources, status="", pending=pending_turn)
        if not still_current():
            return
        if latest is None:
            raise RuntimeError("No answer was returned. Please try again.")
        yield _answer_update(*latest)
    except Exception as exc:
        logger.exception("RAG answer failed")
        if not still_current():
            return
        pending.append({"role": "assistant", "content": f"Request failed: {exc}\n\n"
                        "The question above was not added to the conversation."})
        yield _answer_update(previous, [], status=f"Request failed: {exc}", display=pending)


def on_new_chat():
    return _answer_update([], [])


def on_stop_chat(history, pending):
    if pending is None:
        return (gr.skip(),) * 6 + ("No answer is running.",) + (gr.skip(),) * 4
    previous = pending["history"]
    display = _display(previous) + [{"role": "user", "content": pending["question"]},
        {"role": "assistant", "content": "Generation stopped. This turn was not added to the conversation."}]
    return _answer_update(previous, [], status="Generation stopped.", display=display)


def _safe_source_path(value, workspace):
    if not value:
        return None
    root = Path(get_service(workspace).settings.data_dir).resolve()
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("The source is outside the document library.")
    return str(path) if path.is_file() else None


def on_select_source(selection, sources, workspace):
    if selection is None:
        return None, None, ""
    try:
        index = int(selection)
        if index < 0 or index >= len(sources or []):
            raise ValueError("Select an available reference.")
        source = sources[index]
        original = _safe_source_path(source.get("source_path"), workspace)
        image = _safe_source_path(source.get("image_path"), workspace)
        text = source.get("text", "")
        if not original:
            text = "The original file is unavailable.\n\n" + text
        return original, image, text
    except Exception as exc:
        logger.exception("Opening a RAG source failed")
        return None, None, f"Could not open reference: {exc}"


def preview_document(path, **kwargs):
    from .viewer import preview_document as preview
    return preview(path, **kwargs)


def on_review_selection(document_id, operation_state):
    operation_state["review_document_id"] = document_id
    return _reset_viewer(operation_state, "Opening document..." if document_id else "")


def on_preview_document(document_id, unit, sheet, workspace, operation_state):
    revision = operation_state["revision"]
    if operation_state.get("switching", False):
        return (gr.skip(),) * (VIEWER_OUTPUT_COUNT - 1) + ("Wait for the workspace to open.",)
    if operation_state.get("review_document_id", document_id) != document_id:
        return (gr.skip(),) * VIEWER_OUTPUT_COUNT
    operation_state["viewer_revision"] = operation_state.get("viewer_revision", 0) + 1
    viewer_revision = operation_state["viewer_revision"]

    def still_current():
        return (_current_operation(operation_state, revision)
                and operation_state["viewer_revision"] == viewer_revision
                and operation_state.get("review_document_id", document_id) == document_id)

    try:
        if not document_id:
            return _empty_viewer("Select a source to review.")
        service = get_service(workspace)
        document = service.store.get_document(document_id)
        if not document or document["status"] == "deleted":
            raise ValueError("This document is not available in the active workspace. Refresh the sources.")
        path = _safe_source_path(document["path"], workspace)
        if path is None:
            raise FileNotFoundError("The original document is unavailable.")
        preview = preview_document(path, data_dir=service.settings.data_dir, unit=unit or 1, sheet=sheet)
        if not still_current():
            return (gr.skip(),) * VIEWER_OUTPUT_COUNT
        # Deletion can finish while a local converter is running. A deleted
        # source must not be restored by that converter's delayed response.
        document = service.store.get_document(document_id)
        if not document or document["status"] == "deleted":
            return _empty_viewer("This document was removed from the workspace.")
        frame = pd.DataFrame(preview.table, columns=preview.headers or None)
        state = {"workspace": workspace, "document_id": document_id,
                 "unit": preview.unit, "sheet": preview.sheet}
        return (state, path, gr.update(value=preview.image, visible=preview.image is not None),
                gr.update(value=preview.html, visible=bool(preview.html)),
                gr.update(value=frame, visible=bool(preview.table or preview.headers)),
                gr.update(value=preview.text, visible=bool(preview.text)),
                gr.update(value=preview.unit, label=preview.unit_label, maximum=max(1, preview.unit_count)),
                f"{preview.unit_label} {preview.unit} of {preview.unit_count}",
                gr.update(choices=preview.sheets, value=preview.sheet, visible=bool(preview.sheets)),
                preview.status)
    except Exception as exc:
        logger.exception("Reviewing a RAG document failed")
        if not still_current():
            return (gr.skip(),) * VIEWER_OUTPUT_COUNT
        return _empty_viewer(f"Could not review document: {exc}")


def on_review_document(document_id, workspace, operation_state):
    return on_preview_document(document_id, 1, None, workspace, operation_state)


def on_preview_previous(document_id, unit, sheet, workspace, operation_state):
    return on_preview_document(document_id, max(1, int(unit or 1) - 1), sheet, workspace, operation_state)


def on_preview_next(document_id, unit, sheet, workspace, operation_state):
    return on_preview_document(document_id, int(unit or 1) + 1, sheet, workspace, operation_state)


def on_preview_sheet(document_id, sheet, workspace, operation_state):
    return on_preview_document(document_id, 1, sheet, workspace, operation_state)


def on_save_chat(history, name, workspace):
    try:
        store = _history_store()
        data_dir = get_service(workspace).settings.data_dir
        saved = store.save_chat(history, name, data_dir=data_dir)
        return f"Saved {saved}.", gr.update(choices=store.list_chats(data_dir=data_dir), value=saved)
    except Exception as exc:
        logger.exception("Saving RAG chat failed")
        return f"Could not save chat: {exc}", gr.skip()


def on_load_chat(name, current_history, pending, workspace):
    try:
        if not name:
            raise ValueError("Select a saved chat first.")
        history = _history_store().load_chat(name, data_dir=get_service(workspace).settings.data_dir)
        sources = next((entry.get("sources", []) for entry in reversed(history)
                        if entry.get("role") == "assistant"), [])
        return _answer_update(history, sources, status=f"Loaded {name}.")
    except Exception as exc:
        logger.exception("Loading RAG chat failed")
        if pending is not None:
            # Loading also cancels generation. If the file cannot be opened,
            # remove that canceled partial turn instead of leaving it pending.
            stopped = list(on_stop_chat(current_history, pending))
            stopped[6] = f"Could not load chat: {exc}"
            return tuple(stopped)
        return (gr.skip(),) * 6 + (f"Could not load chat: {exc}",) + (gr.skip(),) * 4


def on_toggle_history(visible):
    visible = not visible
    return gr.update(visible=visible), visible


def build_tab():
    """Build workspace-scoped RAG controls without opening storage at import."""
    history_state = gr.State([])
    state = history_state
    sources_state = gr.State([])
    pending_state = gr.State(None)
    viewer_state = gr.State(None)
    upload_names_state = gr.State({})
    workspace_state = gr.State(None)
    operation_state = gr.State({"revision": 0, "switching": False, "viewer_revision": 0})
    history_column_state = gr.State(True)

    with gr.Row() as outer_row:
        with gr.Column(scale=1) as sources_column:
            workspace_name = gr.Dropdown(label="Workspace", choices=[], allow_custom_value=True,
                                         info="Choose a saved workspace or type a new name, then open it.")
            workspace_button = gr.Button("Create / Open workspace", variant="primary")
            workspace_status = gr.Markdown("Opening workspace...")
            with gr.Accordion("Sources", open=True):
                upload = gr.File(label="Upload documents", file_count="multiple", type="filepath")
                upload_filename = gr.Textbox(label="Filename", interactive=False,
                    info="Edit the name before indexing. Keep the original file extension.")
                upload_selector = gr.Dropdown(label="Uploaded file to rename", choices=[], visible=False)
                upload_editor_path = gr.Textbox(value="", visible=False)
                index_collection = gr.Textbox(value=DEFAULT_COLLECTION, visible=False)
                use_vision = gr.Checkbox(label="Read scanned pages and images with vision", value=False,
                                        info="Uses the OpenAI API for image descriptions and OCR.")
                with gr.Row():
                    index_button = gr.Button("Index documents", variant="primary")
                    index_stop_button = gr.Button("Stop indexing", variant="stop")
                index_status = gr.Markdown()
                collection_filter = gr.Dropdown(label="Search collection", choices=[DEFAULT_COLLECTION],
                                                value=DEFAULT_COLLECTION, visible=False)
                document_filter = gr.CheckboxGroup(label="Sources for chat and tools", choices=[], value=[],
                    elem_classes=["rag-source-checkboxes"],
                    info="Only checked, ready sources are searched. Nothing checked means no sources.")
                with gr.Row():
                    select_all_sources_button = gr.Button("Select all ready")
                    clear_sources_button = gr.Button("Clear")
                with gr.Row():
                    refresh_button = gr.Button("Refresh library")
                    delete_button = gr.Button("Remove checked sources", variant="stop")
                library_table = gr.Dataframe(headers=LIBRARY_COLUMNS, value=[], interactive=False,
                                             max_height=260, wrap=True)
                library_status = gr.Markdown()
                gr.Markdown("Indexing sends extracted text to OpenAI for embeddings. Questions send "
                            "matching passages. Optional vision also sends document images.")
            with gr.Accordion("Document viewer", open=True) as viewer_group:
                review_document = gr.Dropdown(label="Review document", choices=[],
                    info="Review an original source without changing which documents are searched.")
                viewer_original = gr.File(label="Download original", interactive=False)
                with gr.Row():
                    viewer_previous = gr.Button("Previous", min_width=75)
                    viewer_unit = gr.Number(label="Page", value=1, minimum=1, precision=0, min_width=85,
                                            info="Press Enter to jump.")
                    viewer_next = gr.Button("Next", min_width=75)
                viewer_unit_info = gr.Markdown()
                viewer_sheet = gr.Dropdown(label="Sheet", choices=[], visible=False)
                viewer_image = gr.Image(label="Document page", interactive=False, visible=False)
                viewer_html = gr.HTML(visible=False)
                viewer_table = gr.Dataframe(value=[], interactive=False, visible=False, wrap=True)
                viewer_text = gr.Textbox(label="Document text", lines=16, interactive=False, visible=False)
                viewer_status = gr.Markdown()
        with gr.Column(scale=2) as right_column:
            with gr.Row(equal_height=True) as top_row:
                with gr.Column(scale=1) as chat_column:
                    chatbot = gr.Chatbot(buttons=["copy"], min_height=500, latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": r"\[", "right": r"\]", "display": True},
                        {"left": r"\(", "right": r"\)", "display": False},
                    ])
                    with gr.Row():
                        new_chat_button = gr.Button("New RAG chat")
                        stop_button = gr.Button("Stop", variant="stop")
                        toggle_history_column_button = gr.Button("Toggle chat history")
                    status = gr.Markdown()
                    with gr.Accordion("Prompt", open=True) as prompt_group:
                        with gr.Row():
                            question = gr.Textbox(label="Message", placeholder="Ask about your documents...", scale=4)
                            send_button = gr.Button("Send", variant="primary", scale=1, min_width=80)
                        model = gr.Dropdown(label="Model", value=model_config.DEFAULT_MODEL,
                                            choices=[item for item in model_config.MODEL_CHOICES
                                                     if not model_config.requires_web_search(item)])
                        _efforts = model_config.reasoning_effort_choices(model_config.DEFAULT_MODEL)
                        reasoning_effort = gr.Dropdown(label="Reasoning effort",
                            choices=_efforts or [model_config.NO_REASONING_EFFORT],
                            value="medium" if "medium" in _efforts else model_config.NO_REASONING_EFFORT,
                            interactive=bool(_efforts))
                        gr.Markdown("The assistant is instructed to use the selected sources, cite evidence, "
                                    "and say when evidence is insufficient.")
                with gr.Column(scale=1) as details_column:
                    with gr.Group() as history_column:
                        saved_chat = gr.Dropdown(label="Saved RAG chats", choices=[])
                        chat_name = gr.Textbox(label="Chat name", value="RAG chat")
                        with gr.Row() as history_buttons:
                            load_button = gr.Button("Load RAG chat")
                            save_button = gr.Button("Save RAG chat")
                        save_status = gr.Markdown()
                    with gr.Accordion("References from the latest answer", open=True) as references_group:
                        sources_table = gr.Dataframe(headers=SOURCE_COLUMNS, value=[], interactive=False,
                                                     max_height=260, wrap=True)
                        source_selector = gr.Dropdown(label="Open reference", choices=[])
                        original_file = gr.File(label="Original document", interactive=False)
                        source_image = gr.Image(label="Source page or image", interactive=False)
                        source_text = gr.Textbox(label="Source excerpt", lines=12, interactive=False)
            with gr.Row() as bottom_row:
                with gr.Column(scale=1) as tools_column:
                    with gr.Accordion("Tools", open=True):
                        gr.Markdown("Workspace tools will appear here in a future update.")

    answer_outputs = [history_state, chatbot, sources_state, sources_table, source_selector,
                      question, status, original_file, source_image, source_text, pending_state]
    viewer_outputs = [viewer_state, viewer_original, viewer_image, viewer_html, viewer_table,
                      viewer_text, viewer_unit, viewer_unit_info, viewer_sheet, viewer_status]
    review_outputs = [review_document, *viewer_outputs]
    library_outputs = [library_table, collection_filter, document_filter, library_status, *review_outputs]
    workspace_outputs = [workspace_state, workspace_name, workspace_status, *library_outputs[:4],
                         saved_chat, upload, index_collection, index_status, chat_name, save_status,
                         *answer_outputs, *review_outputs,
                         upload_names_state, upload_filename, upload_selector, upload_editor_path]
    # Slow local rendering uses its own queue so opening another workspace can
    # invalidate it immediately without waiting for an Office/PDF converter.
    _storage_queue = {"concurrency_id": "rag-workspace-storage", "concurrency_limit": 1}
    _viewer_queue = {"concurrency_id": "rag-document-viewer", "concurrency_limit": 1}
    _viewer_inputs = [review_document, viewer_unit, viewer_sheet, workspace_state, operation_state]
    review_selection_event = review_document.input(on_review_selection,
        [review_document, operation_state], viewer_outputs, queue=False, trigger_mode="always_last")
    review_event = review_selection_event.then(on_review_document,
        [review_document, workspace_state, operation_state], viewer_outputs,
        trigger_mode="always_last", **_viewer_queue)
    viewer_unit_event = viewer_unit.submit(on_preview_document, _viewer_inputs, viewer_outputs, **_viewer_queue)
    viewer_previous_event = viewer_previous.click(on_preview_previous, _viewer_inputs, viewer_outputs, **_viewer_queue)
    viewer_next_event = viewer_next.click(on_preview_next, _viewer_inputs, viewer_outputs, **_viewer_queue)
    viewer_sheet_event = viewer_sheet.input(on_preview_sheet,
        [review_document, viewer_sheet, workspace_state, operation_state], viewer_outputs, **_viewer_queue)
    viewer_events = [review_event, viewer_unit_event, viewer_previous_event, viewer_next_event, viewer_sheet_event]
    # A cancellation request can arrive after this input's chained preview.
    # Leave preview jobs queued: their selection/revision guards reject stale
    # work. Cancel only navigation here; workspace changes still cancel all.
    review_document.input(fn=None, cancels=viewer_events[1:], queue=False)
    send_event = gr.on(triggers=[question.submit, send_button.click], fn=on_answer,
                      inputs=[question, history_state, collection_filter, document_filter,
                              model, reasoning_effort, workspace_state, operation_state],
                      outputs=answer_outputs, trigger_mode="once", concurrency_limit=1)
    document_filter.input(on_source_selection,
        [document_filter, operation_state, history_state, pending_state], answer_outputs,
        queue=False, cancels=[send_event], trigger_mode="always_last")
    select_all_sources_button.click(on_select_all_sources,
        [workspace_state, collection_filter, operation_state, history_state, pending_state],
        [document_filter, *answer_outputs], queue=False, cancels=[send_event])
    clear_sources_button.click(on_clear_sources, [operation_state, history_state, pending_state],
        [document_filter, *answer_outputs], queue=False, cancels=[send_event])
    stop_button.click(on_stop_chat, [history_state, pending_state], answer_outputs, cancels=[send_event])
    new_chat_button.click(on_new_chat, outputs=answer_outputs, cancels=[send_event])
    toggle_history_column_button.click(on_toggle_history, history_column_state,
                                       [history_column, history_column_state])
    model.change(on_model_change, model, reasoning_effort)
    refresh_button.click(on_refresh, [workspace_state, collection_filter, document_filter,
                                     review_document, operation_state], library_outputs, **_storage_queue)
    upload.change(on_upload_change,
        [upload, upload_filename, upload_selector, upload_editor_path, upload_names_state],
        [upload_names_state, upload_filename, upload_selector, upload_editor_path], queue=False)
    upload_filename.input(on_upload_name, [upload_filename, upload_editor_path, upload_names_state],
                          upload_names_state, queue=False, trigger_mode="always_last")
    upload_selector.input(on_upload_selection,
        [upload_selector, upload_filename, upload_editor_path, upload_names_state],
        [upload_names_state, upload_filename, upload_editor_path], queue=False, trigger_mode="always_last")
    index_event = index_button.click(on_index, [upload, index_collection, use_vision,
                                              workspace_state, operation_state, review_document,
                                              upload_filename, upload_names_state, upload_selector, upload_editor_path],
                                    [index_status, library_table, collection_filter, document_filter, *review_outputs])
    index_stop_button.click(on_stop_index, workspace_state, index_status,
                            cancels=[index_event], queue=False)
    delete_button.click(on_delete_documents, [document_filter, collection_filter, workspace_state,
                                             review_document, operation_state], library_outputs, **_storage_queue)
    source_selector.input(on_select_source, [source_selector, sources_state, workspace_state],
                          [original_file, source_image, source_text], **_storage_queue)
    save_button.click(on_save_chat, [history_state, chat_name, workspace_state],
                      [save_status, saved_chat], **_storage_queue)
    load_button.click(on_load_chat, [saved_chat, history_state, pending_state, workspace_state],
                      answer_outputs, cancels=[send_event], **_storage_queue)
    workspace_cancel_event = workspace_button.click(on_prepare_workspace_switch,
        [workspace_state, operation_state], [operation_state, workspace_status, workspace_button],
        cancels=[send_event, index_event, *viewer_events], queue=False)
    workspace_open_event = workspace_cancel_event.then(on_open_workspace,
        [workspace_name, workspace_state, history_state, pending_state, operation_state],
        workspace_outputs, **_storage_queue)
    workspace_ready_event = workspace_open_event.then(on_finish_workspace_switch, operation_state,
        [operation_state, workspace_button], queue=False)
    load_event = get_blocks_context().root_block.load(on_initial_load, inputs=[operation_state],
                                                     outputs=workspace_outputs, **_storage_queue)
    return SimpleNamespace(**{name: value for name, value in locals().items()
                              if not name.startswith("_")})
