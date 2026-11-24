"""Workspace research controls. Notes remain distinct from primary chat evidence."""

from types import SimpleNamespace
from uuid import uuid4

import gradio as gr
from openai import APIError

from chatbot.log import logger
from . import research


SEARCH_OUTPUT_COUNT = 8
NOTE_OUTPUT_COUNT = 16
OUTPUT_COUNT = SEARCH_OUTPUT_COUNT + NOTE_OUTPUT_COUNT
SEARCH_HEADERS = ["Reference", "Document", "Location", "Excerpt"]


def _current(operation_state, revision):
    return not operation_state.get("switching", False) and operation_state.get("revision", 0) == revision


def _revision(operation_state):
    return operation_state.get("revision", 0) + bool(operation_state.get("switching"))


def _search_empty(message=""):
    return None, message, gr.update(value=[], headers=SEARCH_HEADERS), gr.update(choices=[], value=None), "", None, gr.skip(), gr.skip()


def _search_epoch(operation_state):
    return operation_state.get("research_workspace_epoch", "") + ":" + operation_state.get("research_search_token", "")


def _client_current(client_epoch, operation_state, *, result=False):
    # State inputs are read when a queued callback starts, not when the browser
    # clicked. The hidden browser value identifies the original workspace/view.
    if client_epoch is None:  # Direct Python callers do not carry browser state.
        return True
    if not isinstance(client_epoch, str) or ":" not in client_epoch:
        return False
    return (client_epoch == _search_epoch(operation_state) if result else
            client_epoch.split(":", 1)[0] == operation_state.get("research_workspace_epoch"))


def _choices(notes):
    return [(f"{note['title']} — {research.ORIGINS[note['origin']]}", note["id"]) for note in notes]


def _reference_choices(sources):
    return [(f"[{source['number']}] {source['filename']} — {source['locator']}", str(index))
            for index, source in enumerate(sources)]


def _note_view(workspace, operation_state, notes, note=None, message=""):
    token = uuid4().hex
    export_token = uuid4().hex
    envelope = {"workspace": workspace, "revision": _revision(operation_state), "token": token, "note": note}
    operation_state["research_note_token"] = token
    operation_state["research_note_ids"] = [item["id"] for item in notes]
    operation_state["research_note_export"] = {"token": export_token,
        "inputs": (note["title"] if note else "", note["body"] if note else "", "Note", "md")}
    provenance = "Human note. Notes are saved separately and are not automatically used as chat evidence."
    if note:
        provenance = f"Origin: **{research.ORIGINS[note['origin']]}**. "
        if note["origin"] == "ai":
            provenance += "This is a model interpretation, not independent evidence. "
        if note["web_search"]:
            provenance += "The original answer used web search. "
        if note["user_edited"]:
            provenance += "Edited by the user; retained references may not support added text. "
        provenance += "Notes are not automatically used as chat evidence."
    sources = note["sources"] if note else []
    return (envelope, gr.update(choices=_choices(notes), value=note["id"] if note else None),
            note["title"] if note else "", note["body"] if note else "", note["body"] if note else "",
            provenance, message, gr.update(choices=_reference_choices(sources), value=None), "", None, "", token,
            None, "Note", "md", export_token)


def _note_error(message):
    values = [gr.skip()] * NOTE_OUTPUT_COUNT
    values[6] = message
    return tuple(values)


def _matches(envelope, workspace, operation_state, client_epoch=None):
    return (isinstance(envelope, dict) and envelope.get("workspace") == workspace
            and envelope.get("token") == operation_state.get("research_note_token")
            and (client_epoch is None or client_epoch == envelope.get("token"))
            and _current(operation_state, envelope.get("revision")))


def on_reset(workspace, operation_state):
    revision = operation_state.get("revision", 0)
    try:
        notes = research.list_notes(workspace) if workspace else []
        message = ""
    except (ValueError, OSError, RuntimeError) as exc:
        notes, message = [], f"Could not open notes: {exc}"
    if operation_state.get("revision", 0) != revision:
        return (gr.skip(),) * OUTPUT_COUNT
    operation_state["research_search_token"] = uuid4().hex
    operation_state["research_workspace_epoch"] = uuid4().hex
    search = list(_search_empty())
    search[-2] = ""
    search[-1] = _search_epoch(operation_state)
    return (*search, *_note_view(workspace, operation_state, notes, message=message))


def on_search_options(operation_state, client_epoch=None):
    if not _client_current(client_epoch, operation_state):
        return (gr.skip(),) * SEARCH_OUTPUT_COUNT
    operation_state["research_search_token"] = uuid4().hex
    values = list(_search_empty())
    values[-1] = _search_epoch(operation_state)
    return tuple(values)


def on_search(query, mode, workspace, collection, selected_ids, operation_state, client_epoch=None):
    revision = operation_state.get("revision", 0)
    if not _client_current(client_epoch, operation_state):
        return (gr.skip(),) * SEARCH_OUTPUT_COUNT
    if not _current(operation_state, revision):
        return _search_empty("Wait for the workspace to open.")
    token = uuid4().hex
    operation_state["research_search_token"] = token
    selected = list(selected_ids or [])

    def current():
        return (_current(operation_state, revision) and operation_state.get("research_search_token") == token
                and set(operation_state.get("selected_ids", selected)) == set(selected))

    try:
        sources = research.search_sources(workspace, query, selected, collection, mode,
                                          is_cancelled=lambda: not current())
        if not current():
            return (gr.skip(),) * SEARCH_OUTPUT_COUNT
        envelope = {"workspace": workspace, "revision": revision, "token": token, "sources": sources,
                    "selected_ids": selected}
        rows = [[f"[{item['number']}]", item["filename"], item["locator"], item["text"][:300]] for item in sources]
        message = f"Found {len(sources)} passages. "
        message += "Keyword search is local; no model request was made." if mode == "keyword" else "Hybrid search used one query embedding; no answer was generated."
        return (envelope, message, gr.update(value=rows, headers=SEARCH_HEADERS),
                gr.update(choices=_reference_choices(sources), value=None), "", None, gr.skip(), _search_epoch(operation_state))
    except (ValueError, OSError, RuntimeError, APIError) as exc:
        logger.warning("Research search failed: %s", exc)
        return _search_empty(f"Could not search sources: {exc}") if current() else (gr.skip(),) * SEARCH_OUTPUT_COUNT


def _search_matches(envelope, workspace, operation_state, client_epoch=None):
    return (isinstance(envelope, dict) and envelope.get("workspace") == workspace
            and envelope.get("token") == operation_state.get("research_search_token")
            and _client_current(client_epoch, operation_state, result=True)
            and _current(operation_state, envelope.get("revision"))
            and set(envelope.get("selected_ids", [])) == set(operation_state.get("selected_ids", envelope.get("selected_ids", []))))


def _selected(sources, selection):
    if not isinstance(selection, str) or selection not in {str(index) for index in range(len(sources))}:
        raise ValueError("Select an available reference.")
    return sources[int(selection)]


def on_search_reference(selection, envelope, workspace, operation_state, client_epoch=None):
    if not _search_matches(envelope, workspace, operation_state, client_epoch):
        return gr.skip(), gr.skip()
    if selection is None:
        return "", None
    try:
        source = _selected(envelope["sources"], selection)
        path = research.reference_path(workspace, source)
        if not _search_matches(envelope, workspace, operation_state):
            return gr.skip(), gr.skip()
        return source["text"], path
    except (ValueError, OSError) as exc:
        return (f"Could not open reference: {exc}", None) if _search_matches(envelope, workspace, operation_state) else (gr.skip(), gr.skip())


def on_new(workspace, operation_state, client_epoch=None):
    revision = operation_state.get("revision", 0)
    if (not _current(operation_state, revision)
            or client_epoch is not None and client_epoch != operation_state.get("research_note_token")):
        return (gr.skip(),) * NOTE_OUTPUT_COUNT
    try:
        notes = research.list_notes(workspace)
        if not _current(operation_state, revision):
            return (gr.skip(),) * NOTE_OUTPUT_COUNT
        return _note_view(workspace, operation_state, notes, message="Write a title and note, then Save note.")
    except (ValueError, OSError, RuntimeError) as exc:
        return _note_error(f"Could not create note: {exc}") if _current(operation_state, revision) else (gr.skip(),) * NOTE_OUTPUT_COUNT


def on_filter(query, envelope, workspace, operation_state, client_epoch=None):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return gr.skip(), gr.skip()
    try:
        notes = research.list_notes(workspace, query)
        if not _matches(envelope, workspace, operation_state):
            return gr.skip(), gr.skip()
        operation_state["research_note_ids"] = [item["id"] for item in notes]
        note = envelope.get("note")
        selected = note["id"] if note and note["id"] in operation_state["research_note_ids"] else None
        return gr.update(choices=_choices(notes), value=selected), f"{len(notes)} notes match. Unsaved editor text is unchanged."
    except (ValueError, OSError, RuntimeError) as exc:
        return (gr.skip(), f"Could not search notes: {exc}") if _matches(envelope, workspace, operation_state) else (gr.skip(), gr.skip())


def on_open(selection, envelope, workspace, operation_state, client_epoch=None, *, reload=False):
    if (not _matches(envelope, workspace, operation_state, client_epoch)
            or not isinstance(selection, str) or (selection not in operation_state.get("research_note_ids", [])
                and not (reload and envelope.get("note", {}).get("id") == selection))):
        return (gr.skip(),) * NOTE_OUTPUT_COUNT
    if not reload and envelope.get("note") and envelope["note"]["id"] == selection:
        return (gr.skip(),) * NOTE_OUTPUT_COUNT  # An unchanged dropdown blur must not erase edits.
    try:
        note = research.load_note(workspace, selection)
        notes = research.list_notes(workspace)
        if not _matches(envelope, workspace, operation_state):
            return (gr.skip(),) * NOTE_OUTPUT_COUNT
        return _note_view(workspace, operation_state, notes, note, "Opened note.")
    except (ValueError, OSError, RuntimeError) as exc:
        return _note_error(f"Could not open note: {exc}") if _matches(envelope, workspace, operation_state) else (gr.skip(),) * NOTE_OUTPUT_COUNT


def on_reload(envelope, workspace, operation_state, client_epoch=None):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return (gr.skip(),) * NOTE_OUTPUT_COUNT
    note = envelope.get("note")
    if note is None:
        return _note_error("Open a saved note before reloading it.")
    return on_open(note["id"], envelope, workspace, operation_state, reload=True)


def on_save(title, body, envelope, workspace, operation_state, client_epoch=None):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return (gr.skip(),) * NOTE_OUTPUT_COUNT
    try:
        old = envelope.get("note")
        note = research.save_note(workspace, title, body, identifier=old["id"] if old else None,
                                  expected_revision=old["revision"] if old else None,
                                  is_cancelled=lambda: not _matches(envelope, workspace, operation_state))
        notes = research.list_notes(workspace)
        if not _matches(envelope, workspace, operation_state):
            return (gr.skip(),) * NOTE_OUTPUT_COUNT
        return _note_view(workspace, operation_state, notes, note, "Saved note.")
    except (ValueError, OSError, RuntimeError) as exc:
        return _note_error(f"Could not save note: {exc}") if _matches(envelope, workspace, operation_state) else (gr.skip(),) * NOTE_OUTPUT_COUNT


def on_delete(envelope, workspace, operation_state, client_epoch=None):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return (gr.skip(),) * NOTE_OUTPUT_COUNT
    try:
        note = envelope.get("note")
        if note is None:
            raise ValueError("Open a saved note before deleting it.")
        research.delete_note(workspace, note["id"], note["revision"],
                             is_cancelled=lambda: not _matches(envelope, workspace, operation_state))
        notes = research.list_notes(workspace)
        if not _matches(envelope, workspace, operation_state):
            return (gr.skip(),) * NOTE_OUTPUT_COUNT
        return _note_view(workspace, operation_state, notes, message="Deleted note.")
    except (ValueError, OSError, RuntimeError) as exc:
        return _note_error(f"Could not delete note: {exc}") if _matches(envelope, workspace, operation_state) else (gr.skip(),) * NOTE_OUTPUT_COUNT


def on_capture_answer(history, envelope, workspace, operation_state, pending=None, client_epoch=None):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return (gr.skip(),) * NOTE_OUTPUT_COUNT
    try:
        if pending is not None:
            raise ValueError("Wait for the current answer to finish before saving it as a note.")
        answer = next((item for item in reversed(history or []) if item.get("role") == "assistant"), None)
        if not answer:
            raise ValueError("Complete a chat answer before saving it as a note.")
        note = research.save_note(workspace, "Saved chat answer", answer["content"], origin="ai",
                                  sources=answer.get("sources", []), web_search=answer.get("web_search", False),
                                  is_cancelled=lambda: not _matches(envelope, workspace, operation_state))
        notes = research.list_notes(workspace)
        if not _matches(envelope, workspace, operation_state):
            return (gr.skip(),) * NOTE_OUTPUT_COUNT
        return _note_view(workspace, operation_state, notes, note, "Saved answer as an AI-derived note. You can rename it below.")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        return _note_error(f"Could not save answer: {exc}") if _matches(envelope, workspace, operation_state) else (gr.skip(),) * NOTE_OUTPUT_COUNT


def on_capture_excerpt(selection, search_envelope, note_envelope, workspace, operation_state,
                       search_client_epoch=None, client_epoch=None):
    if not _matches(note_envelope, workspace, operation_state, client_epoch):
        return (gr.skip(),) * NOTE_OUTPUT_COUNT
    if not _search_matches(search_envelope, workspace, operation_state, search_client_epoch):
        return _note_error("Search for a passage and select a result first.")
    try:
        source = _selected(search_envelope["sources"], selection)
        note = research.save_note(workspace, (source["filename"] + " — " + source["locator"])[:200],
                                  source["text"], origin="excerpt", sources=[source],
                                  is_cancelled=lambda: not _matches(note_envelope, workspace, operation_state)
                                  or not _search_matches(search_envelope, workspace, operation_state))
        notes = research.list_notes(workspace)
        if not _matches(note_envelope, workspace, operation_state):
            return (gr.skip(),) * NOTE_OUTPUT_COUNT
        return _note_view(workspace, operation_state, notes, note, "Saved the selected source excerpt.")
    except (ValueError, OSError, RuntimeError) as exc:
        return _note_error(f"Could not save excerpt: {exc}") if _matches(note_envelope, workspace, operation_state) else (gr.skip(),) * NOTE_OUTPUT_COUNT


def on_note_reference(selection, envelope, workspace, operation_state, client_epoch=None):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return gr.skip(), gr.skip()
    try:
        note = envelope.get("note")
        if note is None or selection is None:
            return "", None
        source = _selected(note["sources"], selection)
        path = research.reference_path(workspace, source)
        if not _matches(envelope, workspace, operation_state):
            return gr.skip(), gr.skip()
        return source["text"], path
    except (ValueError, OSError) as exc:
        return (f"Could not open note reference: {exc}", None) if _matches(envelope, workspace, operation_state) else (gr.skip(), gr.skip())


def on_preview(text, envelope, workspace, operation_state, client_epoch=None):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return gr.skip()
    if not isinstance(text, str) or len(text) > research.MAX_NOTE_CHARACTERS:
        return gr.skip()
    return text


def on_note_export_options(title, body, filename, output_format, envelope, workspace, operation_state, client_epoch):
    if not _matches(envelope, workspace, operation_state, client_epoch):
        return gr.skip(), gr.skip()
    token = uuid4().hex
    operation_state["research_note_export"] = {"token": token, "inputs": (title, body, filename, output_format)}
    return None, token


def on_note_export(title, body, filename, output_format, envelope, workspace, operation_state, client_epoch, export_epoch):
    def current():
        export = operation_state.get("research_note_export", {})
        return (_matches(envelope, workspace, operation_state, client_epoch)
                and isinstance(export_epoch, str) and bool(export_epoch)
                and export.get("token") == export_epoch
                and export.get("inputs") == (title, body, filename, output_format))

    if not current():
        return gr.skip(), None
    try:
        note = envelope.get("note")
        if note is None or title != note["title"] or body != note["body"]:
            raise ValueError("Save note edits before exporting.")
        path = research.export_note(workspace, note["id"], note["revision"], output_format, filename,
                                    is_cancelled=lambda: not current())
        return ("Saved note exported.", str(path)) if current() else (gr.skip(), None)
    except (ValueError, OSError, RuntimeError) as exc:
        logger.warning("Note export failed: %s", exc)
        return (f"Could not export note: {exc}", None) if current() else (gr.skip(), None)


def build(workspace, collection, selected_ids, history_state, sources_state, operation_state, pending_state=None):
    """Build controls without touching workspace storage or the OpenAI client."""
    storage_queue = {"concurrency_id": "rag-research-storage", "concurrency_limit": 1}
    with gr.Tabs():
        with gr.Tab("Search"):
            gr.Markdown("Search checked sources directly. Keyword search runs locally; Hybrid uses one query embedding and never generates an answer.")
            search_state = gr.State(None)
            workspace_epoch = gr.Textbox(value="", visible=False, elem_id="rag-research-epoch")
            with gr.Row():
                search_query = gr.Textbox(label="Search sources", placeholder="Find a term or passage", scale=3)
                search_mode = gr.Dropdown(label="Search mode", choices=[("Keyword (local)", "keyword"), ("Hybrid", "hybrid")], value="keyword", scale=1)
            search_button = gr.Button("Search checked sources", variant="primary")
            search_status = gr.Markdown()
            results = gr.Dataframe(headers=SEARCH_HEADERS, value=[], datatype="str", interactive=False, max_height=240)
            search_selector = gr.Dropdown(label="Open search result", choices=[])
            search_excerpt = gr.Textbox(label="Matching passage", lines=6, max_lines=6, interactive=False)
            search_original = gr.File(label="Original source", interactive=False)
        with gr.Tab("Notes"):
            note_state = gr.State(None)
            note_epoch = gr.Textbox(value="", visible=False, elem_id="rag-note-epoch")
            gr.Markdown("Keep your own research notes and saved findings. Notes are separate from source evidence; saving or searching notes makes no model requests.")
            with gr.Row():
                note_filter = gr.Textbox(label="Search notes", placeholder="Search title or text", scale=2)
                note_selector = gr.Dropdown(label="Saved notes", choices=[], scale=3)
            with gr.Row():
                new_button = gr.Button("New note")
                reload_button = gr.Button("Reload saved note")
                save_answer_button = gr.Button("Save latest answer to note")
                save_excerpt_button = gr.Button("Save selected search excerpt to note")
            note_title = gr.Textbox(label="Note title", max_length=200)
            with gr.Row(equal_height=True):
                note_body = gr.Textbox(label="Note text (Markdown)", lines=12, max_lines=18)
                note_preview = gr.Markdown(height=320, sanitize_html=True)
            with gr.Row():
                save_button = gr.Button("Save note", variant="primary")
                delete_button = gr.Button("Delete saved note", variant="stop")
            provenance = gr.Markdown()
            note_status = gr.Markdown()
            with gr.Accordion("Note references", open=False):
                note_reference = gr.Dropdown(label="Open note reference", choices=[])
                note_excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                note_original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                gr.Markdown("Export the saved note with its references and origin. Save edits first.")
                note_filename = gr.Textbox(label="Filename", value="Note")
                note_format = gr.Dropdown(label="Export format", choices=[("Markdown", "md"), ("JSON", "json")], value="md")
                note_export_button = gr.DownloadButton("Export saved note", value=None, interactive=True)
            note_download = gr.File(visible=False, interactive=False)
            note_export_epoch = gr.Textbox(value="", visible=False, elem_id="rag-note-export-epoch")

    search_outputs = [search_state, search_status, results, search_selector, search_excerpt, search_original, search_query, workspace_epoch]
    note_outputs = [note_state, note_selector, note_title, note_body, note_preview, provenance, note_status,
                    note_reference, note_excerpt, note_original, note_filter, note_epoch,
                    note_download, note_filename, note_format, note_export_epoch]
    outputs = [*search_outputs, *note_outputs]
    search_event = gr.on(triggers=[search_query.submit, search_button.click], fn=on_search,
        inputs=[search_query, search_mode, workspace, collection, selected_ids, operation_state, workspace_epoch],
        outputs=search_outputs, trigger_mode="always_last", preprocess=False, **storage_queue)
    gr.on(triggers=[search_query.input, search_mode.input, selected_ids.input, collection.input],
          fn=on_search_options, inputs=[operation_state, workspace_epoch], outputs=search_outputs, queue=False)
    search_selector.input(on_search_reference, [search_selector, search_state, workspace, operation_state, workspace_epoch],
                          [search_excerpt, search_original], queue=False, preprocess=False)
    new_button.click(on_new, [workspace, operation_state, note_epoch], note_outputs, **storage_queue)
    reload_button.click(on_reload, [note_state, workspace, operation_state, note_epoch], note_outputs, **storage_queue)
    note_selector.input(on_open, [note_selector, note_state, workspace, operation_state, note_epoch], note_outputs,
                        preprocess=False, **storage_queue)
    note_filter.input(on_filter, [note_filter, note_state, workspace, operation_state, note_epoch], [note_selector, note_status],
                      trigger_mode="always_last", **storage_queue)
    note_body.input(on_preview, [note_body, note_state, workspace, operation_state, note_epoch], note_preview,
                    queue=False, trigger_mode="always_last")
    save_button.click(on_save, [note_title, note_body, note_state, workspace, operation_state, note_epoch], note_outputs, **storage_queue)
    delete_button.click(on_delete, [note_state, workspace, operation_state, note_epoch], note_outputs, **storage_queue)
    answer_inputs = [history_state, note_state, workspace, operation_state,
                     pending_state if pending_state is not None else gr.State(None), note_epoch]
    save_answer_button.click(on_capture_answer, answer_inputs, note_outputs, **storage_queue)
    save_excerpt_button.click(on_capture_excerpt, [search_selector, search_state, note_state, workspace, operation_state, workspace_epoch, note_epoch],
                              note_outputs, preprocess=False, **storage_queue)
    note_reference.input(on_note_reference, [note_reference, note_state, workspace, operation_state, note_epoch],
                         [note_excerpt, note_original], preprocess=False, queue=False)
    export_inputs = [note_title, note_body, note_filename, note_format, note_state, workspace, operation_state, note_epoch]
    gr.on(triggers=[note_title.input, note_body.input, note_filename.input, note_format.input],
          fn=on_note_export_options, inputs=export_inputs, outputs=[note_download, note_export_epoch],
          queue=False, trigger_mode="always_last")
    note_export_event = note_export_button.click(on_note_export, [*export_inputs, note_export_epoch],
                                                 [note_status, note_download], **storage_queue)
    # Keep the button's own href empty: Gradio otherwise downloads its old
    # value before the callback finishes. Download the newly prepared file.
    from .tools.ui import SUMMARY_DOWNLOAD_JS
    note_download_event = note_export_event.success(fn=None, inputs=note_download, outputs=None,
        js=SUMMARY_DOWNLOAD_JS.replace('"Summary"', '"Note"'), queue=False)
    return SimpleNamespace(**{key: value for key, value in locals().items() if not key.startswith("_")}, reset_callback=on_reset)
