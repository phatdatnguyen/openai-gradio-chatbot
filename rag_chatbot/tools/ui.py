"""Workspace tool controls; storage and generation stay lazy until requested."""

from contextlib import closing
from copy import deepcopy
from functools import partial
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr
import pandas as pd
from openai import APIError

from chatbot.log import logger
from chatbot import config as model_config
from .learning import DEFAULT_LEARNING_COUNT, validate_item_count
from .presentation import DEFAULT_SLIDE_COUNT, validate_slide_count


_JOB_LOCK = threading.Lock()
OUTPUT_COUNT = 10
OUTPUT_COUNTS = {"summary": 12, "spreadsheet": OUTPUT_COUNT, "presentation": 14, "graphics": 11, "plot": 11, "mind_maps": 7,
                 "flashcards": 14, "quizzes": 17, "compare": 10, "evidence_check": 10}
TOTAL_OUTPUT_COUNT = sum(OUTPUT_COUNTS.values())
DEFAULT_SUMMARY_INSTRUCTIONS = "Please help me summarize the documents"
DEFAULT_SPREADSHEET_INSTRUCTIONS = "Please help me to generate a spreadsheet for the documents"
DEFAULT_MIND_MAP_INSTRUCTIONS = "Please help me create a mind map of the documents"
TOOL_LABELS = {"summary": "Summary", "spreadsheet": "Spreadsheet", "presentation": "Presentation", "graphics": "Graphics", "plot": "Plot", "mind_maps": "Mind Map",
               "flashcards": "Flashcards", "quizzes": "Quizzes", "compare": "Compare", "evidence_check": "Evidence Check"}
PLANNED_TOOLS = ()
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": r"\[", "right": r"\]", "display": True},
    {"left": r"\(", "right": r"\)", "display": False},
]


def _artifacts():
    from . import artifacts
    return artifacts


def _generate(*args, **kwargs):
    from .service import generate_tool
    return generate_tool(*args, **kwargs)


def _render_mind_map(payload, zoom=1.0):
    from .mindmap import render_mind_map_html
    return render_mind_map_html(payload, zoom=zoom)


def _enabled(tool_id):
    from plugins import is_enabled
    return is_enabled(tool_id)


def cancel_tool_job(operation_state):
    """Invalidate a session job without pretending its current HTTP call stopped."""
    with _JOB_LOCK:
        job = operation_state.get("tool_job")
        if job:
            job["cancelled"] = True
            return True
    return False


def _current(operation_state, revision, answer_revision=None):
    return (not operation_state.get("switching", False)
            and operation_state.get("revision", 0) == revision
            and (answer_revision is None
                 or operation_state.get("answer_revision", 0) == answer_revision))


def _status(message, tool_id="spreadsheet"):
    return gr.skip(), message, *((gr.skip(),) * (OUTPUT_COUNTS[tool_id] - 2))


def _reference_choices(payload):
    return [(f"[{index + 1}] {source.get('filename', '')} — {source.get('locator', '')}", str(index))
            for index, source in enumerate(payload.get("sources", []))]


def _coverage(payload):
    coverage = payload.get("coverage", {})
    processed = coverage.get("processed_chunks", 0)
    chunks = coverage.get("chunks", 0)
    documents = coverage.get("documents", len(payload.get("documents", [])))
    suffix = ("Changing the source checkboxes affects the next generation."
              if payload.get("tool_id") == "mind_maps" else
              "References below belong to this output; changing the source checkboxes affects the next generation.")
    return f"Sources at generation: **{documents} documents**, **{processed} / {chunks} chunks processed**. {suffix}"


def _literal_markdown(value):
    return re.sub(r"([\\`*_{}\[\]<>()#+.!|~-])", r"\\\1", str(value).replace("\n", " ").replace("\r", " "))


def _render(workspace, payload, message):
    if payload["tool_id"] in {"compare", "evidence_check"}:
        from .review_ui import render_output
        return render_output(workspace, payload, message)
    if payload["tool_id"] == "plot":
        from .plot_ui import render_output
        return render_output(workspace, payload, message)
    if payload["tool_id"] == "graphics":
        from .graphics_ui import render_output
        return render_output(workspace, payload, message)
    if payload["tool_id"] == "presentation":
        from .presentation_ui import render_output
        return render_output(workspace, payload, message)
    if payload["tool_id"] in {"flashcards", "quizzes"}:
        from .learning_ui import render_output
        return render_output(workspace, payload, message)
    store = _artifacts()
    if payload["tool_id"] == "summary":
        editor = store.render_summary(payload, include_sources=False)
        return ({"workspace": workspace, "payload": payload}, message,
                gr.update(value=editor, visible=False), gr.update(value=editor, visible=True),
                _coverage(payload), gr.update(choices=_reference_choices(payload), value=None),
                "", None, gr.update(visible=True, interactive=True), gr.update(visible=False), None, gr.skip())
    elif payload["tool_id"] == "mind_maps":
        preview = _render_mind_map(payload)
        return ({"workspace": workspace, "payload": payload}, message, preview,
                _coverage(payload), 1.0, None, gr.skip())
    else:
        headers, rows = store.spreadsheet_table(payload)
        # Gradio 6.30 exposes row_limits/column_limits but does not implement
        # them. Its supported tuple counts still enforce a fixed grid.
        editor = gr.update(value=rows, headers=headers, row_count=(len(rows), "fixed"),
                           column_count=(len(headers), "fixed"))
        numbers = {source["id"]: index + 1 for index, source in enumerate(payload.get("sources", []))}
        references = []
        for index, row in enumerate(payload.get("content", {}).get("rows", [])):
            cells = "; ".join(f"{_literal_markdown(cell['column'])}: " + (" ".join(
                f"[{numbers[item]}]" for item in cell.get("source_ids", []) if item in numbers)
                or "no evidence found") for cell in row.get("cells", []))
            references.append(f"Row {index + 1}: {_literal_markdown(row.get('filename', ''))}\n\n{cells}")
        preview = "\n\n".join(references)
    return ({"workspace": workspace, "payload": payload}, message, editor, preview,
            _coverage(payload), gr.update(choices=_reference_choices(payload), value=None),
            "", None, None, gr.skip())


def _columns(value):
    columns = [item.strip() for item in re.split(r"[,\n]", value or "") if item.strip()]
    if not columns:
        raise ValueError("Enter at least one spreadsheet column, separated by commas or new lines.")
    if len(columns) > 12 or any(len(item) > 80 for item in columns):
        raise ValueError("Use at most 12 columns, each at most 80 characters.")
    if len(set(item.casefold() for item in columns)) != len(columns):
        raise ValueError("Spreadsheet column names must be unique.")
    return columns


def on_generate(tool_id, instructions, columns_text, workspace, selected_ids, collection,
                model, reasoning_effort, operation_state, item_count=DEFAULT_LEARNING_COUNT,
                slide_count=DEFAULT_SLIDE_COUNT, image_options=None, chart_type="auto", claims_text=""):
    revision = operation_state.get("revision", 0)
    answer_revision = operation_state.get("answer_revision", 0)
    selected_snapshot = frozenset(selected_ids or [])
    token = uuid4().hex
    try:
        if not _current(operation_state, revision):
            raise ValueError("Wait for the workspace to open before generating a tool output.")
        if not workspace:
            raise ValueError("Create or open a workspace first.")
        if not selected_ids:
            raise ValueError("No sources selected. Check at least one ready source before generating.")
        if tool_id == "summary" and operation_state.get("summary_editing"):
            raise ValueError("Save your Summary edits before generating another summary.")
        if not model_config.supports_structured_output(model):
            raise ValueError("Choose a model that supports structured tool outputs, such as gpt-6-luna, in Prompt.")
        if not _enabled(tool_id):
            raise ValueError(f"{TOOL_LABELS[tool_id]} is disabled. Enable it in the Plugins tab.")
        columns = _columns(columns_text) if tool_id == "spreadsheet" else None
        learning_options = ({"item_count": validate_item_count(tool_id, item_count)}
                            if tool_id in {"flashcards", "quizzes"} else {})
        if tool_id == "compare" and not 2 <= len(selected_snapshot) <= 8:
            raise ValueError("Compare needs 2–8 checked source documents.")
        if tool_id == "evidence_check":
            from .review import parse_claims
            parse_claims(claims_text)
            learning_options["claims_text"] = claims_text
        if tool_id == "presentation":
            learning_options["slide_count"] = validate_slide_count(slide_count)
        if tool_id == "graphics":
            from .graphics import validate_image_options
            learning_options["image_options"] = validate_image_options(image_options)
        if tool_id == "plot":
            from .plot import validate_chart_type
            learning_options["chart_type"] = validate_chart_type(chart_type)
        with _JOB_LOCK:
            if operation_state.get("tool_job"):
                raise ValueError("Another tool request is running or stopping. Wait for its current request to finish.")
            operation_state["tool_job_revision"] = operation_state.get("tool_job_revision", 0) + 1
            operation_state["tool_job"] = {"id": token, "cancelled": False, "workspace": workspace}
    except (ValueError, OSError, RuntimeError) as exc:
        yield _status(str(exc), tool_id)
        return

    def is_cancelled():
        job = operation_state.get("tool_job", {})
        return (job.get("id") != token or job.get("cancelled", False)
                or not _current(operation_state, revision, answer_revision)
                or ("selected_ids" in operation_state
                    and frozenset(operation_state["selected_ids"]) != selected_snapshot))

    try:
        yield _status("Preparing all selected sources…", tool_id)
        with closing(_generate(workspace, tool_id, list(selected_ids), instructions=instructions or "",
                               columns=columns, model=model, reasoning_effort=reasoning_effort,
                               is_cancelled=is_cancelled, collection=collection, **learning_options)) as events:
            for event in events:
                if is_cancelled():
                    return
                payload = event.get("result")
                if payload is None:
                    update = _status(event.get("status", "Working…"), tool_id)
                else:
                    if tool_id == "graphics":
                        # The generated raster is transient until its guarded
                        # asset and manifest writes finish. Never expose a
                        # temporary or unvalidated image path to Gradio.
                        payload = _artifacts().save_current_graphics(workspace, payload, is_cancelled=is_cancelled)
                        if is_cancelled():
                            return
                    update = _render(workspace, payload, event.get("status") or "Generated.")
                # Rendering and filesystem-backed output preparation can take
                # time; invalidation may arrive after the event was received.
                if is_cancelled():
                    return
                if payload is not None:
                    store = _artifacts()
                    if tool_id == "graphics":
                        saved = payload
                    else:
                        save_current = (store.save_current_learning if tool_id in {"flashcards", "quizzes"}
                                        else {"summary": store.save_current_summary,
                                              "spreadsheet": store.save_current_spreadsheet,
                                              "presentation": store.save_current_presentation,
                                              "plot": store.save_current_plot,
                                              "compare": store.save_current_review,
                                              "evidence_check": store.save_current_review,
                                              "mind_maps": store.save_current_mind_map}[tool_id])
                        saved = save_current(workspace, payload, is_cancelled=is_cancelled)
                    update = list(update)
                    update[0] = {**update[0], "workspace": workspace, "payload": saved}
                    update[1] = f"Completed. {TOOL_LABELS[tool_id]} saved for this workspace."
                    update = tuple(update)
                    if is_cancelled():
                        return
                    if tool_id == "mind_maps":
                        operation_state["mind_map_revision"] = operation_state.get("mind_map_revision", 0) + 1
                        on_mind_map_export_options(operation_state)
                    elif tool_id == "spreadsheet":
                        on_spreadsheet_export_options(operation_state)
                    elif tool_id == "presentation":
                        from .presentation_ui import publish
                        if not publish(update[0], operation_state, is_cancelled=is_cancelled):
                            return
                    elif tool_id == "graphics":
                        from .graphics_ui import publish
                        if not publish(update[0], operation_state, is_cancelled=is_cancelled):
                            return
                    elif tool_id == "plot":
                        from .plot_ui import publish
                        if not publish(update[0], operation_state, is_cancelled=is_cancelled):
                            return
                    elif tool_id in {"compare", "evidence_check"}:
                        from .review_ui import publish
                        if not publish(update[0], operation_state, is_cancelled=is_cancelled):
                            return
                    elif tool_id in {"flashcards", "quizzes"}:
                        from .learning_ui import publish
                        if not publish(update[0], operation_state, is_cancelled=is_cancelled):
                            return
                    if is_cancelled():
                        return
                yield update
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, APIError) as exc:
        if not is_cancelled():
            logger.exception("Workspace tool generation failed")
            yield _status(f"Could not generate {TOOL_LABELS[tool_id]}: {exc}", tool_id)
    finally:
        with _JOB_LOCK:
            if operation_state.get("tool_job", {}).get("id") == token:
                operation_state.pop("tool_job", None)


def on_stop(operation_state):
    if cancel_tool_job(operation_state):
        return "Stopping after the current request. Its result will be discarded."
    return "No tool generation is running."


def _output_payload(envelope, workspace, tool_id):
    if not envelope or envelope.get("workspace") != workspace:
        raise ValueError("Generate an output in this workspace first.")
    payload = deepcopy(envelope["payload"])
    if payload.get("tool_id") != tool_id:
        raise ValueError("This output belongs to a different tool.")
    return payload


def _edited(envelope, workspace, tool_id, editor):
    payload = _output_payload(envelope, workspace, tool_id)
    if tool_id == "summary":
        if not isinstance(editor, str) or not editor.strip():
            raise ValueError("The Summary cannot be empty.")
        initial = _artifacts().render_summary(payload, include_sources=False)
        if editor != initial:
            payload["edited_text"] = editor
    elif tool_id == "spreadsheet":
        headers, rows = _artifacts().spreadsheet_table(payload)
        if isinstance(editor, pd.DataFrame):
            if list(editor.columns) != headers:
                raise ValueError("Keep the generated spreadsheet columns when editing.")
            values = editor.fillna("").astype(str).values.tolist()
        else:
            values = [["" if value is None else str(value) for value in row] for row in (editor or [])]
        if len(values) != len(rows) or any(len(row) != len(headers) for row in values):
            raise ValueError("Keep the generated spreadsheet rows and columns when editing.")
        if values != rows:
            payload["edited_rows"] = values
    else:
        raise ValueError("This tool's output cannot be edited.")
    return payload


def on_summary_edit(envelope, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    try:
        if not _current(operation_state, revision):
            return (gr.skip(),) * OUTPUT_COUNTS["summary"]
        if not envelope or envelope.get("workspace") != workspace:
            raise ValueError("Generate a Summary in this workspace first.")
        cancel_tool_job(operation_state)
        on_summary_export_options(operation_state)
        result = list(_render(workspace, envelope["payload"], "Edit the Summary, then Save your changes."))
        result[2]["visible"] = True
        result[3]["visible"] = False
        result[8] = gr.update(visible=False)
        result[9] = gr.update(visible=True)
        if not _current(operation_state, revision):
            return (gr.skip(),) * OUTPUT_COUNTS["summary"]
        operation_state["summary_editing"] = True
        return tuple(result)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        return _status(f"Could not edit Summary: {exc}", "summary") if _current(operation_state, revision) else (gr.skip(),) * OUTPUT_COUNTS["summary"]


def on_summary_save(envelope, editor, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    try:
        if not _current(operation_state, revision):
            return (gr.skip(),) * OUTPUT_COUNTS["summary"]
        cancel_tool_job(operation_state)
        payload = _edited(envelope, workspace, "summary", editor)
        result = list(_render(workspace, payload, "Summary changes saved."))
        if not _current(operation_state, revision):
            return (gr.skip(),) * OUTPUT_COUNTS["summary"]
        saved = _artifacts().save_current_summary(workspace, payload,
            is_cancelled=lambda: not _current(operation_state, revision))
        result[0] = {"workspace": workspace, "payload": saved}
        if not _current(operation_state, revision):
            return (gr.skip(),) * OUTPUT_COUNTS["summary"]
        operation_state["summary_editing"] = False
        on_summary_export_options(operation_state)
        return tuple(result)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        logger.exception("Saving current Summary failed")
        return _status(f"Could not save Summary: {exc}", "summary") if _current(operation_state, revision) else (gr.skip(),) * OUTPUT_COUNTS["summary"]


def on_summary_export_options(operation_state):
    operation_state["summary_export_revision"] = operation_state.get("summary_export_revision", 0) + 1
    return None


def on_summary_export(envelope, filename, output_format, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    export_revision = operation_state.get("summary_export_revision", 0)

    def current():
        return (_current(operation_state, revision)
                and operation_state.get("summary_export_revision", 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        if operation_state.get("summary_editing"):
            raise ValueError("Save your Summary edits before exporting.")
        if not envelope or envelope.get("workspace") != workspace:
            raise ValueError("Generate a Summary in this workspace first.")
        path = _artifacts().export_summary(workspace, envelope["payload"], output_format, filename=filename,
                                           is_cancelled=lambda: not current())
        result = "Summary exported.", str(path)
        return result if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting current Summary failed")
        return (f"Could not export Summary: {exc}", None) if current() else (gr.skip(), None)


def on_spreadsheet_save(envelope, editor, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    count = OUTPUT_COUNTS["spreadsheet"]
    try:
        if not _current(operation_state, revision):
            return (gr.skip(),) * count
        payload = _edited(envelope, workspace, "spreadsheet", editor)
        if not _current(operation_state, revision):
            return (gr.skip(),) * count
        cancel_tool_job(operation_state)
        result = list(_render(workspace, payload, "Spreadsheet changes saved."))
        if not _current(operation_state, revision):
            return (gr.skip(),) * count
        saved = _artifacts().save_current_spreadsheet(workspace, payload,
            is_cancelled=lambda: not _current(operation_state, revision))
        result[0] = {"workspace": workspace, "payload": saved}
        if not _current(operation_state, revision):
            return (gr.skip(),) * count
        on_spreadsheet_export_options(operation_state)
        return tuple(result)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        logger.exception("Saving current Spreadsheet failed")
        return _status(f"Could not save Spreadsheet: {exc}", "spreadsheet") if _current(operation_state, revision) else (gr.skip(),) * count


def on_spreadsheet_export_options(operation_state):
    operation_state["spreadsheet_export_revision"] = operation_state.get("spreadsheet_export_revision", 0) + 1
    return None


def on_spreadsheet_export(envelope, editor, filename, output_format, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    export_revision = operation_state.get("spreadsheet_export_revision", 0)

    def current():
        return (_current(operation_state, revision)
                and operation_state.get("spreadsheet_export_revision", 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        payload = _edited(envelope, workspace, "spreadsheet", editor)
        if not current():
            return gr.skip(), None
        path = _artifacts().export_spreadsheet(workspace, payload, output_format, filename=filename,
                                               is_cancelled=lambda: not current())
        result = "Spreadsheet exported.", str(path)
        return result if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting Spreadsheet failed")
        return (f"Could not export Spreadsheet: {exc}", None) if current() else (gr.skip(), None)


def on_mind_map_zoom(zoom, envelope, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    map_revision = operation_state.get("mind_map_revision", 0)
    operation_state["mind_map_view_revision"] = operation_state.get("mind_map_view_revision", 0) + 1
    view_revision = operation_state["mind_map_view_revision"]

    def current():
        return (_current(operation_state, revision)
                and operation_state.get("mind_map_revision", 0) == map_revision
                and operation_state.get("mind_map_view_revision", 0) == view_revision)

    try:
        if not current():
            return gr.skip(), gr.skip()
        if not envelope or envelope.get("workspace") != workspace:
            return gr.skip(), "Generate a Mind Map in this workspace first."
        preview = _render_mind_map(envelope["payload"], zoom=zoom)
        return (preview, gr.skip()) if current() else (gr.skip(), gr.skip())
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        return (gr.skip(), f"Could not display Mind Map: {exc}") if current() else (gr.skip(), gr.skip())


def on_mind_map_export_options(operation_state):
    operation_state["mind_map_export_revision"] = operation_state.get("mind_map_export_revision", 0) + 1
    return None


def on_mind_map_export(envelope, filename, output_format, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    export_revision = operation_state.get("mind_map_export_revision", 0)

    def current():
        return (_current(operation_state, revision)
                and operation_state.get("mind_map_export_revision", 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        if not envelope or envelope.get("workspace") != workspace:
            raise ValueError("Generate a Mind Map in this workspace first.")
        path = _artifacts().export_mind_map(workspace, envelope["payload"], output_format, filename=filename,
                                            is_cancelled=lambda: not current())
        result = "Mind Map exported.", str(path)
        return result if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting Mind Map failed")
        return (f"Could not export Mind Map: {exc}", None) if current() else (gr.skip(), None)


# DownloadButton's installed frontend downloads its previous value before a
# click callback returns. This post-export hook uses the newly returned FileData
# URL instead, so one Export click downloads the prepared file. No DOM lookup
# or untrusted HTML is involved; only files served by this app may be downloaded.
SUMMARY_DOWNLOAD_JS = """(file) => {
    if (!file || !file.url) return;
    const url = new URL(file.url, window.location.href);
    if (url.origin !== window.location.origin || !["http:", "https:"].includes(url.protocol)) return;
    const link = document.createElement("a");
    link.href = url.href;
    link.download = file.orig_name || "Summary";
    document.body.appendChild(link);
    link.click();
    link.remove();
}"""
SPREADSHEET_DOWNLOAD_JS = SUMMARY_DOWNLOAD_JS.replace('"Summary"', '"Spreadsheet"')
MIND_MAP_DOWNLOAD_JS = SUMMARY_DOWNLOAD_JS.replace('"Summary"', '"Mind Map"')


def on_reference(selection, envelope, workspace, operation_state):
    revision = operation_state.get("revision", 0)
    if not _current(operation_state, revision):
        return gr.skip(), gr.skip()
    if not selection or not envelope or envelope.get("workspace") != workspace:
        return "", None
    try:
        sources = envelope["payload"].get("sources", [])
        if not isinstance(selection, str) or selection not in {str(index) for index in range(len(sources))}:
            return gr.skip(), gr.skip()
        index = int(selection)
        if index < 0 or index >= len(sources):
            raise ValueError("Select an available reference.")
        source = sources[index]
        from ..workspaces import workspace_path
        root = workspace_path(workspace).resolve()
        source_root = (root / "sources").resolve()
        path = Path(source.get("source_path") or "")
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(source_root) or not source_root.is_relative_to(root):
            raise ValueError("The original file is outside this workspace's sources.")
        result = source.get("text", ""), str(path) if path.is_file() else None
        return result if _current(operation_state, revision) else (gr.skip(), gr.skip())
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        return (f"Could not open reference: {exc}", None) if _current(operation_state, revision) else (gr.skip(), gr.skip())


def on_reset(workspace, operation_state, *, allow_switching=False):
    revision = operation_state.get("revision", 0)
    job = operation_state.get("tool_job", {})
    if job and not job.get("cancelled") and job.get("workspace") == workspace:
        return (gr.skip(),) * TOTAL_OUTPUT_COUNT
    outputs = []
    learning_envelopes = {}
    review_envelopes = {}
    presentation_envelope = None
    graphics_envelope = None
    plot_envelope = None
    for tool_id in TOOL_LABELS:
        if tool_id in {"compare", "evidence_check"}:
            from . import review_ui
            review_envelopes[tool_id] = None
            try:
                payload = _artifacts().load_current_review(workspace, tool_id) if workspace else None
                message = ""
                if payload is not None:
                    review_outputs = list(_render(workspace, payload, f"Restored this workspace's {TOOL_LABELS[tool_id]}."))
                    review_outputs[-1] = TOOL_LABELS[tool_id]
                    review_envelopes[tool_id] = review_outputs[0]
                    outputs.extend(review_outputs)
                    continue
            except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
                message = f"Could not restore {TOOL_LABELS[tool_id]}: {exc}"
            outputs.extend(review_ui.empty_output(tool_id, message))
            continue
        if tool_id == "plot":
            from . import plot_ui
            try:
                payload = _artifacts().load_current_plot(workspace) if workspace else None
                message = ""
                if payload is not None:
                    plot_outputs = list(_render(workspace, payload, "Restored this workspace's Plot."))
                    plot_outputs[-1] = "Plot"
                    plot_envelope = plot_outputs[0]
                    outputs.extend(plot_outputs)
                    continue
            except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
                message = f"Could not restore Plot: {exc}"
            outputs.extend(plot_ui.empty_output(message))
            continue
        if tool_id == "graphics":
            from . import graphics_ui
            try:
                payload = _artifacts().load_current_graphics(workspace) if workspace else None
                message = ""
                if payload is not None:
                    graphics_outputs = list(_render(workspace, payload, "Restored this workspace's Graphics."))
                    graphics_outputs[-1] = "Graphics"
                    graphics_envelope = graphics_outputs[0]
                    outputs.extend(graphics_outputs)
                    continue
            except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
                message = f"Could not restore Graphics: {exc}"
            outputs.extend(graphics_ui.empty_output(message))
            continue
        if tool_id == "presentation":
            from . import presentation_ui
            try:
                payload = _artifacts().load_current_presentation(workspace) if workspace else None
                message = ""
                if payload is not None:
                    presentation_outputs = list(_render(workspace, payload, "Restored this workspace's Presentation."))
                    presentation_outputs[-1] = "Presentation"
                    presentation_envelope = presentation_outputs[0]
                    outputs.extend(presentation_outputs)
                    continue
            except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
                message = f"Could not restore Presentation: {exc}"
            outputs.extend(presentation_ui.empty_output(message))
            continue
        if tool_id in {"flashcards", "quizzes"}:
            from . import learning_ui
            learning_envelopes[tool_id] = None
            try:
                payload = _artifacts().load_current_learning(workspace, tool_id) if workspace else None
                message = ""
                if payload is not None:
                    learning_outputs = list(_render(workspace, payload, f"Restored this workspace's {TOOL_LABELS[tool_id]}."))
                    learning_outputs[-1] = TOOL_LABELS[tool_id]
                    learning_envelopes[tool_id] = learning_outputs[0]
                    outputs.extend(learning_outputs)
                    continue
            except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
                message = f"Could not restore {TOOL_LABELS[tool_id]}: {exc}"
            outputs.extend(learning_ui.empty_output(tool_id, message))
            continue
        if tool_id == "mind_maps":
            try:
                payload = _artifacts().load_current_mind_map(workspace) if workspace else None
                message = ""
                if payload is not None:
                    map_outputs = list(_render(workspace, payload, "Restored this workspace's Mind Map."))
                    map_outputs[-1] = "Mind Map"
                    outputs.extend(map_outputs)
                    continue
            except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
                message = f"Could not restore Mind Map: {exc}"
            outputs.extend([None, message, "", "", 1.0, None, "Mind Map"])
            continue
        if tool_id == "summary":
            try:
                payload = _artifacts().load_current_summary(workspace) if workspace else None
                message = ""
                if payload is not None:
                    summary_outputs = list(_render(workspace, payload, "Restored this workspace's Summary."))
                    summary_outputs[-1] = "Summary"
                    outputs.extend(summary_outputs)
                    continue
            except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
                message = f"Could not restore Summary: {exc}"
            outputs.extend([None, message, gr.update(value="", visible=False), gr.update(value="", visible=True),
                "", gr.update(choices=[], value=None), "", None, gr.update(visible=True, interactive=False),
                gr.update(visible=False), None, "Summary"])
            continue
        try:
            payload = _artifacts().load_current_spreadsheet(workspace) if workspace else None
            message = ""
            if payload is not None:
                spreadsheet_outputs = list(_render(workspace, payload, "Restored this workspace's Spreadsheet."))
                spreadsheet_outputs[-1] = "Spreadsheet"
                outputs.extend(spreadsheet_outputs)
                continue
        except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
            message = f"Could not restore Spreadsheet: {exc}"
        editor = gr.update(value=[], headers=["Value"],
                    row_count=(0, "fixed"), column_count=(1, "fixed"))
        outputs.extend([None, message, editor, "", "", gr.update(choices=[], value=None), "", None,
                        None, "Spreadsheet"])
    job = operation_state.get("tool_job", {})
    if job and not job.get("cancelled") and job.get("workspace") == workspace:
        return (gr.skip(),) * len(outputs)
    current = (operation_state.get("revision", 0) == revision
               and (allow_switching or not operation_state.get("switching", False)))
    if current:
        operation_state["summary_editing"] = False
        operation_state["summary_export_revision"] = operation_state.get("summary_export_revision", 0) + 1
        operation_state["spreadsheet_export_revision"] = operation_state.get("spreadsheet_export_revision", 0) + 1
        operation_state["mind_map_revision"] = operation_state.get("mind_map_revision", 0) + 1
        on_mind_map_export_options(operation_state)
        if presentation_envelope is None:
            presentation_ui.clear(operation_state)
        else:
            presentation_ui.publish(presentation_envelope, operation_state)
        if graphics_envelope is None:
            graphics_ui.clear(operation_state)
        else:
            graphics_ui.publish(graphics_envelope, operation_state)
        if plot_envelope is None:
            plot_ui.clear(operation_state)
        else:
            plot_ui.publish(plot_envelope, operation_state)
        for tool_id, envelope in learning_envelopes.items():
            if envelope is None:
                learning_ui.clear(tool_id, operation_state)
            else:
                learning_ui.publish(envelope, operation_state)
        for tool_id, envelope in review_envelopes.items():
            if envelope is None:
                review_ui.clear(tool_id, operation_state)
            else:
                review_ui.publish(envelope, operation_state)
    return tuple(outputs) if current else (gr.skip(),) * len(outputs)


def on_workspace_reset(workspace, operation_state):
    """Clear old outputs before finishing the workspace switch and enabling jobs."""
    return on_reset(workspace, operation_state, allow_switching=True)


def on_visual_tool_open(tool_id, workspace, operation_state, event: gr.SelectData = None):
    """Refresh a mounted visual tab from its durable result without generation.

    Images restored while a Gradio tab is inactive can lose their browser preview.
    Rehydrate only this tab after selection, including a fresh served image URL.
    A workspace switch, generation, or newer open must win over a slow disk read.
    """
    if tool_id not in {"graphics", "plot"}:
        raise ValueError("This tool does not support visual-tab restoration.")
    skipped = (gr.skip(),) * OUTPUT_COUNTS[tool_id]
    if event is not None and not event.selected:
        return skipped
    from . import graphics_ui, plot_ui
    view = graphics_ui if tool_id == "graphics" else plot_ui
    restore_key = f"{tool_id}_restore_token"
    with _JOB_LOCK:
        revision = operation_state.get("revision", 0)
        if not _current(operation_state, revision) or operation_state.get("tool_job"):
            return skipped
        previous = dict(operation_state.get(tool_id, {}))
        job_revision = operation_state.get("tool_job_revision", 0)
        token = uuid4().hex
        operation_state[restore_key] = token

    def current():
        return (_current(operation_state, revision) and not operation_state.get("tool_job")
                and operation_state.get("tool_job_revision", 0) == job_revision
                and operation_state.get(restore_key) == token
                and operation_state.get(tool_id, {}) == previous)

    try:
        store = _artifacts()
        load = store.load_current_graphics if tool_id == "graphics" else store.load_current_plot
        payload = load(workspace) if workspace else None
        if not current():
            return skipped
        if payload is None:
            with _JOB_LOCK:
                if not current():
                    return skipped
                view.clear(operation_state)
            return view.empty_output()
        result = view.render_output(workspace, payload, f"Restored this workspace's {TOOL_LABELS[tool_id]}.")
        if not view.publish(result[0], operation_state, is_cancelled=lambda: not current()):
            return skipped
        return result
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Restoring %s tab failed", tool_id)
        return _status(f"Could not restore {TOOL_LABELS[tool_id]}: {exc}", tool_id) if current() else skipped


def on_selection_changed(operation_state):
    if cancel_tool_job(operation_state):
        return ("Source selection changed. The pending tool result will be discarded.",) * len(TOOL_LABELS)
    return (gr.skip(),) * len(TOOL_LABELS)


def _build_summary(workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    payload_state = gr.State(None)
    columns = gr.State("")
    instructions = gr.Textbox(label="Summary instructions", value=DEFAULT_SUMMARY_INSTRUCTIONS, lines=2)
    with gr.Row():
        generate_button = gr.Button("Generate Summary", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            preview = gr.Markdown(latex_delimiters=LATEX_DELIMITERS, height=480, max_height=480,
                                  container=True, padding=True, elem_classes=["rag-summary-preview"])
            editor = gr.Textbox(label="Summary Markdown", lines=20, max_lines=20, visible=False,
                info="Keep citation numbers when preserving a claim. Save applies your edits to this Summary.")
            edit_button = gr.Button("Edit", interactive=False)
            save_button = gr.Button("Save", visible=False)
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Accordion("References", open=True) as references_group:
                reference = gr.Dropdown(label="Open reference", choices=[])
                excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value="Summary")
                output_format = gr.Dropdown(label="Export format", choices=["markdown", "docx", "json"], value="markdown")
                export_button = gr.Button("Export")
    download = gr.File(visible=False, interactive=False)
    outputs = [payload_state, status, editor, preview, coverage, reference, excerpt, original,
               edit_button, save_button, download, filename]
    generation = generate_button.click(partial(on_generate, "summary"),
        [instructions, columns, workspace, selected_ids, collection, model, reasoning_effort, operation_state],
        outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(on_stop, operation_state, status, queue=False)
    edit_event = edit_button.click(on_summary_edit, [payload_state, workspace, operation_state], outputs, queue=False)
    save_event = save_button.click(on_summary_save, [payload_state, editor, workspace, operation_state], outputs,
                                  concurrency_id="rag-tool-storage", concurrency_limit=1)
    export_event = export_button.click(on_summary_export,
        [payload_state, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
                                         js=SUMMARY_DOWNLOAD_JS, queue=False)
    filename.input(on_summary_export_options, operation_state, download, queue=False)
    output_format.input(on_summary_export_options, operation_state, download, queue=False)
    reference.input(on_reference, [reference, payload_state, workspace, operation_state], [excerpt, original], queue=False,
                    preprocess=False)
    return SimpleNamespace(payload_state=payload_state, instructions=instructions, columns=columns,
        generate_button=generate_button, stop_button=stop_button, status=status, coverage=coverage,
        editor=editor, preview=preview, reference=reference, excerpt=excerpt, original=original,
        edit_button=edit_button, save_button=save_button, filename=filename, output_format=output_format,
        export_button=export_button, download=download, generation=generation, outputs=outputs,
        edit_event=edit_event, save_event=save_event, export_event=export_event, download_event=download_event,
        body=body, main_column=main_column, sidebar=sidebar, references_group=references_group,
        export_group=export_group)


def _build_spreadsheet(workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    payload_state = gr.State(None)
    instructions = gr.Textbox(label="Spreadsheet instructions", value=DEFAULT_SPREADSHEET_INSTRUCTIONS, lines=2)
    columns = gr.Textbox(label="Columns", value="Purpose, Methods, Key findings, Limitations",
        info="Separate column names with commas or new lines. One row per source.")
    with gr.Row():
        generate_button = gr.Button("Generate Spreadsheet", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            editor = gr.Dataframe(headers=["Value"], value=[], interactive=True, datatype="str",
                row_count=(0, "fixed"), column_count=(1, "fixed"), wrap=True,
                label="Spreadsheet", max_height=480)
            save_button = gr.Button("Save")
            preview = gr.Markdown(max_height=200)
            gr.Markdown("One Spreadsheet is saved automatically after generation and restored when this workspace opens. "
                        "Edit cell values, then Save to update it. Row order follows the source documents above. "
                        "Edited cells retain the original references and are marked as user edits on export.")
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Accordion("References", open=True) as references_group:
                reference = gr.Dropdown(label="Open reference", choices=[])
                excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value="Spreadsheet")
                output_format = gr.Dropdown(label="Export format", choices=["csv", "xlsx", "json"], value="csv")
                export_button = gr.Button("Export")
    download = gr.File(visible=False, interactive=False)
    outputs = [payload_state, status, editor, preview, coverage, reference, excerpt, original, download, filename]
    generation = generate_button.click(partial(on_generate, "spreadsheet"),
        [instructions, columns, workspace, selected_ids, collection, model, reasoning_effort, operation_state],
        outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(on_stop, operation_state, status, queue=False)
    save_event = save_button.click(on_spreadsheet_save,
        [payload_state, editor, workspace, operation_state], outputs,
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    export_event = export_button.click(on_spreadsheet_export,
        [payload_state, editor, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
                                         js=SPREADSHEET_DOWNLOAD_JS, queue=False)
    filename.input(on_spreadsheet_export_options, operation_state, download, queue=False)
    output_format.input(on_spreadsheet_export_options, operation_state, download, queue=False)
    editor.input(on_spreadsheet_export_options, operation_state, download, queue=False)
    reference.input(on_reference, [reference, payload_state, workspace, operation_state], [excerpt, original], queue=False,
                    preprocess=False)
    return SimpleNamespace(payload_state=payload_state, instructions=instructions, columns=columns,
        generate_button=generate_button, stop_button=stop_button, status=status, coverage=coverage,
        editor=editor, preview=preview, reference=reference, excerpt=excerpt, original=original,
        filename=filename, save_button=save_button, save_event=save_event,
        output_format=output_format, export_button=export_button, download=download,
        generation=generation, outputs=outputs,
        export_event=export_event, download_event=download_event, body=body, main_column=main_column,
        sidebar=sidebar, references_group=references_group, export_group=export_group)


def _build_mind_map(workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    payload_state = gr.State(None)
    columns = gr.State("")
    instructions = gr.Textbox(label="Mind Map instructions", value=DEFAULT_MIND_MAP_INSTRUCTIONS, lines=2)
    with gr.Row():
        generate_button = gr.Button("Generate Mind Map", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            preview = gr.HTML(min_height=480, max_height=520, container=True, padding=False)
            zoom = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Zoom")
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value="Mind Map")
                output_format = gr.Dropdown(label="Export format", choices=["svg", "markdown", "json"], value="svg")
                export_button = gr.Button("Export")
    download = gr.File(visible=False, interactive=False)
    outputs = [payload_state, status, preview, coverage, zoom, download, filename]
    generation = generate_button.click(partial(on_generate, "mind_maps"),
        [instructions, columns, workspace, selected_ids, collection, model, reasoning_effort, operation_state],
        outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(on_stop, operation_state, status, queue=False)
    zoom_event = zoom.input(on_mind_map_zoom, [zoom, payload_state, workspace, operation_state],
                            [preview, status], queue=False, trigger_mode="always_last")
    export_event = export_button.click(on_mind_map_export,
        [payload_state, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
                                         js=MIND_MAP_DOWNLOAD_JS, queue=False)
    filename.input(on_mind_map_export_options, operation_state, download, queue=False)
    output_format.input(on_mind_map_export_options, operation_state, download, queue=False)
    return SimpleNamespace(payload_state=payload_state, instructions=instructions, columns=columns,
        generate_button=generate_button, stop_button=stop_button, status=status, coverage=coverage,
        preview=preview, zoom=zoom, filename=filename, output_format=output_format,
        export_button=export_button, download=download, generation=generation, outputs=outputs,
        zoom_event=zoom_event, export_event=export_event, download_event=download_event,
        body=body, main_column=main_column, sidebar=sidebar, export_group=export_group)


def build_tools(workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    """Build lightweight controls; no library, plugin, or OpenAI access here."""
    gr.Markdown("Tools use the checked sources and the model/reasoning selected in Prompt. "
                "Generation reads every indexed chunk in the selection and may make several API requests. "
                "Use instructions to set the language and focus. Save outputs separately from chats.")
    controls = {}
    all_outputs = []
    from .presentation_ui import build as build_presentation
    from .graphics_ui import build as build_graphics
    from .plot_ui import build as build_plot
    builders = {"summary": _build_summary, "spreadsheet": _build_spreadsheet,
                "presentation": build_presentation, "graphics": build_graphics, "plot": build_plot, "mind_maps": _build_mind_map}
    from .learning_ui import build as build_learning
    builders.update({tool_id: partial(build_learning, tool_id) for tool_id in ("flashcards", "quizzes")})
    from .review_ui import build as build_review
    builders.update({tool_id: partial(build_review, tool_id) for tool_id in ("compare", "evidence_check")})
    with gr.Tabs():
        for tool_id, label in TOOL_LABELS.items():
            with gr.Tab(label) as tab:
                tool = builders[tool_id](workspace, selected_ids, collection, model, reasoning_effort, operation_state)
                controls[tool_id] = tool
                all_outputs.extend(tool.outputs)
            if tool_id in {"graphics", "plot"}:
                tool.tab = tab
                tool.restore_event = tab.select(partial(on_visual_tool_open, tool_id),
                    [workspace, operation_state], tool.outputs, queue=False,
                    trigger_mode="always_last", show_progress="hidden")
        for label in PLANNED_TOOLS:
            with gr.Tab(label):
                gr.Markdown(f"**{label}** is planned for a later stage. " + ", ".join(TOOL_LABELS.values()) + " are available now.")
    return SimpleNamespace(controls=controls, outputs=all_outputs,
                           statuses=[tool.status for tool in controls.values()])
