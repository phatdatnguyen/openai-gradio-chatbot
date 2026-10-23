"""Workspace presentation previews and local navigation; no API calls here."""

from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from chatbot.log import logger
from .presentation import DEFAULT_SLIDE_COUNT, MAX_SLIDES


DEFAULT_INSTRUCTIONS = "Please help me create a presentation from the documents"


def _ui():
    from . import ui
    return ui


def _initial(envelope):
    return {"token": envelope["token"], "workspace": envelope["workspace"], "index": 0,
            "view_revision": 0}


def publish(envelope, operation_state, *, is_cancelled=None):
    """Publish navigation state only after storage or a current workspace restore."""
    with _ui()._JOB_LOCK:
        if is_cancelled is not None and is_cancelled():
            return False
        state = _initial(envelope)
        # Restoration precedes the final workspace-switch revision increment.
        state["workspace_revision"] = operation_state.get("revision", 0) + bool(operation_state.get("switching"))
        operation_state["presentation"] = state
        on_export_options(operation_state)
        _invalidate_reference(operation_state)
    return True


def clear(operation_state):
    operation_state.pop("presentation", None)
    on_export_options(operation_state)
    _invalidate_reference(operation_state)


def _view(envelope, state, message, *, initial=False):
    from .presentation_render import render_slide_html
    ui = _ui()
    payload = envelope["payload"]
    slides = payload["content"]["slides"]
    index = state["index"]
    slide = slides[index]
    choices = [(f"{number + 1}. {item['title']}", str(number)) for number, item in enumerate(slides)]
    tail = ([ui._coverage(payload), gr.update(choices=ui._reference_choices(payload), value=None),
             "", None, None, gr.skip()] if initial else [gr.skip()] * 6)
    return ({**envelope, "view_revision": state["view_revision"]}, message,
            render_slide_html(payload, index), slide["notes"], f"Slide {index + 1} of {len(slides)}",
            gr.update(choices=choices, value=str(index), interactive=True),
            gr.update(interactive=index > 0), gr.update(interactive=index + 1 < len(slides)), *tail)


def render_output(workspace, payload, message):
    envelope = {"workspace": workspace, "payload": payload, "token": uuid4().hex, "view_revision": 0}
    return _view(envelope, _initial(envelope), message, initial=True)


def empty_output(message=""):
    return (None, message, "", "", "", gr.update(choices=[], value=None, interactive=False),
            gr.update(interactive=False), gr.update(interactive=False), "",
            gr.update(choices=[], value=None), "", None, None, "Presentation")


def _matches(envelope, workspace, operation_state, *, view=True):
    state = operation_state.get("presentation")
    return (bool(envelope) and bool(state) and envelope.get("workspace") == workspace
            and state["workspace"] == workspace and envelope.get("payload", {}).get("tool_id") == "presentation"
            and envelope.get("token") == state["token"]
            and _ui()._current(operation_state, state["workspace_revision"])
            and (not view or envelope.get("view_revision") == state["view_revision"]))


def on_navigate(action, selection, envelope, workspace, operation_state):
    ui = _ui()
    skip = (gr.skip(),) * ui.OUTPUT_COUNTS["presentation"]
    with ui._JOB_LOCK:
        if not _matches(envelope, workspace, operation_state):
            return skip
        state = deepcopy(operation_state["presentation"])
    slides = envelope["payload"]["content"]["slides"]
    if action in {"previous", "next"}:
        state["index"] = max(0, min(len(slides) - 1, state["index"] + (1 if action == "next" else -1)))
    elif action == "select":
        if not isinstance(selection, str) or selection not in {str(index) for index in range(len(slides))}:
            return ui._status("Choose an available slide.", "presentation")
        state["index"] = int(selection)
    else:
        return ui._status("Unknown slide action.", "presentation")
    state["view_revision"] += 1
    result = _view(envelope, state, gr.skip())
    with ui._JOB_LOCK:
        if not _matches(envelope, workspace, operation_state):
            return skip
        operation_state["presentation"] = state
    return result


def on_generate(instructions, slide_count, workspace, selected_ids, collection, model, reasoning_effort,
                operation_state):
    yield from _ui().on_generate("presentation", instructions, "", workspace, selected_ids, collection,
                                model, reasoning_effort, operation_state, slide_count=slide_count)


def on_export_options(operation_state):
    operation_state["presentation_export_revision"] = operation_state.get("presentation_export_revision", 0) + 1
    return None


def on_export(envelope, filename, output_format, workspace, operation_state):
    ui = _ui()
    export_revision = operation_state.get("presentation_export_revision", 0)

    def current():
        return (_matches(envelope, workspace, operation_state, view=False)
                and operation_state.get("presentation_export_revision", 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        path = ui._artifacts().export_presentation(workspace, envelope["payload"], output_format, filename=filename,
                                                   is_cancelled=lambda: not current())
        return ("Presentation exported.", str(path)) if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting presentation failed")
        return (f"Could not export Presentation: {exc}", None) if current() else (gr.skip(), None)


def on_reference(selection, envelope, workspace, operation_state):
    ui = _ui()
    with ui._JOB_LOCK:
        if not _matches(envelope, workspace, operation_state, view=False):
            return gr.skip(), gr.skip()
        reference_revision = _invalidate_reference(operation_state)
    result = _ui().on_reference(selection, envelope, workspace, operation_state)
    with ui._JOB_LOCK:
        current = (_matches(envelope, workspace, operation_state, view=False)
                   and operation_state.get("presentation_reference_revision") == reference_revision)
        return result if current else (gr.skip(), gr.skip())


def _invalidate_reference(operation_state):
    revision = operation_state.get("presentation_reference_revision", 0) + 1
    operation_state["presentation_reference_revision"] = revision
    return revision


def build(workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    ui = _ui()
    payload_state = gr.State(None)
    with gr.Row(equal_height=True) as instruction_row:
        instructions = gr.Textbox(label="Presentation instructions", value=DEFAULT_INSTRUCTIONS, lines=2, scale=3)
        slide_count = gr.Slider(label="Number of slides", minimum=1, maximum=MAX_SLIDES, step=1,
                                precision=0, value=DEFAULT_SLIDE_COUNT, scale=1, min_width=180)
    with gr.Row():
        generate_button = gr.Button("Generate Presentation", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            progress = gr.Markdown()
            preview = gr.HTML(min_height=420, max_height=520, container=True, padding=False)
            with gr.Row():
                previous_button = gr.Button("Previous", interactive=False, scale=1)
                slide_selection = gr.Dropdown(label="Slide", choices=[], interactive=False, scale=3)
                next_button = gr.Button("Next", interactive=False, scale=1)
            notes = gr.Textbox(label="Speaker notes", lines=6, max_lines=6, interactive=False)
            gr.Markdown("Saved automatically for this workspace. Export PowerPoint to edit the slides.")
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Accordion("References", open=True) as references_group:
                reference = gr.Dropdown(label="Open reference", choices=[])
                excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value="Presentation")
                output_format = gr.Dropdown(label="Export format",
                    choices=[("PowerPoint (.pptx)", "pptx"), ("Markdown", "markdown"), ("JSON", "json")], value="pptx")
                export_button = gr.Button("Export")
    download = gr.File(visible=False, interactive=False)
    outputs = [payload_state, status, preview, notes, progress, slide_selection, previous_button,
               next_button, coverage, reference, excerpt, original, download, filename]
    navigation_inputs = [slide_selection, payload_state, workspace, operation_state]
    previous_event = previous_button.click(partial(on_navigate, "previous"), navigation_inputs, outputs, queue=False)
    next_event = next_button.click(partial(on_navigate, "next"), navigation_inputs, outputs, queue=False)
    select_event = slide_selection.input(partial(on_navigate, "select"), navigation_inputs, outputs, queue=False)
    generation = generate_button.click(on_generate,
        [instructions, slide_count, workspace, selected_ids, collection, model, reasoning_effort, operation_state],
        outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(ui.on_stop, operation_state, status, queue=False)
    export_event = export_button.click(on_export,
        [payload_state, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
        js=ui.SUMMARY_DOWNLOAD_JS.replace('"Summary"', '"Presentation"'), queue=False)
    filename.input(on_export_options, operation_state, download, queue=False)
    output_format.input(on_export_options, operation_state, download, queue=False)
    reference.input(on_reference, [reference, payload_state, workspace, operation_state], [excerpt, original], queue=False,
                    preprocess=False)
    return SimpleNamespace(payload_state=payload_state, instructions=instructions, slide_count=slide_count,
        instruction_row=instruction_row, generate_button=generate_button, stop_button=stop_button,
        status=status, coverage=coverage, preview=preview, notes=notes, progress=progress,
        slide_selection=slide_selection, previous_button=previous_button, next_button=next_button,
        previous_event=previous_event, next_event=next_event, select_event=select_event,
        reference=reference, excerpt=excerpt, original=original, filename=filename, output_format=output_format,
        export_button=export_button, download=download, generation=generation, outputs=outputs,
        export_event=export_event, download_event=download_event, body=body, main_column=main_column,
        sidebar=sidebar, references_group=references_group, export_group=export_group)
