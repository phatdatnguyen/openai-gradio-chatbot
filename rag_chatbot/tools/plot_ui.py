"""Source-grounded numeric plots with local previews and guarded downloads."""

from types import SimpleNamespace
from itertools import islice
from uuid import uuid4

import gradio as gr

from chatbot.log import logger


DEFAULT_INSTRUCTIONS = "Please help me plot numerical data from the documents"
CHART_CHOICES = [("Auto", "auto"), ("Bar", "bar"), ("Line", "line"), ("Scatter", "scatter")]
EMPTY_HEADERS = ["Series", "Label", "X", "Y", "References", "Evidence"]
MAX_PREVIEW_ROWS = 200


def _ui():
    from . import ui
    return ui


def _preview(payload):
    from .plot_render import render_plot_html
    return render_plot_html(payload)


def publish(envelope, operation_state, *, is_cancelled=None):
    with _ui()._JOB_LOCK:
        if is_cancelled is not None and is_cancelled():
            return False
        operation_state["plot"] = {
            "token": envelope["token"], "workspace": envelope["workspace"],
            # Restoration runs before the workspace switch's final revision bump.
            "workspace_revision": operation_state.get("revision", 0) + bool(operation_state.get("switching")),
        }
        on_export_options(operation_state)
        _invalidate_reference(operation_state)
    return True


def clear(operation_state):
    operation_state.pop("plot", None)
    on_export_options(operation_state)
    _invalidate_reference(operation_state)


def _matches(envelope, workspace, operation_state):
    state = operation_state.get("plot")
    return (bool(envelope) and bool(state) and envelope.get("workspace") == workspace
            and state["workspace"] == workspace and envelope.get("payload", {}).get("tool_id") == "plot"
            and envelope.get("token") == state["token"]
            and _ui()._current(operation_state, state["workspace_revision"]))


def _citations(source_ids, numbers):
    return " ".join(f"[{numbers[source_id]}]" for source_id in source_ids)


def _axis_header(content, axis):
    label = content[f"{axis}_label"] or axis.upper()
    unit = content[f"{axis}_unit"]
    return f"{axis.upper()}: {label}" + (f" ({unit})" if unit else "")


def _table(payload):
    content = payload["content"]
    numbers = {source["id"]: index + 1 for index, source in enumerate(payload["sources"])}
    headers = ["Series", "Label", _axis_header(content, "x"), _axis_header(content, "y"),
               "References", "Evidence"]
    all_points = ((series, point) for series in content["series"] for point in series["points"])
    total = sum(len(series["points"]) for series in content["series"])
    rows = [[series["name"], point["label"], "" if point["x"] is None else point["x"], point["y"],
             _citations(point["source_ids"], numbers), point["evidence"]]
            for series, point in islice(all_points, MAX_PREVIEW_ROWS)]
    label = (f"Plotted data — first {MAX_PREVIEW_ROWS:,} of {total:,} points; export CSV/JSON for all rows"
             if total > MAX_PREVIEW_ROWS else f"Plotted data — {total:,} points")
    return gr.update(value=rows, headers=headers, label=label)


def render_output(workspace, payload, message):
    ui = _ui()
    envelope = {"workspace": workspace, "payload": payload, "token": uuid4().hex}
    numbers = {source["id"]: index + 1 for index, source in enumerate(payload["sources"])}
    limitations = "\n\n".join(
        f"{item['text']} {_citations(item['source_ids'], numbers)}".strip()
        for item in payload["content"]["limitations"])
    return (envelope, message, _preview(payload), _table(payload),
            gr.update(value=limitations, visible=bool(limitations)), ui._coverage(payload),
            gr.update(choices=ui._reference_choices(payload), value=None), "", None, None, gr.skip())


def empty_output(message=""):
    return (None, message, "", gr.update(value=[], headers=EMPTY_HEADERS, label="Plotted data"),
            gr.update(value="", visible=False), "", gr.update(choices=[], value=None), "", None, None, "Plot")


def on_generate(instructions, chart_type, workspace, selected_ids, collection, model, reasoning_effort,
                operation_state):
    yield from _ui().on_generate("plot", instructions, "", workspace, selected_ids, collection,
                                model, reasoning_effort, operation_state, chart_type=chart_type)


def on_export_options(operation_state):
    operation_state["plot_export_revision"] = operation_state.get("plot_export_revision", 0) + 1
    return None


def on_export(envelope, filename, output_format, workspace, operation_state):
    ui = _ui()
    export_revision = operation_state.get("plot_export_revision", 0)

    def current():
        return (_matches(envelope, workspace, operation_state)
                and operation_state.get("plot_export_revision", 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        path = ui._artifacts().export_plot(workspace, envelope["payload"], output_format,
                    filename=filename, is_cancelled=lambda: not current())
        return ("Plot exported.", str(path)) if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting Plot failed")
        return (f"Could not export Plot: {exc}", None) if current() else (gr.skip(), None)


def _invalidate_reference(operation_state):
    revision = operation_state.get("plot_reference_revision", 0) + 1
    operation_state["plot_reference_revision"] = revision
    return revision


def on_reference(selection, envelope, workspace, operation_state):
    ui = _ui()
    with ui._JOB_LOCK:
        if not _matches(envelope, workspace, operation_state):
            return gr.skip(), gr.skip()
        reference_revision = _invalidate_reference(operation_state)
    result = ui.on_reference(selection, envelope, workspace, operation_state)
    with ui._JOB_LOCK:
        current = (_matches(envelope, workspace, operation_state)
                   and operation_state.get("plot_reference_revision") == reference_revision)
        return result if current else (gr.skip(), gr.skip())


def build(workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    ui = _ui()
    payload_state = gr.State(None)
    with gr.Row() as instruction_row:
        instructions = gr.Textbox(label="Plot instructions", value=DEFAULT_INSTRUCTIONS, lines=2, scale=4)
        chart_type = gr.Dropdown(label="Chart type", choices=CHART_CHOICES, value="auto", scale=1)
    with gr.Row():
        generate_button = gr.Button("Generate Plot", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            preview = gr.HTML(label="Plot preview")
            table = gr.Dataframe(headers=EMPTY_HEADERS, datatype="str", value=[],
                                 label="Plotted data", interactive=False, type="array", max_height=320)
            limitations = gr.Textbox(label="Data limitations", lines=4, max_lines=4, interactive=False, visible=False)
            gr.Markdown("Saved automatically for this workspace. Review the extracted values, labels, and units "
                        "against the sources. Charts and exports include every plotted point; the table previews "
                        "the first 200 rows for large results.")
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Accordion("References", open=True) as references_group:
                reference = gr.Dropdown(label="Open reference", choices=[])
                excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value="Plot")
                output_format = gr.Dropdown(label="Export format", value="png",
                    choices=[("PNG", "png"), ("SVG", "svg"), ("PDF", "pdf"), ("CSV", "csv"), ("JSON", "json")])
                export_button = gr.Button("Export")
    download = gr.File(visible=False, interactive=False)
    outputs = [payload_state, status, preview, table, limitations, coverage, reference, excerpt, original, download, filename]
    generation = generate_button.click(on_generate,
        [instructions, chart_type, workspace, selected_ids, collection, model, reasoning_effort, operation_state],
        outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(ui.on_stop, operation_state, status, queue=False)
    export_event = export_button.click(on_export,
        [payload_state, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
        js=ui.SUMMARY_DOWNLOAD_JS.replace('"Summary"', '"Plot"'), queue=False)
    filename.input(on_export_options, operation_state, download, queue=False)
    output_format.input(on_export_options, operation_state, download, queue=False)
    reference.input(on_reference, [reference, payload_state, workspace, operation_state], [excerpt, original], queue=False,
                    preprocess=False)
    return SimpleNamespace(payload_state=payload_state, instructions=instructions, chart_type=chart_type,
        instruction_row=instruction_row, generate_button=generate_button, stop_button=stop_button, status=status,
        coverage=coverage, preview=preview, table=table, limitations=limitations, reference=reference,
        excerpt=excerpt, original=original, filename=filename, output_format=output_format, export_button=export_button,
        download=download, generation=generation, outputs=outputs, export_event=export_event,
        download_event=download_event, body=body, main_column=main_column, sidebar=sidebar,
        references_group=references_group, export_group=export_group)
