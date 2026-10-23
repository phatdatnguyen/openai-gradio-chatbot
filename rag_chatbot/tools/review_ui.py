"""Selected-source comparison and claim review controls with guarded downloads."""

from functools import partial
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from chatbot.log import logger


LABELS = {"compare": "Compare", "evidence_check": "Evidence Check"}
DEFAULT_INSTRUCTIONS = {
    "compare": "Please help me compare the documents",
    "evidence_check": "Please help me check the following claims against the documents",
}
HEADERS = {
    "compare": ["Aspect", "Comparison"],
    "evidence_check": ["Claim", "Assessment", "Rationale", "Quoted evidence", "References"],
}
STATUS_LABELS = {
    "found": "Found", "not_found": "Not found in selected sources", "supported": "Supported",
    "contradicted": "Contradicted", "mixed": "Mixed evidence",
}
REVIEW_NOTICE = (
    "This is a review of the selected sources, not independent fact verification. "
    "An assessment reflects the model's interpretation; review the cited passages. "
    "Not finding evidence does not establish that a claim is false."
)


def _ui():
    from . import ui
    return ui


def _preview(payload):
    from .exports import render_review_markdown
    return render_review_markdown(payload, include_sources=False)


def _label(tool_id):
    if tool_id not in LABELS:
        raise ValueError("Unknown review tool.")
    return LABELS[tool_id]


def publish(envelope, operation_state, *, is_cancelled=None):
    tool_id = envelope["payload"]["tool_id"]
    _label(tool_id)
    with _ui()._JOB_LOCK:
        if is_cancelled is not None and is_cancelled():
            return False
        operation_state[tool_id] = {
            "token": envelope["token"], "workspace": envelope["workspace"],
            # Workspace restoration precedes the final switch revision bump.
            "workspace_revision": operation_state.get("revision", 0) + bool(operation_state.get("switching")),
        }
        on_export_options(tool_id, operation_state)
        _invalidate_reference(tool_id, operation_state)
    return True


def clear(tool_id, operation_state):
    _label(tool_id)
    operation_state.pop(tool_id, None)
    on_export_options(tool_id, operation_state)
    _invalidate_reference(tool_id, operation_state)


def _matches(tool_id, envelope, workspace, operation_state):
    state = operation_state.get(tool_id)
    return (tool_id in LABELS and bool(envelope) and bool(state)
            and envelope.get("workspace") == workspace and state["workspace"] == workspace
            and envelope.get("payload", {}).get("tool_id") == tool_id
            and envelope.get("token") == state["token"]
            and _ui()._current(operation_state, state["workspace_revision"]))


def _citations(source_ids, numbers):
    return " ".join(f"[{numbers[source_id]}]" for source_id in dict.fromkeys(source_ids))


def _table(payload):
    tool_id = payload["tool_id"]
    _label(tool_id)
    numbers = {source["id"]: index + 1 for index, source in enumerate(payload["sources"])}
    if tool_id == "compare":
        source_names = {source["document_id"]: source["filename"] for source in payload["sources"]}
        documents = payload["documents"]
        names = [document.get("name") or document.get("filename")
                 or source_names.get(document["id"], document["id"]) for document in documents]
        headers = ["Aspect", *[f"{name} (document {index + 1})" if names.count(name) > 1 else name
                               for index, name in enumerate(names)], "Comparison"]
        rows = []
        for aspect in payload["content"]["aspects"]:
            findings = {finding["document_id"]: finding for finding in aspect["findings"]}
            cells = []
            for document in documents:
                finding = findings[document["id"]]
                prefix = "Found" if finding["status"] == "found" else "Not found"
                cells.append(f"{prefix}: {finding['text']} {_citations(finding['source_ids'], numbers)}".strip())
            synthesis = aspect["synthesis"]
            rows.append([aspect["aspect"], *cells,
                         f"{synthesis['text']} {_citations(synthesis['source_ids'], numbers)}".strip()])
    else:
        headers = HEADERS[tool_id]
        rows = [[claim["text"], STATUS_LABELS[claim["verdict"]], claim["rationale"],
                 "\n\n".join(f"{item['stance'].capitalize()}: {item['quote']}"
                             for item in claim["evidence"]),
                 _citations([item["source_id"] for item in claim["evidence"]], numbers)]
                for claim in payload["content"]["claims"]]
    return gr.update(value=rows, headers=headers)


def render_output(workspace, payload, message):
    ui = _ui()
    _label(payload["tool_id"])
    envelope = {"workspace": workspace, "payload": payload, "token": uuid4().hex}
    return (envelope, message, _preview(payload), _table(payload), ui._coverage(payload),
            gr.update(choices=ui._reference_choices(payload), value=None), "", None, None, gr.skip())


def empty_output(tool_id, message=""):
    label = _label(tool_id)
    return (None, message, "", gr.update(value=[], headers=HEADERS[tool_id]), "",
            gr.update(choices=[], value=None), "", None, None, label)


def on_generate(tool_id, instructions, claims_text, workspace, selected_ids, collection, model,
                reasoning_effort, operation_state):
    _label(tool_id)
    options = {"claims_text": claims_text} if tool_id == "evidence_check" else {}
    yield from _ui().on_generate(tool_id, instructions, "", workspace, selected_ids, collection,
                                model, reasoning_effort, operation_state, **options)


def on_export_options(tool_id, operation_state):
    _label(tool_id)
    key = f"{tool_id}_export_revision"
    operation_state[key] = operation_state.get(key, 0) + 1
    return None


def on_export(tool_id, envelope, filename, output_format, workspace, operation_state):
    label = _label(tool_id)
    ui = _ui()
    key = f"{tool_id}_export_revision"
    export_revision = operation_state.get(key, 0)

    def current():
        return (_matches(tool_id, envelope, workspace, operation_state)
                and operation_state.get(key, 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        path = ui._artifacts().export_review(workspace, envelope["payload"], output_format,
                    filename=filename, is_cancelled=lambda: not current())
        return (f"{label} exported.", str(path)) if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting %s failed", label)
        return (f"Could not export {label}: {exc}", None) if current() else (gr.skip(), None)


def _invalidate_reference(tool_id, operation_state):
    key = f"{tool_id}_reference_revision"
    revision = operation_state.get(key, 0) + 1
    operation_state[key] = revision
    return revision


def on_reference(tool_id, selection, envelope, workspace, operation_state):
    ui = _ui()
    with ui._JOB_LOCK:
        if not _matches(tool_id, envelope, workspace, operation_state):
            return gr.skip(), gr.skip()
        reference_revision = _invalidate_reference(tool_id, operation_state)
    result = ui.on_reference(selection, envelope, workspace, operation_state)
    with ui._JOB_LOCK:
        current = (_matches(tool_id, envelope, workspace, operation_state)
                   and operation_state.get(f"{tool_id}_reference_revision") == reference_revision)
        return result if current else (gr.skip(), gr.skip())


def build(tool_id, workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    label = _label(tool_id)
    ui = _ui()
    payload_state = gr.State(None)
    instructions = gr.Textbox(label=f"{label} instructions", value=DEFAULT_INSTRUCTIONS[tool_id], lines=2)
    claims = gr.Textbox(label="Claims to check", value="", lines=5, max_lines=8,
                        placeholder="Enter one claim per line (up to 20 claims).",
                        info="Each claim is checked against all selected documents.",
                        visible=tool_id == "evidence_check")
    gr.Markdown("Select 2–8 documents to compare." if tool_id == "compare" else
                "Enter up to 20 claims, one per line, and select the documents to review.")
    with gr.Row():
        generate_button = gr.Button(f"Generate {label}", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            preview = gr.Markdown(label=f"{label} result", latex_delimiters=[], height=480, max_height=480,
                                  container=True, padding=True, elem_classes=["rag-summary-preview"])
            table = gr.Dataframe(headers=HEADERS[tool_id], datatype="str", value=[],
                                 label="Comparison findings" if tool_id == "compare" else "Claim assessments",
                                 interactive=False, type="array", max_height=320)
            gr.Markdown("Saved automatically for this workspace. " + REVIEW_NOTICE)
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Accordion("References", open=True) as references_group:
                reference = gr.Dropdown(label="Open reference", choices=[])
                excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value=label)
                output_format = gr.Dropdown(label="Export format", value="md",
                    choices=[("Markdown", "md"), ("CSV", "csv"), ("JSON", "json")])
                export_button = gr.Button("Export")
    download = gr.File(visible=False, interactive=False)
    outputs = [payload_state, status, preview, table, coverage, reference, excerpt, original, download, filename]
    generation = generate_button.click(partial(on_generate, tool_id),
        [instructions, claims, workspace, selected_ids, collection, model, reasoning_effort, operation_state],
        outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(ui.on_stop, operation_state, status, queue=False)
    export_event = export_button.click(partial(on_export, tool_id),
        [payload_state, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
        js=ui.SUMMARY_DOWNLOAD_JS.replace('"Summary"', f'"{label}"'), queue=False)
    filename.input(partial(on_export_options, tool_id), operation_state, download, queue=False)
    output_format.input(partial(on_export_options, tool_id), operation_state, download, queue=False)
    reference.input(partial(on_reference, tool_id), [reference, payload_state, workspace, operation_state],
                    [excerpt, original], queue=False, preprocess=False)
    return SimpleNamespace(payload_state=payload_state, instructions=instructions, claims=claims,
        generate_button=generate_button, stop_button=stop_button, status=status, coverage=coverage,
        preview=preview, table=table, reference=reference, excerpt=excerpt, original=original, filename=filename,
        output_format=output_format, export_button=export_button, download=download, generation=generation,
        outputs=outputs, export_event=export_event, download_event=download_event, body=body,
        main_column=main_column, sidebar=sidebar, references_group=references_group, export_group=export_group)
