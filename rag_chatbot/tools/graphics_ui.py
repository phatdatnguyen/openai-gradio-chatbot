"""Source-grounded image controls, saved previews, and session-safe exports."""

from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from chatbot import config
from chatbot.log import logger


DEFAULT_INSTRUCTIONS = "Please help me create a graphic for the documents"


def _ui():
    from . import ui
    return ui


def publish(envelope, operation_state, *, is_cancelled=None):
    with _ui()._JOB_LOCK:
        if is_cancelled is not None and is_cancelled():
            return False
        operation_state["graphics"] = {
            "token": envelope["token"], "workspace": envelope["workspace"],
            # Workspace restoration precedes the switch's final revision bump.
            "workspace_revision": operation_state.get("revision", 0) + bool(operation_state.get("switching")),
        }
        on_export_options(operation_state)
        _invalidate_reference(operation_state)
    return True


def clear(operation_state):
    operation_state.pop("graphics", None)
    on_export_options(operation_state)
    _invalidate_reference(operation_state)


def _matches(envelope, workspace, operation_state):
    state = operation_state.get("graphics")
    return (bool(envelope) and bool(state) and envelope.get("workspace") == workspace
            and state["workspace"] == workspace and envelope.get("payload", {}).get("tool_id") == "graphics"
            and envelope.get("token") == state["token"]
            and _ui()._current(operation_state, state["workspace_revision"]))


def render_output(workspace, payload, message):
    ui = _ui()
    path = ui._artifacts().graphics_image_path(workspace, payload)
    envelope = {"workspace": workspace, "payload": payload, "token": uuid4().hex}
    content = payload["content"]
    return (envelope, message, str(path), content["caption"], content["prompt"], ui._coverage(payload),
            gr.update(choices=ui._reference_choices(payload), value=None), "", None, None, gr.skip())


def empty_output(message=""):
    return (None, message, None, "", "", "", gr.update(choices=[], value=None), "", None, None, "Graphics")


def on_image_model_change(image_model):
    model = config.get_image_model_config(image_model)
    return (gr.update(choices=model["size_choices"], value=model["default_size"]),
            gr.update(choices=model["quality_choices"], value=model["default_quality"]),
            gr.update(choices=model["background_choices"], value=model["default_background"]))


def on_image_format_change(output_format):
    return gr.update(interactive=output_format in config.COMPRESSIBLE_IMAGE_FORMATS)


def on_generate(instructions, image_model, image_size, image_quality, image_background,
                image_output_format, image_output_compression, image_moderation,
                workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    image_options = {"image_model": image_model, "image_size": image_size, "image_quality": image_quality,
                     "image_background": image_background, "image_output_format": image_output_format,
                     "image_output_compression": image_output_compression, "image_moderation": image_moderation}
    yield from _ui().on_generate("graphics", instructions, "", workspace, selected_ids, collection,
                                model, reasoning_effort, operation_state, image_options=image_options)


def on_export_options(operation_state):
    operation_state["graphics_export_revision"] = operation_state.get("graphics_export_revision", 0) + 1
    return None


def on_export(envelope, filename, output_format, workspace, operation_state):
    ui = _ui()
    export_revision = operation_state.get("graphics_export_revision", 0)

    def current():
        return (_matches(envelope, workspace, operation_state)
                and operation_state.get("graphics_export_revision", 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        path = ui._artifacts().export_current_graphics(workspace, envelope["payload"], output_format,
                        filename=filename, is_cancelled=lambda: not current())
        return ("Graphics exported.", str(path)) if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting graphics failed")
        return (f"Could not export Graphics: {exc}", None) if current() else (gr.skip(), None)


def _invalidate_reference(operation_state):
    revision = operation_state.get("graphics_reference_revision", 0) + 1
    operation_state["graphics_reference_revision"] = revision
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
                   and operation_state.get("graphics_reference_revision") == reference_revision)
        return result if current else (gr.skip(), gr.skip())


def build(workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    ui = _ui()
    payload_state = gr.State(None)
    instructions = gr.Textbox(label="Graphics instructions", value=DEFAULT_INSTRUCTIONS, lines=2)
    with gr.Accordion("Image options", open=True) as image_options_group:
        defaults = config.get_image_model_config(config.DEFAULT_IMAGE_MODEL)
        with gr.Row():
            image_model = gr.Dropdown(label="Image model", choices=config.IMAGE_MODEL_CHOICES,
                                      value=config.DEFAULT_IMAGE_MODEL)
            image_size = gr.Dropdown(label="Image size", choices=defaults["size_choices"], value=defaults["default_size"])
            image_quality = gr.Dropdown(label="Image quality", choices=defaults["quality_choices"],
                                        value=defaults["default_quality"])
        with gr.Row():
            image_background = gr.Dropdown(label="Image background", choices=defaults["background_choices"],
                                           value=defaults["default_background"])
            image_output_format = gr.Dropdown(label="Image format", choices=config.IMAGE_OUTPUT_FORMAT_CHOICES, value="png")
            image_output_compression = gr.Slider(label="Compression", minimum=0, maximum=100, step=1,
                                                  value=100, interactive=False)
            image_moderation = gr.Dropdown(label="Moderation", choices=config.IMAGE_MODERATION_CHOICES, value="auto")
        gr.Markdown("The Prompt model prepares a brief from your selected sources. The image model creates the graphic.")
    with gr.Row():
        generate_button = gr.Button("Generate Graphics", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            preview = gr.Image(label="Graphics", type="filepath", height=480, interactive=False,
                               buttons=["fullscreen"])
            caption = gr.Textbox(label="Caption", lines=3, max_lines=3, interactive=False)
            with gr.Accordion("Image brief", open=False):
                brief = gr.Textbox(label="Generated image prompt", lines=6, max_lines=6, interactive=False)
            gr.Markdown("Saved automatically for this workspace. References describe the evidence used for the brief; "
                        "check the generated image's labels and details against the sources.")
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Accordion("References", open=True) as references_group:
                reference = gr.Dropdown(label="Open reference", choices=[])
                excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value="Graphics")
                output_format = gr.Dropdown(label="Export format", value="image",
                    choices=[("Image (original format)", "image"), ("Markdown", "markdown"), ("JSON", "json")])
                export_button = gr.Button("Export")
    download = gr.File(visible=False, interactive=False)
    outputs = [payload_state, status, preview, caption, brief, coverage, reference, excerpt, original, download, filename]
    generation = generate_button.click(on_generate,
        [instructions, image_model, image_size, image_quality, image_background, image_output_format,
         image_output_compression, image_moderation, workspace, selected_ids, collection, model,
         reasoning_effort, operation_state], outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(ui.on_stop, operation_state, status, queue=False)
    model_event = image_model.change(on_image_model_change, image_model,
                                    [image_size, image_quality, image_background], queue=False)
    format_event = image_output_format.change(on_image_format_change, image_output_format,
                                             image_output_compression, queue=False)
    export_event = export_button.click(on_export,
        [payload_state, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
        js=ui.SUMMARY_DOWNLOAD_JS.replace('"Summary"', '"Graphics"'), queue=False)
    filename.input(on_export_options, operation_state, download, queue=False)
    output_format.input(on_export_options, operation_state, download, queue=False)
    reference.input(on_reference, [reference, payload_state, workspace, operation_state], [excerpt, original], queue=False,
                    preprocess=False)
    return SimpleNamespace(payload_state=payload_state, instructions=instructions, image_model=image_model,
        image_size=image_size, image_quality=image_quality, image_background=image_background,
        image_output_format=image_output_format, image_output_compression=image_output_compression,
        image_moderation=image_moderation, image_options_group=image_options_group, model_event=model_event,
        format_event=format_event, generate_button=generate_button, stop_button=stop_button,
        status=status, coverage=coverage, preview=preview, caption=caption, brief=brief,
        reference=reference, excerpt=excerpt, original=original, filename=filename, output_format=output_format,
        export_button=export_button, download=download, generation=generation, outputs=outputs,
        export_event=export_event, download_event=download_event, body=body, main_column=main_column,
        sidebar=sidebar, references_group=references_group, export_group=export_group)
