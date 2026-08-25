"""Gradio web UI for the OpenAI chatbot.

Layout and event wiring only -- model metadata lives in config.py, API calls in
api.py, document parsing in readers.py, token accounting in tokens.py and history
persistence in history_store.py.
"""

import pandas as pd
import gradio as gr

import api
import config
import history_store
import readers
from log import logger, setup_logging, truncate
from messages import normalize_text, replace_history_content
from tokens import InputTooLargeError

setup_logging()

HISTORY_COLUMNS = ["File name"]


def history_dataframe():
    return pd.DataFrame(history_store.get_history_file_list(), columns=HISTORY_COLUMNS)


def build_image_option_updates(image_model):
    model_config = config.get_image_model_config(image_model)
    return (
        gr.Dropdown(label="Image size", value=model_config["default_size"],
                    choices=model_config["size_choices"], interactive=True),
        gr.Dropdown(label="Image quality", value=model_config["default_quality"],
                    choices=model_config["quality_choices"], interactive=True),
        gr.Dropdown(label="Image background", value=model_config["default_background"],
                    choices=model_config["background_choices"], interactive=True),
        gr.Dropdown(label="Input fidelity", value=model_config["default_input_fidelity"],
                    choices=model_config["input_fidelity_choices"],
                    interactive=model_config["input_fidelity_interactive"]),
    )


def on_image_output_format_change(image_output_format):
    compression_enabled = image_output_format in config.COMPRESSIBLE_IMAGE_FORMATS
    return gr.Slider(label="Compression", minimum=0, maximum=100, step=1, value=100,
                     interactive=compression_enabled)


def on_llm_model_change(llm_model):
    """Re-scope the web-search and reasoning-effort controls to the chosen model.

    Only these two dropdowns are rebuilt. The image upload and the generate-image
    checkbox are deliberately left alone -- rebuilding them used to throw away an
    image the user had already staged.
    """
    if config.requires_web_search(llm_model):
        web_search = gr.Dropdown(label="Web search", value="medium",
                                 choices=["low", "medium", "high"], interactive=True)
    elif config.supports_web_search(llm_model):
        web_search = gr.Dropdown(label="Web search", value=config.WEB_SEARCH_OFF,
                                 choices=config.WEB_SEARCH_CONTEXT_CHOICES, interactive=True)
    else:
        web_search = gr.Dropdown(label="Web search", value=config.WEB_SEARCH_OFF,
                                 choices=config.WEB_SEARCH_CONTEXT_CHOICES, interactive=False)

    effort_choices = config.reasoning_effort_choices(llm_model)
    reasoning_effort = gr.Dropdown(
        label="Reasoning effort",
        value=config.NO_REASONING_EFFORT if effort_choices else None,
        choices=effort_choices or [config.NO_REASONING_EFFORT],
        interactive=bool(effort_choices),
    )

    return web_search, reasoning_effort


def make_reset_inputs():
    """Clear the per-message inputs after a send.

    The generate-image checkbox is intentionally left untouched so that generating
    several images in a row does not mean re-ticking the box every time.
    """
    return (
        gr.Textbox(label="Message", placeholder="Type a message or question...",
                   autofocus=True, scale=4, value=None),
        gr.Image(label="Upload an image", sources=["upload", "clipboard"],
                 type="filepath", value=None),
        gr.File(label="Upload a document", type="filepath", value=None),
        gr.Textbox(label="Link", value=None),
    )


def _keep_inputs():
    """Placeholder for the input components on intermediate streaming yields."""
    return (gr.skip(), gr.skip(), gr.skip(), gr.skip())


def _describe_failure(exc):
    """Turn an exception into something worth showing in the UI."""
    if isinstance(exc, InputTooLargeError):
        return str(exc)
    if isinstance(exc, readers.EmptyDocumentError):
        return str(exc)
    if isinstance(exc, RuntimeError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def on_user_input(llm_model, web_search, temperature, top_p, text, image, document, url,
                  history, generate_image, system_prompt, reasoning_effort, stream_output,
                  image_model, image_size, image_quality, image_background,
                  image_output_format, image_output_compression, image_moderation,
                  image_input_fidelity):
    """Handle a send. Generator so streamed replies render as they arrive."""
    history = list(history or [])
    prompt_text = normalize_text(text)

    if not (prompt_text or image or document or url):
        gr.Warning("Enter a message, add a link, or upload a file/image before sending.")
        yield (history, replace_history_content(history)) + _keep_inputs()
        return

    if image and not generate_image and not config.supports_vision(llm_model):
        gr.Warning(
            f"{llm_model} cannot read images. Pick a vision-capable model, or tick "
            "'Generate or edit image' to send it to an image model instead."
        )
        yield (history, replace_history_content(history)) + _keep_inputs()
        return

    if generate_image and (document or url):
        gr.Warning("Image generation uses only the prompt and the uploaded image; "
                   "the document/link was ignored.")

    # --- build the user turn -------------------------------------------------
    try:
        if generate_image:
            if not prompt_text:
                gr.Warning("Enter a prompt before generating or editing an image.")
                yield (history, replace_history_content(history)) + _keep_inputs()
                return
            user_message = api.build_user_message(prompt_text, image_path=image)
        else:
            user_message = api.build_user_message(
                prompt_text, image_path=image, document_path=document, url=url
            )
    except Exception as exc:
        logger.exception("Failed to build the user message")
        gr.Warning(_describe_failure(exc))
        yield (history, replace_history_content(history)) + _keep_inputs()
        return

    history.append(user_message)
    logger.info("User: %s", truncate(user_message["content"], 500))

    # Show the user turn and clear the inputs straight away.
    yield (history, replace_history_content(history)) + make_reset_inputs()

    # --- get the reply -------------------------------------------------------
    try:
        if generate_image:
            ai_message = api.generate_image(
                prompt_text, image, image_model, image_size, image_quality,
                image_background, image_output_format, image_output_compression,
                image_moderation, image_input_fidelity,
            )
            history.append({"role": "assistant", "content": ai_message})
            yield (history, replace_history_content(history)) + _keep_inputs()

        elif stream_output:
            history.append({"role": "assistant", "content": ""})
            chunks = []
            for delta in api.stream_chat(
                llm_model, history[:-1], web_search=web_search, temperature=temperature,
                top_p=top_p, system_prompt=system_prompt, reasoning_effort=reasoning_effort,
            ):
                chunks.append(delta)
                history[-1]["content"] = "".join(chunks)
                yield (history, replace_history_content(history)) + _keep_inputs()

            history[-1]["content"] = "".join(chunks).strip()
            if not history[-1]["content"]:
                history.pop()
                gr.Warning("The model returned an empty response.")
            yield (history, replace_history_content(history)) + _keep_inputs()

        else:
            ai_message = api.send_chat(
                llm_model, history, web_search=web_search, temperature=temperature,
                top_p=top_p, system_prompt=system_prompt, reasoning_effort=reasoning_effort,
            )
            if ai_message:
                history.append({"role": "assistant", "content": ai_message})
            else:
                gr.Warning("The model returned an empty response.")
            yield (history, replace_history_content(history)) + _keep_inputs()

        logger.info("AI: %s", truncate(history[-1]["content"] if history else "", 500))

    except Exception as exc:
        logger.exception("Request to %s failed", llm_model)
        gr.Warning(_describe_failure(exc))
        # Roll the pending turn back out so it is not silently resent next time.
        if history and history[-1].get("role") == "assistant":
            history.pop()
        if history and history[-1].get("role") == "user":
            history.pop()
        yield (history, replace_history_content(history)) + _keep_inputs()


def on_new_chat_click():
    return [], []


def on_toggle_history_column(state):
    state = not state
    return gr.update(visible=state), state


def on_select_history_file(file_list, evt: gr.SelectData):
    row_index = evt.index[0] if evt.index else None
    return history_store.select_history_file(file_list, row_index)


def on_save_history(history, history_file_name):
    try:
        status = history_store.save_history(history, history_file_name)
    except Exception as exc:
        logger.exception("Saving history failed")
        status = f"Error saving history: {_describe_failure(exc)}"
    return status, history_dataframe()


def on_load_history(history_file_name):
    try:
        history, status = history_store.load_history(history_file_name)
        return history, replace_history_content(history), status
    except Exception as exc:
        logger.exception("Loading history failed")
        return [], [], f"Error loading history: {_describe_failure(exc)}"


def on_delete_history(history_file_name):
    try:
        status = history_store.delete_history(history_file_name)
    except Exception as exc:
        logger.exception("Deleting history failed")
        status = f"Error deleting history: {_describe_failure(exc)}"
    return status, history_dataframe()


with gr.Blocks(title="OpenAI Chatbot") as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(buttons=["copy"], min_height=800)
            state = gr.State([])
            history_column_state = gr.State(True)
            with gr.Row(equal_height=True):
                new_chat_button = gr.Button(value="New chat")
                stop_button = gr.Button(value="Stop", variant="stop")
                toggle_history_column_button = gr.Button(value="Toggle chat history")
        with gr.Column(scale=1) as history_column:
            history_files_list = gr.DataFrame(label="Chat history files",
                                              value=history_dataframe(), max_height=260)
            history_file_name = gr.Textbox(label="File name",
                                           value=history_store.DEFAULT_HISTORY_NAME)
            load_button = gr.Button(value="Load chat history")
            load_status = gr.Markdown(value="")
            save_button = gr.Button(value="Save chat history")
            save_status = gr.Markdown(value="")
            delete_button = gr.Button(value="Delete chat history", variant="stop")
            delete_status = gr.Markdown(value="")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion(label="Prompt"):
                with gr.Row(equal_height=True):
                    text_input = gr.Textbox(label="Message",
                                            placeholder="Type a message or question...",
                                            autofocus=True, scale=4)
                    send_button = gr.Button(value="Send", variant="primary", scale=1,
                                            min_width=80)
                system_prompt = gr.Textbox(label="System prompt",
                                           placeholder="Optional instructions for the AI...",
                                           lines=2)
                llm_model = gr.Dropdown(label="Model", value=config.DEFAULT_MODEL,
                                        choices=config.MODEL_CHOICES)
                web_search = gr.Dropdown(label="Web search", value=config.WEB_SEARCH_OFF,
                                         choices=config.WEB_SEARCH_CONTEXT_CHOICES)
                _default_effort_choices = config.reasoning_effort_choices(config.DEFAULT_MODEL)
                reasoning_effort = gr.Dropdown(
                    label="Reasoning effort",
                    value=config.NO_REASONING_EFFORT,
                    choices=_default_effort_choices or [config.NO_REASONING_EFFORT],
                    interactive=bool(_default_effort_choices),
                )
                stream_output = gr.Checkbox(label="Stream responses", value=True)
                temperature = gr.Slider(label="Temperature", minimum=0, maximum=2,
                                        step=0.01, value=1)
                top_p = gr.Slider(label="Top-p", minimum=0, maximum=1, step=0.01, value=1)
        with gr.Column(scale=1):
            with gr.Accordion(label="Input"):
                image_input = gr.Image(label="Upload an image",
                                       sources=["upload", "clipboard"], type="filepath")
                document_input = gr.File(label="Upload a document", type="filepath")
                url_input = gr.Textbox(label="Link")
        with gr.Column(scale=1):
            with gr.Accordion(label="Image generation and editing"):
                _default_image_config = config.get_image_model_config(config.DEFAULT_IMAGE_MODEL)
                generate_image = gr.Checkbox(label="Generate or edit image", value=False)
                image_model = gr.Dropdown(label="Image model",
                                          value=config.DEFAULT_IMAGE_MODEL,
                                          choices=config.IMAGE_MODEL_CHOICES)
                image_size = gr.Dropdown(label="Image size",
                                         value=_default_image_config["default_size"],
                                         choices=_default_image_config["size_choices"])
                image_quality = gr.Dropdown(label="Image quality",
                                            value=_default_image_config["default_quality"],
                                            choices=_default_image_config["quality_choices"])
                image_background = gr.Dropdown(label="Image background",
                                               value=_default_image_config["default_background"],
                                               choices=_default_image_config["background_choices"])
                image_output_format = gr.Dropdown(label="Output format", value="png",
                                                  choices=config.IMAGE_OUTPUT_FORMAT_CHOICES)
                image_output_compression = gr.Slider(label="Compression", minimum=0,
                                                     maximum=100, step=1, value=100,
                                                     interactive=False)
                image_moderation = gr.Dropdown(label="Moderation", value="auto",
                                               choices=config.IMAGE_MODERATION_CHOICES)
                image_input_fidelity = gr.Dropdown(
                    label="Input fidelity",
                    value=_default_image_config["default_input_fidelity"],
                    choices=_default_image_config["input_fidelity_choices"],
                    interactive=_default_image_config["input_fidelity_interactive"],
                )

    user_input_inputs = [
        llm_model, web_search, temperature, top_p, text_input, image_input, document_input,
        url_input, state, generate_image, system_prompt, reasoning_effort, stream_output,
        image_model, image_size, image_quality, image_background, image_output_format,
        image_output_compression, image_moderation, image_input_fidelity,
    ]
    user_input_outputs = [state, chatbot, text_input, image_input, document_input, url_input]

    submit_event = text_input.submit(on_user_input, user_input_inputs, user_input_outputs)
    click_event = send_button.click(on_user_input, user_input_inputs, user_input_outputs)
    stop_button.click(None, None, None, cancels=[submit_event, click_event])

    new_chat_button.click(on_new_chat_click, [], [chatbot, state])
    toggle_history_column_button.click(on_toggle_history_column, history_column_state,
                                       [history_column, history_column_state])

    llm_model.change(on_llm_model_change, llm_model, [web_search, reasoning_effort])
    image_model.change(build_image_option_updates, image_model,
                       [image_size, image_quality, image_background, image_input_fidelity])
    image_output_format.change(on_image_output_format_change, image_output_format,
                               image_output_compression)

    history_files_list.select(on_select_history_file, history_files_list, history_file_name)
    load_button.click(on_load_history, history_file_name, [state, chatbot, load_status])
    save_button.click(on_save_history, [state, history_file_name],
                      [save_status, history_files_list])
    delete_button.click(on_delete_history, history_file_name,
                        [delete_status, history_files_list])

if __name__ == "__main__":
    config.check_model_tables()
    if config.client is None:
        logger.warning(config.MISSING_KEY_MESSAGE)
    demo.launch(max_file_size=100 * gr.FileSize.MB)
