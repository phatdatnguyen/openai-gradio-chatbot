"""All OpenAI API access, behind one routing decision.

``process_document`` and ``process_text`` used to contain a byte-for-byte identical
34-line web-search / responses / chat branch, and ``process_image`` had a fourth,
incomplete copy that never checked RESPONSES_API_MODELS at all. Everything now goes
through ``_build_call()``, so a capability fix lands in exactly one place.
"""

import os

import requests

import config
import readers
from log import logger, truncate
from messages import (
    build_document_block,
    build_link_block,
    image_file_to_data_url,
    join_parts,
    normalize_text,
    prepare_chat_messages,
    prepare_responses_input,
)
from tokens import trim_history

import html2text

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)
URL_FETCH_TIMEOUT_SECONDS = 15

CHAT_SURFACE = "chat"
RESPONSES_SURFACE = "responses"


class ApiKeyMissingError(RuntimeError):
    pass


def fetch_url_text(url):
    """Fetch a page and convert it to markdown-ish text."""
    response = requests.get(
        url, headers={"User-Agent": BROWSER_USER_AGENT}, timeout=URL_FETCH_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return html2text.html2text(response.text)


def build_user_message(prompt_text, image_path=None, document_path=None, url=None):
    """Assemble one user history entry from whatever inputs were supplied.

    Unlike the old if/elif chain this combines inputs instead of discarding the
    lower-priority ones, so a prompt + document + link all make it through.
    """
    prompt_text = normalize_text(prompt_text)
    parts = [prompt_text]

    if document_path:
        document_text = readers.read_document(document_path)
        parts.append(build_document_block(os.path.basename(document_path), document_text))

    if url:
        parts.append(build_link_block(url, fetch_url_text(url)))

    message = {"role": "user", "content": join_parts(*parts)}
    if image_path:
        message["image_url"] = image_file_to_data_url(image_path)
    return message


def _fold_system_prompt_into_first_user_message(messages, system_prompt):
    """For o1-mini / o3-mini, which reject system messages entirely.

    Prepending the instructions to the first user turn keeps the user's intent
    rather than silently dropping it.
    """
    folded = [dict(message) for message in messages]
    for message in folded:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = f"{system_prompt}\n\n{content}"
            elif isinstance(content, list):
                content.insert(0, {"type": "text", "text": system_prompt})
            return folded
    return [{"role": "user", "content": system_prompt}] + folded


def _build_call(llm_model, history, web_search, temperature, top_p, system_prompt,
                reasoning_effort):
    """Decide the API surface and build its kwargs. Returns ``(surface, kwargs)``."""
    system_prompt = normalize_text(system_prompt)
    use_search = bool(web_search) and web_search != config.WEB_SEARCH_OFF

    if use_search and not config.supports_web_search(llm_model):
        logger.warning("%s does not support web search; ignoring it.", llm_model)
        use_search = False

    # Web search is only exposed through the Responses API, so it forces that surface.
    use_responses = use_search or config.uses_responses_api(llm_model)
    surface = RESPONSES_SURFACE if use_responses else CHAT_SURFACE

    kwargs = {"model": llm_model}

    if config.supports_temperature(llm_model):
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        logger.debug("Omitting temperature/top_p: %s is a reasoning model.", llm_model)

    effort = reasoning_effort
    if effort and effort != config.NO_REASONING_EFFORT and config.is_reasoning_model(llm_model):
        if surface == RESPONSES_SURFACE:
            kwargs["reasoning"] = {"effort": effort}
        else:
            kwargs["reasoning_effort"] = effort

    if surface == RESPONSES_SURFACE:
        payload = prepare_responses_input(history)
        if system_prompt and not config.supports_system_message(llm_model):
            payload = _fold_system_prompt_into_first_user_message(payload, system_prompt)
            system_prompt = ""
        kwargs["input"] = trim_history(
            payload, llm_model, web_search=use_search, system_prompt=system_prompt
        )
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if use_search:
            kwargs["tools"] = [{
                "type": config.WEB_SEARCH_TOOL_TYPE,
                "search_context_size": web_search,
            }]
    else:
        payload = prepare_chat_messages(history)
        trimmed = trim_history(payload, llm_model, system_prompt=system_prompt)
        if system_prompt:
            if config.supports_system_message(llm_model):
                trimmed = [{"role": "system", "content": system_prompt}] + trimmed
            else:
                trimmed = _fold_system_prompt_into_first_user_message(trimmed, system_prompt)
        kwargs["messages"] = trimmed

    logger.info(
        "%s via %s API (web_search=%s, effort=%s)",
        llm_model, surface, web_search, reasoning_effort,
    )
    return surface, kwargs


def send_chat(llm_model, history, web_search=config.WEB_SEARCH_OFF, temperature=1.0,
              top_p=1.0, system_prompt="", reasoning_effort=None):
    """Blocking call. Returns the assistant's reply text."""
    surface, kwargs = _build_call(
        llm_model, history, web_search, temperature, top_p, system_prompt, reasoning_effort
    )
    client = config.get_client()

    if surface == RESPONSES_SURFACE:
        response = client.responses.create(**kwargs)
        return (response.output_text or "").strip()

    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def stream_chat(llm_model, history, web_search=config.WEB_SEARCH_OFF, temperature=1.0,
                top_p=1.0, system_prompt="", reasoning_effort=None):
    """Streaming call. Yields text deltas as they arrive."""
    surface, kwargs = _build_call(
        llm_model, history, web_search, temperature, top_p, system_prompt, reasoning_effort
    )
    client = config.get_client()

    if surface == RESPONSES_SURFACE:
        with client.responses.stream(**kwargs) as stream:
            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
        return

    stream = client.chat.completions.create(stream=True, **kwargs)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text


def generate_image(prompt_text, image_path, image_model, image_size, image_quality,
                   image_background, image_output_format, image_output_compression,
                   image_moderation, image_input_fidelity):
    """Generate or edit an image. Returns the assistant message text.

    Raises RuntimeError when the API returns no image data, rather than the old
    behaviour of silently doing nothing.
    """
    client = config.get_client()

    image_kwargs = {
        "model": image_model,
        "prompt": prompt_text,
        "size": image_size,
        "quality": image_quality,
        "background": image_background,
        "output_format": image_output_format,
    }
    if image_output_format in config.COMPRESSIBLE_IMAGE_FORMATS:
        image_kwargs["output_compression"] = int(image_output_compression)

    is_edit = bool(image_path)
    if is_edit:
        # gpt-image-2 does not take an input_fidelity setting.
        if image_model != "gpt-image-2":
            image_kwargs["input_fidelity"] = image_input_fidelity
        logger.info("Editing image with %s", image_model)
        with open(image_path, "rb") as input_image_file:
            response = client.images.edit(image=input_image_file, **image_kwargs)
    else:
        # images.edit does not accept a moderation setting; images.generate does.
        image_kwargs["moderation"] = image_moderation
        logger.info("Generating image with %s", image_model)
        response = client.images.generate(**image_kwargs)

    image_response = response.data[0] if response.data else None
    base64_image = getattr(image_response, "b64_json", None) if image_response else None

    if not base64_image:
        raise RuntimeError(
            "The image API returned no image. This usually means the prompt was "
            "refused by moderation - try rewording it."
        )

    revised_prompt = getattr(image_response, "revised_prompt", None)
    action_label = "Edited image" if is_edit else "Generated image"
    ai_message = f"{action_label} with `{image_model}`."
    if revised_prompt and revised_prompt != prompt_text:
        ai_message += f"\n\nRevised prompt:\n{revised_prompt}"
    ai_message += f"\n\n![Image](data:image/{image_output_format};base64,{base64_image})"

    logger.debug("Image result: %s", truncate(ai_message, 200))
    return ai_message
