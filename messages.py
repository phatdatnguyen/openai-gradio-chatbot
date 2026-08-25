"""Conversion between the app's internal history format and the OpenAI payloads.

Internal history is a list of dicts: ``{"role", "content"}`` plus an optional
``image_url`` (a data URL) on user turns. That shape is what gets saved to JSON
and what the Gradio chatbot renders, after ``replace_history_content()`` strips
the bulky inlined document/link blocks back down to a short label.
"""

import base64
import io
import json
import re

IMAGE_DATA_URL_PREFIX = "data:image/png;base64,"
DATA_URL_PATTERN = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,")

DOCUMENT_BLOCK_TEMPLATE = "<<<DOCUMENT_CONTENT>>>\nFile: {name}\n{text}\n<<<END_DOCUMENT>>>"
LINK_BLOCK_TEMPLATE = "<<<LINK_CONTENT>>>\nURL: {url}\n{text}\n<<<END_LINK>>>"

_DOCUMENT_BLOCK_PATTERN = re.compile(
    r"(?s)<<<DOCUMENT_CONTENT>>>\nFile: (.*?)\n.*?<<<END_DOCUMENT>>>"
)
_LINK_BLOCK_PATTERN = re.compile(r"(?s)<<<LINK_CONTENT>>>\nURL: (.*?)\n.*?<<<END_LINK>>>")


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def serialize_message_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_file_to_data_url(image_path):
    """Read an image file and return it as a PNG data URL, closing the handle."""
    from PIL import Image  # local import keeps module import cost down

    with Image.open(image_path) as image:
        return f"{IMAGE_DATA_URL_PREFIX}{encode_image(image)}"


def is_data_url(value):
    return bool(value) and bool(DATA_URL_PATTERN.match(value))


def build_document_block(file_name, document_text):
    return DOCUMENT_BLOCK_TEMPLATE.format(name=file_name, text=document_text)


def build_link_block(url, link_text):
    return LINK_BLOCK_TEMPLATE.format(url=url, text=link_text)


def join_parts(*parts):
    return "\n".join(part for part in parts if part)


def prepare_chat_messages(history):
    """Convert history to a Chat Completions ``messages`` list."""
    messages = []
    for message in history or []:
        content = message.get("content", "")
        image_url = message.get("image_url")
        if message.get("role") == "user" and image_url:
            user_content = []
            if content:
                user_content.append({"type": "text", "text": content})
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": message.get("role", "user"), "content": content})
    return messages


def prepare_responses_input(history):
    """Convert history to a Responses API ``input`` list."""
    messages = []
    for message in history or []:
        content = message.get("content", "")
        image_url = message.get("image_url")
        role = message.get("role", "user")
        if role == "user" and image_url:
            user_content = []
            if content:
                user_content.append({"type": "input_text", "text": content})
            user_content.append({"type": "input_image", "image_url": image_url})
            messages.append({"role": "user", "content": user_content})
        else:
            # Plain string content is valid for every role here; only the image
            # case needs the typed content parts.
            messages.append({"role": role, "content": content})
    return messages


def replace_history_content(history):
    """Build the display copy of history for the Gradio chatbot.

    Collapses the inlined document and link blocks down to a one-line label, and
    appends any attached image as markdown so it renders in the transcript.
    """
    replaced_history = []
    for message in history or []:
        content = serialize_message_value(message.get("content", ""))

        content = _DOCUMENT_BLOCK_PATTERN.sub(
            lambda match: f"[File: {match.group(1)}]", content
        )
        content = _LINK_BLOCK_PATTERN.sub(lambda match: f"[URL: {match.group(1)}]", content)

        image_url = message.get("image_url")
        if image_url:
            content += f"\n![Image]({image_url})"

        replaced_history.append({
            "role": message.get("role", "user"),
            "content": content,
        })

    return replaced_history
