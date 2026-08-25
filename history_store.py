"""Saving and loading chat history.

Images used to be persisted as inline base64 data URLs, which made a single saved
chat 2.8 MB. They are now written next to the JSON under ``history/assets/<name>/``
and referenced by relative path. Loading still understands the old inline format,
so existing saved chats keep working untouched.
"""

import base64
import glob
import json
import os
import re
import shutil

from log import logger
from messages import is_data_url, normalize_text

HISTORY_DIR = "history"
ASSETS_DIR_NAME = "assets"
DEFAULT_HISTORY_NAME = "Chat history"
MAX_HISTORY_NAME_LENGTH = 100

_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_DATA_URL_HEADER = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,")

# Generated images arrive as markdown embedded in the assistant's content
# (``![Image](data:image/png;base64,...)``), which is where the bulk of a saved
# chat's size actually comes from -- not the image_url field.
_CONTENT_DATA_URL = re.compile(r"data:image/([a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)")
_ASSET_REFERENCE = re.compile(
    r"assets/[^\s)\"']+\.(?:png|jpg|jpeg|webp|gif)", re.IGNORECASE
)
IMAGE_EXTENSION_ALIASES = {"jpeg": "jpg"}


def sanitize_history_file_name(history_file_name):
    if hasattr(history_file_name, "value"):
        history_file_name = history_file_name.value
    sanitized = _UNSAFE_NAME_CHARS.sub("_", normalize_text(history_file_name))
    # Strip leading dots so a name like ".." cannot escape the history directory.
    sanitized = sanitized.lstrip(".").strip()
    return sanitized[:MAX_HISTORY_NAME_LENGTH] or DEFAULT_HISTORY_NAME


def history_file_path(safe_name):
    return os.path.join(HISTORY_DIR, f"{safe_name}.json")


def assets_dir(safe_name):
    return os.path.join(HISTORY_DIR, ASSETS_DIR_NAME, safe_name)


def get_history_file_list():
    names = [
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(HISTORY_DIR, "*.json"))
    ]
    return sorted(names, key=str.lower)


def _split_data_url(data_url):
    """Return ``(extension, base64_payload)`` for a data URL, or None."""
    match = _DATA_URL_HEADER.match(data_url or "")
    if not match:
        return None
    extension = match.group(1).lower()
    return IMAGE_EXTENSION_ALIASES.get(extension, extension), data_url[match.end():]


def _write_asset(target_dir, safe_name, file_name, payload):
    """Decode and write one image, returning its history-relative reference."""
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, file_name), "wb") as image_file:
        image_file.write(base64.b64decode(payload))
    return f"{ASSETS_DIR_NAME}/{safe_name}/{file_name}"


def _externalize_images(history, safe_name):
    """Write inline images to disk, returning a history that references them by path.

    Covers both the ``image_url`` field (user uploads) and data URLs embedded in
    the markdown of ``content`` (generated images) -- the latter is where nearly
    all of a saved chat's weight sits.
    """
    target_dir = assets_dir(safe_name)
    counts = {"written": 0}
    externalized = []

    for index, message in enumerate(history):
        message = dict(message)

        image_url = message.get("image_url")
        parsed = _split_data_url(image_url) if image_url else None
        if parsed:
            extension, payload = parsed
            try:
                message["image_url"] = _write_asset(
                    target_dir, safe_name, f"{index:04d}.{extension}", payload
                )
                counts["written"] += 1
            except (OSError, ValueError) as exc:
                # Better to keep the inline copy than lose the image.
                logger.warning("Could not externalize image %d, keeping inline: %s", index, exc)

        content = message.get("content")
        if isinstance(content, str) and "data:image/" in content:
            embedded = [0]

            def replace(match):
                extension = match.group(1).lower()
                extension = IMAGE_EXTENSION_ALIASES.get(extension, extension)
                file_name = f"{index:04d}_{embedded[0]}.{extension}"
                embedded[0] += 1
                try:
                    reference = _write_asset(target_dir, safe_name, file_name, match.group(2))
                except (OSError, ValueError) as exc:
                    logger.warning("Could not externalize embedded image: %s", exc)
                    return match.group(0)
                counts["written"] += 1
                return reference

            message["content"] = _CONTENT_DATA_URL.sub(replace, content)

        externalized.append(message)

    if counts["written"]:
        logger.info("Wrote %d image(s) to %s", counts["written"], target_dir)
    return externalized


def _read_asset_as_data_url(reference):
    """Read a history-relative asset path back into a data URL, or None if missing."""
    asset_path = os.path.join(HISTORY_DIR, reference.replace("/", os.sep))
    try:
        with open(asset_path, "rb") as image_file:
            payload = base64.b64encode(image_file.read()).decode("utf-8")
    except OSError as exc:
        logger.warning("Missing history asset %s: %s", asset_path, exc)
        return None
    extension = os.path.splitext(asset_path)[1].lstrip(".").lower() or "png"
    if extension == "jpg":
        extension = "jpeg"
    return f"data:image/{extension};base64,{payload}"


def _inline_images(history):
    """Turn relative asset paths back into data URLs for display.

    Inline base64 values are passed through unchanged, which is what keeps
    pre-existing saved histories loading correctly.
    """
    inlined = []
    for message in history:
        message = dict(message)

        image_url = message.get("image_url")
        if image_url and not is_data_url(image_url):
            data_url = _read_asset_as_data_url(image_url)
            if data_url:
                message["image_url"] = data_url
            else:
                message.pop("image_url", None)

        content = message.get("content")
        if isinstance(content, str) and f"{ASSETS_DIR_NAME}/" in content:
            message["content"] = _ASSET_REFERENCE.sub(
                lambda match: _read_asset_as_data_url(match.group(0)) or match.group(0),
                content,
            )

        inlined.append(message)
    return inlined


def save_history(history, history_file_name):
    """Persist history. Returns a status message."""
    if not history:
        raise ValueError("There is nothing to save yet.")

    safe_name = sanitize_history_file_name(history_file_name)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    stored = _externalize_images(history, safe_name)
    path = history_file_path(safe_name)
    with open(path, "w", encoding="utf8") as file:
        json.dump(stored, file, indent=4, ensure_ascii=False)

    size_kb = os.path.getsize(path) / 1024
    logger.info("Saved %s (%.1f KB)", path, size_kb)
    return f"Saved **{safe_name}** ({size_kb:,.1f} KB)."


def load_history(history_file_name):
    """Load history. Returns ``(history, status_message)``."""
    safe_name = sanitize_history_file_name(history_file_name)
    path = history_file_path(safe_name)
    with open(path, "r", encoding="utf8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError(f"{safe_name}.json does not contain a chat history.")

    history = _inline_images(history)
    logger.info("Loaded %s (%d messages)", path, len(history))
    return history, f"Loaded **{safe_name}** ({len(history)} messages)."


def delete_history(history_file_name):
    """Delete a saved history and its asset directory. Returns a status message."""
    safe_name = sanitize_history_file_name(history_file_name)
    path = history_file_path(safe_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved history named {safe_name}.")

    os.remove(path)
    target_dir = assets_dir(safe_name)
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)

    logger.info("Deleted %s", path)
    return f"Deleted **{safe_name}**."


def select_history_file(file_list, row_index):
    """Resolve a DataFrame row index to a history name using the rendered value.

    Reads from the DataFrame that is actually on screen rather than re-globbing the
    directory, so the click cannot land on a different row than the one displayed.
    """
    try:
        rows = file_list.iloc[:, 0].tolist()
    except AttributeError:
        rows = list(file_list or [])

    if row_index is None or row_index < 0 or row_index >= len(rows):
        return ""
    return str(rows[row_index])
