"""Separate, versioned RAG conversations and their citation snapshots."""

import json
import os
import tempfile
from pathlib import Path

from chatbot.history_store import sanitize_history_file_name
from .config import Settings


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _directory(data_dir):
    root = Path(data_dir or Settings().data_dir).resolve()
    target = (root / "chats").resolve()
    if not target.is_relative_to(root):
        raise ValueError("Chat history must stay inside the RAG data directory.")
    return target


def list_chats(*, data_dir=None):
    directory = _directory(data_dir)
    return sorted((path.stem for path in directory.glob("*.json")
                   if not path.name.startswith(".pending-") and path.is_file()
                   and not path.is_symlink() and path.resolve().parent == directory), key=str.lower)


def _validate(history):
    if not isinstance(history, list):
        raise ValueError("RAG history must be a list of messages.")
    result = []
    for message in history:
        if (not isinstance(message, dict) or message.get("role") not in ("user", "assistant")
                or not isinstance(message.get("content"), str)):
            raise ValueError("Invalid message in RAG history.")
        item = {"role": message["role"], "content": message["content"]}
        if "web_search" in message:
            if type(message["web_search"]) is not bool:
                raise ValueError("Invalid web-search provenance in RAG history.")
            item["web_search"] = message["web_search"]
        if "sources" in message:
            sources = message["sources"]
            if not isinstance(sources, list):
                raise ValueError("Invalid citation list in RAG history.")
            numbers = set()
            for source in sources:
                if (not isinstance(source, dict) or type(source.get("number")) is not int
                        or source["number"] < 1
                        or any(not isinstance(source.get(key), str) for key in
                               ("document_id", "filename", "locator", "text", "source_path"))
                        or source.get("image_path") is not None and not isinstance(source["image_path"], str)):
                    raise ValueError("Invalid citation in RAG history.")
                if source["number"] in numbers:
                    raise ValueError("Duplicate citation number in RAG history.")
                numbers.add(source["number"])
            item["sources"] = [dict(source) for source in sources]
        result.append(item)
    return result


def save_chat(history, name, *, data_dir=None):
    name = sanitize_history_file_name(name)
    directory = _directory(data_dir)
    path = (directory / f"{name}.json").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Invalid chat filename.")
    atomic_json(path, {"version": 1, "messages": _validate(history)})
    return name


def load_chat(name, *, data_dir=None):
    directory = _directory(data_dir)
    path = (directory / f"{sanitize_history_file_name(name)}.json").resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Invalid chat filename.")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("Unsupported RAG history format.")
    return _validate(value.get("messages"))
