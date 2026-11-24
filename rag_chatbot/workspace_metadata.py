"""Small local workspace descriptions with guarded, optimistic updates."""

import json
import os
from pathlib import Path
import tempfile
import threading

from .workspaces import _linked, workspace_path


MAX_DESCRIPTION = 2000
MAX_BYTES = 16_384
_LOCK = threading.RLock()


def _path(workspace):
    root = workspace_path(workspace)
    path = root / "workspace.json"
    if path.exists() or path.is_symlink():
        if _linked(path) or not path.is_file() or path.resolve().parent != root:
            raise ValueError("Workspace settings must be an ordinary file inside the workspace.")
    return path


def _description(value):
    if (not isinstance(value, str) or len(value) > MAX_DESCRIPTION
            or any(ord(char) < 32 and char not in "\n\r\t" or 0xD800 <= ord(char) <= 0xDFFF
                   for char in value)):
        raise ValueError("Workspace description must be text of at most 2,000 characters.")
    return value


def _read(path):
    if not path.exists():
        return {"version": 1, "revision": 0, "description": ""}
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Workspace settings exceed the supported size.")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid workspace settings.") from exc
    if (not isinstance(value, dict) or set(value) != {"version", "revision", "description"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["revision"]) is not int or value["revision"] < 0):
        raise ValueError("Unsupported workspace settings format.")
    _description(value["description"])
    return value


def load(workspace):
    with _LOCK:
        return _read(_path(workspace))


def save(workspace, description, expected_revision, *, is_cancelled=None):
    description = _description(description)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Reload this workspace before saving its description.")
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The cancellation guard must be callable.")

    def check():
        if is_cancelled and is_cancelled():
            raise RuntimeError("Workspace description save cancelled.")

    with _LOCK:
        check()
        path = _path(workspace)
        current = _read(path)
        if current["revision"] != expected_revision:
            raise ValueError("The description changed in another session. Reload it before saving.")
        value = {"version": 1, "revision": expected_revision + 1, "description": description}
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        if len(raw) > MAX_BYTES:
            raise ValueError("Workspace settings exceed the supported size.")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".description-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            check()
            _path(workspace)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return value
