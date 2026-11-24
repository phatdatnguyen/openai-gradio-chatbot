"""Bounded, app-wide instruction presets for trusted Studio tools.

Presets contain plain prompt text only. They never install code, select source
files, or invoke a model. Per-record revisions prevent lost edits across sessions.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import unicodedata
from uuid import uuid4


SETTINGS_PATH = Path(__file__).resolve().parents[2] / ".cache" / "studio-presets.json"
MAX_PRESETS = 200
MAX_INSTRUCTION_CHARACTERS = 8000
MAX_SETTINGS_BYTES = 4 * 1024 * 1024
_ID = re.compile(r"[0-9a-f]{32}\Z")
_BAD_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff\ufffe\uffff]")
_LOCK = threading.RLock()


class PresetCancelled(RuntimeError):
    """A superseded UI action cannot replace preset preferences."""


def _tool(tool_id):
    from plugins.manager import CATALOG

    if not isinstance(tool_id, str) or tool_id not in {entry["id"] for entry in CATALOG if entry["available"]}:
        raise ValueError("Choose an available Studio tool for this preset.")
    return tool_id


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid instruction preset identifier.")
    return value


def _name(value):
    if not isinstance(value, str):
        raise ValueError("Enter a preset name.")
    value = unicodedata.normalize("NFC", value.strip())
    if (not value or len(value) > 80 or len(value.encode("utf-8", errors="replace")) > 240
            or any(ord(character) < 32 for character in value) or _BAD_TEXT.search(value)):
        raise ValueError("Preset names need 1–80 characters (at most 240 UTF-8 bytes), without control characters.")
    return value


def _instructions(value):
    if (not isinstance(value, str) or not value.strip() or len(value) > MAX_INSTRUCTION_CHARACTERS
            or _BAD_TEXT.search(value)):
        raise ValueError("Preset instructions need nonempty text of at most 8,000 characters.")
    return value


def _revision(value):
    if type(value) is not int or not 1 <= value <= 1_000_000_000:
        raise ValueError("Invalid instruction preset revision.")
    return value


def _check_cancelled(callback):
    if callback is not None and not callable(callback):
        raise ValueError("Preset cancellation guard must be callable.")
    if callback is not None and callback():
        raise PresetCancelled("Instruction preset action cancelled.")


def _path():
    path = Path(SETTINGS_PATH).absolute()
    for member in (*reversed(path.parents), path):
        if member.exists() or member.is_symlink():
            attributes = getattr(member.lstat(), "st_file_attributes", 0)
            if member.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError("Instruction preset storage cannot use linked paths.")
    if path.parent.exists() and not path.parent.is_dir():
        raise ValueError("Instruction preset storage directory is invalid.")
    if path.exists() and not path.is_file():
        raise ValueError("Instruction preset storage must be a file.")
    return path


def _validate(value):
    if (not isinstance(value, dict) or set(value) != {"version", "presets"}
            or type(value["version"]) is not int or value["version"] != 1
            or not isinstance(value["presets"], list) or len(value["presets"]) > MAX_PRESETS):
        raise ValueError("Invalid instruction preset storage format.")
    identifiers, names = set(), set()
    for record in value["presets"]:
        if not isinstance(record, dict) or set(record) != {"id", "tool_id", "name", "instructions", "revision"}:
            raise ValueError("Invalid instruction preset record.")
        identifier = _identifier(record["id"])
        tool_id = _tool(record["tool_id"])
        name = _name(record["name"])
        _instructions(record["instructions"])
        _revision(record["revision"])
        key = (tool_id, name.casefold())
        if identifier in identifiers or key in names or name != record["name"]:
            raise ValueError("Instruction presets contain duplicate identifiers or names.")
        identifiers.add(identifier)
        names.add(key)
    return value


def _read():
    path = _path()
    if not path.exists():
        return {"version": 1, "presets": []}
    with path.open("rb") as handle:
        data = handle.read(MAX_SETTINGS_BYTES + 1)
    if len(data) > MAX_SETTINGS_BYTES:
        raise ValueError("Instruction preset storage exceeds 4 MiB.")
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Could not read instruction preset JSON.") from exc
    return _validate(value)


def _write(value, is_cancelled):
    _validate(value)
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    if len(data) > MAX_SETTINGS_BYTES:
        raise ValueError("Instruction preset storage exceeds 4 MiB.")
    _check_cancelled(is_cancelled)
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-presets-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _check_cancelled(is_cancelled)
        if _path() != path:
            raise ValueError("Instruction preset storage changed during saving.")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def list_presets(tool_id):
    """Read only this tool's presets, with no directory creation or model calls."""
    _tool(tool_id)
    with _LOCK:
        return deepcopy(sorted((item for item in _read()["presets"] if item["tool_id"] == tool_id),
                               key=lambda item: (item["name"].casefold(), item["id"])))


def get_preset(tool_id, preset_id):
    _identifier(preset_id)
    return next((item for item in list_presets(tool_id) if item["id"] == preset_id), None)


def save_preset(tool_id, name, instructions, *, preset_id=None, expected_revision=None, is_cancelled=None):
    """Create or update one record while preserving other sessions' changes."""
    _tool(tool_id)
    name, instructions = _name(name), _instructions(instructions)
    _check_cancelled(is_cancelled)
    if preset_id is not None:
        _identifier(preset_id)
        _revision(expected_revision)
    elif expected_revision is not None:
        raise ValueError("A new preset cannot replace a saved preset revision.")
    with _LOCK:
        value = _read()
        current = next((item for item in value["presets"] if item["id"] == preset_id), None)
        if preset_id is not None and (current is None or current["tool_id"] != tool_id
                                      or current["revision"] != expected_revision):
            raise ValueError("This preset changed in another session. Refresh and select it again before saving.")
        if any(item["tool_id"] == tool_id and item["name"].casefold() == name.casefold()
               and item["id"] != preset_id for item in value["presets"]):
            raise ValueError("A preset with this name already exists for this tool. Select it to update it, or use a new name.")
        if current is None and len(value["presets"]) >= MAX_PRESETS:
            raise ValueError(f"Keep at most {MAX_PRESETS} instruction presets. Delete an unused preset first.")
        result = {"id": preset_id or uuid4().hex, "tool_id": tool_id, "name": name,
                  "instructions": instructions, "revision": current["revision"] + 1 if current else 1}
        value["presets"] = [item for item in value["presets"] if item["id"] != result["id"]] + [result]
        _write(value, is_cancelled)
        return deepcopy(result)


def delete_preset(tool_id, preset_id, *, expected_revision, is_cancelled=None):
    _tool(tool_id)
    _identifier(preset_id)
    _revision(expected_revision)
    _check_cancelled(is_cancelled)
    with _LOCK:
        value = _read()
        current = next((item for item in value["presets"] if item["id"] == preset_id), None)
        if current is None or current["tool_id"] != tool_id or current["revision"] != expected_revision:
            raise ValueError("This preset changed in another session. Refresh and select it again before deleting.")
        value["presets"] = [item for item in value["presets"] if item["id"] != preset_id]
        _write(value, is_cancelled)
