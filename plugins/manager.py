"""Small bundled tool catalog and local enable/disable preferences.

This module never installs packages, imports third-party tool modules, or makes
network calls. Adding executable plugins requires a reviewed code change.
"""

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import stat
import threading

from rag_chatbot.history import atomic_json


SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".cache" / "plugins.json"
_LOCK = threading.RLock()
_MAX_SETTINGS_BYTES = 64 * 1024
CATALOG = (
    {"id": "summary", "name": "Summary", "description": "Cited summaries of selected sources.",
     "available": True, "version": "1.0", "exports": (("DOCX", "python-docx", "docx"),)},
    {"id": "spreadsheet", "name": "Spreadsheet", "description": "Extract a cited row for each selected source.",
     "available": True, "version": "1.0", "exports": (("XLSX", "openpyxl", "openpyxl"),)},
    {"id": "presentation", "name": "Presentation", "description": "Cited slide decks with speaker notes and PowerPoint export.",
     "available": True, "version": "1.0", "exports": (("PPTX", "python-pptx", "pptx"),)},
    {"id": "graphics", "name": "Graphics", "description": "Generate images from cited visual briefs based on selected sources.",
     "available": True, "version": "1.0", "exports": ()},
    {"id": "plot", "name": "Plot", "description": "Plot cited numerical data as bar, line, or scatter charts.",
     "available": True, "version": "1.0", "exports": (("Plot previews / PNG / SVG / PDF", "matplotlib", "matplotlib"),)},
    {"id": "mind_maps", "name": "Mind Map", "description": "Map concepts and relationships in selected sources.",
     "available": True, "version": "1.0", "exports": ()},
    {"id": "flashcards", "name": "Flashcards", "description": "Study cards grounded in selected sources.",
     "available": True, "version": "1.0", "exports": ()},
    {"id": "quizzes", "name": "Quizzes", "description": "Questions with source-backed explanations and local scoring.",
     "available": True, "version": "1.0", "exports": ()},
    {"id": "compare", "name": "Compare", "description": "Compare cited findings, methods, and disagreements across selected documents.",
     "available": True, "version": "1.0", "exports": ()},
    {"id": "evidence_check", "name": "Evidence Check", "description": "Review claims against selected sources with supporting and conflicting quotations.",
     "available": True, "version": "1.0", "exports": ()},
)


def _entry(tool_id):
    for entry in CATALOG:
        if entry["id"] == tool_id:
            return entry
    raise ValueError("Unknown local tool.")


def _safe_path():
    path = Path(SETTINGS_PATH)
    for member in (path.parent, path):
        if member.exists() or member.is_symlink():
            attributes = getattr(member.lstat(), "st_file_attributes", 0)
            if member.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError("Plugin settings cannot use linked paths.")
    if path.parent.exists() and not path.parent.is_dir():
        raise ValueError("Plugin settings directory is invalid.")
    if path.exists() and not path.is_file():
        raise ValueError("Plugin settings file is invalid.")
    return path


def _read_settings():
    path = _safe_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = handle.read(_MAX_SETTINGS_BYTES + 1)
    if len(data) > _MAX_SETTINGS_BYTES:
        raise ValueError("Plugin settings exceed the supported size.")
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid local plugin settings JSON.") from exc
    if (not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1
            or not isinstance(value.get("enabled"), dict)):
        raise ValueError("Unsupported local plugin settings format.")
    available_ids = {entry["id"] for entry in CATALOG if entry["available"]}
    if any(key not in available_ids or type(enabled) is not bool for key, enabled in value["enabled"].items()):
        raise ValueError("Invalid local plugin enable/disable preferences.")
    return value["enabled"]


def is_enabled(tool_id):
    entry = _entry(tool_id)
    with _LOCK:
        return entry["available"] and _read_settings().get(tool_id, True)


def _dependency(export, distribution, module):
    try:
        version = importlib.metadata.version(distribution)
        available = importlib.util.find_spec(module) is not None
    except (importlib.metadata.PackageNotFoundError, ValueError, ImportError, AttributeError):
        version, available = None, False
    return {"export": export, "package": distribution, "installed": available, "version": version}


def get_status(tool_id):
    entry = _entry(tool_id)
    enabled = is_enabled(tool_id)
    return {"id": tool_id, "name": entry["name"], "description": entry["description"],
            "available": entry["available"], "enabled": enabled,
            "status": "Enabled" if enabled else "Disabled" if entry["available"] else "Planned",
            "version": entry.get("version", "—"),
            "dependencies": [_dependency(*dependency) for dependency in entry.get("exports", ())]}


def list_statuses():
    return [get_status(entry["id"]) for entry in CATALOG]


def set_enabled(tool_id, enabled):
    set_enabled_many({tool_id: enabled})
    return get_status(tool_id)


def set_enabled_many(changes):
    """Atomically merge explicit toggles without replacing unrelated settings."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Provide at least one plugin preference change.")
    # Validate the whole edit before writing anything (including checkbox-column
    # actions that affect several rows at once).
    for tool_id, enabled in changes.items():
        entry = _entry(tool_id)
        if not entry["available"]:
            raise ValueError(f"{entry['name']} is planned and cannot be enabled yet.")
        if type(enabled) is not bool:
            raise ValueError("Enabled must be a boolean.")
    with _LOCK:
        preferences = _read_settings()
        preferences.update(changes)
        atomic_json(_safe_path(), {"version": 1, "enabled": preferences})
