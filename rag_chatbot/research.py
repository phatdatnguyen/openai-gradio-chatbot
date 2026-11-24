"""Local research notes and explicitly scoped library search, without generation."""

from copy import deepcopy
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from uuid import UUID, uuid4

from chatbot.log import logger
from .workspaces import workspace_path


MAX_NOTE_BYTES = 2 * 1024 * 1024
MAX_NOTE_CHARACTERS = 100_000
MAX_NOTES = 1000
MAX_NOTE_EXPORT_BYTES = 16 * 1024 * 1024
_LOCK = threading.RLock()
ORIGINS = {"human": "Human", "ai": "AI-derived", "excerpt": "Source excerpt"}


def _guard(is_cancelled):
    if is_cancelled is not None:
        if not callable(is_cancelled):
            raise ValueError("The cancellation guard must be callable.")
        if is_cancelled():
            raise RuntimeError("Research action cancelled because its workspace or selection changed.")


def _linked(path):
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _child(parent, name, *, directory=False, create=False):
    path = parent / name
    if path.exists() or path.is_symlink():
        if _linked(path) or path.resolve().parent != parent.resolve():
            raise ValueError("Research files cannot use linked paths or leave their workspace.")
        if directory != path.is_dir() or not directory and not path.is_file():
            raise ValueError("Invalid research storage path.")
    elif create:
        path.mkdir()
    return path


def _directory(workspace, create=False):
    return _child(workspace_path(workspace), "notes", directory=True, create=create)


def _identifier(value):
    if not isinstance(value, str):
        raise ValueError("Invalid research note identifier.")
    try:
        if str(UUID(value)) != value:
            raise ValueError("Invalid research note identifier.")
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid research note identifier.") from exc
    return value


def _text(value, label, maximum, *, blank=False):
    if (not isinstance(value, str) or len(value) > maximum or "\x00" in value
            or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
            or not blank and not value.strip()):
        raise ValueError(f"{label} must contain {'0' if blank else '1'}–{maximum:,} characters.")
    return value


def _source_relative(workspace, value):
    root = workspace_path(workspace)
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("References must stay inside this workspace's sources directory.") from exc
    if not parts or parts[0] != "sources" or any(part in {".", ".."} for part in parts):
        raise ValueError("References must stay inside this workspace's sources directory.")
    current = root
    for part in parts:
        current = current / part
        if (current.exists() or current.is_symlink()) and _linked(current):
            raise ValueError("References cannot use linked files or directories.")
    if not current.resolve().is_relative_to(root / "sources"):
        raise ValueError("References must stay inside this workspace's sources directory.")
    return Path(*parts).as_posix()


def reference_path(workspace, source):
    value = source.get("source_path")
    if not isinstance(value, str) or not value:
        return None
    path = workspace_path(workspace) / _source_relative(workspace, value)
    return str(path) if path.is_file() else None


def _sources(workspace, sources):
    if not isinstance(sources, list) or len(sources) > 100:
        raise ValueError("A note may retain at most 100 source references.")
    result = []
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            raise ValueError("Invalid note reference.")
        item = {key: _text(source.get(key), f"Reference {key}", limit, blank=key == "text")
                for key, limit in (("document_id", 512), ("filename", 512), ("locator", 2000),
                                   ("text", 200_000), ("source_path", 4000))}
        item["source_path"] = _source_relative(workspace, item["source_path"])
        number = source.get("number", index)
        if type(number) is not int or not 1 <= number <= 10000:
            raise ValueError("Invalid note reference number.")
        item["number"] = number
        if isinstance(source.get("id"), str):
            item["id"] = _text(source["id"], "Reference identifier", 512)
        if source.get("image_path"):
            item["image_path"] = _source_relative(workspace, _text(source["image_path"], "Reference image path", 4000))
        result.append(item)
    if len({item["number"] for item in result}) != len(result):
        raise ValueError("Note references must have unique numbers.")
    return result


def _validate(workspace, value):
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int or value.get("schema_version") != 1
            or type(value.get("revision")) is not int or value["revision"] < 1):
        raise ValueError("Unsupported research note format.")
    _identifier(value.get("id"))
    _text(value.get("title"), "Note title", 200)
    _text(value.get("body"), "Note text", MAX_NOTE_CHARACTERS)
    if not isinstance(value.get("origin"), str) or value["origin"] not in ORIGINS:
        raise ValueError("Invalid note origin.")
    if type(value.get("user_edited")) is not bool or type(value.get("web_search")) is not bool:
        raise ValueError("Invalid note provenance.")
    for key in ("created_at", "updated_at"):
        try:
            datetime.fromisoformat(value[key])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("Invalid note timestamp.") from exc
    value["sources"] = _sources(workspace, value.get("sources"))
    return value


def load_note(workspace, identifier):
    with _LOCK:
        path = _child(_directory(workspace), f"{_identifier(identifier)}.json")
        with path.open("rb") as handle:
            data = handle.read(MAX_NOTE_BYTES + 1)
        if len(data) > MAX_NOTE_BYTES:
            raise ValueError("The research note exceeds the supported size.")
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Invalid research note JSON.") from exc
        value = _validate(workspace, value)
        if value["id"] != identifier:
            raise ValueError("The research note does not match its filename.")
        return value


def list_notes(workspace, query=""):
    query = _text(query, "Note search", 1000, blank=True).strip().casefold()
    with _LOCK:
        directory = _directory(workspace)
        if not directory.exists():
            return []
        paths = [path for path in directory.glob("*.json") if not path.name.startswith(".pending-")]
        if len(paths) > MAX_NOTES:
            raise ValueError(f"This workspace exceeds {MAX_NOTES:,} supported notes.")
        result = []
        for path in paths:
            try:
                value = load_note(workspace, path.stem)
            except (ValueError, OSError) as exc:
                logger.warning("Skipping invalid research note %s: %s", path.name, exc)
                continue
            if query and query not in (value["title"] + "\n" + value["body"]).casefold():
                continue
            result.append({key: value[key] for key in ("id", "title", "origin", "revision", "updated_at")})
        return sorted(result, key=lambda item: (item["updated_at"], item["id"]), reverse=True)


def save_note(workspace, title, body, *, identifier=None, expected_revision=None,
              origin="human", sources=None, web_search=False, is_cancelled=None):
    """Create a note or update one only if the caller still owns its revision."""
    _guard(is_cancelled)
    _text(title, "Note title", 200)
    _text(body, "Note text", MAX_NOTE_CHARACTERS)
    with _LOCK:
        now = datetime.now(timezone.utc).isoformat()
        if identifier is not None:
            old = load_note(workspace, identifier)
            if type(expected_revision) is not int or old["revision"] != expected_revision:
                raise ValueError("This note changed in another session. Reload it before saving.")
            value = {**old, "title": title, "body": body, "revision": old["revision"] + 1,
                     "updated_at": now, "user_edited": old["user_edited"] or title != old["title"] or body != old["body"]}
        else:
            if len(list_notes(workspace)) >= MAX_NOTES:
                raise ValueError(f"A workspace may contain at most {MAX_NOTES:,} notes.")
            value = {"schema_version": 1, "id": str(uuid4()), "revision": 1, "title": title, "body": body,
                     "created_at": now, "updated_at": now, "origin": origin, "user_edited": False,
                     "sources": deepcopy(sources or []), "web_search": web_search}
        value = _validate(workspace, value)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
        if len(encoded) > MAX_NOTE_BYTES:
            raise ValueError("This note and its references exceed the 2 MiB storage limit.")
        _guard(is_cancelled)
        directory = _directory(workspace, create=True)
        path = _child(directory, f"{value['id']}.json")
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _guard(is_cancelled)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return value


def delete_note(workspace, identifier, expected_revision, *, is_cancelled=None):
    with _LOCK:
        _guard(is_cancelled)
        current = load_note(workspace, identifier)
        if type(expected_revision) is not int or current["revision"] != expected_revision:
            raise ValueError("This note changed in another session. Reload it before deleting.")
        path = _child(_directory(workspace), f"{_identifier(identifier)}.json")
        _guard(is_cancelled)
        path.unlink()


def _note_export_bytes(note, output_format):
    """Portable saved-note snapshot; source paths have already been made relative."""
    value = {key: note[key] for key in (
        "schema_version", "id", "revision", "title", "body", "origin", "user_edited",
        "web_search", "created_at", "updated_at", "sources")}
    if output_format == "json":
        return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")

    def plain(text):
        text = html.escape(text, quote=False)
        for character in "\\`*_{}[]()#+-.!|":
            text = text.replace(character, "\\" + character)
        return text

    lines = [f"# {plain(note['title']).replace(chr(10), ' ').replace(chr(13), ' ')}", "",
             f"Origin: {ORIGINS[note['origin']]}", f"Note ID: {note['id']}",
             f"Revision: {note['revision']}", f"Created: {note['created_at']}",
             f"Updated: {note['updated_at']}",
             f"User edited: {'Yes' if note['user_edited'] else 'No'}",
             f"Original answer used web search: {'Yes' if note['web_search'] else 'No'}", "",
             "Notes are not automatically used as chat evidence."]
    if note["origin"] == "ai":
        lines.append("This is a model interpretation, not independent evidence.")
    if note["user_edited"]:
        lines.append("Retained references may not support text added during editing.")
    lines.extend(["", note["body"], "", "## References", ""])
    for source in note["sources"]:
        lines.extend([f"### [{source['number']}] {plain(source['filename'])}", "",
                      f"Location: {plain(source['locator'])}",
                      f"Document ID: {plain(source['document_id'])}",
                      f"Workspace-relative source: {plain(source['source_path'])}", ""])
        lines.extend("> " + plain(line) for line in source["text"].splitlines())
        lines.append("")
    if not note["sources"]:
        lines.append("No source references were saved with this note.")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def export_note(workspace, identifier, expected_revision, output_format="md", filename="Note", *, is_cancelled=None):
    """Export only the saved revision, preserving old downloads on cancellation."""
    _guard(is_cancelled)
    if output_format not in ("md", "json"):
        raise ValueError("Choose Markdown or JSON for note export.")
    # This shared validator has only standard-library imports; Office/plot
    # exporters remain lazy and are never invoked for a note.
    from .tools.exports import _export_filename
    name = _export_filename(filename, output_format, ("md", "markdown", "json"))
    with _LOCK:
        note = load_note(workspace, identifier)
        if type(expected_revision) is not int or note["revision"] != expected_revision:
            raise ValueError("This note changed in another session. Reload it before exporting.")

        def check():
            _guard(is_cancelled)
            if load_note(workspace, identifier) != note:
                raise ValueError("This note changed in another session. Reload it before exporting.")

        encoded = _note_export_bytes(note, output_format)
        if len(encoded) > MAX_NOTE_EXPORT_BYTES:
            raise ValueError("This note export exceeds the supported size.")
        check()
        directory = _child(_directory(workspace), "exports", directory=True, create=True)
        path = _child(directory, name)
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            check()
            directory = _child(_directory(workspace), "exports", directory=True)
            path = _child(directory, name)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return path


def search_sources(workspace, query, selected_ids, collection, mode="keyword", *, limit=20,
                   is_cancelled=None, service=None):
    """Return cited matches; keyword search never opens Qdrant or calls a model."""
    _guard(is_cancelled)
    query = _text(query, "Search query", 2000).strip()
    if mode not in ("keyword", "hybrid"):
        raise ValueError("Choose Keyword or Hybrid search.")
    if (not isinstance(selected_ids, (list, tuple)) or not selected_ids
            or any(not isinstance(item, str) or not item for item in selected_ids)):
        raise ValueError("Check at least one ready source to search.")
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Search must request between 1 and 50 results.")
    if service is None:
        from .service import get_service
        service = get_service(workspace)
    selected = list(dict.fromkeys(selected_ids))
    documents = {item["id"]: item for item in service.store.list_documents(collection)
                 if item["status"] == "ready" and item["fingerprint"] == service.settings.fingerprint()
                 and item.get("generation")}
    if any(identifier not in documents for identifier in selected):
        raise ValueError("A checked source is unavailable or needs reindexing. Refresh the source list.")
    snapshots = [{key: documents[identifier][key] for key in
                  ("id", "content_hash", "fingerprint", "generation", "collection", "chunk_count")}
                 for identifier in selected]

    def check():
        _guard(is_cancelled)
        service.store.validate_source_snapshot(snapshots)

    check()
    vector = None
    if mode == "hybrid":
        service.store.check_connection()
        check()
        vector = service._vectors([query])[0]
        check()
    evidence = service.store.search(query, vector, collection, document_ids=selected, limit=limit, candidates=max(40, limit))
    check()
    return [{"number": index, "id": item.chunk.id, "document_id": item.chunk.document_id,
             "filename": item.filename, "locator": item.chunk.locator, "text": item.chunk.text,
             "source_path": item.source_path, "image_path": item.chunk.image_path}
            for index, item in enumerate(evidence, 1)]
