"""Current and versioned Studio outputs, separate from chat histories."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from uuid import UUID, uuid4

from chatbot.log import logger
from rag_chatbot.workspaces import workspace_path


MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_LOCK = threading.RLock()
_LEARNING_TOOL_NAMES = {"flashcards": "Flashcards", "quizzes": "Quizzes"}
_REVIEW_TOOL_NAMES = {"compare": "Compare", "evidence_check": "Evidence Check"}
_TOOL_IDS = {"summary", "spreadsheet", "mind_maps", "presentation", "graphics", "plot",
             *_LEARNING_TOOL_NAMES, *_REVIEW_TOOL_NAMES}


def _linked(path):
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _child(parent, name, *, directory=False, create=False):
    """Check each path component, including dangling links and Windows junctions."""
    path = parent / name
    if path.exists() or path.is_symlink():
        if _linked(path) or path.resolve().parent != parent.resolve():
            raise ValueError("Studio files cannot use linked paths or leave their workspace.")
        if (directory and not path.is_dir()) or (not directory and not path.is_file()):
            raise ValueError("Invalid Studio storage path.")
    elif create:
        if not directory:
            raise ValueError("Only Studio directories can be created here.")
        path.mkdir()
    return path


def _directory(workspace, *, create=False):
    root = workspace_path(workspace)
    return _child(root, "artifacts", directory=True, create=create)


def _artifact_id(value):
    if not isinstance(value, str):
        raise ValueError("Invalid artifact identifier.")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid artifact identifier.") from exc
    if str(parsed) != value:
        raise ValueError("Invalid artifact identifier.")
    return value


def _text(value, label):
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label} in Studio output.")
    return value


def _validate(payload, *, persisted=False):
    if (not isinstance(payload, dict) or not isinstance(payload.get("tool_id"), str)
            or payload["tool_id"] not in _TOOL_IDS):
        raise ValueError("Unsupported Studio output type.")
    if not _text(payload.get("title"), "title").strip() or len(payload["title"]) > 512:
        raise ValueError("Studio output titles must contain 1–512 characters.")
    if persisted:
        if type(payload.get("version")) is not int or payload["version"] != 1:
            raise ValueError("Unsupported Studio output version.")
        _artifact_id(payload.get("id"))
        timestamp = _text(payload.get("created_at"), "creation time")
        try:
            datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise ValueError("Invalid Studio output creation time.") from exc
    if "updated_at" in payload:
        try:
            datetime.fromisoformat(_text(payload["updated_at"], "update time"))
        except ValueError as exc:
            raise ValueError("Invalid Studio output update time.") from exc
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Studio sources must be a list.")
    source_ids = set()
    source_documents = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Invalid Studio source.")
        for key in ("id", "document_id", "filename", "locator", "text", "source_path"):
            _text(source.get(key), f"source {key}")
        if not source["id"] or source["id"] in source_ids:
            raise ValueError("Studio sources must have unique identifiers.")
        source_ids.add(source["id"])
        source_documents[source["id"]] = source["document_id"]
        for key in ("source_path", "image_path"):
            path = source.get(key)
            if path is not None:
                _text(path, key)
                if Path(path).is_absolute() or ".." in Path(path).parts or "\\" in path or ":" in path:
                    raise ValueError("Studio source paths must be relative workspace paths.")
    if not isinstance(payload.get("documents"), list) or any(
            not isinstance(item, dict) for item in payload["documents"]):
        raise ValueError("Studio document snapshots must be a list of objects.")
    document_ids = []
    for document in payload["documents"]:
        identifier = _text(document.get("id"), "document identifier")
        if not identifier or identifier in document_ids:
            raise ValueError("Studio documents must have unique identifiers.")
        document_ids.append(identifier)
    if not set(source_documents.values()).issubset(document_ids):
        raise ValueError("Studio sources must belong to the saved document snapshot.")
    for key in ("coverage", "settings"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"Studio {key} must be an object.")
    _text(payload.get("model"), "model")

    def check_citations(item):
        citations = item.get("source_ids")
        if (not isinstance(citations, list) or any(not isinstance(value, str) for value in citations)
                or not set(citations).issubset(source_ids)):
            raise ValueError("Studio output cites an unknown source.")

    content = payload.get("content")
    if not isinstance(content, dict):
        raise ValueError("Invalid Studio content.")
    if payload["tool_id"] == "summary":
        sections = content.get("sections")
        if not isinstance(sections, list):
            raise ValueError("Summary sections must be a list.")
        for section in sections:
            if not isinstance(section, dict) or not isinstance(section.get("paragraphs"), list):
                raise ValueError("Invalid Summary section.")
            _text(section.get("heading"), "section heading")
            for paragraph in section["paragraphs"]:
                if not isinstance(paragraph, dict):
                    raise ValueError("Invalid Summary paragraph.")
                _text(paragraph.get("text"), "paragraph")
                check_citations(paragraph)
        if "edited_text" in payload:
            _text(payload["edited_text"], "edited Summary")
    elif payload["tool_id"] == "mind_maps":
        from .mindmap import validate_mind_map
        validate_mind_map(content, allowed_source_ids=source_ids)
    elif payload["tool_id"] == "presentation":
        from .presentation import validate_presentation
        validate_presentation(content, allowed_source_ids=source_ids)
    elif payload["tool_id"] == "graphics":
        from .graphics import validate_graphics
        validate_graphics(content, allowed_source_ids=source_ids, persisted=True)
        if {source_documents[identifier] for identifier in content["source_ids"]} != set(document_ids):
            raise ValueError("Graphics must cite every selected source document.")
    elif payload["tool_id"] == "plot":
        from .plot import validate_chart_type, validate_plot
        validate_plot(content, allowed_source_ids=source_ids, sources=sources)
        validate_chart_type(payload["settings"].get("chart_type", "auto"))
        cited = {identifier for series in content["series"] for point in series["points"]
                 for identifier in point["source_ids"]}
        cited.update(identifier for item in content["limitations"] for identifier in item["source_ids"])
        if {source_documents[identifier] for identifier in cited} != set(document_ids):
            raise ValueError("Plot must represent every selected document with numeric data or a cited limitation.")
    elif payload["tool_id"] in _LEARNING_TOOL_NAMES:
        from .learning import validate_learning_content
        validate_learning_content(payload["tool_id"], content, allowed_source_ids=source_ids)
    elif payload["tool_id"] == "compare":
        from .review import validate_compare
        validate_compare(content, documents=payload["documents"], sources=sources)
    elif payload["tool_id"] == "evidence_check":
        from .review import validate_evidence_check
        validate_evidence_check(content, claims=payload["settings"].get("claims"), sources=sources)
    else:
        columns, rows = content.get("columns"), content.get("rows")
        if (not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns)
                or len(columns) != len(set(columns)) or not isinstance(rows, list)):
            raise ValueError("Invalid Spreadsheet columns or rows.")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("cells"), list):
                raise ValueError("Invalid Spreadsheet row.")
            _text(row.get("document_id"), "row document")
            _text(row.get("filename"), "row filename")
            if row["document_id"] not in document_ids:
                raise ValueError("Spreadsheet row must belong to the saved document snapshot.")
            cells = row["cells"]
            if len(cells) != len(columns) or any(not isinstance(cell, dict) for cell in cells):
                raise ValueError("Spreadsheet row does not match its columns.")
            if [cell.get("column") for cell in cells] != columns:
                raise ValueError("Spreadsheet cells must follow the declared columns.")
            for cell in cells:
                _text(cell.get("value"), "cell")
                check_citations(cell)
                if any(source_documents[source_id] != row["document_id"] for source_id in cell["source_ids"]):
                    raise ValueError("Spreadsheet cells must cite their own source document.")
        if "edited_rows" in payload:
            edited = payload["edited_rows"]
            if (not isinstance(edited, list) or len(edited) != len(rows)
                    or any(not isinstance(row, list) or len(row) != len(columns)
                           or any(not isinstance(cell, str) for cell in row) for row in edited)):
                raise ValueError("Edited Spreadsheet must keep the original row and column count.")
    return payload


def _relative_paths(payload, root):
    if not isinstance(payload.get("sources"), list) or not isinstance(payload.get("documents"), list):
        raise ValueError("Studio sources and document snapshots must be lists.")
    for source in payload.get("sources", []):
        if not isinstance(source, dict):
            continue
        for key in ("source_path", "image_path"):
            value = source.get(key)
            if isinstance(value, str) and value:
                path = Path(value)
                if path.is_absolute():
                    try:
                        source[key] = path.resolve().relative_to(root).as_posix()
                    except ValueError as exc:
                        raise ValueError("Studio sources must belong to their workspace.") from exc
                else:
                    source[key] = path.as_posix()
    for document in payload.get("documents", []):
        if isinstance(document, dict) and isinstance(document.get("path"), str):
            path = Path(document["path"])
            if path.is_absolute():
                try:
                    document["path"] = path.resolve().relative_to(root).as_posix()
                except ValueError as exc:
                    raise ValueError("Studio documents must belong to their workspace.") from exc


def _encoded(value):
    try:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError("Studio output must contain valid finite JSON data.") from exc
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Studio output is too large to save (maximum 16 MiB).")
    return data


def _atomic_bytes(path, data, *, before_replace=None):
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_artifact(workspace, payload):
    """Save a new immutable snapshot; edits and regeneration receive new IDs."""
    with _LOCK:
        if not isinstance(payload, dict):
            raise ValueError("Studio output must be an object.")
        value = json.loads(_encoded(payload))
        root = workspace_path(workspace)
        _relative_paths(value, root)
        value.update(id=str(uuid4()), version=1, created_at=datetime.now(timezone.utc).isoformat())
        _validate(value, persisted=True)
        encoded = _encoded(value)
        directory = _directory(workspace, create=True)
        target = _child(directory, value["id"], directory=True, create=True)
        _atomic_bytes(target / "manifest.json", encoded)
        return value


def load_artifact(workspace, artifact_id):
    with _LOCK:
        directory = _directory(workspace)
        target = _child(directory, _artifact_id(artifact_id), directory=True)
        manifest = _child(target, "manifest.json")
        value = _read_manifest(manifest)
        if value["id"] != artifact_id:
            raise ValueError("Studio output identifier does not match its directory.")
        return value


def _read_manifest(path):
    with path.open("rb") as handle:
        data = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("Studio output exceeds the supported size.")
    try:
        value = json.loads(data, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid Studio output JSON.") from exc
    return _validate(value, persisted=True)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value {value} is not supported.")


def list_artifacts(workspace, tool_id=None):
    if tool_id is not None and (not isinstance(tool_id, str) or tool_id not in _TOOL_IDS):
        raise ValueError("Unsupported Studio tool.")
    with _LOCK:
        directory = _directory(workspace)
        if not directory.exists():
            return []
        result = []
        for child in directory.iterdir():
            if child.name in _TOOL_IDS:
                # Current tool outputs and direct downloads are separate from saved versions.
                continue
            try:
                _artifact_id(child.name)
                value = load_artifact(workspace, child.name)
            except (ValueError, OSError) as exc:
                logger.warning("Skipping invalid Studio output %s: %s", child.name, exc)
                continue
            if tool_id is None or value["tool_id"] == tool_id:
                result.append({key: value[key] for key in ("id", "version", "title", "tool_id", "created_at")})
        return sorted(result, key=lambda item: (item["created_at"], item["id"]), reverse=True)


def _summary_directory(workspace, *, create=False):
    return _child(_directory(workspace, create=create), "summary", directory=True, create=create)


def _existing_summary(workspace):
    """Read the current Summary or the newest legacy version without writing."""
    current = _child(_summary_directory(workspace), "current.json")
    if current.exists():
        value = _read_manifest(current)
        if value["tool_id"] != "summary":
            raise ValueError("The current Summary file contains a different tool's output.")
        return value, True
    legacy = list_artifacts(workspace, "summary")
    return (load_artifact(workspace, legacy[0]["id"]), False) if legacy else (None, False)


def load_current_summary(workspace):
    """Load the single Summary, copying the latest legacy output once if needed."""
    with _LOCK:
        value, current_exists = _existing_summary(workspace)
        if value is not None and not current_exists:
            value.setdefault("updated_at", value["created_at"])
            target = _child(_summary_directory(workspace, create=True), "current.json")
            _atomic_bytes(target, _encoded(value))
        return value


def save_current_summary(workspace, payload, *, is_cancelled=None):
    """Replace the current Summary atomically, retaining its identity and creation time."""
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Summary cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Summary save cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "summary":
            raise ValueError("Only a Summary can be saved as the current Summary.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value)
        existing, _ = _existing_summary(workspace)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        target = _child(_summary_directory(workspace, create=True), "current.json")
        _atomic_bytes(target, _encoded(value), before_replace=ensure_active)
        return value


def _export_guard(is_cancelled, label):
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError(f"The {label} cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError(f"{label} export cancelled.")

    return ensure_active


def export_summary(workspace, payload, format, filename="Summary", *, is_cancelled=None):
    """Export the supplied Summary without saving or changing its current version."""
    from .exports import export_bytes, summary_export_filename

    ensure_active = _export_guard(is_cancelled, "Summary")
    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "summary":
            raise ValueError("Only a Summary can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        extension, data = export_bytes(value, format)
        name = summary_export_filename(filename, extension)
        directory = _child(_summary_directory(workspace, create=True), "exports", directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def _spreadsheet_directory(workspace, *, create=False):
    return _child(_directory(workspace, create=create), "spreadsheet", directory=True, create=create)


def _existing_spreadsheet(workspace):
    """Read the current Spreadsheet or newest saved version without writing."""
    current = _child(_spreadsheet_directory(workspace), "current.json")
    if current.exists():
        value = _read_manifest(current)
        if value["tool_id"] != "spreadsheet":
            raise ValueError("The current Spreadsheet file contains a different tool's output.")
        return value, True
    legacy = list_artifacts(workspace, "spreadsheet")
    return (load_artifact(workspace, legacy[0]["id"]), False) if legacy else (None, False)


def load_current_spreadsheet(workspace):
    """Load the current table, adopting the latest saved version only if absent."""
    with _LOCK:
        value, current_exists = _existing_spreadsheet(workspace)
        if value is not None and not current_exists:
            value.setdefault("updated_at", value["created_at"])
            target = _child(_spreadsheet_directory(workspace, create=True), "current.json")
            _atomic_bytes(target, _encoded(value))
        return value


def save_current_spreadsheet(workspace, payload, *, is_cancelled=None):
    """Replace the current table atomically, retaining its identity and provenance."""
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Spreadsheet cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Spreadsheet save cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "spreadsheet":
            raise ValueError("Only a Spreadsheet can be saved as the current Spreadsheet.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value)
        existing, _ = _existing_spreadsheet(workspace)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        target = _child(_spreadsheet_directory(workspace, create=True), "current.json")
        _atomic_bytes(target, _encoded(value), before_replace=ensure_active)
        return value


def export_spreadsheet(workspace, payload, format, filename="Spreadsheet", *, is_cancelled=None):
    """Export the current editable table directly, without creating a saved version."""
    from .exports import export_bytes, spreadsheet_export_filename

    ensure_active = _export_guard(is_cancelled, "Spreadsheet")
    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "spreadsheet":
            raise ValueError("Only a Spreadsheet can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        extension, data = export_bytes(value, format)
        name = spreadsheet_export_filename(filename, extension)
        tool_directory = _spreadsheet_directory(workspace, create=True)
        directory = _child(tool_directory, "exports", directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def _mind_map_directory(workspace, *, create=False):
    return _child(_directory(workspace, create=create), "mind_maps", directory=True, create=create)


def load_current_mind_map(workspace):
    """Load this workspace's one current Mind Map without creating directories."""
    with _LOCK:
        current = _child(_mind_map_directory(workspace), "current.json")
        if not current.exists():
            return None
        value = _read_manifest(current)
        if value["tool_id"] != "mind_maps":
            raise ValueError("The current Mind Map file contains a different tool's output.")
        return value


def save_current_mind_map(workspace, payload, *, is_cancelled=None):
    """Publish one current map; cancellation cannot overwrite the previous map."""
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Mind Map cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Mind Map save cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "mind_maps":
            raise ValueError("Only a Mind Map can be saved as the current Mind Map.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value)
        existing = load_current_mind_map(workspace)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        target = _child(_mind_map_directory(workspace, create=True), "current.json")
        _atomic_bytes(target, _encoded(value), before_replace=ensure_active)
        return value


def export_mind_map(workspace, payload, format, filename="Mind Map", *, is_cancelled=None):
    """Export trusted SVG, Markdown, or JSON without creating saved versions."""
    from .exports import export_bytes, mind_map_export_filename

    ensure_active = _export_guard(is_cancelled, "Mind Map")
    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "mind_maps":
            raise ValueError("Only a Mind Map can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        extension, data = export_bytes(value, format)
        name = mind_map_export_filename(filename, extension)
        directory = _child(_mind_map_directory(workspace, create=True), "exports", directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def _presentation_directory(workspace, *, create=False):
    return _child(_directory(workspace, create=create), "presentation", directory=True, create=create)


def load_current_presentation(workspace):
    """Read the current deck without creating files or changing its content."""
    with _LOCK:
        current = _child(_presentation_directory(workspace), "current.json")
        if not current.exists():
            return None
        value = _read_manifest(current)
        if value["tool_id"] != "presentation":
            raise ValueError("The current Presentation file contains a different tool's output.")
        return value


def save_current_presentation(workspace, payload, *, is_cancelled=None):
    """Replace this workspace's deck atomically while preserving its identity."""
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Presentation cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Presentation save cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "presentation":
            raise ValueError("Only a Presentation can be saved as the current Presentation.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value)
        existing = load_current_presentation(workspace)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        target = _child(_presentation_directory(workspace, create=True), "current.json")
        _atomic_bytes(target, _encoded(value), before_replace=ensure_active)
        return value


def export_presentation(workspace, payload, format="pptx", filename="Presentation", *, is_cancelled=None):
    """Download the current deck as text or editable slides without saving a version."""
    from .exports import export_bytes, presentation_export_filename

    ensure_active = _export_guard(is_cancelled, "Presentation")
    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "presentation":
            raise ValueError("Only a Presentation can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        extension, data = export_bytes(value, format)
        name = presentation_export_filename(filename, extension)
        directory = _child(_presentation_directory(workspace, create=True), "exports", directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def _plot_directory(workspace, *, create=False):
    return _child(_directory(workspace, create=create), "plot", directory=True, create=create)


def load_current_plot(workspace):
    """Restore only numeric data, without importing an optional renderer."""
    with _LOCK:
        current = _child(_plot_directory(workspace), "current.json")
        if not current.exists():
            return None
        value = _read_manifest(current)
        if value["tool_id"] != "plot":
            raise ValueError("The current Plot file contains a different tool's output.")
        return value


def save_current_plot(workspace, payload, *, is_cancelled=None):
    """Atomically replace the current plot while preserving its identity."""
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Plot cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Plot save cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "plot":
            raise ValueError("Only a Plot can be saved as the current Plot.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value)
        existing = load_current_plot(workspace)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        target = _child(_plot_directory(workspace, create=True), "current.json")
        _atomic_bytes(target, _encoded(value), before_replace=ensure_active)
        return value


def export_plot(workspace, payload, format="png", filename="Plot", *, is_cancelled=None):
    """Export trusted numeric data locally without modifying the saved result."""
    from .exports import export_bytes, plot_export_filename

    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Plot cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Plot export cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "plot":
            raise ValueError("Only a Plot can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        # Validate the basename before spending time rendering the chart.
        normalized_format = str(format).lower().lstrip(".")
        name = plot_export_filename(filename, normalized_format)
        extension, data = export_bytes(value, normalized_format)
        if extension != normalized_format:
            raise ValueError("Unexpected Plot export format.")
        ensure_active()
        directory = _child(_plot_directory(workspace, create=True), "exports", directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def _graphics_directory(workspace, *, create=False):
    return _child(_directory(workspace, create=create), "graphics", directory=True, create=create)


def graphics_image_path(workspace, payload):
    """Resolve and validate a Graphics asset only inside this workspace."""
    from .graphics import MAX_IMAGE_BYTES, validate_raster

    with _LOCK:
        if not isinstance(payload, dict) or payload.get("tool_id") != "graphics":
            raise ValueError("Only a Graphics output has a Graphics image.")
        _validate(payload, persisted="id" in payload)
        content = payload["content"]
        directory = _child(_graphics_directory(workspace), "assets", directory=True)
        path = _child(directory, Path(content["asset_path"]).name)
        with path.open("rb") as handle:
            data = handle.read(MAX_IMAGE_BYTES + 1)
        width, height = validate_raster(data, content["output_format"])
        if (width, height) != (content["width"], content["height"]):
            raise ValueError("The Graphics image does not match its saved dimensions.")
        return path


def _load_graphics_manifest(workspace):
    current = _child(_graphics_directory(workspace), "current.json")
    if not current.exists():
        return None
    value = _read_manifest(current)
    if value["tool_id"] != "graphics":
        raise ValueError("The current Graphics file contains a different tool's output.")
    return value


def load_current_graphics(workspace):
    with _LOCK:
        value = _load_graphics_manifest(workspace)
        if value is None:
            return None
        graphics_image_path(workspace, value)
        return value


def save_current_graphics(workspace, payload, *, is_cancelled=None):
    """Publish a verified immutable raster and one atomic manifest; retain old output on failure."""
    from .graphics import decode_image_asset, validate_graphics

    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Graphics cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Graphics save cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "graphics":
            raise ValueError("Only Graphics can be saved as the current Graphics output.")
        # The binary response is transient and may be larger than a manifest.
        value = json.loads(_encoded({key: item for key, item in payload.items() if key != "image_data"}))
        _relative_paths(value, workspace_path(workspace))
        validate_graphics(value.get("content"))
        data, metadata = decode_image_asset(payload.get("image_data"))
        ensure_active()
        name = f"{uuid4()}.{metadata['output_format']}"
        value["content"].update(metadata, asset_path=f"artifacts/graphics/assets/{name}")
        _validate(value)
        # A damaged/missing previous raster must not prevent a fresh valid replacement.
        existing = _load_graphics_manifest(workspace)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        encoded = _encoded(value)
        directory = _graphics_directory(workspace, create=True)
        assets = _child(directory, "assets", directory=True, create=True)
        image = _child(assets, name)
        current = _child(directory, "current.json")
        ensure_active()
        _atomic_bytes(image, data, before_replace=ensure_active)
        try:
            _atomic_bytes(current, encoded, before_replace=ensure_active)
        except (OSError, ValueError, RuntimeError):
            # Only this operation's new immutable asset can be removed.
            image.unlink(missing_ok=True)
            raise
        return value


def export_current_graphics(workspace, payload, format="image", filename="Graphics", *, is_cancelled=None):
    """Download the original raster or its cited brief and provenance without saving a version."""
    from .exports import export_bytes, graphics_export_filename
    from .graphics import MAX_IMAGE_BYTES

    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The Graphics cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Graphics export cancelled.")

    with _LOCK:
        ensure_active()
        if not isinstance(payload, dict) or payload.get("tool_id") != "graphics":
            raise ValueError("Only Graphics can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        if format == "image":
            path = graphics_image_path(workspace, value)
            with path.open("rb") as handle:
                data = handle.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Graphics image exceeds the supported size.")
            extension = value["content"]["output_format"]
        else:
            extension, data = export_bytes(value, format)
        name = graphics_export_filename(filename, extension)
        directory = _child(_graphics_directory(workspace, create=True), "exports", directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def _learning_directory(workspace, tool_id, *, create=False):
    if not isinstance(tool_id, str) or tool_id not in _LEARNING_TOOL_NAMES:
        raise ValueError("Only Flashcards and Quizzes use learning output storage.")
    return _child(_directory(workspace, create=create), tool_id, directory=True, create=create)


def load_current_learning(workspace, tool_id):
    """Read one current learning set without creating directories or study state."""
    with _LOCK:
        current = _child(_learning_directory(workspace, tool_id), "current.json")
        if not current.exists():
            return None
        value = _read_manifest(current)
        if value["tool_id"] != tool_id:
            raise ValueError("The current learning file contains a different tool's output.")
        return value


def save_current_learning(workspace, payload, *, is_cancelled=None):
    """Atomically replace a current learning set, retaining its stable identity."""
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The learning output cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Learning output save cancelled.")

    with _LOCK:
        ensure_active()
        tool_id = payload.get("tool_id") if isinstance(payload, dict) else None
        if not isinstance(tool_id, str) or tool_id not in _LEARNING_TOOL_NAMES:
            raise ValueError("Only Flashcards and Quizzes can be saved as a learning output.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value)
        existing = load_current_learning(workspace, tool_id)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        target = _child(_learning_directory(workspace, tool_id, create=True), "current.json")
        _atomic_bytes(target, _encoded(value), before_replace=ensure_active)
        return value


def export_learning(workspace, payload, format, filename, *, is_cancelled=None):
    """Download a learning set without changing its saved result or study state."""
    from .exports import export_bytes, learning_export_filename

    ensure_active = _export_guard(is_cancelled, "Study tool")
    with _LOCK:
        ensure_active()
        tool_id = payload.get("tool_id") if isinstance(payload, dict) else None
        if not isinstance(tool_id, str) or tool_id not in _LEARNING_TOOL_NAMES:
            raise ValueError("Only Flashcards and Quizzes can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        extension, data = export_bytes(value, format)
        name = learning_export_filename(filename, extension)
        directory = _child(_learning_directory(workspace, tool_id, create=True), "exports",
                           directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def _review_directory(workspace, tool_id, *, create=False):
    if not isinstance(tool_id, str) or tool_id not in _REVIEW_TOOL_NAMES:
        raise ValueError("Only Compare and Evidence Check use review output storage.")
    return _child(_directory(workspace, create=create), tool_id, directory=True, create=create)


def load_current_review(workspace, tool_id):
    """Read the single current review without creating files or making model calls."""
    with _LOCK:
        current = _child(_review_directory(workspace, tool_id), "current.json")
        if not current.exists():
            return None
        value = _read_manifest(current)
        if value["tool_id"] != tool_id:
            raise ValueError("The current review file contains a different tool's output.")
        return value


def save_current_review(workspace, payload, *, is_cancelled=None):
    """Atomically publish a review, retaining the existing identity and creation time."""
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The review output cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Review output save cancelled.")

    with _LOCK:
        ensure_active()
        tool_id = payload.get("tool_id") if isinstance(payload, dict) else None
        if not isinstance(tool_id, str) or tool_id not in _REVIEW_TOOL_NAMES:
            raise ValueError("Only Compare and Evidence Check can be saved as a review output.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value)
        existing = load_current_review(workspace, tool_id)
        now = datetime.now(timezone.utc).isoformat()
        value.update(id=existing["id"] if existing else str(uuid4()), version=1,
                     created_at=existing["created_at"] if existing else now, updated_at=now)
        _validate(value, persisted=True)
        target = _child(_review_directory(workspace, tool_id, create=True), "current.json")
        _atomic_bytes(target, _encoded(value), before_replace=ensure_active)
        return value


def export_review(workspace, payload, format="md", filename=None, *, is_cancelled=None):
    """Download a complete review snapshot without modifying the saved result."""
    from .exports import export_bytes, review_export_filename

    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("The review output cancellation guard must be callable.")

    def ensure_active():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Review output export cancelled.")

    with _LOCK:
        ensure_active()
        tool_id = payload.get("tool_id") if isinstance(payload, dict) else None
        if not isinstance(tool_id, str) or tool_id not in _REVIEW_TOOL_NAMES:
            raise ValueError("Only Compare and Evidence Check can be exported here.")
        value = json.loads(_encoded(payload))
        _relative_paths(value, workspace_path(workspace))
        _validate(value, persisted="id" in value)
        normalized_format = str(format).lower().lstrip(".")
        if normalized_format == "markdown":
            normalized_format = "md"
        name = review_export_filename(_REVIEW_TOOL_NAMES[tool_id] if filename is None else filename,
                                      normalized_format)
        extension, data = export_bytes(value, normalized_format)
        if extension != normalized_format:
            raise ValueError("Unexpected review export format.")
        ensure_active()
        directory = _child(_review_directory(workspace, tool_id, create=True), "exports",
                           directory=True, create=True)
        output = _child(directory, name)
        _atomic_bytes(output, data, before_replace=ensure_active)
        return output


def citation_label(payload, source_ids):
    numbers = {source["id"]: number for number, source in enumerate(payload["sources"], 1)}
    return " ".join(f"[{numbers[value]}]" for value in dict.fromkeys(source_ids) if value in numbers)


def _markdown_text(value):
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|])", r"\\\1", value)


def source_appendix(payload):
    lines = ["## Sources", ""]
    for number, source in enumerate(payload["sources"], 1):
        lines.append(f"{number}. {_markdown_text(source['filename'])} — {_markdown_text(source['locator'])}")
    return "\n".join(lines)


def render_summary(payload, *, include_sources=True):
    if payload.get("tool_id") != "summary":
        raise ValueError("This output is not a Summary.")
    if "edited_text" in payload:
        text = payload["edited_text"]
    else:
        parts = [f"# {_markdown_text(payload['title'])}"]
        for section in payload["content"]["sections"]:
            parts.append(f"## {_markdown_text(section['heading'])}")
            for paragraph in section["paragraphs"]:
                citations = citation_label(payload, paragraph["source_ids"])
                parts.append(f"{paragraph['text']} {citations}".strip())
        text = "\n\n".join(parts)
    if include_sources:
        text += "\n\n" + source_appendix(payload)
    return text


def spreadsheet_table(payload):
    if payload.get("tool_id") != "spreadsheet":
        raise ValueError("This output is not a Spreadsheet.")
    rows = payload.get("edited_rows")
    if rows is None:
        rows = [[cell["value"] for cell in row["cells"]] for row in payload["content"]["rows"]]
    return list(payload["content"]["columns"]), [list(row) for row in rows]


def export_artifact(workspace, artifact_id, format):
    from .exports import export_bytes

    with _LOCK:
        value = load_artifact(workspace, artifact_id)
        extension, data = export_bytes(value, format)
        target = _child(_directory(workspace), artifact_id, directory=True)
        output = _child(target, f"output.{extension}")
        _atomic_bytes(output, data)
        return output
