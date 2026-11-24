"""Session-owned RAG autosaves, separate from manually named snapshots."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from uuid import uuid4

from chatbot.log import logger
from .history import _validate


MAX_BYTES = 16 * 1024 * 1024
_NAME = re.compile(r"Autosaved \d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2} [0-9a-f]{32}\Z")
_LOCK = threading.RLock()


class AutosaveCancelled(RuntimeError):
    """A stopped or superseded answer cannot replace an autosaved chat."""


def new_session(workspace):
    identifier = uuid4().hex
    return {"workspace": workspace, "id": identifier,
            "name": f"Autosaved {datetime.now(timezone.utc):%Y-%m-%d %H-%M-%S} {identifier}",
            "saved": False}


def display_name(name):
    """Short timestamp labels keep the full unique filename as callback value."""
    if isinstance(name, str) and _NAME.fullmatch(name):
        timestamp, identifier = name.rsplit(" ", 1)
        return f"{timestamp} UTC · {identifier[:8]}"
    return name


def _linked(path):
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _directory(data_dir, *, create=False):
    root = Path(data_dir).resolve(strict=True)
    directory = root / "chats"
    if directory.exists() or directory.is_symlink():
        if _linked(directory) or not directory.is_dir() or directory.resolve().parent != root:
            raise ValueError("Autosaved chats must use the workspace's own chats directory.")
    elif create:
        directory.mkdir()
    return directory


def _path(directory, name):
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("Invalid autosaved chat name.")
    path = directory / f"{name}.json"
    if path.exists() or path.is_symlink():
        if _linked(path) or not path.is_file() or path.resolve().parent != directory:
            raise ValueError("Autosaved chats cannot use linked or external files.")
    return path


def save(messages, session, *, data_dir, is_cancelled=None):
    """Write only a complete accepted conversation, checking before replacement."""
    def check():
        if is_cancelled is not None:
            if not callable(is_cancelled):
                raise ValueError("The autosave cancellation check must be callable.")
            if is_cancelled():
                raise AutosaveCancelled("The completed answer was superseded before autosave.")

    check()
    validated = _validate(messages)
    if not validated or validated[-1]["role"] != "assistant" or not validated[-1]["content"].strip():
        raise ValueError("Autosave requires a completed assistant answer.")
    if not isinstance(session, dict):
        raise ValueError("Invalid autosave session.")
    name = session.get("name")
    identifier = session.get("id")
    if (not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier)
            or not isinstance(name, str) or not name.endswith(" " + identifier)):
        raise ValueError("Invalid autosave session.")
    value = {"version": 1, "messages": validated,
             "autosave": {"id": identifier, "updated_at": datetime.now(timezone.utc).isoformat()}}
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("This conversation exceeds the 16 MiB autosave limit. Save a named copy manually.")
    with _LOCK:
        directory = _directory(data_dir, create=True)
        path = _path(directory, name)
        check()
        descriptor, temporary_name = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            check()
            _path(_directory(data_dir), name)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return name


def latest(*, data_dir):
    """Return the latest valid auto conversation; a bad file cannot block opening."""
    with _LOCK:
        directory = _directory(data_dir)
        if not directory.exists():
            return None
        candidates = []
        for path in directory.glob("Autosaved *.json"):
            try:
                path = _path(directory, path.stem)
                candidates.append((path.lstat().st_mtime_ns, path))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping invalid autosaved chat %s: %s", path.name, exc)
        # Successful atomic replacement changes the file's mtime. Open only
        # the newest valid candidate, rather than parsing every full history
        # each time a workspace opens. Ties have deterministic filename order.
        for _modified, path in sorted(candidates, key=lambda item: (item[0], item[1].name), reverse=True):
            try:
                path = _path(directory, path.stem)
                with path.open("rb") as handle:
                    raw = handle.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("Autosaved conversation exceeds the size limit.")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("Invalid autosaved conversation object.")
                metadata = value.get("autosave", {})
                identifier = metadata.get("id") if isinstance(metadata, dict) else None
                if (type(value.get("version")) is not int or value["version"] != 1
                        or not isinstance(identifier, str) or not path.stem.endswith(" " + identifier)
                        or not re.fullmatch(r"[0-9a-f]{32}", identifier)):
                    raise ValueError("Invalid autosaved conversation metadata.")
                updated = datetime.fromisoformat(metadata["updated_at"])
                if updated.tzinfo is None:
                    raise ValueError("Autosave timestamps require a timezone.")
                messages = _validate(value.get("messages"))
                if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
                    raise ValueError("Autosaved conversation has no completed answer.")
                return {"name": path.stem, "messages": messages}
            except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
                logger.warning("Skipping invalid autosaved chat %s: %s", path.name, exc)
        return None
