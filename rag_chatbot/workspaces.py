"""Named, isolated RAG working directories below the configured data root.

Importing this module and listing workspaces never create a directory. Creating
the initial Default workspace is an explicit initial-load operation.
"""

from dataclasses import replace
import hashlib
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
import unicodedata

from .config import Settings


MAX_WORKSPACE_NAME_LENGTH = 64
MAX_WORKSPACE_NAME_BYTES = 255
_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED_NAME = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.IGNORECASE)
_LOCK = threading.RLock()


def validate_workspace(name):
    """Validate one directory name; never sanitize an unsafe path into a name."""
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Enter a workspace name without leading or trailing spaces.")
    name = unicodedata.normalize("NFC", name)
    if (len(name) > MAX_WORKSPACE_NAME_LENGTH
            or len(name.encode("utf-8")) > MAX_WORKSPACE_NAME_BYTES
            or name.startswith(".") or name.endswith(".")
            or _INVALID_NAME.search(name) or _RESERVED_NAME.match(name)):
        raise ValueError("Use a workspace name of at most 64 characters (255 UTF-8 bytes) without path separators, "
                         "reserved Windows names, or trailing dots.")
    return name


def _key(name):
    return unicodedata.normalize("NFC", name).casefold()


def _root(data_dir=None):
    root = Path(data_dir if data_dir is not None else Settings().data_dir).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("The RAG data root must be a directory.")
    return root


def _linked(path):
    # is_symlink alone misses Windows directory junctions on older Python.
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _valid_directory(path, root):
    if _linked(path) or not path.is_dir() or path.resolve().parent != root:
        return False
    for name, directory in (("sources", True), ("chats", True), ("library.sqlite3", False)):
        member = path / name
        if not member.exists() or _linked(member) or member.resolve().parent != path.resolve():
            return False
        if directory != member.is_dir():
            return False
        if not directory and not member.is_file():
            return False
    try:
        with (path / "library.sqlite3").open("rb") as catalog:
            if catalog.read(16) != b"SQLite format 3\x00":
                return False
    except OSError:
        return False
    return True


def _matching_child(root, name):
    if not root.exists():
        return None
    matches = [child for child in root.iterdir() if _key(child.name) == _key(name)]
    if len(matches) > 1:
        raise ValueError("Workspace names that differ only by case are ambiguous. Rename one directory.")
    return matches[0] if matches else None


def list_workspaces(*, data_dir=None):
    """List real workspace directories, excluding temporary/unrelated folders."""
    with _LOCK:
        root = _root(data_dir)
        if not root.exists():
            return []
        found = {}
        for path in root.iterdir():
            try:
                validate_workspace(path.name)
            except ValueError:
                continue
            if not _valid_directory(path, root):
                continue
            key = _key(path.name)
            if key in found:
                raise ValueError("Workspace names that differ only by case are ambiguous. Rename one directory.")
            found[key] = path.name
        return sorted(found.values(), key=_key)


def create_workspace(name, *, data_dir=None):
    """Create a complete workspace atomically and return its canonical name."""
    name = validate_workspace(name)
    with _LOCK:
        root = _root(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        existing = _matching_child(root, name)
        if existing is not None:
            raise ValueError(f"A file or workspace named {existing.name!r} already exists.")
        target = root / name
        temporary = Path(tempfile.mkdtemp(prefix=".creating-", dir=root)).resolve()
        try:
            (temporary / "chats").mkdir()
            # Catalog initialization is local only; LibraryStore opens its vector
            # client lazily, so workspace creation makes no network/API request.
            from .store import LibraryStore
            store = LibraryStore(replace(Settings(), data_dir=temporary))
            store.close()
            temporary.rename(target)
        finally:
            if temporary.exists():
                if temporary.parent != root or not temporary.name.startswith(".creating-"):
                    raise ValueError("Refusing to clean a temporary workspace outside its data root.")
                shutil.rmtree(temporary)
        return name


def workspace_path(name, create=False, *, data_dir=None):
    """Open an existing workspace, resolving name casing without escaping its root."""
    name = validate_workspace(name)
    with _LOCK:
        root = _root(data_dir)
        candidate = _matching_child(root, name)
        if candidate is None:
            if create:
                name = create_workspace(name, data_dir=root)
                candidate = root / name
            else:
                raise FileNotFoundError(f"Workspace {name!r} does not exist. Create it first.")
        if not _valid_directory(candidate, root):
            raise ValueError("The selected directory is not a valid workspace or contains unsafe links.")
        return candidate.resolve()


def default_workspace(*, data_dir=None):
    """Choose an existing workspace, creating Default only for an empty library."""
    with _LOCK:
        existing = list_workspaces(data_dir=data_dir)
        if existing:
            return next((name for name in existing if _key(name) == "default"), existing[0])
        return create_workspace("Default", data_dir=data_dir)


def workspace_settings(name, *, settings=None):
    """Derive isolated SQL/source/chat paths and a stable vector namespace."""
    base = settings or Settings()
    directory = workspace_path(name, data_dir=base.data_dir)
    identity = hashlib.sha256(str(directory).casefold().encode("utf-8")).hexdigest()[:32]
    namespace = f"workspace-{identity}"
    local_path = None
    if base.qdrant_local_path:
        local_root = Path(base.qdrant_local_path).resolve()
        destination = local_root / namespace
        if destination.exists() and _linked(destination):
            raise ValueError("Workspace vector files cannot use a linked directory.")
        destination = destination.resolve()
        if destination.parent != local_root:
            raise ValueError("Workspace vector files must stay inside the configured local Qdrant directory.")
        local_path = str(destination)
    return replace(base, data_dir=directory, qdrant_local_path=local_path, namespace=namespace)
