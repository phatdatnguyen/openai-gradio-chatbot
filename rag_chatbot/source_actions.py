"""Pasted-text sources and retries through the existing indexing pipeline."""

from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from chatbot.log import logger
from .store import validate_display_name
from .workspaces import workspace_path


MAX_TEXT_BYTES = 1024 * 1024
OUTPUT_COUNT = 5


class SourceActionCancelled(RuntimeError):
    """An action no longer belongs to the visible workspace."""


def _guard(is_cancelled):
    if is_cancelled is not None:
        if not callable(is_cancelled):
            raise ValueError("The cancellation guard must be callable.")
        if is_cancelled():
            raise SourceActionCancelled("Source indexing stopped because its workspace changed.")


def validate_text_source(title, text):
    """Keep the pasted text verbatim and return a portable TXT/MD name and bytes."""
    if (not isinstance(title, str) or not title.strip()
            or any(0xD800 <= ord(char) <= 0xDFFF for char in title)):
        raise ValueError("Enter a source filename ending in .txt or .md.")
    suffix = Path(title).suffix.lower() or ".txt"
    if suffix not in {".txt", ".md"}:
        raise ValueError("Pasted text filenames must end in .txt or .md.")
    name = validate_display_name("original" + suffix, title)
    if len(name.encode("utf-8")) > 255:
        raise ValueError("Source filenames must fit within 255 UTF-8 bytes.")
    if (not isinstance(text, str) or not text.strip() or "\x00" in text
            or len(text) > MAX_TEXT_BYTES
            or any(0xD800 <= ord(char) <= 0xDFFF for char in text)):
        raise ValueError("Paste nonempty text of at most 1 MiB, without null characters.")
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_TEXT_BYTES:
        raise ValueError("Pasted text is limited to 1 MiB of UTF-8 text.")
    return name, encoded


def _root(workspace, service):
    root = workspace_path(workspace)
    if Path(service.settings.data_dir).resolve() != root:
        raise ValueError("The indexing service does not belong to this workspace.")
    return root


def _linked(path):
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _original(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("The source original is missing. Upload the document again.")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("Source originals must stay inside this workspace's sources directory.") from exc
    if not parts or parts[0] != "sources" or any(part in {".", ".."} for part in parts):
        raise ValueError("Source originals must stay inside this workspace's sources directory.")
    current = root
    for part in parts:
        current = current / part
        if (current.exists() or current.is_symlink()) and _linked(current):
            raise ValueError("Source originals cannot use linked files or directories.")
    if not current.is_file() or not current.resolve().is_relative_to(root / "sources"):
        raise ValueError("The source original is missing. Upload the document again.")
    return current


def _file_revision(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


@contextmanager
def _staged_text(root, suffix, content, is_cancelled):
    _guard(is_cancelled)
    # This temporary directory is inside ignored workspace data. Only the
    # single file created here is ever removed; managed originals are retained.
    directory = Path(tempfile.mkdtemp(prefix=".text-source-", dir=root))
    path = directory / ("original" + suffix)
    try:
        if directory.resolve().parent != root or _linked(directory):
            raise ValueError("Invalid temporary text-source directory.")
        _guard(is_cancelled)
        with path.open("xb") as handle:
            handle.write(content)
        _guard(is_cancelled)
        yield path
    finally:
        if (directory.exists() and directory.resolve().parent == root
                and not _linked(directory) and directory.name.startswith(".text-source-")):
            path.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                logger.warning("Could not remove the temporary text-source directory: %s", directory.name)


def index_text(workspace, title, text, collection, service, *, is_cancelled=None):
    """Index a bounded text original, retaining normal deduplication and caches."""
    _guard(is_cancelled)
    name, content = validate_text_source(title, text)
    root = _root(workspace, service)
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError("Open a workspace before indexing text.")
    with _staged_text(root, Path(name).suffix.lower(), content, is_cancelled) as path:
        with closing(service.ingest([str(path)], collection, display_names=[name],
                                    is_cancelled=is_cancelled)) as progress:
            for message in progress:
                _guard(is_cancelled)
                yield str(message)


def retry_sources(workspace, selected_ids, collection, service, *, use_vision=False, is_cancelled=None):
    """Retry only validated checked originals; ready documents keep their caches."""
    _guard(is_cancelled)
    root = _root(workspace, service)
    if (not isinstance(selected_ids, (list, tuple)) or not selected_ids
            or any(not isinstance(value, str) or not value for value in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("Check at least one source to retry indexing.")
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError("Open a workspace before retrying indexing.")
    paths, names, snapshots, file_revisions = [], [], [], []
    # Validate the entire selection before ingest can stage a document or make
    # a paid request. Never turn an empty selection into all workspace sources.
    for identifier in selected_ids:
        _guard(is_cancelled)
        document = service.store.get_document(identifier)
        if not document or document.get("status") == "deleted" or document.get("collection") != collection:
            raise ValueError("A checked source is no longer in this workspace and collection. Refresh the library.")
        path = _original(root, str(service.store.original_path(identifier)))
        file_revision = _file_revision(path)
        if (service.store._hash_file(path) != document.get("content_hash")
                or _file_revision(path) != file_revision):
            raise ValueError("A source original has changed. Upload the original document again to repair it.")
        names.append(validate_display_name(path, document.get("name")))
        paths.append(str(path))
        snapshots.append({key: document.get(key) for key in ("id", "path", "content_hash", "collection")})
        file_revisions.append(file_revision)

    def cancelled():
        if is_cancelled is not None and is_cancelled():
            return True
        for snapshot, file_revision in zip(snapshots, file_revisions):
            document = service.store.get_document(snapshot["id"])
            if (not document or document.get("status") == "deleted"
                    or any(document.get(key) != value for key, value in snapshot.items())):
                return True
            if _file_revision(_original(root, str(service.store.original_path(snapshot["id"])))) != file_revision:
                return True
        return False

    _guard(cancelled)
    with closing(service.ingest(paths, collection, use_vision=use_vision, display_names=names,
                                is_cancelled=cancelled)) as progress:
        for message in progress:
            _guard(cancelled)
            yield str(message)


def on_reset(workspace, operation):
    token = uuid4().hex
    operation["source_actions_token"] = token
    operation["source_actions_job"] = None
    state = {"workspace": workspace,
             "workspace_revision": operation.get("revision", 0) + bool(operation.get("switching")),
             "token": token, "job": None, "completed": False}
    return "Pasted text.txt", "", "", state, token


def _current(state, workspace, operation, epoch):
    return (isinstance(state, dict) and state.get("workspace") == workspace and bool(workspace)
            and state.get("workspace_revision") == operation.get("revision", 0)
            and isinstance(epoch, str) and bool(epoch)
            and epoch == state.get("token") == operation.get("source_actions_token")
            and not operation.get("switching"))


def can_refresh(state, workspace, operation, epoch):
    """A root-owned library refresh may follow only a completed current job."""
    return (_current(state, workspace, operation, epoch) and state.get("completed") is True
            and bool(state.get("job")) and state.get("job") == operation.get("source_actions_job"))


def _run(action, state, workspace, operation, epoch):
    if not _current(state, workspace, operation, epoch):
        yield gr.skip(), gr.skip()
        return
    job = uuid4().hex
    operation["source_actions_job"] = job
    envelope = {**state, "job": job, "completed": False}

    def cancelled():
        return (not _current(envelope, workspace, operation, epoch)
                or operation.get("source_actions_job") != job)

    yield "Preparing source indexing…", envelope
    try:
        _guard(cancelled)
        with closing(action(cancelled)) as progress:
            for message in progress:
                _guard(cancelled)
                yield message, gr.skip()
        _guard(cancelled)
        yield gr.skip(), {**envelope, "completed": True}
    except SourceActionCancelled:
        return
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        if not cancelled():
            logger.warning("Source indexing action failed: %s", exc)
            yield f"Could not index sources: {exc}", envelope


def on_text(title, text, collection, state, workspace, operation, epoch, *, get_service):
    def action(cancelled):
        validate_text_source(title, text)
        _guard(cancelled)
        service = get_service(workspace)
        yield from index_text(workspace, title, text, collection, service, is_cancelled=cancelled)
    yield from _run(action, state, workspace, operation, epoch)


def on_retry(collection, use_vision, state, workspace, operation, epoch, *, get_service):
    def action(cancelled):
        _guard(cancelled)
        selected_ids = list(operation.get("selected_ids") or [])
        if not selected_ids:
            raise ValueError("Check at least one source to retry indexing.")
        service = get_service(workspace)
        yield from retry_sources(workspace, selected_ids, collection, service,
                                 use_vision=use_vision, is_cancelled=cancelled)
    yield from _run(action, state, workspace, operation, epoch)


def build(workspace, collection, use_vision, operation, *, get_service):
    """Mount inside Sources; the caller owns library refresh and Stop controls."""
    with gr.Accordion("Add pasted text", open=False):
        text_title = gr.Textbox(label="Source filename", value="Pasted text.txt",
                               info="Use .txt or .md; a missing extension becomes .txt.")
        text_body = gr.Textbox(label="Text", lines=5, max_lines=8,
                              placeholder="Paste text to use as a source for chat and tools.",
                              info="Up to 1 MiB. Indexing may use paid embeddings. Notes remain separate.")
        text_button = gr.Button("Index pasted text", size="sm")
    retry_button = gr.Button("Retry checked sources", size="sm")
    status = gr.Markdown()
    state = gr.State(None)
    workspace_epoch = gr.Textbox(value="", visible=False, elem_id="rag-source-actions-epoch")
    action_outputs = [status, state]
    queue = {"concurrency_id": "rag-source-indexing", "concurrency_limit": 1}
    text_event = text_button.click(partial(on_text, get_service=get_service),
        [text_title, text_body, collection, state, workspace, operation, workspace_epoch], action_outputs, **queue)
    retry_event = retry_button.click(partial(on_retry, get_service=get_service),
        [collection, use_vision, state, workspace, operation, workspace_epoch], action_outputs, **queue)
    return SimpleNamespace(text_title=text_title, text_body=text_body, text_button=text_button,
        retry_button=retry_button, status=status, state=state, workspace_epoch=workspace_epoch,
        action_outputs=action_outputs, outputs=[text_title, text_body, status, state, workspace_epoch],
        reset_callback=on_reset, text_event=text_event, retry_event=retry_event,
        index_events=[text_event, retry_event])
