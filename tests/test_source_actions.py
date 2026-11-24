"""Text sources, safe cached retries, and browser/workspace cancellation guards."""

import asyncio
from dataclasses import replace
from pathlib import Path
from functools import partial

import gradio as gr
from gradio.state_holder import SessionState
import pytest
from qdrant_client import QdrantClient

from rag_chatbot import source_actions as actions
from rag_chatbot.config import Settings
from rag_chatbot.service import RAGService
from rag_chatbot.store import LibraryStore
from rag_chatbot.workspaces import create_workspace, workspace_path


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Text")
    create_workspace("Other")
    settings = replace(Settings(), data_dir=workspace_path("Text"), embedding_dimensions=3,
                       chunk_size=64, chunk_overlap=8)
    store = LibraryStore(settings, vector_client=QdrantClient(":memory:"))
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0, 0.2, 0.1] for _ in texts]

    result = RAGService(settings, store=store, embedder=embed)
    result.embedding_calls = calls
    yield result
    store.close()


def add(service, title="Pasted text.txt", text="This experiment has marker QX-1397 and 40 participants."):
    messages = list(actions.index_text("Text", title, text, "Default", service))
    assert "1 indexed" in messages[-1]
    return service.store.list_documents("Default")[-1]


def test_text_is_a_managed_searchable_source_and_deduplicates_without_paid_calls(service, monkeypatch):
    text = "# Results\n\nMarker QX-1397: 40 participants.\n<script>alert('literal')</script>\n"
    document = add(service, "Results.md", text)
    assert document["name"] == "Results.md"
    assert Path(document["path"]).read_bytes() == text.encode("utf-8")
    assert document["status"] == "ready" and document["generation"]
    results = service.store.search("participants", None, "Default", [document["id"]])
    assert results and results[0].chunk.document_id == document["id"]
    assert results[0].source_path == document["path"]
    assert not list(workspace_path("Text").glob(".text-source-*"))
    assert not (workspace_path("Text") / "notes").exists()
    calls = len(service.embedding_calls)
    monkeypatch.setattr(service.store, "check_connection", lambda: pytest.fail("Ready deduplication needs no Qdrant"))
    messages = list(actions.index_text("Text", "Renamed.md", text, "Default", service))
    assert "1 unchanged" in messages[-1]
    assert len(service.embedding_calls) == calls
    assert service.store.get_document(document["id"])["name"] == "Renamed.md"


@pytest.mark.parametrize("title,text", [
    ("", "body"), ("../escape.txt", "body"), ("CON.txt", "body"),
    ("Report.pdf", "body"), (" report.txt", "body"), ("trailing.", "body"),
    ("é" * 126 + ".txt", "body"), ("\ud800.txt", "body"),
    ("Report", ""), ("Report", " \n"), ("Report", "\x00body"),
    ("Report", "bad\udfff"), ("Report", "a" * (actions.MAX_TEXT_BYTES + 1)),
    ("Report", "é" * (actions.MAX_TEXT_BYTES // 2 + 1)),
], ids=["blank-name", "traversal", "reserved", "extension", "spaces", "trailing-dot",
        "long-utf8-name", "surrogate-name", "blank", "whitespace", "null", "surrogate",
        "oversize-characters", "oversize-utf8"])
def test_invalid_text_inputs_fail_before_storage_or_api(service, monkeypatch, title, text):
    monkeypatch.setattr(service, "ingest", lambda *a, **k: pytest.fail("Invalid inputs must never index"))
    with pytest.raises(ValueError):
        list(actions.index_text("Text", title, text, "Default", service))
    assert service.store.list_documents() == []
    assert not list(workspace_path("Text").glob(".text-source-*"))


def test_valid_text_name_and_exact_byte_bound():
    name, content = actions.validate_text_source("My text", "é" * (actions.MAX_TEXT_BYTES // 2))
    assert name == "My text.txt" and len(content) == actions.MAX_TEXT_BYTES


def test_cancelled_text_never_stages_and_generator_close_cleans_only_own_temp(service):
    with pytest.raises(actions.SourceActionCancelled):
        list(actions.index_text("Text", "Title", "body", "Default", service, is_cancelled=lambda: True))
    assert service.store.list_documents() == []
    unrelated = workspace_path("Text") / "keep.txt"
    unrelated.write_text("Keep me", encoding="utf-8")
    job = actions.index_text("Text", "Title", "body", "Default", service)
    assert next(job).startswith("Preparing")
    assert list(workspace_path("Text").glob(".text-source-*"))
    job.close()
    assert not list(workspace_path("Text").glob(".text-source-*"))
    document = service.store.list_documents()[0]
    assert document["status"] == "failed" and Path(document["path"]).is_file()
    assert unrelated.read_text(encoding="utf-8") == "Keep me"
    assert not service.is_indexing and service.embedding_calls == []


def test_text_workspace_guard_stops_before_embeddings(service, monkeypatch):
    cancelled = False

    def connected():
        nonlocal cancelled
        cancelled = True

    monkeypatch.setattr(service.store, "check_connection", connected)
    with pytest.raises(actions.SourceActionCancelled):
        list(actions.index_text("Text", "Title", "body", "Default", service,
                                is_cancelled=lambda: cancelled))
    assert service.embedding_calls == []
    assert not list(workspace_path("Text").glob(".text-source-*"))
    assert not service.is_indexing


def test_unavailable_vectors_fail_before_paid_embeddings_and_preserve_retry_original(service, monkeypatch):
    def disconnected():
        raise OSError("Qdrant unavailable")

    monkeypatch.setattr(service.store, "check_connection", disconnected)
    result = list(actions.index_text("Text", "Title", "body", "Default", service))
    assert any("No embedding request was sent" in message for message in result)
    assert service.embedding_calls == []
    document = service.store.list_documents()[0]
    assert document["status"] == "failed" and Path(document["path"]).is_file()
    assert not list(workspace_path("Text").glob(".text-source-*"))


def test_retry_failed_and_ready_sources_reuses_embeddings_and_keeps_names(service):
    first = add(service, "First.txt", "The first experiment measured 14 participants.")
    second = add(service, "Second.txt", "The second experiment measured 40 participants.")
    service.store.stage_document(first["path"], "Default", service.settings.fingerprint(), force=True)
    service.store.fail_document(first["id"], "Transient failure")
    calls = len(service.embedding_calls)
    messages = list(actions.retry_sources("Text", [first["id"], second["id"]], "Default", service))
    assert "1 indexed, 1 unchanged, 0 failed" in messages[-1]
    assert len(service.embedding_calls) == calls
    assert len(service.store.list_documents()) == 2
    assert service.store.get_document(first["id"])["name"] == "First.txt"
    assert service.store.get_document(first["id"])["status"] == "ready"


@pytest.mark.parametrize("selection", [[], ["missing"], ["same", "same"], "one", [None]])
def test_retry_rejects_invalid_selection_before_indexing(service, monkeypatch, selection):
    monkeypatch.setattr(service, "ingest", lambda *a, **k: pytest.fail("No invalid retries"))
    with pytest.raises(ValueError):
        list(actions.retry_sources("Text", selection, "Default", service))


@pytest.mark.parametrize("damage", ["missing", "changed", "collection", "deleted", "outside"])
def test_retry_rejects_invalid_originals_before_api(service, monkeypatch, damage):
    document = add(service)
    original = Path(document["path"])
    if damage == "missing":
        original.unlink()
    elif damage == "changed":
        original.write_text("Changed source", encoding="utf-8")
    elif damage == "collection":
        with service.store._db:
            service.store._db.execute("UPDATE documents SET collection='Other' WHERE id=?", (document["id"],))
    elif damage == "deleted":
        service.store.delete_document(document["id"])
    else:
        with service.store._db:
            service.store._db.execute("UPDATE documents SET path='library.sqlite3' WHERE id=?", (document["id"],))
    monkeypatch.setattr(service, "ingest", lambda *a, **k: pytest.fail("No invalid original retries"))
    with pytest.raises(ValueError):
        list(actions.retry_sources("Text", [document["id"]], "Default", service))


def test_retry_rejects_service_for_another_workspace(service):
    document = add(service)
    with pytest.raises(ValueError, match="does not belong"):
        list(actions.retry_sources("Other", [document["id"]], "Default", service))


def test_retry_rechecks_source_removal_before_paid_work(service, monkeypatch):
    document = add(service)
    service.store.stage_document(document["path"], "Default", service.settings.fingerprint(), force=True)
    service.store.fail_document(document["id"], "Transient failure")
    calls = len(service.embedding_calls)

    def connected():
        service.store.delete_document(document["id"])

    monkeypatch.setattr(service.store, "check_connection", connected)
    with pytest.raises(actions.SourceActionCancelled):
        list(actions.retry_sources("Text", [document["id"]], "Default", service))
    assert len(service.embedding_calls) == calls
    assert not service.is_indexing


def test_retry_checks_links_before_using_original(service):
    document = add(service)
    original = Path(document["path"])
    target = original.with_name("actual.txt")
    original.rename(target)
    try:
        original.symlink_to(target)
    except OSError:
        target.rename(original)
        pytest.skip("Creating symlinks requires privileges on this platform")
    with pytest.raises(ValueError, match="linked"):
        list(actions.retry_sources("Text", [document["id"]], "Default", service))


def test_original_path_checks_unresolved_catalog_components(service, monkeypatch):
    document = add(service)
    original = Path(document["path"])
    actual_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == original or actual_is_symlink(path))
    with pytest.raises(ValueError, match="linked"):
        service.store.original_path(document["id"])
    monkeypatch.setattr(Path, "is_symlink", actual_is_symlink)
    unsafe = f"sources/{document['id']}/../../sources/{document['id']}/original.txt"
    with service.store._db:
        service.store._db.execute("UPDATE documents SET path=? WHERE id=?", (unsafe, document["id"]))
    # A resolved document view cannot reveal the traversal; the raw accessor can.
    assert Path(service.store.get_document(document["id"])["path"]) == original
    with pytest.raises(ValueError, match="sources directory"):
        service.store.original_path(document["id"])


def test_ui_actions_use_canonical_selection_and_keep_editor_text(service):
    document = add(service)
    operation = {"revision": 3, "selected_ids": [document["id"]]}
    reset = actions.on_reset("Text", operation)
    messages = list(actions.on_retry("Default", False, reset[3], "Text", operation, reset[4],
                                    get_service=lambda _: service))
    assert any("unchanged" in str(status) for status, _ in messages)
    final = messages[-1][1]
    assert actions.can_refresh(final, "Text", operation, reset[4])
    assert operation["selected_ids"] == [document["id"]]
    operation["revision"] += 1
    assert not actions.can_refresh(final, "Text", operation, reset[4])


def test_stale_action_rejects_before_service_lookup(service):
    operation = {"revision": 1}
    old = actions.on_reset("Text", operation)
    operation["revision"] += 1
    new = actions.on_reset("Other", operation)
    no_service = lambda _: pytest.fail("Queued old content must never access another workspace")
    assert list(actions.on_text("Old", "Old body", "Default", new[3], "Other", operation, old[4],
                                get_service=no_service)) == [(gr.skip(), gr.skip())]
    assert list(actions.on_retry("Default", False, new[3], "Other", operation, old[4],
                                 get_service=no_service)) == [(gr.skip(), gr.skip())]


def test_workspace_switch_during_action_drops_late_status(service, monkeypatch):
    operation = {"revision": 1}
    reset = actions.on_reset("Text", operation)

    def changed(*args, **kwargs):
        operation["revision"] += 1
        assert kwargs["is_cancelled"]()
        yield "Late old workspace result"

    monkeypatch.setattr(actions, "index_text", changed)
    result = list(actions.on_text("Title", "body", "Default", reset[3], "Text", operation, reset[4],
                                  get_service=lambda _: service))
    assert len(result) == 1 and "Preparing" in result[0][0]
    assert not actions.can_refresh(result[0][1], "Text", operation, reset[4])


@pytest.mark.parametrize("callback", [actions.on_text, actions.on_retry])
def test_real_gradio_queue_rejects_old_client_epoch_with_new_server_states(callback):
    operation = {"revision": 1}
    old = actions.on_reset("Text", operation)
    operation["revision"] += 1
    new = actions.on_reset("Other", operation)
    with gr.Blocks() as app:
        workspace = gr.State("Other")
        collection = gr.Textbox("Default", visible=False)
        vision = gr.Checkbox(False)
        op_state = gr.State(operation)
        controls = actions.build(workspace, collection, vision, op_state,
                                 get_service=lambda _: pytest.fail("No stale indexing"))
    assert len(controls.outputs) == actions.OUTPUT_COUNT
    state = SessionState(app)
    state[workspace._id] = "Other"
    state[op_state._id] = operation
    state[controls.state._id] = new[3]
    function = next(value for value in app.fns.values()
                    if isinstance(value.fn, partial) and value.fn.func is callback)
    values = {controls.text_title._id: "Old source", controls.text_body._id: "Old body",
              collection._id: "Default", vision._id: False, controls.workspace_epoch._id: old[4]}

    async def invoke():
        return await app.process_api(function, [values.get(component._id) for component in function.inputs], state=state)

    result = asyncio.run(invoke())
    assert result["data"][0] in (gr.skip(), {"__type__": "update", "props": gr.skip()})
    assert result["data"][1] is None


def test_reset_has_no_storage_dependency_and_predicts_finish_revision():
    operation = {"revision": 6, "switching": True}
    result = actions.on_reset("Unavailable", operation)
    assert len(result) == actions.OUTPUT_COUNT
    assert result[3]["workspace_revision"] == 7
    assert not actions._current(result[3], "Unavailable", operation, result[4])
    operation.update(revision=7, switching=False)
    assert actions._current(result[3], "Unavailable", operation, result[4])
