"""Local Notes persistence and direct Search without paid generation."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rag_chatbot import research
from rag_chatbot.config import Settings
from rag_chatbot.schemas import Chunk
from rag_chatbot.store import LibraryStore
from rag_chatbot.workspaces import create_workspace, workspace_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    create_workspace("Other")
    return workspace_path("Research")


@pytest.fixture
def source(workspace):
    path = workspace / "sources" / "report.txt"
    path.write_text("Measured sample size: 40.", encoding="utf-8")
    return {"number": 1, "document_id": "d1", "filename": "report.txt", "locator": "Paragraph 1",
            "text": "Measured sample size: 40.", "source_path": str(path), "image_path": None}


def test_note_roundtrip_edit_conflict_search_and_workspace_isolation(workspace, source):
    assert research.list_notes("Research") == []
    assert not (workspace / "notes").exists()
    note = research.save_note("Research", "Sample finding", "The sample has 40 participants [1].", origin="ai", sources=[source], web_search=True)
    assert research.load_note("Research", note["id"]) == note
    assert note["sources"][0]["source_path"] == "sources/report.txt"
    assert len(research.list_notes("Research", "PARTICIPANTS")) == 1
    assert research.list_notes("Research", "not mentioned") == []
    assert research.list_notes("Other") == []
    edited = research.save_note("Research", "My edited finding", "Review this result.", identifier=note["id"], expected_revision=1)
    assert edited["revision"] == 2 and edited["user_edited"]
    assert edited["origin"] == "ai" and edited["web_search"]
    assert edited["sources"] == note["sources"]
    with pytest.raises(ValueError, match="another session"):
        research.save_note("Research", "Stale", "Stale", identifier=note["id"], expected_revision=1)
    with pytest.raises(ValueError, match="another session"):
        research.delete_note("Research", note["id"], 1)
    research.delete_note("Research", note["id"], 2)
    assert research.list_notes("Research") == []


@pytest.mark.parametrize("identifier", ["../outside", "x", "", "1" * 36, None])
def test_note_identifiers_cannot_escape_storage(workspace, identifier):
    with pytest.raises(ValueError, match="identifier"):
        research.load_note("Research", identifier)


@pytest.mark.parametrize("path", ["chats/private.json", "library.sqlite3", "sources/../chats/private.json", "../outside.txt"])
def test_notes_refuse_non_source_reference_paths(workspace, source, path):
    with pytest.raises(ValueError, match="sources directory"):
        research.save_note("Research", "A note", "Text", sources=[{**source, "source_path": path}])
    assert not (workspace / "notes").exists()


def test_missing_original_retains_note_snapshot_but_has_no_download(workspace, source):
    note = research.save_note("Research", "Excerpt", source["text"], origin="excerpt", sources=[source])
    Path(source["source_path"]).unlink()
    assert research.load_note("Research", note["id"])["sources"][0]["text"] == source["text"]
    assert research.reference_path("Research", note["sources"][0]) is None


@pytest.mark.parametrize("stop_at", [1, 2, 3])
def test_cancelled_note_update_preserves_prior_file(workspace, stop_at):
    note = research.save_note("Research", "Original", "Keep this")
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) == stop_at

    with pytest.raises(RuntimeError, match="cancelled"):
        research.save_note("Research", "Replacement", "Do not save", identifier=note["id"], expected_revision=1,
                           is_cancelled=cancelled)
    assert research.load_note("Research", note["id"]) == note
    assert not list((workspace / "notes").glob(".pending-*"))


@pytest.mark.parametrize("field,value", [("title", ""), ("title", "x" * 201), ("body", ""), ("body", "x" * 100001)],
                         ids=["blank-title", "long-title", "blank-body", "long-body"])
def test_note_bounds_fail_before_writing(workspace, field, value):
    values = {"title": "Title", "body": "Content", field: value}
    with pytest.raises(ValueError):
        research.save_note("Research", **values)
    assert not (workspace / "notes").exists()


def test_invalid_files_and_pending_writes_are_not_listed(workspace):
    note = research.save_note("Research", "Valid", "Text")
    directory = workspace / "notes"
    (directory / ".pending-test.json").write_text("{}")
    (directory / f"{uuid4()}.json").write_text("broken")
    (directory / f"{uuid4()}.json").mkdir()
    assert [item["id"] for item in research.list_notes("Research")] == [note["id"]]


def test_notes_reject_linked_storage(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "notes").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symbolic links requires platform privileges")
    with pytest.raises(ValueError, match="linked"):
        research.save_note("Research", "No", "No")
    assert list(outside.iterdir()) == []


@pytest.fixture
def indexed(workspace, tmp_path):
    settings = replace(Settings(), data_dir=workspace, embedding_dimensions=3)
    store = LibraryStore(settings)
    uploaded = tmp_path / "study.txt"
    uploaded.write_text("The spectrum has a peak at 26.5 degrees.", encoding="utf-8")
    document = store.stage_document(uploaded, "Science", settings.fingerprint())
    chunk = Chunk(str(uuid4()), document["id"], uploaded.read_text(encoding="utf-8"), "Page 2")
    with store._db:
        store._db.execute("INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES(?,?,?,?,?,?)",
                          (chunk.id, document["id"], chunk.text, chunk.locator, chunk.kind, 0))
        store._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=1 WHERE id=?", (document["id"],))
    service = SimpleNamespace(store=store, settings=settings, _vectors=lambda _: pytest.fail("No embeddings"))
    yield service, document, chunk
    store.close()


def test_keyword_search_uses_local_index_and_retains_citation(indexed):
    service, document, chunk = indexed
    results = research.search_sources("Research", "spectrum", [document["id"]], "Science", service=service)
    assert len(results) == 1
    assert results[0]["id"] == chunk.id and results[0]["text"] == chunk.text
    assert results[0]["locator"] == "Page 2"
    assert service.store._vectors is None


@pytest.mark.parametrize("selected,collection", [([], "Science"), (["stale"], "Science"), (None, "Science")])
def test_search_missing_selection_never_embeds(indexed, selected, collection):
    service, _, _ = indexed
    with pytest.raises(ValueError):
        research.search_sources("Research", "spectrum", selected, collection, "hybrid", service=service)
    assert service.store._vectors is None


@pytest.mark.parametrize("change", ["collection", "fingerprint", "generation", "status"])
def test_search_rejects_stale_or_wrong_scope_before_embedding(indexed, change):
    service, document, _ = indexed
    values = {"collection": "Other", "fingerprint": "old", "generation": "", "status": "deleted"}
    with service.store._db:
        service.store._db.execute(f"UPDATE documents SET {change}=? WHERE id=?", (values[change], document["id"]))
    with pytest.raises(ValueError, match="unavailable"):
        research.search_sources("Research", "spectrum", [document["id"]], "Science", "hybrid", service=service)


def test_hybrid_checks_connection_and_embeds_once_without_generation(indexed, monkeypatch):
    service, document, _ = indexed
    calls = []
    monkeypatch.setattr(service.store, "check_connection", lambda: calls.append("connection"))
    service._vectors = lambda texts: calls.append(texts) or [[1.0, 0.0, 0.0]]
    monkeypatch.setattr(service.store, "search", lambda *args, **kwargs: calls.append((args, kwargs)) or [])
    assert research.search_sources("Research", "spectrum", [document["id"]], "Science", "hybrid", service=service) == []
    assert calls[:2] == ["connection", ["spectrum"]]
    assert calls[2][0][1] == [1.0, 0.0, 0.0]
    assert calls[2][1]["document_ids"] == [document["id"]]


def test_catalog_change_during_embedding_discards_search(indexed, monkeypatch):
    service, document, _ = indexed
    monkeypatch.setattr(service.store, "check_connection", lambda: None)

    def embed(_):
        with service.store._db:
            service.store._db.execute("UPDATE documents SET generation='changed' WHERE id=?", (document["id"],))
        return [[1.0, 0.0, 0.0]]

    service._vectors = embed
    monkeypatch.setattr(service.store, "search", lambda *args, **kwargs: pytest.fail("Do not search a changed snapshot"))
    with pytest.raises(ValueError, match="changed"):
        research.search_sources("Research", "spectrum", [document["id"]], "Science", "hybrid", service=service)
