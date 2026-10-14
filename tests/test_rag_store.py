"""Offline tests of SQLite publication and persistent local Qdrant retrieval."""

from dataclasses import replace
from pathlib import Path
import sqlite3
import uuid

import pytest
from qdrant_client import QdrantClient, models

from rag_chatbot.config import Settings
from rag_chatbot.schemas import Chunk
from rag_chatbot.store import LibraryStore


@pytest.fixture
def store(tmp_path):
    settings = Settings(data_dir=tmp_path / "library", embedding_dimensions=3)
    result = LibraryStore(settings, vector_client=QdrantClient(path=str(tmp_path / "vectors")))
    yield result
    result.close()


def stage(store, tmp_path, text="source document", collection="science", name="paper.txt"):
    source = tmp_path / name
    source.write_text(text, encoding="utf-8")
    return store.stage_document(source, collection, store.settings.fingerprint())


def publish(store, tmp_path, text="BRCA1 mutation experiment", collection="science",
            name="paper.txt", vector=None):
    document = stage(store, tmp_path, text=text, collection=collection, name=name)
    chunk = Chunk(str(uuid.uuid4()), document["id"], text, "page 1")
    store.commit_document(document["id"], [chunk], [vector or [1.0, 0.0, 0.0]])
    return document, chunk


def test_lazy_catalog_does_not_contact_qdrant(tmp_path, monkeypatch):
    def unexpected_client(*args, **kwargs):
        raise AssertionError("Opening the catalog must not contact Qdrant")
    monkeypatch.setattr("qdrant_client.QdrantClient", unexpected_client)
    store = LibraryStore(Settings(data_dir=tmp_path / "library"))
    try:
        assert store.list_documents() == []
        assert store.collections() == []
    finally:
        store.close()


def test_configured_local_mode_and_readiness_need_no_server(tmp_path):
    settings = Settings(data_dir=tmp_path / "library", embedding_dimensions=3,
                        qdrant_local_path=str(tmp_path / "local-qdrant"))
    store = LibraryStore(settings)
    try:
        assert store._vectors is None
        name = store.check_connection()
        assert store.vector_client.collection_exists(name)
        _, chunk = publish(store, tmp_path)
        assert store.search("BRCA1", [1, 0, 0], "science")[0].chunk == chunk
    finally:
        store.close()


def test_persistence_and_durable_original_after_restart(tmp_path):
    settings = Settings(data_dir=tmp_path / "library", embedding_dimensions=3)
    vector_path = str(tmp_path / "vectors")
    store = LibraryStore(settings, vector_client=QdrantClient(path=vector_path))
    document, chunk = publish(store, tmp_path)
    (tmp_path / "paper.txt").unlink()
    store.close()
    reopened = LibraryStore(settings, vector_client=QdrantClient(path=vector_path))
    try:
        assert Path(reopened.get_document(document["id"])["path"]).read_text() == chunk.text
        assert reopened.list_documents()[0]["status"] == "ready"
        found = reopened.search("BRCA1", [1, 0, 0], "science")
        assert found[0].chunk.id == chunk.id
        assert found[0].filename == "paper.txt"
        assert Path(found[0].source_path).is_file()
        assert reopened.get_chunk(chunk.id) == chunk
    finally:
        reopened.close()


def test_hash_collection_and_fingerprint_deduplication(store, tmp_path):
    document, chunk = publish(store, tmp_path)
    same = stage(store, tmp_path, text=chunk.text, name="renamed.txt")
    assert same["id"] == document["id"] and same["unchanged"]
    other_collection = stage(store, tmp_path, text=chunk.text, collection="biology")
    assert other_collection["id"] != document["id"]
    changed = stage(store, tmp_path, text="new contents", name="paper.txt")
    assert changed["id"] != document["id"]
    version = store.stage_document(tmp_path / "paper.txt", "science", "different-fingerprint")
    assert version["id"] != changed["id"]
    assert store.collections() == ["biology", "science"]


def test_failed_embedding_is_not_visible_and_can_be_retried(store, tmp_path):
    document = stage(store, tmp_path)
    store.fail_document(document["id"], "embedding quota unavailable")
    assert store.get_document(document["id"])["status"] == "failed"
    assert store.search("source", [1, 0, 0], "science") == []
    retry = stage(store, tmp_path)
    assert retry["id"] == document["id"] and not retry["unchanged"]
    assert retry["status"] == "staging" and retry["error"] == ""
    chunk = Chunk("retried-chunk", retry["id"], "Successful retry", "page 2")
    store.commit_document(retry["id"], [chunk], [[1, 0, 0]])
    assert store.search("successful", [1, 0, 0], "science")[0].chunk == chunk


def test_interrupted_stage_can_be_retried(store, tmp_path):
    document = stage(store, tmp_path)
    retry = stage(store, tmp_path)
    assert retry["id"] == document["id"] and not retry["unchanged"]


def test_scope_filters_dense_and_lexical_candidates_before_limit(store, tmp_path):
    wanted, wanted_chunk = publish(store, tmp_path, text="selected enzyme", vector=[0.8, 0.2, 0])
    publish(store, tmp_path, text="enzyme enzyme enzyme", collection="other", name="other.txt")
    publish(store, tmp_path, text="enzyme enzyme", name="excluded.txt")
    result = store.search("enzyme", [1, 0, 0], "science", [wanted["id"]], limit=1, candidates=1)
    assert [item.chunk.id for item in result] == [wanted_chunk.id]
    assert store.search("enzyme", [1, 0, 0], "science", []) == []


def test_fingerprint_change_hides_incompatible_index(store, tmp_path):
    document, chunk = publish(store, tmp_path)
    original = store.settings
    store.settings = replace(original, chunk_size=original.chunk_size + 1)
    assert store.search("BRCA1", [1, 0, 0], "science") == []
    assert store.get_chunk(chunk.id) is None
    assert store.get_document(document["id"])["status"] == "ready"


def test_stale_generations_cannot_starve_valid_dense_results(store, tmp_path):
    document, chunk = publish(store, tmp_path, text="published observation", vector=[0.8, 0.2, 0])
    collection_name = store.ensure_index()
    store.vector_client.upsert(collection_name, points=[models.PointStruct(
        id=str(uuid.uuid4()), vector=[1.0, 0.0, 0.0],
        payload={"document_id": document["id"], "chunk_id": "stale-chunk",
                 "collection": "science", "generation": "failed-generation"},
    )], wait=True)
    results = store.search("unmatched query", [1, 0, 0], "science", limit=1, candidates=1)
    assert [item.chunk.id for item in results] == [chunk.id]


def test_deleted_document_is_hidden_even_when_vector_cleanup_fails(store, tmp_path, monkeypatch):
    deleted, _ = publish(store, tmp_path, text="deleted enzyme")
    _, retained = publish(store, tmp_path, text="retained enzyme", name="keep.txt", vector=[0.8, 0.2, 0])
    def offline(*args, **kwargs):
        raise RuntimeError("Qdrant offline")
    monkeypatch.setattr(store.vector_client, "delete", offline)
    store.delete_document(deleted["id"])
    assert store.get_document(deleted["id"])["status"] == "deleted"
    assert Path(deleted["path"]).exists()
    results = store.search("enzyme", [1, 0, 0], "science", limit=1, candidates=1)
    assert [item.chunk.id for item in results] == [retained.id]
    assert all(row["id"] != deleted["id"] for row in store.list_documents())


def test_reimport_after_delete_gets_new_identity(store, tmp_path):
    document, chunk = publish(store, tmp_path)
    store.delete_document(document["id"])
    reimport = stage(store, tmp_path, text=chunk.text)
    assert reimport["id"] != document["id"] and not reimport["unchanged"]


def test_partial_qdrant_upload_is_never_published(store, tmp_path, monkeypatch):
    document = stage(store, tmp_path)
    chunks = [Chunk(f"part-{index}", document["id"], f"part text {index}", "page 1") for index in range(129)]
    original_upsert = store.vector_client.upsert
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("second batch interrupted")
        return original_upsert(*args, **kwargs)
    monkeypatch.setattr(store.vector_client, "upsert", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.commit_document(document["id"], chunks, [[1, 0, 0]] * 129)
    assert store.get_document(document["id"])["status"] == "failed"
    assert store.get_document(document["id"])["chunk_count"] == 0
    assert store.search("part", [1, 0, 0], "science") == []
    monkeypatch.setattr(store.vector_client, "upsert", original_upsert)
    retry = stage(store, tmp_path)
    replacement = Chunk("replacement", retry["id"], "new successful part", "page 2")
    store.commit_document(retry["id"], [replacement], [[0.8, 0.2, 0]])
    assert store.search("part", [1, 0, 0], "science", limit=1, candidates=1)[0].chunk == replacement


def test_catalog_failure_rolls_back_chunks_and_fts_after_vector_upload(store, tmp_path):
    document = stage(store, tmp_path)
    store._db.executescript("""
        CREATE TRIGGER simulate_catalog_error BEFORE INSERT ON chunks
        WHEN new.text='bad' BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END;
    """)
    chunks = [Chunk("first", document["id"], "first successful chunk", "page 1"),
              Chunk("second", document["id"], "bad", "page 2")]
    with pytest.raises(sqlite3.IntegrityError, match="simulated write failure"):
        store.commit_document(document["id"], chunks, [[1, 0, 0], [1, 0, 0]])
    assert store._db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert store.get_document(document["id"])["status"] == "failed"
    assert store.search("first", [1, 0, 0], "science") == []


@pytest.mark.parametrize("query", ['"BRCA1" AND NOT (', 'text:BRCA1 OR *', 'BRCA1; DROP TABLE chunks;', 'BRCA1 NEAR("mutation",2)', '"*:()'])
def test_user_queries_are_not_interpreted_as_fts_syntax(store, tmp_path, query):
    publish(store, tmp_path)
    results = store.search(query, None, "science")
    assert isinstance(results, list)
    assert len(store.list_documents()) == 1


def test_hybrid_retrieval_retains_rare_exact_term(store, tmp_path):
    _, exact = publish(store, tmp_path, text="Rare marker QX-1397 measured at 10 μM", vector=[0, 1, 0])
    publish(store, tmp_path, text="General laboratory experiments", name="general.txt", vector=[1, 0, 0])
    results = store.search("QX-1397", [1, 0, 0], "science", limit=2, candidates=2)
    assert results[0].chunk.id == exact.id


@pytest.mark.parametrize("vector", [[1, 0], [float("nan"), 0, 1], [0, 0, 0]])
def test_invalid_embedding_cannot_publish(store, tmp_path, vector):
    document = stage(store, tmp_path)
    chunk = Chunk("invalid-vector", document["id"], "text", "page 1")
    with pytest.raises(ValueError):
        store.commit_document(document["id"], [chunk], [vector])
    assert store.get_document(document["id"])["status"] == "failed"


def test_external_image_path_cannot_be_published(store, tmp_path):
    document = stage(store, tmp_path)
    chunk = Chunk("unsafe-path", document["id"], "text", "page 1", image_path="../../outside.png")
    with pytest.raises(ValueError, match="inside"):
        store.commit_document(document["id"], [chunk], [[1, 0, 0]])


def test_collection_name_never_becomes_a_filesystem_path(store, tmp_path):
    document = stage(store, tmp_path, collection="../../outside")
    assert Path(document["path"]).resolve().is_relative_to(store.data_dir)
    assert not (tmp_path / "outside").exists()
