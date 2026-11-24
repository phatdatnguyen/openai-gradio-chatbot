"""Per-invocation indexing cancellation through vector publication."""

from pathlib import Path

from PIL import Image
import pytest
from qdrant_client import QdrantClient

from rag_chatbot.config import Settings
from rag_chatbot.service import RAGService
from rag_chatbot.store import LibraryStore


@pytest.fixture
def service(tmp_path):
    settings = Settings(data_dir=tmp_path / "library", embedding_dimensions=3)
    store = LibraryStore(settings, vector_client=QdrantClient(path=str(tmp_path / "vectors")))
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0, 0.2, 0.1] for _ in texts]

    result = RAGService(store=store, embedder=embed)
    result.test_embeddings = calls
    yield result
    store.close()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("Research measured seven samples.", encoding="utf-8")
    return path


def ingest(service, source, **kwargs):
    return list(service.ingest([source], "science", **kwargs))


def test_pre_cancelled_import_never_stages_or_contacts_vectors(service, source, monkeypatch):
    monkeypatch.setattr(service.store, "stage_document", lambda *a, **k: pytest.fail("No staging"))
    monkeypatch.setattr(service.store, "check_connection", lambda: pytest.fail("No vector request"))
    progress = ingest(service, source, is_cancelled=lambda: True)
    assert len(progress) == 1 and "Indexing stopped" in progress[0]
    assert service.test_embeddings == [] and not service.is_indexing


def test_invalid_callback_fails_before_staging(service, source):
    with pytest.raises(ValueError, match="callable"):
        ingest(service, source, is_cancelled=True)
    assert service.store.list_documents() == []
    assert service.test_embeddings == []


@pytest.mark.parametrize("after", ["stage", "connection"])
def test_cancelled_readiness_does_not_start_extraction_or_embeddings(service, source, monkeypatch, after):
    cancelled = [False]
    target = "stage_document" if after == "stage" else "check_connection"
    original = getattr(service.store, target)

    def superseded(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled[0] = True
        return result

    monkeypatch.setattr(service.store, target, superseded)
    monkeypatch.setattr("rag_chatbot.service.extract_elements", lambda *a, **k: pytest.fail("No extraction"))
    progress = ingest(service, source, is_cancelled=lambda: cancelled[0])
    assert "Indexing stopped" in progress[-1]
    assert service.test_embeddings == []
    assert service.store.list_documents()[0]["status"] == "failed"


def test_cancelled_embedding_keeps_paid_cache_without_publishing_and_retry_is_independent(service, source):
    cancelled = [False]
    original = service.embedder

    def embed(texts):
        result = original(texts)
        cancelled[0] = True
        return result

    service.embedder = embed
    progress = ingest(service, source, is_cancelled=lambda: cancelled[0])
    assert "Indexing stopped" in progress[-1]
    document = service.store.list_documents()[0]
    assert document["status"] == "failed" and not document["generation"]
    assert list(Path(document["path"]).parent.glob("embeddings/*.json"))
    assert len(service.test_embeddings) == 1
    # This call has no cancellation callback; the old invocation's True value
    # must not leak into the new job or force paid work to be repeated.
    progress = ingest(service, source)
    assert "1 indexed" in progress[-1] and len(service.test_embeddings) == 1
    assert service.store.list_documents()[0]["status"] == "ready"


def test_cancelled_vision_keeps_paid_cache_and_stops_before_embeddings(service, tmp_path):
    source = tmp_path / "figure.png"
    Image.new("RGB", (12, 12), "white").save(source)
    cancelled = [False]
    vision_calls = []

    def vision(path):
        vision_calls.append(path)
        cancelled[0] = True
        return "The image shows seven samples."

    service.vision = vision
    progress = ingest(service, source, use_vision=True, is_cancelled=lambda: cancelled[0])
    assert "Indexing stopped" in progress[-1] and len(vision_calls) == 1
    assert service.test_embeddings == []
    document = service.store.list_documents()[0]
    assert document["status"] == "failed"
    assert list(Path(document["path"]).parent.glob("vision/*.json"))
    progress = ingest(service, source, use_vision=True)
    assert "1 indexed" in progress[-1] and len(vision_calls) == 1


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cancel_during_vector_upsert_never_publishes_catalog(service, source, monkeypatch, cleanup_fails):
    cancelled = [False]
    original = service.store.vector_client.upsert

    def upsert(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled[0] = True
        return result

    monkeypatch.setattr(service.store.vector_client, "upsert", upsert)
    if cleanup_fails:
        def delete(*args, **kwargs):
            raise RuntimeError("Cleanup offline")
        monkeypatch.setattr(service.store.vector_client, "delete", delete)
    progress = ingest(service, source, is_cancelled=lambda: cancelled[0])
    assert "Indexing stopped" in progress[-1]
    document = service.store.list_documents()[0]
    assert document["status"] == "failed" and document["generation"] == ""
    assert document["chunk_count"] == 0
    assert service.store.search("samples", [1.0, 0.2, 0.1], "science") == []
    collection = service.store._collection_name(document["fingerprint"], document["dimensions"])
    assert service.store.vector_client.count(collection).count == (1 if cleanup_fails else 0)


def test_cancel_between_sources_does_not_stage_next_source(service, source, tmp_path):
    other = tmp_path / "second.txt"
    other.write_text("Second source.", encoding="utf-8")
    cancelled = [False]
    progress = service.ingest([source, other], "science", is_cancelled=lambda: cancelled[0])
    for update in progress:
        if update.startswith("Indexed"):
            cancelled[0] = True
    documents = service.store.list_documents()
    assert len(documents) == 1 and documents[0]["status"] == "ready"
    assert len(service.test_embeddings) == 1


def test_cancel_after_profile_write_stops_before_vector_publication(service, source, monkeypatch):
    from rag_chatbot import service as service_module

    cancelled = [False]
    original = service_module.atomic_json

    def write(path, value):
        result = original(path, value)
        if Path(path).name == "indexed-profile.json":
            cancelled[0] = True
        return result

    monkeypatch.setattr(service_module, "atomic_json", write)
    monkeypatch.setattr(service.store, "commit_document", lambda *a, **k: pytest.fail("No publication"))
    progress = ingest(service, source, is_cancelled=lambda: cancelled[0])
    assert "Indexing stopped" in progress[-1]
    assert service.store.list_documents()[0]["status"] == "failed"
