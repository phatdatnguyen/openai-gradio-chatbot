"""Editable catalog source names preserve originals, identity, and paid caches."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag_chatbot.config import Settings
from rag_chatbot.schemas import SourceElement
from rag_chatbot.service import RAGService
from rag_chatbot.store import LibraryStore, validate_display_name


class Embeddings:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.2, 0.1] for _ in texts]


@pytest.fixture
def service(tmp_path):
    settings = Settings(data_dir=tmp_path / "library", embedding_dimensions=3,
                        chunk_size=64, chunk_overlap=8)
    store = LibraryStore(settings, vector_client=QdrantClient(path=str(tmp_path / "vectors")))
    result = RAGService(settings, store=store, embedder=Embeddings(),
                        streamer=lambda *args, **kwargs: iter(["The marker is documented [1]."]))
    yield result
    assert not result.is_indexing
    store.close()


def upload(tmp_path, name="upload.txt", text="Marker QX-1397 has a measured value of 7."):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_custom_name_is_used_in_progress_search_and_citations_without_renaming_original(service, tmp_path):
    path = upload(tmp_path)
    original = path.read_bytes()
    progress = list(service.ingest([str(path)], "science", display_names=["Experiment notes"]))
    assert "1 indexed" in progress[-1]
    assert any("Experiment notes.txt" in update for update in progress)
    document = service.store.list_documents()[0]
    assert document["name"] == "Experiment notes.txt"
    assert Path(document["path"]).name == "original.txt"
    assert path.name == "upload.txt" and path.read_bytes() == original
    assert Path(document["path"]).read_bytes() == original
    found = service.store.search("QX-1397", [1, 0.2, 0.1], "science")
    assert found[0].filename == "Experiment notes.txt"
    history, citations = list(service.answer("Which marker?", [], "science", model="gpt-4o"))[-1]
    assert citations[0]["filename"] == "Experiment notes.txt"
    assert history[-1]["sources"][0]["filename"] == "Experiment notes.txt"


def test_duplicate_content_explicit_rename_only_changes_catalog_metadata(service, tmp_path, monkeypatch):
    path = upload(tmp_path)
    list(service.ingest([str(path)], "science", display_names=["First title"]))
    before = service.store.list_documents()[0]
    directory = Path(before["path"]).parent
    cached = {file.relative_to(directory): file.read_bytes() for file in directory.rglob("*") if file.is_file()}
    calls = len(service.embedder.calls)
    with monkeypatch.context() as patch:
        patch.setattr(service.store, "check_connection", lambda: pytest.fail("Metadata rename must not contact Qdrant"))
        patch.setattr("rag_chatbot.store.shutil.copyfile", lambda *args: pytest.fail("Must not copy the original again"))
        result = list(service.ingest([str(path)], "science", display_names=["Renamed paper.TXT"]))
    assert "1 unchanged" in result[-1]
    after = service.store.list_documents()[0]
    assert after["id"] == before["id"] and after["path"] == before["path"]
    assert after["generation"] == before["generation"] and after["status"] == "ready"
    assert after["name"] == "Renamed paper.TXT" and len(service.embedder.calls) == calls
    assert {file.relative_to(directory): file.read_bytes() for file in directory.rglob("*") if file.is_file()} == cached
    found = service.store.search("QX-1397", [1, 0.2, 0.1], "science")
    assert found[0].filename == "Renamed paper.TXT"
    assert service.store.vector_client.count(service.store.ensure_index()).count == 1


@pytest.mark.parametrize("overrides", [None, [None], [""], ["   "]])
def test_reimport_without_explicit_override_preserves_existing_custom_name(service, tmp_path, overrides):
    path = upload(tmp_path)
    list(service.ingest([str(path)], "science", display_names=["My chosen name.txt"]))
    copied_upload = upload(tmp_path, "different original basename.txt")
    progress = list(service.ingest([str(copied_upload)], "science", display_names=overrides))
    assert "1 unchanged" in progress[-1]
    assert service.store.list_documents()[0]["name"] == "My chosen name.txt"
    assert len(service.embedder.calls) == 1


def test_blank_override_for_new_document_uses_uploaded_basename(service, tmp_path):
    path = upload(tmp_path, "Original name.txt")
    list(service.ingest([str(path)], "science", display_names=[""]))
    assert service.store.list_documents()[0]["name"] == "Original name.txt"


@pytest.mark.parametrize("name", ["../escape.txt", "folder/name.txt", "folder\\name.txt", "C:\\name.txt",
                                  "C:name.txt", "name:stream.txt", "NUL.txt", "Con.TXT", "NUL .txt",
                                  "PRN", "COM1", "LPT9.txt", "COM¹.txt", "CONIN$", "CONOUT$.txt",
                                  "name?.txt", "name*.txt", "<name>.txt", 'name".txt', "name|.txt",
                                  " leading.txt", "trailing.txt ", "name.", ".", "..", "name\n.txt",
                                  "name\x00.txt", "name\x7f.txt", "name\x85.txt", "\n", "\t", "paper.pdf",
                                  "x" * 252, "😀" * 126, 123, {}])
def test_all_names_are_validated_before_first_copy_qdrant_or_embedding(service, tmp_path, monkeypatch, name):
    first = upload(tmp_path, "first.txt", "First source")
    second = upload(tmp_path, "second.txt", "Second source")
    monkeypatch.setattr(service.store, "check_connection", lambda: pytest.fail("No vector calls on invalid names"))
    monkeypatch.setattr("rag_chatbot.store.shutil.copyfile", lambda *args: pytest.fail("No partial batch copying"))
    with pytest.raises(ValueError):
        list(service.ingest([str(first), str(second)], "science", display_names=["Good name", name]))
    assert not service.embedder.calls
    assert service.store.list_documents() == []
    assert list((service.settings.data_dir / "sources").iterdir()) == []


@pytest.mark.parametrize("names", [[], ["only one"], ["one", "two", "three"], "not a list", {"one": "two"}])
def test_display_name_count_or_container_mismatch_cannot_partially_index(service, tmp_path, monkeypatch, names):
    first = upload(tmp_path, "first.txt")
    second = upload(tmp_path, "second.txt", "Second source")
    monkeypatch.setattr(service.store, "stage_document", lambda *args, **kwargs: pytest.fail("Must validate batch count first"))
    with pytest.raises(ValueError, match="exactly one source name"):
        list(service.ingest([first, second], "science", display_names=names))
    assert not service.embedder.calls and not service.is_indexing


def test_direct_stage_rejects_unsafe_name_before_copy_or_hash(service, tmp_path, monkeypatch):
    path = upload(tmp_path)
    monkeypatch.setattr(service.store, "_hash_file", lambda *args: pytest.fail("Name validation must precede file work"))
    with pytest.raises(ValueError):
        service.store.stage_document(path, "science", service.settings.fingerprint(), display_name="../bad.txt")
    assert not service.store.list_documents()


def test_two_uploads_with_same_original_basename_keep_independent_custom_names(service, tmp_path):
    first = upload(tmp_path, "first/report.txt", "Marker ALPHA is 7.")
    second = upload(tmp_path, "second/report.txt", "Marker BETA is 19.")
    result = list(service.ingest([first, second], "science", display_names=["Alpha result", "Beta result"]))
    assert "2 indexed" in result[-1]
    documents = service.store.list_documents()
    assert {document["name"] for document in documents} == {"Alpha result.txt", "Beta result.txt"}
    assert len({document["id"] for document in documents}) == 2
    assert first.name == second.name == "report.txt"
    assert first.read_text(encoding="utf-8") == "Marker ALPHA is 7."
    assert second.read_text(encoding="utf-8") == "Marker BETA is 19."


def test_vision_upgrade_and_failed_retry_keep_requested_name_and_document_id(service, tmp_path, monkeypatch):
    path = upload(tmp_path)
    list(service.ingest([path], "science", display_names=["Initial name"]))
    original = service.store.list_documents()[0]
    calls = []
    real_stage = service.store.stage_document

    def stage(*args, **kwargs):
        calls.append(kwargs)
        return real_stage(*args, **kwargs)

    monkeypatch.setattr(service.store, "stage_document", stage)
    monkeypatch.setattr("rag_chatbot.service.extract_elements", lambda *args, **kwargs: [SourceElement("Enhanced evidence", "Page 1")])
    result = list(service.ingest([path], "science", use_vision=True, display_names=["Enhanced name"]))
    assert "1 indexed" in result[-1]
    assert calls[-1] == {"force": True, "display_name": "Enhanced name.txt"}
    upgraded = service.store.list_documents()[0]
    assert upgraded["id"] == original["id"] and upgraded["name"] == "Enhanced name.txt"
    service.store.stage_document(path, "science", service.settings.fingerprint(), force=True)
    service.store.fail_document(upgraded["id"], "temporary failure")
    # A failed row can be renamed and retried without discarding its durable caches.
    result = list(service.ingest([path], "science", use_vision=True, display_names=["Retried title"]))
    assert "1 indexed" in result[-1]
    retried = service.store.list_documents()[0]
    assert retried["id"] == original["id"] and retried["name"] == "Retried title.txt"


def test_extension_and_windows_length_rules_are_applied_after_appending_suffix():
    assert validate_display_name("upload.PDF", "Renamed source") == "Renamed source.PDF"
    assert validate_display_name("upload.pdf", "Renamed source.PDF") == "Renamed source.PDF"
    assert validate_display_name("upload.txt", "x" * 251) == "x" * 251 + ".txt"
    assert validate_display_name("upload.txt", ".hidden") == ".hidden.txt"
    assert validate_display_name("upload.txt", "COM10") == "COM10.txt"
    with pytest.raises(ValueError, match="original file extension"):
        validate_display_name("upload.xlsx", "Data.csv")


def test_name_change_persists_after_catalog_restart(service, tmp_path):
    path = upload(tmp_path)
    list(service.ingest([path], "science", display_names=["Persistent title"]))
    # A separate catalog reader is safe and does not need to open vector storage.
    reopened = LibraryStore(service.settings)
    try:
        assert reopened.list_documents()[0]["name"] == "Persistent title.txt"
    finally:
        reopened.close()
