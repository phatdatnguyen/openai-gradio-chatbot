"""Named workspace isolation with durable SQL and actual local vector stores."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag_chatbot import history, service, workspaces
from rag_chatbot.config import Settings
from rag_chatbot.schemas import Chunk
from rag_chatbot.store import LibraryStore


@pytest.fixture
def environment(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "workspaces", embedding_dimensions=3,
                        chunk_size=64, chunk_overlap=8,
                        qdrant_local_path=str(tmp_path / "vectors"))
    monkeypatch.setattr(service, "Settings", lambda: settings)
    monkeypatch.setattr(workspaces, "Settings", lambda: settings)
    # Do not close or overwrite services belonging to other application sessions.
    original = service._SERVICES
    monkeypatch.setattr(service, "_SERVICES", {})
    yield settings
    for instance in service._SERVICES.values():
        assert not instance.is_indexing
        instance.store.close()
    assert original is not service._SERVICES


def fake_embeddings(texts):
    return [[1.0, 0.2, 0.1] for _ in texts]


def fake_stream(model, messages, **kwargs):
    yield "The indexed marker is described in the source [1]."


def open_service(name):
    instance = service.get_service(name)
    instance.embedder = fake_embeddings
    instance.streamer = fake_stream
    return instance


def index(instance, path):
    assert "1 indexed" in list(instance.ingest([str(path)], "science"))[-1]
    return instance.store.list_documents("science")[0]


@pytest.mark.parametrize("name", [None, 12, "", " ", ".", "..", "../outside", "a/b", "a\\b",
                                  "C:\\outside", "C:outside", "a:b", "trailing.", " spaced", "spaced ",
                                  ".hidden", "NUL", "con.txt", "LPT1", "COM9.log", "COM¹", "LPT².txt",
                                  "a?b", "a*b", 'a"b', "a|b", "a<b", "a>b", "a\nname", "a\x00b",
                                  "x" * 65, "\U0001f600" * 64])
def test_unsafe_workspace_names_are_rejected_without_creating_paths(environment, name):
    with pytest.raises(ValueError):
        workspaces.create_workspace(name)
    assert not environment.data_dir.exists()


@pytest.mark.parametrize("name", ["x" * 64, "\U0001f600" * 63 + "\u754c"])
def test_workspace_names_within_character_and_utf8_limits_can_be_created(environment, name):
    # A component may be at most 255 bytes on common Linux filesystems, even
    # when its Unicode character count is within the application's limit.
    assert workspaces.create_workspace(name) == name
    assert workspaces.workspace_path(name).name == name
    assert workspaces.list_workspaces() == [name]


def test_listing_is_read_only_and_default_is_created_only_when_requested(environment):
    assert workspaces.list_workspaces() == []
    assert not environment.data_dir.exists()
    assert workspaces.default_workspace() == "Default"
    assert workspaces.list_workspaces() == ["Default"]
    directory = workspaces.workspace_path("default")
    assert directory == environment.data_dir / "Default"
    assert {"sources", "chats", "library.sqlite3"}.issubset(child.name for child in directory.iterdir())
    assert workspaces.default_workspace() == "Default"


def test_creation_canonicalizes_unicode_and_prevents_case_collisions(environment):
    assert workspaces.create_workspace("Research café") == "Research café"
    assert workspaces.workspace_path("RESEARCH CAFE\u0301").name == "Research café"
    with pytest.raises(ValueError, match="already exists"):
        workspaces.create_workspace("research café")
    assert workspaces.default_workspace() == "Research café"
    assert workspaces.list_workspaces() == ["Research café"]
    assert not (environment.data_dir / "Default").exists()


def test_listing_requires_real_complete_workspace_not_unrelated_directories(environment):
    root = environment.data_dir
    root.mkdir()
    (root / "not-a-workspace").mkdir()
    (root / "plain-file").write_text("unrelated", encoding="utf-8")
    invalid = root / "bad-catalog"
    (invalid / "sources").mkdir(parents=True)
    (invalid / "chats").mkdir()
    (invalid / "library.sqlite3").write_text("not a SQLite catalog", encoding="utf-8")
    workspaces.create_workspace("Zulu")
    workspaces.create_workspace("Alpha")
    assert workspaces.list_workspaces() == ["Alpha", "Zulu"]
    with pytest.raises(ValueError, match="not a valid workspace"):
        workspaces.workspace_path("bad-catalog")
    with pytest.raises(ValueError, match="already exists"):
        workspaces.create_workspace("PLAIN-FILE")
    with pytest.raises(FileNotFoundError):
        workspaces.workspace_path("unknown")
    assert workspaces.workspace_path("New", create=True).name == "New"


def test_linked_workspace_or_members_are_not_opened(environment, monkeypatch):
    workspaces.create_workspace("Research")
    original = workspaces._linked
    directory = environment.data_dir / "Research"
    for target in (directory, directory / "sources", directory / "chats", directory / "library.sqlite3"):
        monkeypatch.setattr(workspaces, "_linked", lambda path, target=target: path == target or original(path))
        assert workspaces.list_workspaces() == []
        with pytest.raises(ValueError, match="unsafe links"):
            workspaces.workspace_path("Research")


def test_actual_directory_symlink_is_rejected(environment, tmp_path):
    workspaces.create_workspace("Research")
    target = environment.data_dir / "Research"
    linked = environment.data_dir / "Linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows host.")
    assert workspaces.list_workspaces() == ["Research"]
    with pytest.raises(ValueError, match="unsafe links"):
        workspaces.workspace_path("Linked")


def test_local_vector_paths_reject_linked_directories(environment, monkeypatch):
    workspaces.create_workspace("Research")
    scoped = workspaces.workspace_settings("Research")
    destination = Path(scoped.qdrant_local_path)
    destination.mkdir(parents=True)
    original = workspaces._linked
    monkeypatch.setattr(workspaces, "_linked", lambda path: path == destination or original(path))
    with pytest.raises(ValueError, match="linked directory"):
        workspaces.workspace_settings("Research")


def test_get_service_is_thread_safe_canonical_and_does_not_create_unknown_workspaces(environment):
    workspaces.create_workspace("Research")
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(service.get_service, ["Research", "RESEARCH", "research"] * 8))
    assert len({id(instance) for instance in instances}) == 1
    assert instances[0].settings.data_dir == environment.data_dir / "Research"
    assert instances[0].store._vectors is None
    assert len(service._SERVICES) == 1
    with pytest.raises(FileNotFoundError):
        service.get_service("Missing")
    assert not (environment.data_dir / "Missing").exists()


def test_direct_default_service_initializes_only_an_empty_workspace_root(environment):
    instance = service.get_service()
    assert instance.settings.data_dir == environment.data_dir / "Default"
    assert workspaces.list_workspaces() == ["Default"]
    assert service.get_service("DEFAULT") is instance


def test_two_cached_services_isolate_documents_chats_and_source_files(environment, tmp_path):
    for name in ("Alpha", "Beta"):
        workspaces.create_workspace(name)
    alpha, beta = open_service("Alpha"), open_service("Beta")
    assert alpha is not beta
    assert alpha.settings.namespace != beta.settings.namespace
    assert alpha.settings.qdrant_local_path != beta.settings.qdrant_local_path
    document = tmp_path / "experiment.txt"
    document.write_text("Alpha marker QX-1397 is measured at 7.", encoding="utf-8")
    alpha_doc = index(alpha, document)
    document.write_text("Beta marker BX-777 is measured at 19.", encoding="utf-8")
    beta_doc = index(beta, document)
    assert alpha_doc["id"] != beta_doc["id"]
    assert alpha.store.get_document(beta_doc["id"]) is None
    assert beta.store.get_document(alpha_doc["id"]) is None
    alpha_messages, alpha_sources = list(alpha.answer("Which marker?", [], "science", model="gpt-4o"))[-1]
    beta_messages, beta_sources = list(beta.answer("Which marker?", [], "science", model="gpt-4o"))[-1]
    assert {source["document_id"] for source in alpha_sources} == {alpha_doc["id"]}
    assert {source["document_id"] for source in beta_sources} == {beta_doc["id"]}
    assert "QX-1397" in alpha_sources[0]["text"] and "BX-777" in beta_sources[0]["text"]
    assert "QX-1397" in Path(alpha_sources[0]["source_path"]).read_text(encoding="utf-8")
    for current, other in ((alpha, beta_sources[0]), (beta, alpha_sources[0])):
        with pytest.raises(ValueError, match="configured data directory"):
            current.store._managed_path(other["source_path"])
    history.save_chat(alpha_messages, "same chat", data_dir=alpha.settings.data_dir)
    history.save_chat(beta_messages, "same chat", data_dir=beta.settings.data_dir)
    assert history.list_chats(data_dir=alpha.settings.data_dir) == ["same chat"]
    assert history.load_chat("same chat", data_dir=alpha.settings.data_dir)[-1]["sources"] == alpha_sources
    assert history.load_chat("same chat", data_dir=beta.settings.data_dir)[-1]["sources"] == beta_sources
    alpha.store.delete_document(alpha_doc["id"])
    assert alpha.store.list_documents() == []
    assert beta.store.get_document(beta_doc["id"])["status"] == "ready"
    assert beta.store.search("BX-777", [1.0, 0.2, 0.1], "science")


def test_opening_other_workspace_does_not_close_active_job_and_cancel_is_local(environment, tmp_path):
    workspaces.create_workspace("Alpha")
    workspaces.create_workspace("Beta")
    source = tmp_path / "experiment.txt"
    source.write_text("The marker is QX-1397.", encoding="utf-8")
    alpha = open_service("Alpha")
    progress = alpha.ingest([str(source)], "science")
    try:
        assert next(progress).startswith("Preparing")
        assert alpha.is_indexing
        beta = open_service("Beta")
        assert alpha.is_indexing and not beta.is_indexing
        index(beta, source)
        assert alpha.is_indexing
        alpha.cancel_ingestion()
        assert not beta._cancel_indexing.is_set()
        assert "stopped" in list(progress)[-1]
    finally:
        progress.close()
    assert not alpha.is_indexing
    assert alpha.store.list_documents()[0]["status"] == "failed"
    assert beta.store.list_documents()[0]["status"] == "ready"
    assert "1 indexed" in list(alpha.ingest([str(source)], "science"))[-1]
    assert service.get_service("Alpha") is alpha and service.get_service("Beta") is beta


def test_vector_names_are_stable_isolated_and_persist_after_restart(environment, tmp_path):
    workspaces.create_workspace("Alpha")
    workspaces.create_workspace("Beta")
    alpha_settings = workspaces.workspace_settings("Alpha")
    beta_settings = workspaces.workspace_settings("Beta")
    assert workspaces.workspace_settings("ALPHA") == alpha_settings
    assert alpha_settings.fingerprint() == beta_settings.fingerprint()
    stores = []
    try:
        alpha = LibraryStore(alpha_settings)
        beta = LibraryStore(beta_settings)
        stores.extend((alpha, beta))
        alpha_name, beta_name = alpha.ensure_index(), beta.ensure_index()
        assert alpha_name != beta_name
        source = tmp_path / "report.txt"
        source.write_text("Durable marker QX-1397.", encoding="utf-8")
        document = alpha.stage_document(source, "science", alpha_settings.fingerprint())
        alpha.commit_document(document["id"], [Chunk("chunk-one", document["id"], source.read_text(), "Page 1")],
                              [[1.0, 0.2, 0.1]])
        alpha.close()
        stores.remove(alpha)
        reopened = LibraryStore(workspaces.workspace_settings("alpha"))
        stores.append(reopened)
        assert reopened.ensure_index() == alpha_name
        assert reopened.search("QX-1397", [1.0, 0.2, 0.1], "science")[0].chunk.document_id == document["id"]
        assert not beta.search("QX-1397", [1.0, 0.2, 0.1], "science")
    finally:
        for store in stores:
            store.close()


def test_namespaced_server_collection_mapping_preserves_standalone_compatibility(environment, tmp_path):
    workspaces.create_workspace("Alpha")
    workspaces.create_workspace("Beta")
    # One injected vector client models one server hosting both namespace collections.
    client = QdrantClient(path=str(tmp_path / "shared-vectors"))
    stores = [LibraryStore(replace(workspaces.workspace_settings(name), qdrant_local_path=None),
                           vector_client=client) for name in ("Alpha", "Beta")]
    try:
        names = [store.ensure_index() for store in stores]
        assert len(set(names)) == 2
        assert {item.name for item in client.get_collections().collections} == set(names)
        source = tmp_path / "shared.txt"
        source.write_text("The marker is QX-1397.", encoding="utf-8")
        documents = []
        for position, store in enumerate(stores):
            document = store.stage_document(source, "science", store.settings.fingerprint())
            documents.append(document)
            store.commit_document(document["id"], [Chunk(f"shared-{position}", document["id"],
                                  source.read_text(), "Page 1")], [[1.0, 0.2, 0.1]])
        for store, document in zip(stores, documents):
            assert {e.chunk.document_id for e in store.search("QX-1397", [1.0, 0.2, 0.1], "science")} == {document["id"]}
        stores[0].delete_document(documents[0]["id"])
        assert client.count(names[0]).count == 0
        assert client.count(names[1]).count == 1
        assert stores[1].search("QX-1397", [1.0, 0.2, 0.1], "science")[0].chunk.document_id == documents[1]["id"]
        direct = replace(environment, data_dir=tmp_path / "standalone", namespace="", qdrant_local_path=None)
        with_standalone = LibraryStore(direct, vector_client=client)
        stores.append(with_standalone)
        assert with_standalone.ensure_index() not in names
    finally:
        # The stores normally own their clients; this test explicitly shares one.
        for store in stores:
            store._vectors = None
            store.close()
        client.close()
