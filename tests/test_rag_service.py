"""Offline service integration with real extraction and persistent local Qdrant."""

from dataclasses import replace
from pathlib import Path
import copy
import json

from PIL import Image
import pytest
from qdrant_client import QdrantClient

from chatbot import api, tokens
from chatbot.messages import prepare_chat_messages, prepare_responses_input
from rag_chatbot.config import Settings
from rag_chatbot.schemas import Chunk, Evidence, SourceElement
from rag_chatbot.service import RAGService, SYSTEM_PROMPT
from rag_chatbot.store import LibraryStore


class Embeddings:
    def __init__(self):
        self.calls = []
        self.fail_call = None

    def __call__(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("temporary embedding failure")
        return [[1.0, 0.2, 0.1] for _ in texts]


class Stream:
    def __init__(self, response="The measured value is 7 [1]."):
        self.response = response
        self.calls = []
        self.closed = False

    def __call__(self, model, history, **kwargs):
        self.calls.append((model, copy.deepcopy(history), kwargs))
        try:
            yield self.response[:10]
            yield self.response[10:]
        finally:
            self.closed = True


@pytest.fixture
def service(tmp_path):
    settings = Settings(data_dir=tmp_path / "library", embedding_dimensions=3,
                        chunk_size=64, chunk_overlap=8)
    store = LibraryStore(settings, vector_client=QdrantClient(path=str(tmp_path / "vectors")))
    result = RAGService(settings, store=store, embedder=Embeddings(), streamer=Stream(),
                        vision=lambda path: "A diagram labels marker QX-1397.")
    yield result
    store.close()


def ingest(service, tmp_path, text="The measured value is 7. Marker QX-1397 is present.",
           name="experiment.txt", collection="science"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    progress = list(service.ingest([str(path)], collection))
    assert "1 indexed" in progress[-1]
    return path, service.store.list_documents(collection)[-1]


def test_full_ingest_query_persists_original_question_and_sources(service, tmp_path):
    _, document = ingest(service, tmp_path)
    history = [{"role": "user", "content": "Which marker was studied?"},
               {"role": "assistant", "content": "Marker QX-1397 [1]."}]
    before = copy.deepcopy(history)
    updated, sources = list(service.answer("What was its value?", history, "science", model="gpt-4o"))[-1]
    assert history == before
    assert updated[-2] == {"role": "user", "content": "What was its value?"}
    assert updated[-1]["content"] == "The measured value is 7 [1]."
    assert sources[0]["document_id"] == document["id"]
    assert Path(sources[0]["source_path"]).is_file()
    request = service.streamer.calls[0][1]
    assert "Document evidence" in request[-1]["content"]
    assert "measured value is 7" in request[-1]["content"]
    assert service.streamer.calls[0][2]["system_prompt"] == SYSTEM_PROMPT
    assert "Which marker was studied?" in service.embedder.calls[-1][0]
    assert service.streamer.closed


def test_duplicate_import_does_not_repeat_embeddings(service, tmp_path):
    path, document = ingest(service, tmp_path)
    count = len(service.embedder.calls)
    messages = list(service.ingest([str(path)], "science"))
    assert "1 unchanged" in messages[-1]
    assert len(service.embedder.calls) == count
    assert len(service.store.list_documents("science")) == 1


def test_injected_store_supplies_settings_when_not_repeated(service):
    injected = RAGService(store=service.store, embedder=service.embedder, streamer=service.streamer)
    assert injected.settings == service.store.settings
    assert len(injected._vectors(["question"])[0]) == 3


@pytest.mark.parametrize("model,attaches_image", [("gpt-4o", True), ("gpt-4", False)])
def test_real_image_ingestion_retrieves_description_and_compatible_image(service, tmp_path, model, attaches_image):
    path = tmp_path / "diagram.png"
    Image.new("RGB", (24, 16), "white").save(path)
    progress = list(service.ingest([str(path)], "science", use_vision=True))
    assert "1 indexed" in progress[-1]
    updated, sources = list(service.answer("Which marker appears?", [], "science", model=model))[-1]
    assert "QX-1397" in sources[0]["text"]
    assert Path(sources[0]["image_path"]).is_file()
    current = service.streamer.calls[-1][1][-1]
    assert ("image_url" in current) is attaches_image
    if attaches_image:
        assert "Attached image belongs to source [1]" in current["content"]
    assert updated[-2]["content"] == "Which marker appears?"


def test_retry_uses_successful_embedding_batches(service, tmp_path, monkeypatch):
    def chunks(elements, doc_id, *args):
        return [Chunk(f"{doc_id}-{index}", doc_id, f"paragraph {index}", f"Page {index}") for index in range(33)]
    monkeypatch.setattr("rag_chatbot.service.chunk_elements", chunks)
    service.embedder.fail_call = 2
    path = tmp_path / "many.txt"
    path.write_text("many sections", encoding="utf-8")
    first = list(service.ingest([str(path)], "science"))
    assert "1 failed" in first[-1]
    assert service.store.list_documents()[0]["status"] == "failed"
    second = list(service.ingest([str(path)], "science"))
    assert "1 indexed" in second[-1]
    assert [len(batch) for batch in service.embedder.calls] == [32, 1, 1]
    assert service.store.list_documents()[0]["chunk_count"] == 33


def test_cancelling_index_marks_pending_document_and_releases_lock(service, tmp_path):
    path = tmp_path / "cancel.txt"
    path.write_text("An interrupted experiment.", encoding="utf-8")
    progress = service.ingest([str(path)], "science")
    assert next(progress).startswith("Preparing")
    progress.close()
    document = service.store.list_documents()[0]
    assert document["status"] == "failed" and "stopped" in document["error"]
    assert not service.embedder.calls
    assert "1 indexed" in list(service.ingest([str(path)], "science"))[-1]


def test_cancelling_after_embedding_reuses_cache(service, tmp_path):
    path = tmp_path / "cancel-after-embedding.txt"
    path.write_text("A cached result.", encoding="utf-8")
    progress = service.ingest([str(path)], "science")
    while not next(progress).startswith("Embedding"):
        pass
    progress.close()
    assert len(service.embedder.calls) == 1
    assert "1 indexed" in list(service.ingest([str(path)], "science"))[-1]
    assert len(service.embedder.calls) == 1


def test_readiness_failure_precedes_paid_extraction_and_embedding(service, tmp_path, monkeypatch):
    def offline():
        raise ConnectionError("offline")
    monkeypatch.setattr(service.store, "check_connection", offline)
    path = tmp_path / "document.txt"
    path.write_text("A result.", encoding="utf-8")
    progress = list(service.ingest([str(path)], "science", use_vision=True))
    assert "No embedding request was sent" in "\n".join(progress)
    assert not service.embedder.calls
    assert service.store.list_documents()[0]["status"] == "failed"


def test_ocr_option_change_invalidates_extraction_cache_on_retry(service, tmp_path, monkeypatch):
    extraction_options = []
    def extract(path, *, vision_describe=None):
        extraction_options.append(vision_describe is not None)
        return [SourceElement("Figure details" if vision_describe else "Text only", "Page 1")]
    monkeypatch.setattr("rag_chatbot.service.extract_elements", extract)
    service.embedder.fail_call = 1
    path = tmp_path / "options.txt"
    path.write_text("source", encoding="utf-8")
    assert "1 failed" in list(service.ingest([str(path)], "science", use_vision=False))[-1]
    assert "1 indexed" in list(service.ingest([str(path)], "science", use_vision=True))[-1]
    assert extraction_options == [False, True]
    assert service.store.search("Figure", None, "science")[0].chunk.text == "Figure details"


def test_ready_native_document_can_upgrade_to_vision_without_duplicate_or_downgrade(service, tmp_path, monkeypatch):
    extraction_options = []
    def extract(path, *, vision_describe=None):
        extraction_options.append(vision_describe is not None)
        return [SourceElement("Enhanced figure details" if vision_describe else "Native prose", "Page 1")]
    monkeypatch.setattr("rag_chatbot.service.extract_elements", extract)
    path, document = ingest(service, tmp_path)
    assert "1 indexed" in list(service.ingest([str(path)], "science", use_vision=True))[-1]
    assert extraction_options == [False, True]
    assert len(service.store.list_documents()) == 1
    assert service.store.list_documents()[0]["id"] == document["id"]
    assert service.store.search("Enhanced", None, "science")[0].chunk.text == "Enhanced figure details"
    assert "1 unchanged" in list(service.ingest([str(path)], "science", use_vision=False))[-1]
    assert "1 unchanged" in list(service.ingest([str(path)], "science", use_vision=True))[-1]
    assert extraction_options == [False, True]


def test_new_vision_model_refreshes_ready_image_profile(service, tmp_path, monkeypatch):
    calls = []
    def extract(path, *, vision_describe=None):
        calls.append(service.settings.vision_model)
        return [SourceElement(f"Read with {service.settings.vision_model}", "Image")]
    monkeypatch.setattr("rag_chatbot.service.extract_elements", extract)
    path = tmp_path / "vision-source.txt"
    path.write_text("source", encoding="utf-8")
    assert "1 indexed" in list(service.ingest([str(path)], "science", use_vision=True))[-1]
    service.settings = replace(service.settings, vision_model="another-vision-model")
    assert "1 indexed" in list(service.ingest([str(path)], "science", use_vision=True))[-1]
    assert len(calls) == 2 and calls[-1] == "another-vision-model"


def test_cached_batches_are_validated_individually_not_just_by_combined_count(service, tmp_path, monkeypatch):
    def chunks(elements, doc_id, *args):
        return [Chunk(f"{doc_id}-{index}", doc_id, f"paragraph {index}", f"Page {index}") for index in range(33)]
    monkeypatch.setattr("rag_chatbot.service.chunk_elements", chunks)
    path = tmp_path / "batches.txt"
    path.write_text("source", encoding="utf-8")
    progress = service.ingest([str(path)], "science")
    while "33/33" not in next(progress):
        pass
    progress.close()
    document = service.store.list_documents()[0]
    cached_files = list((Path(document["path"]).parent / "embeddings").glob("*.json"))
    for file in cached_files:
        values = json.loads(file.read_text(encoding="utf-8"))
        values = values[:-1] if len(values) == 32 else values + values
        file.write_text(json.dumps(values), encoding="utf-8")
    assert "1 indexed" in list(service.ingest([str(path)], "science"))[-1]
    assert [len(batch) for batch in service.embedder.calls] == [32, 1, 32, 1]


@pytest.mark.parametrize("invalid_vectors", [[[1, 0]], [[float("nan"), 0, 1]], [["bad", 0, 1]]])
def test_invalid_cached_embedding_values_are_recomputed(service, tmp_path, invalid_vectors):
    path = tmp_path / "bad-cache.txt"
    path.write_text("source", encoding="utf-8")
    progress = service.ingest([str(path)], "science")
    while not next(progress).startswith("Embedding"):
        pass
    progress.close()
    document = service.store.list_documents()[0]
    cached = next((Path(document["path"]).parent / "embeddings").glob("*.json"))
    cached.write_text(json.dumps(invalid_vectors), encoding="utf-8")
    assert "1 indexed" in list(service.ingest([str(path)], "science"))[-1]
    assert len(service.embedder.calls) == 2


def test_stop_prevents_following_vision_requests_and_documents_but_retains_paid_cache(service, tmp_path, monkeypatch):
    calls = []
    def vision(path):
        calls.append(path)
        service.cancel_ingestion()
        return "Paid image result"
    service.vision = vision
    def extract(path, *, vision_describe=None):
        root = Path(path).parent
        for index in range(2):
            image = root / f"image-{index}.png"
            Image.new("RGB", (2, 2), (index, 0, 0)).save(image)
            vision_describe(str(image))
        return [SourceElement("Both images", "Page 1")]
    monkeypatch.setattr("rag_chatbot.service.extract_elements", extract)
    paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path in paths:
        path.write_text(path.name, encoding="utf-8")
    progress = list(service.ingest([str(path) for path in paths], "science", use_vision=True))
    assert "Indexing stopped" in progress[-1]
    assert len(calls) == 1 and not service.embedder.calls
    assert len(service.store.list_documents()) == 1
    document = service.store.list_documents()[0]
    assert document["status"] == "failed"
    assert len(list((Path(document["path"]).parent / "vision").glob("*.json"))) == 1
    # A later job clears the stop flag and reuses the first completed OCR call.
    service.vision = lambda path: calls.append(path) or "Second paid result"
    assert "1 indexed" in list(service.ingest([str(paths[0])], "science", use_vision=True))[-1]
    assert len(calls) == 2


def test_stop_after_embedding_call_prevents_next_batch_and_reuses_saved_vectors(service, tmp_path, monkeypatch):
    def chunks(elements, doc_id, *args):
        return [Chunk(f"{doc_id}-{index}", doc_id, f"paragraph {index}", "Page 1") for index in range(33)]
    monkeypatch.setattr("rag_chatbot.service.chunk_elements", chunks)
    embedder = service.embedder
    def stopping_embedder(texts):
        values = embedder(texts)
        service.cancel_ingestion()
        return values
    service.embedder = stopping_embedder
    path = tmp_path / "stopped.txt"
    path.write_text("source", encoding="utf-8")
    assert "Indexing stopped" in list(service.ingest([str(path)], "science"))[-1]
    assert [len(batch) for batch in embedder.calls] == [32]
    service.embedder = embedder
    assert "1 indexed" in list(service.ingest([str(path)], "science"))[-1]
    assert [len(batch) for batch in embedder.calls] == [32, 1]


def test_current_question_survives_lexical_query_term_limit(service):
    old = " ".join(f"oldterm{index}" for index in range(200))
    query = service._query("Where is QX-1397?", [{"role": "user", "content": old}])
    assert query.startswith("Current question: Where is QX-1397?")
    assert '"qx-1397"' in service.store._fts_query(query)


def test_citations_and_retrieval_stay_in_selected_document_scope(service, tmp_path):
    _, wanted = ingest(service, tmp_path, text="Selected source says seven.", name="wanted.txt")
    ingest(service, tmp_path, text="Excluded source says ninety.", name="excluded.txt")
    ingest(service, tmp_path, text="Private other collection.", name="other.txt", collection="other")
    _, sources = list(service.answer("What is the result?", [], "science", [wanted["id"]], model="gpt-4o"))[-1]
    assert {source["document_id"] for source in sources} == {wanted["id"]}
    request = service.streamer.calls[-1][1][-1]["content"]
    assert "Excluded source" not in request and "Private other" not in request


def test_empty_document_selection_never_embeds_or_generates(service, tmp_path):
    ingest(service, tmp_path)
    calls = len(service.embedder.calls)
    updated, sources = list(service.answer("What is the result?", [], "science", [], model="gpt-4o"))[-1]
    assert sources == [] and "couldn't find indexed evidence" in updated[-1]["content"]
    assert len(service.embedder.calls) == calls and not service.streamer.calls


def test_invalid_citation_ids_are_not_preserved_as_verified_sources(service, tmp_path):
    ingest(service, tmp_path)
    service.streamer = Stream("Supported [1], invented [999].")
    updated, sources = list(service.answer("Explain", [], "science", model="gpt-4o"))[-1]
    assert "[1]" in updated[-1]["content"] and "[999]" not in updated[-1]["content"]
    assert "unverified citation" in updated[-1]["content"]
    assert [source["number"] for source in sources] == [1]


def test_answer_cancellation_closes_upstream_stream(service, tmp_path):
    ingest(service, tmp_path)
    replies = service.answer("Explain", [], "science", model="gpt-4o")
    next(replies)
    replies.close()
    assert service.streamer.closed


def test_stream_failure_closes_stream_and_does_not_mutate_prior_history(service, tmp_path):
    ingest(service, tmp_path)
    prior = [{"role": "user", "content": "Old question"},
             {"role": "assistant", "content": "Old answer"}]
    saved = copy.deepcopy(prior)
    closed = []
    def failing_stream(*args, **kwargs):
        try:
            yield "Partial answer"
            raise RuntimeError("connection lost")
        finally:
            closed.append(True)
    service.streamer = failing_stream
    with pytest.raises(RuntimeError, match="connection lost"):
        list(service.answer("New question", prior, "science", model="gpt-4o"))
    assert closed == [True] and prior == saved


@pytest.mark.parametrize("model,prepare,surface_key", [
    ("gpt-4o", prepare_chat_messages, "messages"),
    ("gpt-5.1-codex", prepare_responses_input, "input"),
])
def test_request_budget_accounts_for_actual_multimodal_api_shape(service, tmp_path, monkeypatch, model, prepare, surface_key):
    image_path = service.settings.data_dir / "figure.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    evidence = [Evidence(Chunk("figure", "document", "Figure shows seven.", "Figure 1",
                               image_path=str(image_path)), "figure.png", 1.0, str(image_path))]
    monkeypatch.setattr(tokens, "reserve_for_output", lambda model: 0)
    initial, _ = service._request("Describe it.", [], evidence, model)
    context = tokens.count_tokens(initial, model) + tokens.count_message_tokens(
        {"role": "system", "content": SYSTEM_PROMPT}, model)
    assert tokens.count_tokens(prepare(initial), model) > tokens.count_tokens(initial, model)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)
    request, sources = service._request("Describe it.", [], evidence, model)
    _, kwargs = api._build_call(model, request, "None", 1, 1, SYSTEM_PROMPT, None)
    assert sources and kwargs[surface_key][-1]["role"] == "user"


def test_large_old_history_is_trimmed_without_losing_current_evidence(service, tmp_path, monkeypatch):
    _, document = ingest(service, tmp_path)
    evidence = service.store.search("marker", None, "science")
    prior = [{"role": "user", "content": "Old question " * 500},
             {"role": "assistant", "content": "Old response " * 500}]
    monkeypatch.setattr(tokens, "reserve_for_output", lambda model: 0)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: 500)
    request, sources = service._request("Which marker?", prior, evidence, "gpt-4o")
    assert len(request) == 1 and "QX-1397" in request[-1]["content"]
    assert sources[0]["document_id"] == document["id"]
    assert len(prior) == 2


def test_prior_source_snapshots_are_not_replayed_as_new_evidence(service, tmp_path):
    ingest(service, tmp_path)
    prior = [{"role": "user", "content": "Previous question"},
             {"role": "assistant", "content": "Previous answer [1].", "sources": [{"text": "OLD SECRET PASSAGE"}]}]
    list(service.answer("Next question", prior, "science", model="gpt-4o"))
    request = service.streamer.calls[-1][1]
    assert all("sources" not in message for message in request)
    assert "OLD SECRET PASSAGE" not in str(request)
