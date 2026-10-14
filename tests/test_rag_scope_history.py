"""Selected sources constrain prior assistant context as well as fresh retrieval."""

import copy

import pytest
from qdrant_client import QdrantClient

from rag_chatbot.config import Settings
from rag_chatbot.service import RAGService
from rag_chatbot.store import LibraryStore


@pytest.fixture
def service(tmp_path):
    settings = Settings(data_dir=tmp_path / "library", embedding_dimensions=3,
                        chunk_size=64, chunk_overlap=8)
    store = LibraryStore(settings, vector_client=QdrantClient(path=str(tmp_path / "vectors")))
    vectors, requests = [], []

    def embed(texts):
        vectors.append(list(texts))
        return [[1.0, 0.2, 0.1] for _ in texts]

    def stream(model, history, **kwargs):
        requests.append(copy.deepcopy(history))
        yield "The selected document supports this answer [1]."

    result = RAGService(settings, store=store, embedder=embed, streamer=stream)
    result.test_requests, result.test_vectors = requests, vectors
    documents = []
    for filename, text in (("a.txt", "Alpha evidence includes private quantity 17."),
                           ("b.txt", "Beta evidence includes public quantity 42.")):
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        assert "1 indexed" in list(result.ingest([path], "science"))[-1]
        documents.append(next(document for document in store.list_documents() if document["name"] == filename))
    result.test_documents = documents
    yield result
    store.close()


def exchange(question, answer, document_ids):
    return [{"role": "user", "content": question},
            {"role": "assistant", "content": answer,
             "sources": [{"document_id": identity} for identity in document_ids]}]


def test_deselecting_source_omits_all_prior_answers_but_retains_questions_and_visible_history(service):
    alpha, beta = service.test_documents
    history = (exchange("Which marker was tested first?", "A_PRIVATE_PRIOR_FACT", [alpha["id"]]) +
               exchange("And the second marker?", "B_ANSWER_MAY_HAVE_INHERITED_A_FACT", [beta["id"]]))
    original = copy.deepcopy(history)
    updated, sources = list(service.answer("What is its quantity?", history, "science", [beta["id"]], model="gpt-4o"))[-1]
    request = service.test_requests[-1]
    contents = "\n".join(message["content"] for message in request)
    assert "A_PRIVATE_PRIOR_FACT" not in contents
    assert "B_ANSWER_MAY_HAVE_INHERITED_A_FACT" not in contents
    assert "Alpha evidence" not in contents and "Beta evidence" in contents
    assert "Which marker was tested first?" in contents and "And the second marker?" in contents
    assert "And the second marker?" in service.test_vectors[-1][0]
    assert all(message["role"] != "assistant" for message in request)
    assert {source["document_id"] for source in sources} == {beta["id"]}
    assert history == original and updated[:len(history)] == original


@pytest.mark.parametrize("document_ids", [None, "both"])
def test_compatible_prior_answers_remain_available_for_followups(service, document_ids):
    alpha, beta = service.test_documents
    history = exchange("Describe both markers", "COMPATIBLE_ANSWER", [alpha["id"], beta["id"]])
    selected = [alpha["id"], beta["id"]] if document_ids == "both" else None
    list(service.answer("Which quantity is larger?", history, "science", selected, model="gpt-4o"))
    assert any(message["role"] == "assistant" and message["content"] == "COMPATIBLE_ANSWER"
               for message in service.test_requests[-1])


@pytest.mark.parametrize("sources", [None, [], "not a citation list", [{}], [None],
                                     [{"document_id": "missing-document"}], [{"document_id": ["bad"]}]])
def test_unattributed_or_invalid_old_answers_do_not_enter_request(service, sources):
    _, beta = service.test_documents
    previous = {"role": "assistant", "content": "UNVERIFIABLE_OLD_FACT"}
    if sources is not None:
        previous["sources"] = sources
    history = [{"role": "user", "content": "A previous user question"}, previous]
    original = copy.deepcopy(history)
    list(service.answer("Follow up", history, "science", [beta["id"]], model="gpt-4o"))
    assert "UNVERIFIABLE_OLD_FACT" not in str(service.test_requests[-1])
    assert "A previous user question" in str(service.test_requests[-1])
    assert history == original


def test_mixed_citation_answer_is_excluded_when_one_source_is_unchecked(service):
    alpha, beta = service.test_documents
    history = exchange("Compare them", "MIXED_SCOPE_ANSWER", [alpha["id"], beta["id"]])
    list(service.answer("What about B?", history, "science", [beta["id"]], model="gpt-4o"))
    assert "MIXED_SCOPE_ANSWER" not in str(service.test_requests[-1])


@pytest.mark.parametrize("unavailable", ["deleted", "staging", "other_collection", "old_fingerprint"])
def test_all_sources_scope_still_excludes_answers_from_unavailable_document(service, tmp_path, unavailable):
    alpha, beta = service.test_documents
    if unavailable == "deleted":
        service.store.delete_document(alpha["id"])
    elif unavailable == "staging":
        service.store.stage_document(tmp_path / "a.txt", "science", service.settings.fingerprint(), force=True)
    elif unavailable == "other_collection":
        with service.store._db:
            service.store._db.execute("UPDATE documents SET collection='other' WHERE id=?", (alpha["id"],))
    else:
        with service.store._db:
            service.store._db.execute("UPDATE documents SET fingerprint='old-version' WHERE id=?", (alpha["id"],))
    history = exchange("Previous question", "UNAVAILABLE_DOCUMENT_FACT", [alpha["id"]])
    updated, sources = list(service.answer("New question", history, "science", None, model="gpt-4o"))[-1]
    assert "UNAVAILABLE_DOCUMENT_FACT" not in str(service.test_requests[-1])
    assert {source["document_id"] for source in sources} == {beta["id"]}
    assert updated[1]["content"] == "UNAVAILABLE_DOCUMENT_FACT"


def test_empty_selection_makes_no_embedding_or_generation_call_and_preserves_history(service):
    alpha, _ = service.test_documents
    history = exchange("Previous question", "PREVIOUS_VISIBLE_ANSWER", [alpha["id"]])
    count = len(service.test_vectors)
    updated, sources = list(service.answer("New question", history, "science", [], model="gpt-4o"))[-1]
    assert sources == [] and not service.test_requests
    assert len(service.test_vectors) == count
    assert updated[:len(history)] == history
