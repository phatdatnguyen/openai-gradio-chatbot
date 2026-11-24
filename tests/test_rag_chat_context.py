"""Whole-source chat budgets, snapshot validity, and cancellation without APIs."""

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest

from chatbot import config, tokens
from rag_chatbot import chat_context
from rag_chatbot.config import Settings
from rag_chatbot.service import ChatCancelled, RAGService, WEB_SYSTEM_PROMPT
from rag_chatbot.store import LibraryStore


@pytest.fixture
def service(tmp_path, monkeypatch):
    store = LibraryStore(Settings(data_dir=tmp_path / "library"))
    calls = []

    def unexpected(*args, **kwargs):
        pytest.fail("Full selected sources must not use Qdrant or embeddings")

    def sender(model, history, **kwargs):
        calls.append((model, copy.deepcopy(history), kwargs))
        return "The complete evidence is available [1]."

    def streamer(model, history, **kwargs):
        calls.append((model, copy.deepcopy(history), kwargs))
        yield "The complete evidence "
        yield "is available [1]."

    result = RAGService(store=store, embedder=unexpected, streamer=streamer, sender=sender)
    result.test_calls = calls
    monkeypatch.setattr(store, "check_connection", unexpected)
    monkeypatch.setattr(store, "search", unexpected)
    yield result
    store.close()


def publish(service, tmp_path, passages, *, name="paper.txt", collection="science"):
    """Seed the authoritative ready catalog; no vector client is constructed."""
    path = tmp_path / name
    path.write_text("\n".join(passages), encoding="utf-8")
    document = service.store.stage_document(path, collection, service.settings.fingerprint())
    with service.store._db:
        for index, text in enumerate(passages):
            service.store._db.execute(
                "INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
                (uuid4().hex, document["id"], text, f"Section {index + 1}", "text", index))
        service.store._db.execute(
            "UPDATE documents SET status='ready',chunk_count=?,generation=? WHERE id=?",
            (len(passages), uuid4().hex, document["id"]))
    return document["id"]


def answer(service, selected, **kwargs):
    options = {"model": "gpt-4o", "stream_output": False, "context_mode": "full", **kwargs}
    return list(service.answer("Explain all sections.", [], "science", document_ids=selected, **options))


def evidence_from_call(service):
    content = service.test_calls[-1][1][-1]["content"]
    return json.loads(content.split("(quoted JSON data):\n", 1)[1].split("\n\nUser question:", 1)[0])


def test_full_context_retains_all_selected_passages_in_source_order(service, tmp_path):
    first = [f"Passage {index}: complete value {index + 10}." for index in range(11)]
    a = publish(service, tmp_path, first, name="A.txt")
    b = publish(service, tmp_path, ["Second document unique finding."], name="B.txt")
    publish(service, tmp_path, ["Unselected confidential text."], name="excluded.txt")
    messages, sources = answer(service, [b, a])[-1]
    assert len(sources) == 12  # Relevant mode normally keeps at most eight.
    assert [source["number"] for source in sources] == list(range(1, 13))
    assert [source["text"] for source in sources] == ["Second document unique finding.", *first]
    assert [source["document_id"] for source in sources] == [b, *([a] * 11)]
    assert [item["text"] for item in evidence_from_call(service)] == [source["text"] for source in sources]
    assert "Unselected confidential" not in str(service.test_calls)
    assert all(Path(source["source_path"]).is_file() for source in sources)
    assert messages[-1]["context_mode"] == "full" and "web_search" not in messages[-1]
    assert service.store._vectors is None


@pytest.mark.parametrize("selected", [None, [], (), "bad", ["missing"]])
@pytest.mark.parametrize("web_search", [config.WEB_SEARCH_OFF, "medium"])
def test_full_requires_explicit_current_selection_before_generation(service, selected, web_search):
    with pytest.raises(ValueError):
        answer(service, selected, web_search=web_search)
    assert service.test_calls == []


@pytest.mark.parametrize("mode", [None, "all", {}, False, 1])
def test_invalid_mode_is_rejected_before_storage_or_paid_work(service, monkeypatch, mode):
    monkeypatch.setattr(service.store, "snapshot_sources", lambda *a, **k: pytest.fail("No reads"))
    with pytest.raises(ValueError, match="chat context"):
        answer(service, ["unused"], context_mode=mode)
    assert service.test_calls == []


@pytest.mark.parametrize("field,value", [("status", "failed"), ("status", "deleted"),
                                        ("fingerprint", "old-index"), ("generation", "")])
def test_stale_selected_source_fails_instead_of_being_omitted(service, tmp_path, field, value):
    selected = publish(service, tmp_path, ["Evidence."])
    with service.store._db:
        service.store._db.execute(f"UPDATE documents SET {field}=? WHERE id=?", (value, selected))
    with pytest.raises(ValueError, match="unavailable"):
        answer(service, [selected])
    assert service.test_calls == []


def test_full_context_rejects_source_from_another_collection(service, tmp_path):
    selected = publish(service, tmp_path, ["Other collection."], collection="other")
    with pytest.raises(ValueError, match="active collection"):
        answer(service, [selected])
    assert service.test_calls == []


@pytest.mark.parametrize("limit", ["documents", "chunks", "characters", "tokens"])
def test_full_context_bounds_reject_before_generation(service, tmp_path, limit):
    if limit == "documents":
        selected = [publish(service, tmp_path, [f"Document {index}."], name=f"doc{index}.txt")
                    for index in range(chat_context.MAX_DOCUMENTS + 1)]
    else:
        passages = (["Passage"] * (chat_context.MAX_CHUNKS + 1) if limit == "chunks" else
                    ["x" * (chat_context.MAX_SOURCE_CHARACTERS + 1)] if limit == "characters" else
                    ["token " * (chat_context.MAX_EVIDENCE_TOKENS + 1)])
        selected = [publish(service, tmp_path, passages)]
    with pytest.raises(ValueError):
        answer(service, selected)
    assert service.test_calls == []


@pytest.mark.parametrize("model,web_search", [("gpt-4o", config.WEB_SEARCH_OFF),
                                            ("gpt-5.1-codex", config.WEB_SEARCH_OFF),
                                            ("gpt-4o", "medium")])
def test_full_model_context_overflow_never_truncates_evidence(service, tmp_path, monkeypatch, model, web_search):
    selected = publish(service, tmp_path, ["Complete source text. " * 250])
    monkeypatch.setattr(tokens, "reserve_for_output", lambda model: 0)
    monkeypatch.setattr(config, "get_max_context_tokens", lambda *args: 500)
    with pytest.raises(ValueError, match="No source text was omitted"):
        answer(service, [selected], model=model, web_search=web_search)
    assert service.test_calls == []


def test_full_mode_trims_old_conversation_but_keeps_all_current_evidence(service, tmp_path, monkeypatch):
    selected = publish(service, tmp_path, ["First section.", "Last section."])
    prior = [{"role": "user", "content": "Old question " * 500}]
    before = copy.deepcopy(prior)
    monkeypatch.setattr(tokens, "reserve_for_output", lambda model: 0)
    monkeypatch.setattr(config, "get_max_context_tokens", lambda *args: 500)
    messages, sources = list(service.answer("Explain.", prior, "science", document_ids=[selected],
                                          context_mode="full", model="gpt-4o", stream_output=False))[-1]
    assert len(service.test_calls[0][1]) == 1 and len(evidence_from_call(service)) == 2
    assert sources[1]["text"] == "Last section."
    assert prior == before and messages[:1] == prior


@pytest.mark.parametrize("field,value", [("fingerprint", "changed"), ("generation", "new-publication"),
                                        ("status", "deleted")])
@pytest.mark.parametrize("when", ["preflight", "generation"])
def test_changed_snapshot_never_publishes_an_answer(service, tmp_path, monkeypatch, field, value, when):
    selected = publish(service, tmp_path, ["Evidence."])

    def change():
        with service.store._db:
            service.store._db.execute(f"UPDATE documents SET {field}=? WHERE id=?", (value, selected))

    if when == "preflight":
        original = chat_context.prepare_request

        def prepare(*args, **kwargs):
            result = original(*args, **kwargs)
            change()
            return result

        monkeypatch.setattr(chat_context, "prepare_request", prepare)
    else:
        original = service.sender

        def send(*args, **kwargs):
            result = original(*args, **kwargs)
            change()
            return result

        service.sender = send
    with pytest.raises(ValueError, match="changed"):
        answer(service, [selected])
    assert len(service.test_calls) == (1 if when == "generation" else 0)


@pytest.mark.parametrize("change", ["cancel", "generation", "delete"])
def test_stream_cancellation_and_source_changes_close_stream(service, tmp_path, change):
    selected = publish(service, tmp_path, ["Evidence."])
    cancelled = [False]
    closed = []
    consumed = []

    def stream(*args, **kwargs):
        try:
            consumed.append(1)
            yield "Partial answer"
            consumed.append(2)
            yield " that must not appear."
        finally:
            closed.append(True)

    service.streamer = stream
    replies = service.answer("Explain.", [], "science", document_ids=[selected], model="gpt-4o",
                             context_mode="full", is_cancelled=lambda: cancelled[0])
    assert next(replies)[0][-1]["content"] == "Partial answer"
    if change == "cancel":
        cancelled[0] = True
    else:
        with service.store._db:
            field, value = ("generation", "changed") if change == "generation" else ("status", "deleted")
            service.store._db.execute(f"UPDATE documents SET {field}=? WHERE id=?", (value, selected))
    with pytest.raises((ChatCancelled, ValueError)):
        next(replies)
    assert closed == [True] and consumed == [1]


def test_pre_cancelled_and_invalid_callbacks_do_no_work(service, monkeypatch):
    monkeypatch.setattr(service.store, "snapshot_sources", lambda *a, **k: pytest.fail("No snapshot read"))
    with pytest.raises(ChatCancelled):
        answer(service, ["unused"], is_cancelled=lambda: True)
    with pytest.raises(ValueError, match="callable"):
        answer(service, ["unused"], is_cancelled=True)
    assert service.test_calls == []


@pytest.mark.parametrize("after", ["connection", "embedding"])
def test_relevant_mode_checks_cancellation_between_paid_stages(service, tmp_path, monkeypatch, after):
    selected = publish(service, tmp_path, ["Evidence."])
    cancelled = [False]
    embeddings = []

    def connect():
        if after == "connection":
            cancelled[0] = True

    def embed(texts):
        embeddings.append(texts)
        cancelled[0] = True
        return [[1.0] * service.settings.embedding_dimensions]

    monkeypatch.setattr(service.store, "check_connection", connect)
    service.embedder = embed
    with pytest.raises(ChatCancelled):
        answer(service, [selected], context_mode="relevant", is_cancelled=lambda: cancelled[0])
    assert len(embeddings) == (1 if after == "embedding" else 0)
    assert service.test_calls == []


@pytest.mark.parametrize("stream_output", [False, True])
def test_full_context_web_search_preserves_provenance_and_complete_sources(service, tmp_path, stream_output):
    selected = publish(service, tmp_path, ["Local first.", "Local last."])
    final, sources = answer(service, [selected], stream_output=stream_output, web_search="medium")[-1]
    assert final[-1]["context_mode"] == "full" and final[-1]["web_search"] is True
    assert len(sources) == len(evidence_from_call(service)) == 2
    options = service.test_calls[0][2]
    assert options["web_search"] == "medium" and options["system_prompt"] == WEB_SYSTEM_PROMPT
    assert ("final_text_callback" in options) is stream_output


def test_full_context_strips_out_of_scope_assistant_history(service, tmp_path):
    selected = publish(service, tmp_path, ["Current source."])
    history = [{"role": "user", "content": "Old question"},
               {"role": "assistant", "content": "OUTSIDE ANSWER", "web_search": True, "sources": []}]
    list(service.answer("Explain.", history, "science", document_ids=[selected], model="gpt-4o",
                        context_mode="full", stream_output=False))
    assert "OUTSIDE ANSWER" not in str(service.test_calls[0][1])
    assert service.test_calls[0][1][0]["content"] == "Old question"
