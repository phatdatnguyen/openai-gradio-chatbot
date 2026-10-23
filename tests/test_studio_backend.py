"""Studio evidence and generation checks; all model requests are local fakes."""

from dataclasses import replace
from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from chatbot import config
from rag_chatbot.config import Settings
from rag_chatbot.store import LibraryStore
from rag_chatbot.tools import service
from rag_chatbot.tools.evidence import bounded_batches
from rag_chatbot.tools.learning import DEFAULT_LEARNING_COUNT, MAX_FLASHCARDS, MAX_QUIZ_QUESTIONS
from rag_chatbot.tools.presentation import DEFAULT_SLIDE_COUNT, MAX_SLIDES


@pytest.fixture
def catalog(tmp_path):
    store = LibraryStore(Settings(data_dir=tmp_path / "workspace"))
    yield store
    store.close()


def publish(catalog, tmp_path, name="paper.txt", texts=None, collection="science"):
    texts = texts or ["Measured value 10 nm."]
    path = tmp_path / name
    path.write_text("\n".join(texts), encoding="utf-8")
    document = catalog.stage_document(path, collection, catalog.settings.fingerprint())
    with catalog._db:
        for index, text in enumerate(texts):
            # Reverse-sort IDs deliberately: citation UUID order is not document order.
            catalog._db.execute(
                "INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
                (f"{document['id']}-{len(texts) - index}", document["id"], text, f"page {index + 1}", "text", index),
            )
        catalog._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=? WHERE id=?",
                            (len(texts), document["id"]))
    return document["id"]


def model_fake(calls):
    def generate(model, history, schema, **options):
        prompt = json.loads(history[-1]["content"])
        calls.append((options["schema_name"], prompt))
        evidence = prompt["evidence"]
        name = options["schema_name"]
        if name == "tool_evidence_notes":
            return {"notes": [{"source_id": source["source_id"], "text": source["text"]}
                              for source in evidence]}
        if name == "tool_summary":
            return {"title": "Research summary", "sections": [{"heading": "Findings", "paragraphs":
                [{"text": point["text"], "source_ids": point["source_ids"]} for point in evidence["notes"]]}]}
        if name == "tool_spreadsheet_row":
            source_ids = evidence["notes"][0]["source_ids"]
            return {"cells": [{"column": column, "value": "10 nm" if index == 0 else "Not found",
                               "source_ids": source_ids if index == 0 else []}
                              for index, column in enumerate(prompt["requested_columns"])]}
        if name == "tool_mind_map":
            return {"title": "Research Mind Map", "nodes": [
                {"id": "root", "parent_id": "", "label": "Research findings", "description": "", "source_ids": []},
                *[{"id": f"node{index}", "parent_id": "root", "label": f"Finding {index}",
                   "description": point["text"][:500], "source_ids": point["source_ids"]}
                  for index, point in enumerate(evidence["notes"], 1)]]}
        if name in {"tool_flashcards", "tool_quizzes"}:
            # Fake exactly the requested count while keeping all input citations,
            # even when fewer items than source notes/documents are requested.
            notes, count = evidence["notes"], prompt["requested_count"]
            items = []
            for index in range(count):
                group = notes[index::count] or [notes[index % len(notes)]]
                text = " / ".join(point["text"][:40] for point in group)
                source_ids = list(dict.fromkeys(source_id for point in group for source_id in point["source_ids"]))
                if name == "tool_flashcards":
                    items.append({"front": f"What is finding {index + 1}?", "back": text, "source_ids": source_ids})
                else:
                    items.append({"question": f"What does finding {index + 1} report?",
                                  "options": [text[:500], "Choice B", "Choice C", "Choice D"], "correct_option": 0,
                                  "explanation": text, "source_ids": source_ids})
            key = "cards" if name == "tool_flashcards" else "questions"
            return {"title": "Research flashcards" if name == "tool_flashcards" else "Research quiz", key: items}
        if name == "tool_presentation":
            notes, count = evidence["notes"], prompt["requested_slide_count"]
            slides = []
            for index in range(count):
                group = notes[index::count] or [notes[index % len(notes)]]
                text = " / ".join(point["text"][:40] for point in group)
                source_ids = list(dict.fromkeys(source_id for point in group for source_id in point["source_ids"]))
                slides.append({"title": f"Slide {index + 1}", "bullets": [text[:180]],
                               "notes": text[:1500], "source_ids": source_ids})
            return {"title": "Research presentation", "slides": slides}
        if name == "tool_consolidated_notes":
            return {"points": [{"text": "Consolidated findings", "source_ids": [identifier
                                for point in evidence for identifier in point["source_ids"]]}]}
        raise AssertionError(f"Unexpected request: {name}")
    return generate


def setup_generation(monkeypatch, catalog):
    calls = []
    monkeypatch.setattr(service, "get_service", lambda workspace: SimpleNamespace(store=catalog))
    monkeypatch.setattr(service.api, "generate_structured", model_fake(calls))
    return calls


def test_snapshot_preserves_selection_and_chunk_order_without_vectors(catalog, tmp_path):
    first = publish(catalog, tmp_path, texts=["First section", "Second section", "Third section"])
    second = publish(catalog, tmp_path, "second.txt", ["Other document"])
    snapshot = catalog.snapshot_sources([second, first], "science")
    assert [item["id"] for item in snapshot["documents"]] == [second, first]
    assert [item["text"] for item in snapshot["sources"]] == [
        "Other document", "First section", "Second section", "Third section"]
    assert catalog._vectors is None
    assert catalog.get_chunk(snapshot["sources"][0]["id"]).text == "Other document"


@pytest.mark.parametrize("invalid", [[], None, "all", ["missing"]])
def test_empty_or_stale_selection_never_becomes_all(catalog, tmp_path, invalid):
    publish(catalog, tmp_path)
    with pytest.raises(ValueError):
        catalog.snapshot_sources(invalid, "science")


def test_collection_fingerprint_and_published_generation_required(catalog, tmp_path):
    identifier = publish(catalog, tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        catalog.snapshot_sources([identifier], "other")
    with catalog._db:
        catalog._db.execute("UPDATE documents SET generation='' WHERE id=?", (identifier,))
    with pytest.raises(ValueError, match="unavailable"):
        catalog.snapshot_sources([identifier])
    with catalog._db:
        catalog._db.execute("UPDATE documents SET generation='published' WHERE id=?", (identifier,))
    catalog.settings = replace(catalog.settings, chunk_size=999)
    with pytest.raises(ValueError, match="unavailable"):
        catalog.snapshot_sources([identifier])


def test_generation_changes_invalidate_snapshot(catalog, tmp_path):
    identifier = publish(catalog, tmp_path)
    snapshot = catalog.snapshot_sources([identifier])
    with catalog._db:
        catalog._db.execute("UPDATE documents SET generation='replacement' WHERE id=?", (identifier,))
    with pytest.raises(ValueError, match="changed"):
        catalog.validate_source_snapshot(snapshot["documents"])


def test_legacy_chunk_order_migration(tmp_path):
    settings = Settings(data_dir=tmp_path / "workspace")
    original = LibraryStore(settings)
    identifier = publish(original, tmp_path, texts=["first", "second", "third"])
    original.close()
    with sqlite3.connect(settings.data_dir / "library.sqlite3") as db:
        db.execute("DROP INDEX chunks_order")
        db.execute("ALTER TABLE chunks DROP COLUMN ordinal")
    reopened = LibraryStore(settings)
    try:
        snapshot = reopened.snapshot_sources([identifier])
        assert [source["text"] for source in snapshot["sources"]] == ["first", "second", "third"]
        assert reopened.search("second", None, "science")[0].chunk.text == "second"
    finally:
        reopened.close()


def test_summary_reads_all_chunks_and_converts_stable_references(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path, texts=[f"Finding number {index}" for index in range(20)])
    calls = setup_generation(monkeypatch, catalog)
    events = list(service.generate_tool("work", "summary", [identifier], collection="science"))
    result = events[-1]["result"]
    assert result["coverage"] == {"documents": 1, "chunks": 20, "processed_chunks": 20}
    mapped = [item for name, prompt in calls if name == "tool_evidence_notes" for item in prompt["evidence"]]
    assert len(mapped) == 20  # Existing chat retrieval returns at most eight.
    assert result["model"] == config.DEFAULT_MODEL
    cited = {source_id for section in result["content"]["sections"]
             for paragraph in section["paragraphs"] for source_id in paragraph["source_ids"]}
    assert cited == {source["id"] for source in result["sources"]}
    assert all(event["result"] is None for event in events[:-1])


def test_spreadsheet_has_one_row_per_document_and_citations_per_cell(catalog, tmp_path, monkeypatch):
    first = publish(catalog, tmp_path)
    second = publish(catalog, tmp_path, "second.txt", ["Another measurement 10 nm"])
    setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", "spreadsheet", [second, first],
                                       columns=["Measurement", "Author"]))[-1]["result"]
    assert [row["document_id"] for row in result["content"]["rows"]] == [second, first]
    for row in result["content"]["rows"]:
        assert row["cells"][0]["source_ids"]
        assert row["cells"][1] == {"column": "Author", "value": "Not found", "source_ids": []}


@pytest.mark.parametrize("tool_id", ["summary", "mind_maps", "flashcards", "quizzes", "presentation"])
def test_no_calls_on_empty_selection_or_budget_error(catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path, texts=["x" * 500_001])
    calls = setup_generation(monkeypatch, catalog)
    with pytest.raises(ValueError, match="at least one"):
        list(service.generate_tool("work", tool_id, []))
    with pytest.raises(ValueError, match="too large"):
        list(service.generate_tool("work", tool_id, [identifier]))
    assert calls == []


@pytest.mark.parametrize("tool_id", ["summary", "flashcards", "quizzes", "presentation"])
def test_cancel_during_request_prevents_further_calls_and_result(catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    stopped = False
    fake = service.api.generate_structured
    def cancelling(*args, **kwargs):
        nonlocal stopped
        result = fake(*args, **kwargs)
        stopped = True
        return result
    monkeypatch.setattr(service.api, "generate_structured", cancelling)
    with pytest.raises(service.ToolCancelled):
        list(service.generate_tool("work", tool_id, [identifier], is_cancelled=lambda: stopped))
    assert len(calls) == 1


@pytest.mark.parametrize("tool_id", ["summary", "flashcards", "quizzes", "presentation"])
def test_reindex_during_final_request_does_not_publish_result(catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    def changing(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == f"tool_{tool_id}":
            with catalog._db:
                catalog._db.execute("UPDATE documents SET status='staging' WHERE id=?", (identifier,))
        return result
    monkeypatch.setattr(service.api, "generate_structured", changing)
    with pytest.raises(ValueError, match="changed"):
        list(service.generate_tool("work", tool_id, [identifier]))
    assert len(calls) == 2


def test_invalid_model_citations_fail_closed(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path)
    setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    def invalid(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_summary":
            result["sections"][0]["paragraphs"][0]["source_ids"] = ["invented"]
        return result
    monkeypatch.setattr(service.api, "generate_structured", invalid)
    with pytest.raises(ValueError, match="citations"):
        list(service.generate_tool("work", "summary", [identifier]))


@pytest.mark.parametrize("remaining", [0, 1])
def test_missing_map_chunks_retain_original_evidence_for_synthesis(catalog, tmp_path, monkeypatch, remaining):
    identifier = publish(catalog, tmp_path, texts=["first", "second"])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured

    def omit_notes(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_evidence_notes":
            result["notes"] = result["notes"][:remaining]
        return result

    monkeypatch.setattr(service.api, "generate_structured", omit_notes)
    result = list(service.generate_tool("work", "summary", [identifier]))[-1]["result"]
    assert calls[-1][1]["evidence"]["notes"] == [
        {"text": "first", "source_ids": ["s1"]}, {"text": "second", "source_ids": ["s2"]}]
    assert result["coverage"] == {"documents": 1, "chunks": 2, "processed_chunks": 2}
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_summary"]


def test_missing_note_recovery_preserves_full_text_input_order_and_exact_citations():
    original_text = "Complete evidence with significant spacing.\n" * 100
    batch = [{"source_id": "s1", "text": "First passage."},
             {"source_id": "s2", "text": original_text},
             {"source_id": "s3", "text": "Third passage."}]
    value = {"notes": [{"source_id": "s3", "text": "Third note."},
                       {"source_id": "s1", "text": "First note."}]}
    original_batch, original_value = deepcopy(batch), deepcopy(value)
    notes = service._notes(value, batch)
    assert notes == [{"text": "First note.", "source_ids": ["s1"]},
                     {"text": original_text, "source_ids": ["s2"]},
                     {"text": "Third note.", "source_ids": ["s3"]}]
    notes[1]["source_ids"].append("edited-after-validation")
    assert batch == original_batch and value == original_value


@pytest.mark.parametrize("rows", [None, {}, [None], [{}],
    [{"source_id": "s1", "text": "",}],
    [{"source_id": "s1", "text": "x" * 2001}],
    [{"source_id": "s1", "text": "Note.", "extra": True}]])
def test_missing_note_recovery_rejects_malformed_output(rows):
    with pytest.raises(ValueError):
        service._notes({"notes": rows}, [{"source_id": "s1", "text": "Original."}])


@pytest.mark.parametrize("identifiers", [["unknown"], [None], [[]], [""], ["s1", "s1"]])
def test_missing_note_recovery_rejects_invalid_identifiers(identifiers):
    rows = [{"source_id": identifier, "text": "A valid note."} for identifier in identifiers]
    with pytest.raises(ValueError, match="invalid evidence section identifier"):
        service._notes({"notes": rows}, [{"source_id": "s1", "text": "First."},
                                        {"source_id": "s2", "text": "Second."}])


@pytest.mark.parametrize("can_shrink", [False, True])
def test_recovered_original_passages_still_follow_consolidation_budget(catalog, tmp_path, monkeypatch, can_shrink):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured

    def omit_notes(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_evidence_notes":
            return {"notes": []}
        if kwargs["schema_name"] == "tool_consolidated_notes" and not can_shrink:
            return {"points": deepcopy(calls[-1][1]["evidence"])}
        return result

    monkeypatch.setattr(service.api, "generate_structured", omit_notes)
    if can_shrink:
        result = list(service.generate_tool("work", "summary", [identifier]))[-1]["result"]
        assert result["coverage"]["processed_chunks"] == 50
        assert calls[-1][0] == "tool_summary"
    else:
        with pytest.raises(ValueError, match="without dropping source notes"):
            list(service.generate_tool("work", "summary", [identifier]))
        assert calls[-1][0] == "tool_consolidated_notes"
    assert any(name == "tool_consolidated_notes" for name, _ in calls)


def test_bounded_batches_keep_all_inputs_and_reject_one_oversized(monkeypatch):
    monkeypatch.setattr("rag_chatbot.tools.evidence.token_size", lambda value, model: len(value))
    items = ["aa", "bbbb", "cc", "dd"]
    batches = bounded_batches(items, "fake", budget=10)
    assert [item for batch in batches for item in batch] == items
    assert len(batches) > 1
    with pytest.raises(ValueError, match="too large"):
        bounded_batches(["x" * 20], "fake", budget=10)


def test_large_summary_consolidates_notes_without_losing_input_coverage(catalog, tmp_path, monkeypatch):
    texts = [("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)]
    identifier = publish(catalog, tmp_path, texts=texts)
    calls = setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", "summary", [identifier]))[-1]["result"]
    assert any(name == "tool_consolidated_notes" for name, _ in calls)
    assert result["coverage"]["processed_chunks"] == 50
    cited = {source_id for section in result["content"]["sections"]
             for paragraph in section["paragraphs"] for source_id in paragraph["source_ids"]}
    assert cited == {source["id"] for source in result["sources"]}


def test_consolidated_points_keep_every_original_note_for_missing_citations():
    batch = [
        {"text": "First finding and its limitation.", "source_ids": ["s1", "s2"]},
        {"text": "A different finding from overlapping evidence.", "source_ids": ["s2", "s3"]},
        {"text": "Already represented evidence.", "source_ids": ["s4"]},
    ]
    value = {"points": [{"text": "Shortened finding.", "source_ids": ["s1", "s3", "s4"]}]}
    original_batch, original_value = deepcopy(batch), deepcopy(value)
    points = service._points(value, batch)
    assert points == [*value["points"], *batch[:2]]
    assert points[0]["source_ids"] == ["s1", "s3", "s4"]  # No fabricated attachment of s2.
    points[1]["source_ids"].append("edited-after-validation")
    assert batch == original_batch and value == original_value


@pytest.mark.parametrize("citations", [[], ["unknown"], ["s1", "s1"], "s1", [None]])
def test_consolidation_recovery_still_rejects_invalid_citations(citations):
    batch = [{"text": "First fact.", "source_ids": ["s1"]},
             {"text": "Second fact.", "source_ids": ["s2"]}]
    with pytest.raises(ValueError, match="citations"):
        service._points({"points": [{"text": "Summary.", "source_ids": citations}]}, batch)


@pytest.mark.parametrize("points", [[], None, [{}], [{"text": "", "source_ids": ["s1"]}],
                                     [{"text": "x" * 2001, "source_ids": ["s1"]}]])
def test_consolidation_recovery_still_rejects_malformed_points(points):
    with pytest.raises(ValueError):
        service._points({"points": points}, [{"text": "Source note.", "source_ids": ["s1"]}])


@pytest.mark.parametrize("tool_id", ["summary", "spreadsheet", "mind_maps", "flashcards", "quizzes", "presentation"])
def test_omitted_consolidation_citations_carry_original_notes_into_final_request(
        catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path, "slides.txt",
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    retained, responses = [], []

    def omit_one_source(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_consolidated_notes":
            batch = calls[-1][1]["evidence"]
            assert len(batch) > 1
            value["points"][0]["source_ids"] = value["points"][0]["source_ids"][:-1]
            retained.append(deepcopy(batch[-1]))
            responses.append((value, deepcopy(value)))
        elif kwargs["schema_name"] == "tool_quizzes":
            for question in value["questions"]:
                question["options"][0] = question["options"][0][:500]
        return value

    monkeypatch.setattr(service.api, "generate_structured", omit_one_source)
    result = list(service.generate_tool("work", tool_id, [identifier]))[-1]["result"]
    final_notes = calls[-1][1]["evidence"]["notes"]
    assert retained and all(note in final_notes for note in retained)
    assert {source_id for note in final_notes for source_id in note["source_ids"]} == {
        f"s{index}" for index in range(1, 51)}
    assert all(value == original for value, original in responses)
    assert result["coverage"] == {"documents": 1, "chunks": 50, "processed_chunks": 50}
    # Recovery stays local: one evidence pass, one consolidation pass, one synthesis.
    original_notes = [{"text": item["text"], "source_ids": [item["source_id"]]}
                      for name, prompt in calls if name == "tool_evidence_notes" for item in prompt["evidence"]]
    assert sum(name == "tool_consolidated_notes" for name, _ in calls) == len(
        bounded_batches(original_notes, config.DEFAULT_MODEL))
    assert result["settings"]["model_requests"] == len(calls)
    assert len(result["sources"]) == 50
    if tool_id == "flashcards":
        assert {source_id for card in result["content"]["cards"] for source_id in card["source_ids"]} == {
            source["id"] for source in result["sources"]}


@pytest.mark.parametrize("response", ["unchanged", "mostly_omitted"])
def test_nonshrinking_consolidation_stops_without_retry_or_final_output(catalog, tmp_path, monkeypatch, response):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured

    def unchanged(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_consolidated_notes":
            batch = calls[-1][1]["evidence"]
            if response == "mostly_omitted":
                return {"points": [{"text": "x y z " * 300, "source_ids": batch[0]["source_ids"]}]}
            return {"points": deepcopy(batch)}
        return value

    monkeypatch.setattr(service.api, "generate_structured", unchanged)
    with pytest.raises(ValueError, match="without dropping source notes"):
        list(service.generate_tool("work", "flashcards", [identifier]))
    assert calls[-1][0] == "tool_consolidated_notes"
    original_notes = [{"text": item["text"], "source_ids": [item["source_id"]]}
                      for name, prompt in calls if name == "tool_evidence_notes" for item in prompt["evidence"]]
    assert sum(name == "tool_consolidated_notes" for name, _ in calls) == len(
        bounded_batches(original_notes, config.DEFAULT_MODEL))


def test_nonshrinking_batch_is_retained_when_other_batches_make_room(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    retained = []

    def unchanged_tail(*args, **kwargs):
        value = fake(*args, **kwargs)
        if (kwargs["schema_name"] == "tool_consolidated_notes"
                and sum(name == "tool_consolidated_notes" for name, _ in calls) > 1):
            points = deepcopy(calls[-1][1]["evidence"])
            retained.extend(deepcopy(points))
            return {"points": points}
        return value

    monkeypatch.setattr(service.api, "generate_structured", unchanged_tail)
    result = list(service.generate_tool("work", "flashcards", [identifier]))[-1]["result"]
    assert retained and all(note in calls[-1][1]["evidence"]["notes"] for note in retained)
    assert {source_id for card in result["content"]["cards"] for source_id in card["source_ids"]} == {
        source["id"] for source in result["sources"]}


@pytest.mark.parametrize("budget", [200, 380])
def test_consolidation_has_five_round_limit_and_accepts_fit_in_last_round(catalog, tmp_path, monkeypatch, budget):
    identifier = publish(catalog, tmp_path, texts=["x" * 99 + str(index) for index in range(4)])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    # A deterministic small budget isolates the round boundary from tokenizer changes.
    monkeypatch.setattr(service, "BATCH_TOKEN_BUDGET", budget)
    monkeypatch.setattr(service, "token_size", lambda notes, model: sum(len(point["text"]) for point in notes))
    monkeypatch.setattr(service, "bounded_batches", lambda notes, model: [notes[:2], notes[2:]])

    def small_reduction(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_consolidated_notes":
            return {"points": [{**point, "text": point["text"][:-1]}
                               for point in calls[-1][1]["evidence"]]}
        return value

    monkeypatch.setattr(service.api, "generate_structured", small_reduction)
    if budget == 200:
        with pytest.raises(ValueError, match="too large to consolidate safely"):
            list(service.generate_tool("work", "flashcards", [identifier]))
        assert calls[-1][0] == "tool_consolidated_notes"
    else:
        result = list(service.generate_tool("work", "flashcards", [identifier]))[-1]["result"]
        assert result["coverage"]["processed_chunks"] == 4
        assert calls[-1][0] == "tool_flashcards"
    assert sum(name == "tool_consolidated_notes" for name, _ in calls) == 10


def test_consolidation_recovery_respects_total_request_limit(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    _, _, batches = service.prepare_snapshot(catalog, [identifier], None, config.DEFAULT_MODEL)
    request_limit = sum(len(items) for items in batches.values()) + 1
    monkeypatch.setattr(service, "MAX_MODEL_CALLS", request_limit)

    def omitted(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_consolidated_notes":
            value["points"][0]["source_ids"] = value["points"][0]["source_ids"][:-1]
        return value

    monkeypatch.setattr(service.api, "generate_structured", omitted)
    with pytest.raises(ValueError, match="request limit"):
        list(service.generate_tool("work", "flashcards", [identifier]))
    assert len(calls) == request_limit
    assert calls[-1][0] == "tool_consolidated_notes"


@pytest.mark.parametrize("change", ["cancel", "reindex"])
def test_consolidation_recovery_honors_cancellation_and_snapshot_guard(catalog, tmp_path, monkeypatch, change):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    stopped = False

    def interrupt(*args, **kwargs):
        nonlocal stopped
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_consolidated_notes":
            value["points"][0]["source_ids"] = value["points"][0]["source_ids"][:-1]
            if change == "cancel":
                stopped = True
            else:
                with catalog._db:
                    catalog._db.execute("UPDATE documents SET generation='new-generation' WHERE id=?", (identifier,))
        return value

    monkeypatch.setattr(service.api, "generate_structured", interrupt)
    exception = service.ToolCancelled if change == "cancel" else ValueError
    with pytest.raises(exception, match="stopped" if change == "cancel" else "changed"):
        list(service.generate_tool("work", "flashcards", [identifier], is_cancelled=lambda: stopped))
    assert sum(name == "tool_consolidated_notes" for name, _ in calls) == 1
    assert calls[-1][0] == "tool_consolidated_notes"


def test_spreadsheet_rejects_citation_from_different_document(catalog, tmp_path, monkeypatch):
    first = publish(catalog, tmp_path)
    second = publish(catalog, tmp_path, "second.txt", ["Different source evidence"])
    setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    def cross_document(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_spreadsheet_row":
            result["cells"][0]["source_ids"] = ["s2"]
        return result
    monkeypatch.setattr(service.api, "generate_structured", cross_document)
    with pytest.raises(ValueError, match="citations"):
        list(service.generate_tool("work", "spreadsheet", [first, second]))


def test_summary_cannot_silently_omit_selected_document(catalog, tmp_path, monkeypatch):
    first = publish(catalog, tmp_path)
    second = publish(catalog, tmp_path, "second.txt", ["Different source evidence"])
    setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    def omission(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_summary":
            result["sections"][0]["paragraphs"] = result["sections"][0]["paragraphs"][:1]
        return result
    monkeypatch.setattr(service.api, "generate_structured", omission)
    with pytest.raises(ValueError, match="omitted a selected document"):
        list(service.generate_tool("work", "summary", [first, second]))


def test_mind_map_reads_every_selected_chunk_and_preserves_citations(catalog, tmp_path, monkeypatch):
    first = publish(catalog, tmp_path, texts=[f"Selected finding {index}" for index in range(20)])
    second = publish(catalog, tmp_path, "second.txt", ["Second selected document finding"])
    calls = setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", "mind_maps", [first, second],
                                       instructions="Connect the findings", collection="science"))[-1]["result"]
    assert result["tool_id"] == "mind_maps" and result["title"] == "Research Mind Map"
    assert result["coverage"] == {"documents": 2, "chunks": 21, "processed_chunks": 21}
    assert result["content"]["nodes"][0]["parent_id"] == ""
    assert result["content"]["nodes"][0]["source_ids"] == []
    source_ids = {source["id"] for source in result["sources"]}
    cited = {identifier for node in result["content"]["nodes"] for identifier in node["source_ids"]}
    assert cited == source_ids
    mapped = [item for name, prompt in calls if name == "tool_evidence_notes" for item in prompt["evidence"]]
    assert len(mapped) == 21
    assert calls[-1][0] == "tool_mind_map"
    assert calls[-1][1]["user_instructions"] == "Connect the findings"
    assert result["settings"]["collection"] == "science"


@pytest.mark.parametrize("identifiers", [
    ["1", "1.1", "1.2"],
    ["Main topic", "Methods / XRD", "Results: crystals"],
    ["根节点", "方法", "結果"],
    ["topic_" * 20, "method_" * 20, "finding_" * 20],
    ["n2", "n1", "n3"],
    ["s1", "s2", "s3"],
    ['root\" onclick=\"alert(1)', "child<script>bad</script>", "result&finding"],
])
def test_mind_map_normalizes_generated_ids_without_retry_or_changing_content(catalog, tmp_path, monkeypatch, identifiers):
    from rag_chatbot.tools.mindmap import render_mind_map_svg

    document = publish(catalog, tmp_path, texts=["First finding", "Second finding"])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    original = []

    def unusual_ids(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_mind_map":
            for index, node in enumerate(value["nodes"]):
                node["id"] = identifiers[index]
                node["parent_id"] = identifiers[index - 1] if index else ""
            original.extend([value, deepcopy(value)])
        return value

    monkeypatch.setattr(service.api, "generate_structured", unusual_ids)
    result = list(service.generate_tool("work", "mind_maps", [document]))[-1]["result"]
    nodes = result["content"]["nodes"]
    assert [node["id"] for node in nodes] == ["n1", "n2", "n3"]
    assert [node["parent_id"] for node in nodes] == ["", "n1", "n2"]
    assert original[0] == original[1]  # The raw model response remains unchanged.
    assert [(node["label"], node["description"]) for node in nodes] == [
        (node["label"], node["description"]) for node in original[0]["nodes"]]
    assert {source_id for node in nodes for source_id in node["source_ids"]} == {
        source["id"] for source in result["sources"]}
    assert 'id="node-n1"' in render_mind_map_svg(result)
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_mind_map"]


@pytest.mark.parametrize("invalid", ["duplicate_id", "missing_parent", "two_roots", "cycle",
                                      "too_deep", "empty_id", "long_label", "long_description", "too_many"])
def test_mind_map_rejects_invalid_generated_tree(catalog, tmp_path, monkeypatch, invalid):
    identifier = publish(catalog, tmp_path, texts=["Finding one", "Finding two"])
    setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    def bad_tree(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] != "tool_mind_map":
            return result
        nodes = result["nodes"]
        if invalid == "duplicate_id":
            nodes[2]["id"] = nodes[1]["id"]
        elif invalid == "missing_parent":
            nodes[1]["parent_id"] = "missing"
        elif invalid == "two_roots":
            nodes[1]["parent_id"] = ""
        elif invalid == "cycle":
            nodes[1]["parent_id"], nodes[2]["parent_id"] = nodes[2]["id"], nodes[1]["id"]
        elif invalid == "too_deep":
            nodes[:] = [nodes[0]] + [{"id": f"level{index}", "parent_id": "root" if index == 1 else f"level{index - 1}",
                "label": "Nested concept", "description": "Finding", "source_ids": ["s1"]} for index in range(1, 6)]
        elif invalid == "empty_id":
            nodes[1]["id"] = ""
        elif invalid == "long_label":
            nodes[1]["label"] = "x" * 81
        elif invalid == "long_description":
            nodes[1]["description"] = "x" * 501
        elif invalid == "too_many":
            nodes[:] = [nodes[0]] + [{"id": f"concept{index}", "parent_id": "root", "label": "Concept",
                "description": "Finding", "source_ids": ["s1"]} for index in range(60)]
        return result
    monkeypatch.setattr(service.api, "generate_structured", bad_tree)
    with pytest.raises(ValueError):
        list(service.generate_tool("work", "mind_maps", [identifier]))


@pytest.mark.parametrize("source_ids", [["invented"], [], ["s1", "s1"]])
def test_mind_map_rejects_invalid_node_citations(catalog, tmp_path, monkeypatch, source_ids):
    identifier = publish(catalog, tmp_path)
    setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    def invalid_citations(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_mind_map":
            result["nodes"][1]["source_ids"] = deepcopy(source_ids)
        return result
    monkeypatch.setattr(service.api, "generate_structured", invalid_citations)
    with pytest.raises(ValueError, match="citation"):
        list(service.generate_tool("work", "mind_maps", [identifier]))


def test_mind_map_cannot_omit_selected_document(catalog, tmp_path, monkeypatch):
    first = publish(catalog, tmp_path)
    second = publish(catalog, tmp_path, "second.txt", ["Second source"])
    setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    def omit_document(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_mind_map":
            result["nodes"] = result["nodes"][:2]
        return result
    monkeypatch.setattr(service.api, "generate_structured", omit_document)
    with pytest.raises(ValueError, match="omitted a selected document"):
        list(service.generate_tool("work", "mind_maps", [first, second]))


def test_mind_map_cancellation_during_synthesis_discards_result(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    cancelled = False
    def cancel_final(*args, **kwargs):
        nonlocal cancelled
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_mind_map":
            cancelled = True
        return result
    monkeypatch.setattr(service.api, "generate_structured", cancel_final)
    with pytest.raises(service.ToolCancelled):
        list(service.generate_tool("work", "mind_maps", [identifier], is_cancelled=lambda: cancelled))
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_mind_map"]


def test_mind_map_consolidates_large_selection_without_losing_coverage(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", "mind_maps", [identifier]))[-1]["result"]
    assert any(name == "tool_consolidated_notes" for name, _ in calls)
    assert result["coverage"]["processed_chunks"] == 50
    cited = {source_id for node in result["content"]["nodes"] for source_id in node["source_ids"]}
    assert cited == {source["id"] for source in result["sources"]}


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
@pytest.mark.parametrize("requested", [None, 1, 7.0, "maximum"])
def test_learning_generation_uses_exact_requested_count_and_persists_it(
        catalog, tmp_path, monkeypatch, tool_id, key, requested):
    first = publish(catalog, tmp_path)
    second = publish(catalog, tmp_path, "second.txt", ["Another document measurement"])
    calls = setup_generation(monkeypatch, catalog)
    if requested == "maximum":
        requested = MAX_FLASHCARDS if tool_id == "flashcards" else MAX_QUIZ_QUESTIONS
    expected = DEFAULT_LEARNING_COUNT if requested is None else int(requested)
    arguments = {} if requested is None else {"item_count": requested}
    result = list(service.generate_tool("work", tool_id, [first, second],
                                       instructions="Focus on measurements", **arguments))[-1]["result"]
    assert len(result["content"][key]) == expected
    assert result["settings"]["item_count"] == expected
    assert type(result["settings"]["item_count"]) is int
    assert all(prompt["requested_count"] == expected for _, prompt in calls)
    assert f"Return exactly {expected}" in calls[-1][1]["task"]
    assert "multiple documents" in calls[-1][1]["task"]
    assert "increasing the preferred" not in calls[-1][1]["task"]
    assert {source_id for item in result["content"][key] for source_id in item["source_ids"]} == {
        source["id"] for source in result["sources"]}
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", f"tool_{tool_id}"]


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
@pytest.mark.parametrize("requested", [None, True, "20", 0, 61, 1.5, float("nan"), float("inf")])
def test_invalid_learning_count_fails_before_workspace_access_or_paid_calls(monkeypatch, tool_id, requested):
    monkeypatch.setattr(service, "get_service", lambda *args: pytest.fail("Invalid count accessed the workspace"))
    monkeypatch.setattr(service.api, "generate_structured", lambda *args, **kwargs: pytest.fail("Invalid count made an API call"))
    with pytest.raises(ValueError, match="count must be a whole number"):
        list(service.generate_tool("work", tool_id, ["selected-document"], item_count=requested))


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
@pytest.mark.parametrize("difference", [-1, 1])
def test_learning_count_mismatch_fails_without_retry_or_truncation(
        catalog, tmp_path, monkeypatch, tool_id, key, difference):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured

    def wrong_count(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == f"tool_{tool_id}":
            if difference < 0:
                result[key].pop()
            else:
                extra = deepcopy(result[key][0])
                extra["front" if tool_id == "flashcards" else "question"] = "Unexpected extra question?"
                result[key].append(extra)
        return result

    monkeypatch.setattr(service.api, "generate_structured", wrong_count)
    with pytest.raises(ValueError, match="requested exactly 3"):
        list(service.generate_tool("work", tool_id, [identifier], item_count=3))
    assert [name for name, _ in calls] == ["tool_evidence_notes", f"tool_{tool_id}"]


@pytest.mark.parametrize("tool_id", ["summary", "spreadsheet", "mind_maps", "presentation"])
def test_other_tools_ignore_learning_count(catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", tool_id, [identifier], item_count="unused"))[-1]["result"]
    assert "item_count" not in result["settings"]
    assert all("requested_count" not in prompt for _, prompt in calls)


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
def test_learning_reads_all_chunks_and_maps_citations_without_mutating_response(
        catalog, tmp_path, monkeypatch, tool_id, key):
    first = publish(catalog, tmp_path, texts=[f"Finding number {index}" for index in range(20)])
    second = publish(catalog, tmp_path, "second.txt", ["Another document measurement"])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    original = []

    def remember_response(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == f"tool_{tool_id}":
            original.extend([value, deepcopy(value)])
        return value

    monkeypatch.setattr(service.api, "generate_structured", remember_response)
    events = list(service.generate_tool("work", tool_id, [first, second],
                                      instructions="Focus on the measurements", collection="science"))
    result = events[-1]["result"]
    assert result["tool_id"] == tool_id
    assert result["coverage"] == {"documents": 2, "chunks": 21, "processed_chunks": 21}
    assert result["model"] == config.DEFAULT_MODEL
    assert all(event["result"] is None for event in events[:-1])
    assert len([item for name, prompt in calls if name == "tool_evidence_notes"
                for item in prompt["evidence"]]) == 21
    assert {source_id for item in result["content"][key] for source_id in item["source_ids"]} == {
        source["id"] for source in result["sources"]}
    assert calls[-1][0] == f"tool_{tool_id}"
    assert calls[-1][1]["user_instructions"] == "Focus on the measurements"
    assert result["settings"]["collection"] == "science"
    assert original[0] == original[1]
    if tool_id == "quizzes":
        # Session practice state must not alias the model's response arrays.
        result["content"][key][0]["options"][0] = "Edited after generation"
        assert original[0] == original[1]


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
@pytest.mark.parametrize("failure", ["omitted_document", "unknown_citation", "empty_citations",
                                     "wrong_structure", "duplicate_prompt", "too_many"])
def test_invalid_learning_response_fails_without_retry(catalog, tmp_path, monkeypatch, tool_id, key, failure):
    first = publish(catalog, tmp_path)
    second = publish(catalog, tmp_path, "second.txt", ["Another document measurement"])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured

    def invalid(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] != f"tool_{tool_id}":
            return value
        if failure == "omitted_document":
            # Keep the exact requested count so the source-coverage check,
            # rather than the count check, rejects this response.
            for item in value[key]:
                item["source_ids"] = ["s1"]
        elif failure == "unknown_citation":
            value[key][0]["source_ids"] = ["not-a-source"]
        elif failure == "empty_citations":
            value[key][0]["source_ids"] = []
        elif failure == "wrong_structure":
            del value[key][0]["back" if tool_id == "flashcards" else "correct_option"]
        elif failure == "duplicate_prompt":
            field = "front" if tool_id == "flashcards" else "question"
            value[key][1][field] = value[key][0][field]
        elif failure == "too_many":
            value[key] *= 31
        return value

    monkeypatch.setattr(service.api, "generate_structured", invalid)
    with pytest.raises(ValueError, match="omitted a selected document" if failure == "omitted_document" else None):
        list(service.generate_tool("work", tool_id, [first, second]))
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", f"tool_{tool_id}"]


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_learning_cancellation_during_final_request_discards_result(catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    stopped = False

    def cancel_final(*args, **kwargs):
        nonlocal stopped
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == f"tool_{tool_id}":
            stopped = True
        return value

    monkeypatch.setattr(service.api, "generate_structured", cancel_final)
    with pytest.raises(service.ToolCancelled):
        list(service.generate_tool("work", tool_id, [identifier], is_cancelled=lambda: stopped))
    assert len(calls) == 2


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
def test_learning_consolidates_without_losing_source_coverage(catalog, tmp_path, monkeypatch, tool_id, key):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", tool_id, [identifier]))[-1]["result"]
    assert any(name == "tool_consolidated_notes" for name, _ in calls)
    assert result["coverage"]["processed_chunks"] == 50
    assert {source_id for item in result["content"][key] for source_id in item["source_ids"]} == {
        source["id"] for source in result["sources"]}


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_learning_preflights_request_budget(catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    monkeypatch.setattr(service, "MAX_MODEL_CALLS", 1)
    with pytest.raises(ValueError, match="too many model requests"):
        list(service.generate_tool("work", tool_id, [identifier]))
    assert calls == []


@pytest.mark.parametrize("requested", [None, 1, 7.0, MAX_SLIDES])
def test_presentation_exact_slide_count_full_source_coverage_and_unmodified_response(
        catalog, tmp_path, monkeypatch, requested):
    first = publish(catalog, tmp_path, texts=[f"Finding number {index}" for index in range(20)])
    second = publish(catalog, tmp_path, "second.txt", ["Another document measurement"])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    original = []

    def remember_response(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_presentation":
            original.extend([value, deepcopy(value)])
        return value

    monkeypatch.setattr(service.api, "generate_structured", remember_response)
    expected = DEFAULT_SLIDE_COUNT if requested is None else int(requested)
    arguments = {} if requested is None else {"slide_count": requested}
    events = list(service.generate_tool("work", "presentation", [first, second],
                                       instructions="Explain the findings", collection="science", **arguments))
    result = events[-1]["result"]
    assert all(event["result"] is None for event in events[:-1])
    assert result["tool_id"] == "presentation" and result["model"] == config.DEFAULT_MODEL
    assert result["coverage"] == {"documents": 2, "chunks": 21, "processed_chunks": 21}
    assert len(result["content"]["slides"]) == expected
    assert result["settings"]["slide_count"] == expected
    assert type(result["settings"]["slide_count"]) is int
    assert result["settings"]["collection"] == "science"
    assert all(prompt["requested_slide_count"] == expected for _, prompt in calls)
    assert "requested_count" not in calls[-1][1] and "item_count" not in result["settings"]
    assert calls[-1][1]["user_instructions"] == "Explain the findings"
    assert f"Return exactly {expected} slides" in calls[-1][1]["task"]
    assert "Do not add an extra cover" in calls[-1][1]["task"]
    assert {source_id for slide in result["content"]["slides"] for source_id in slide["source_ids"]} == {
        source["id"] for source in result["sources"]}
    assert original[0] == original[1]
    result["content"]["slides"][0]["bullets"][0] = "Edited after generation"
    result["content"]["slides"][0]["source_ids"].append("another-source")
    assert original[0] == original[1]
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", "tool_presentation"]


@pytest.mark.parametrize("count", [None, True, "10", 0, MAX_SLIDES + 1, 1.5, float("nan"), float("inf")])
def test_invalid_presentation_count_fails_before_workspace_access_or_paid_calls(monkeypatch, count):
    monkeypatch.setattr(service, "get_service", lambda *args: pytest.fail("Invalid count accessed the workspace"))
    monkeypatch.setattr(service.api, "generate_structured", lambda *args, **kwargs: pytest.fail("Invalid count made an API call"))
    with pytest.raises(ValueError, match="slide count must be a whole number"):
        list(service.generate_tool("work", "presentation", ["selected-document"], slide_count=count))


@pytest.mark.parametrize("failure", ["omitted_document", "unknown_citation", "empty_citations",
                                     "wrong_structure", "duplicate_title", "too_many_bullets",
                                     "too_long_bullets", "too_long_notes", "too_few_slides", "too_many_slides"])
def test_invalid_presentation_output_fails_without_retry(catalog, tmp_path, monkeypatch, failure):
    first = publish(catalog, tmp_path)
    second = publish(catalog, tmp_path, "second.txt", ["Another document measurement"])
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured

    def invalid(*args, **kwargs):
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] != "tool_presentation":
            return value
        slides = value["slides"]
        if failure == "omitted_document":
            for slide in slides:
                slide["source_ids"] = ["s1"]
        elif failure == "unknown_citation":
            slides[0]["source_ids"] = ["not-a-source"]
        elif failure == "empty_citations":
            slides[0]["source_ids"] = []
        elif failure == "wrong_structure":
            del slides[0]["notes"]
        elif failure == "duplicate_title":
            slides[1]["title"] = slides[0]["title"]
        elif failure == "too_many_bullets":
            slides[0]["bullets"] = ["A short bullet"] * 6
        elif failure == "too_long_bullets":
            slides[0]["bullets"] = ["x" * 181]
        elif failure == "too_long_notes":
            slides[0]["notes"] = "x" * 1501
        elif failure == "too_few_slides":
            slides.pop()
        elif failure == "too_many_slides":
            slides.append({**deepcopy(slides[0]), "title": "Unexpected extra slide"})
        return value

    monkeypatch.setattr(service.api, "generate_structured", invalid)
    match = "omitted a selected document" if failure == "omitted_document" else (
        "requested exactly 10" if failure in {"too_few_slides", "too_many_slides"} else None)
    with pytest.raises(ValueError, match=match):
        list(service.generate_tool("work", "presentation", [first, second]))
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", "tool_presentation"]


def test_presentation_final_request_cancellation_discards_output(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    fake = service.api.generate_structured
    stopped = False

    def cancel_final(*args, **kwargs):
        nonlocal stopped
        value = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_presentation":
            stopped = True
        return value

    monkeypatch.setattr(service.api, "generate_structured", cancel_final)
    with pytest.raises(service.ToolCancelled):
        list(service.generate_tool("work", "presentation", [identifier], is_cancelled=lambda: stopped))
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_presentation"]


def test_presentation_consolidates_large_selection_with_all_citations(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path,
        texts=[("Evidence measurements and limitations. " * 40) + str(index) for index in range(50)])
    calls = setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", "presentation", [identifier]))[-1]["result"]
    assert any(name == "tool_consolidated_notes" for name, _ in calls)
    assert result["coverage"]["processed_chunks"] == 50
    assert {source_id for slide in result["content"]["slides"] for source_id in slide["source_ids"]} == {
        source["id"] for source in result["sources"]}


def test_presentation_preflights_request_budget(catalog, tmp_path, monkeypatch):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    monkeypatch.setattr(service, "MAX_MODEL_CALLS", 1)
    with pytest.raises(ValueError, match="too many model requests"):
        list(service.generate_tool("work", "presentation", [identifier]))
    assert calls == []


@pytest.mark.parametrize("tool_id", ["summary", "spreadsheet", "mind_maps", "flashcards", "quizzes"])
def test_other_tools_ignore_presentation_slide_count(catalog, tmp_path, monkeypatch, tool_id):
    identifier = publish(catalog, tmp_path)
    calls = setup_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", tool_id, [identifier], slide_count="unused"))[-1]["result"]
    assert "slide_count" not in result["settings"]
    assert all("requested_slide_count" not in prompt for _, prompt in calls)
