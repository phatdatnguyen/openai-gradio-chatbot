"""Bounded whole-source evidence preparation for Studio tools."""

import json

from chatbot import tokens


BATCH_TOKEN_BUDGET = 6000
MAX_DOCUMENTS = 40
MAX_CHUNKS = 512
MAX_SOURCE_CHARACTERS = 500_000


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def token_size(value, model):
    return tokens.count_text_tokens(json_text(value), model)


def bounded_batches(items, model, *, budget=BATCH_TOKEN_BUDGET):
    """Preserve every item and its ordering; reject oversized individual items."""
    batches, batch = [], []
    # Counting each serialized item plus framing is conservative for a JSON list.
    count = 2
    for item in items:
        size = token_size(item, model) + 2
        if size + 2 > budget:
            raise ValueError("An evidence section is too large for a tool batch. "
                             "Reindex with smaller chunks or select smaller sources.")
        if batch and count + size > budget:
            batches.append(batch)
            batch, count = [], 2
        batch.append(item)
        count += size
    if batch:
        batches.append(batch)
    return batches


def prepare_snapshot(store, selected_ids, collection, model):
    snapshot = store.snapshot_sources(
        selected_ids, collection, max_documents=MAX_DOCUMENTS, max_chunks=MAX_CHUNKS,
        max_characters=MAX_SOURCE_CHARACTERS,
    )
    # Short, run-local identifiers reduce prompt overhead; persisted references
    # always contain the original stable chunk IDs.
    aliases = {f"s{index}": source for index, source in enumerate(snapshot["sources"], 1)}
    document_inputs = {document["id"]: [] for document in snapshot["documents"]}
    for alias, source in aliases.items():
        document_inputs[source["document_id"]].append({
            "source_id": alias, "filename": source["filename"],
            "locator": source["locator"], "text": source["text"],
        })
    # Preflight every batch before any paid request. A too-large late document
    # must not cost the user calls for the earlier documents.
    batches = {identifier: bounded_batches(items, model)
               for identifier, items in document_inputs.items()}
    return snapshot, aliases, batches


def validate_source_ids(value, allowed, *, allow_empty=False):
    if (not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            or len(value) != len(set(value)) or any(item not in allowed for item in value)
            or (not value and not allow_empty)):
        raise ValueError("The tool returned missing, duplicated, or unknown source citations.")
    return value


def require_object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("The tool returned an invalid output structure.")
    return value


def require_text(value, *, limit=6000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("The tool returned empty or oversized text.")
    return value
