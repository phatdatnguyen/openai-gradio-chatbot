"""Complete indexed-text context for explicitly selected chat sources."""

import json

from chatbot import config, tokens
from chatbot.messages import prepare_chat_messages, prepare_responses_input


CONTEXT_MODES = ("relevant", "full")
MAX_DOCUMENTS = 40
MAX_CHUNKS = 512
MAX_SOURCE_CHARACTERS = 500_000
MAX_EVIDENCE_TOKENS = 32_000


def snapshot_sources(store, document_ids, collection):
    """Read every indexed passage in the explicit selection without vectors."""
    if not document_ids:
        raise ValueError("Full selected sources requires at least one checked indexed source, "
                         "even when web search is enabled.")
    return store.snapshot_sources(document_ids, collection, max_documents=MAX_DOCUMENTS,
                                  max_chunks=MAX_CHUNKS, max_characters=MAX_SOURCE_CHARACTERS)


def prepare_request(question, history, snapshot, model, *, web_search, system_prompt):
    """Preflight complete evidence, trimming only older conversational turns.

    This mode supplies indexed text, including indexed image descriptions/OCR.
    Original images remain available in references but are not attached en masse.
    """
    sources = [
        {"number": index, "document_id": item["document_id"],
         "filename": item["filename"], "locator": item["locator"],
         "text": item["text"], "source_path": item["source_path"],
         "image_path": item.get("image_path")}
        for index, item in enumerate(snapshot["sources"], 1)
    ]
    if not sources:
        raise ValueError("The selected sources have no indexed passages. Reindex them before chatting.")
    payload = [{key: source[key] for key in ("number", "filename", "locator", "text")}
               for source in sources]
    evidence = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    count = tokens.count_text_tokens(evidence, model)
    if count > MAX_EVIDENCE_TOKENS:
        raise ValueError(f"Full selected sources needs {count:,} evidence tokens, exceeding the "
                         f"{MAX_EVIDENCE_TOKENS:,}-token limit. Select fewer sources or use "
                         "Relevant passages. No source text was omitted.")
    current = {"role": "user", "content": (
        "All indexed passages from the selected documents (quoted JSON data):\n"
        + evidence + "\n\nUser question:\n" + question)}
    use_search = bool(web_search) and web_search != config.WEB_SEARCH_OFF
    prepare = prepare_responses_input if use_search or config.uses_responses_api(model) else prepare_chat_messages
    try:
        # The newest turn is indivisible: a too-small model must not silently
        # drop passages, fall back to retrieval, or spend a generation request.
        tokens.trim_history(prepare([current]), model, web_search=use_search,
                            system_prompt=system_prompt)
    except tokens.InputTooLargeError as exc:
        raise ValueError("The full selected sources and question do not fit this model's usable context. "
                         "Select fewer sources, shorten the question, choose a larger-context model, "
                         "or use Relevant passages. No source text was omitted.") from exc
    prior = [{"role": message["role"], "content": message["content"]} for message in history]
    request = [*prior, current]
    trimmed = tokens.trim_history(prepare(request), model, web_search=use_search,
                                 system_prompt=system_prompt)
    return request[-len(trimmed):], sources
