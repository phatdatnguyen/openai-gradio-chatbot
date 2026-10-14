# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Setup and Running

Use Python 3.10+ on Windows, macOS, or Linux.

```sh
# Create the environment (use python3 on macOS/Linux before activation)
python -m venv chatbot-env
```

Activate it with `.\chatbot-env\Scripts\Activate.ps1` in Windows PowerShell,
`chatbot-env\Scripts\activate.bat` in Command Prompt, or
`. chatbot-env/bin/activate` on macOS/Linux. Activation is optional if commands use
the environment's Python executable directly.

```sh
# Install dependencies
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip check

# Run the app
python webui.py

# Run the tests
python -m pytest -q
```

API key is read from `OPENAI_API_KEY`, then `chatbot/api_key.py` (`API_KEY = "sk-..."`),
with legacy root `api_key.py` support. `api_key.py` is gitignored and should never be
committed. A missing key no longer crashes at import — the UI still launches and
`config.get_client()` raises a message explaining what to set.

Launch with `start_webui.bat` on Windows or `sh start_webui.sh` on macOS/Linux.
Both use the local `chatbot-env` interpreter and resolve the project directory.

For RAG server mode, start Docker Desktop (Windows/macOS) or Docker Engine with
the Compose plugin (Linux), then run `docker compose up -d`.
For a small embedded trial, set `QDRANT_LOCAL_PATH`; never run multiple app processes
against the same embedded path. RAG indexing checks Qdrant before paid embeddings.
Tests inject embeddings/chat responses and local Qdrant, never send paid requests.

Gradio is pinned at 6.30.0; older 6.13.0 freezes on Dataframe updates. Keep runtime
requirements platform-neutral and guard `pywin32` with its Windows marker. Native
document readers do not require an Office installation. Optional Office PDF previews
use Word/PowerPoint on Windows and LibreOffice on macOS/Linux; the latter can be
configured with `LIBREOFFICE_PATH`. All converter imports/processes stay lazy.
See Readme.md for installation and fallback behavior.

`.github/workflows/tests.yml` defines Python 3.10/3.13 tests and localhost smoke on
Windows/macOS/Linux, plus real LibreOffice conversion on Linux. Run
`RUN_LIBREOFFICE_SMOKE=1 python -m pytest -q tests/test_libreoffice_integration.py`
on macOS/Linux to opt into the real converter check; ordinary tests mock converters.

## Architecture

| Module | Role |
|---|---|
| `webui.py` | Root two-tab Gradio shell. No API calls or model metadata. |
| `chatbot/chatbot.py` | Original Chat layout, callbacks, and `build_tab()`. |
| `chatbot/config.py` | All model tables, capability predicates, shared `OpenAI` client, and `check_model_tables()`. |
| `chatbot/api.py` | Every OpenAI call, including RAG embeddings and vision. `_build_call()` routes generation. |
| `chatbot/messages.py` | History ↔ API payload conversion, image encoding, display formatting. |
| `chatbot/readers.py` | Original attachment text extraction from PDF/DOCX/XLSX/PPTX/HTML/TXT. |
| `chatbot/tokens.py` | Tokenizer resolution, token counting, `trim_history()`. |
| `chatbot/history_store.py` | Save/load/delete original chat history, with image externalization. |
| `chatbot/log.py` | Shared `logger`. Use this, not `print()`. |
| `rag_chatbot/chatbot.py` | Independent RAG tab, library and citation UI. |
| `rag_chatbot/service.py` | Indexing, cached extraction/embedding batches, retrieval and evidence assembly. |
| `rag_chatbot/ingestion.py` | Source-aware readers, opt-in vision callbacks, token chunks; no API calls. |
| `rag_chatbot/viewer.py` | Bounded on-demand document previews, scoped to workspace sources; no API calls. |
| `rag_chatbot/office_preview.py` | Optional local Office-to-PDF conversion and cached previews; originals stay unchanged. |
| `rag_chatbot/store.py` | SQLite catalog/FTS5, Qdrant vectors, scoped hybrid search. |
| `rag_chatbot/history.py` | Separate versioned RAG histories with citation snapshots. |
| `rag_chatbot/workspaces.py` | Safe named working directories under RAG_DATA_DIR, per-directory settings and vector namespaces. |
| `rag_chatbot/config.py` | RAG settings and index fingerprint. |
| `tests/` | Pure logic, mocked API, local storage, UI wiring, and optional conversion/RPC tests. |

Both tabs obtain their initial model from `chatbot.config.DEFAULT_MODEL`, currently
`gpt-6-luna`; keep this shared. RAG selected-reference blocks live directly below
Open reference inside the References from the latest answer accordion.

### Adding a new model

1. Add it to `MODEL_CHOICES` and `MODEL_TOKEN_LIMITS` in `chatbot/config.py`.
2. Add it to whichever of `RESPONSES_API_MODELS`, `WEB_SEARCH_MODELS`,
   `MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH`, `MODEL_MAX_OUTPUT_TOKENS`,
   `VISION_DISABLED_MODELS` apply.
3. Run `pytest -q`. `check_model_tables()` will fail on drift — notably any `*-pro` or
   `*-codex*` model missing from `RESPONSES_API_MODELS`, which is a mistake that has
   been made before.

Capability predicates (`supports_temperature`, `supports_vision`,
`supports_system_message`, `is_reasoning_model`, `reasoning_effort_choices`) are
prefix-driven, so a new `gpt-5.x`, `gpt-6.x`, or `o-series` model is usually classified correctly
with no extra work. Conventional chat models in the gpt-5 line must be listed in
`NON_REASONING_GPT5_MODELS` or they will be treated as reasoning models.

### API call routing

All three surfaces are chosen in `api._build_call()`:

| Condition | Surface |
|---|---|
| Web search requested (and supported) | `client.responses` — search is Responses-only |
| Model in `RESPONSES_API_MODELS` | `client.responses` |
| Otherwise | `client.chat.completions` |
| Image generation / editing | `client.images.generate` / `.edit` via `api.generate_image()` |

`_build_call()` also omits `temperature`/`top_p` for reasoning models, folds the system
prompt into the first user turn for models that reject system messages, and shapes
reasoning effort per surface (`reasoning={"effort":…}` vs `reasoning_effort=…`).

### Message / history flow

1. Gradio hands `on_user_input` the text plus any image/document/link.
2. `api.build_user_message()` combines them into one user turn — documents and links are
   inlined as `<<<DOCUMENT_CONTENT>>>` / `<<<LINK_CONTENT>>>` blocks, images as a data URL
   on `image_url`.
3. `tokens.trim_history()` drops the oldest turns to fit the model's context, charging the
   system prompt and reserving room for output. It raises `InputTooLargeError` rather than
   sending an empty payload.
4. `api.stream_chat()` (or `send_chat()`) runs the request.
5. `messages.replace_history_content()` builds the display copy, collapsing the bulky
   inlined blocks back to `[File: …]` / `[URL: …]` labels.

`on_user_input` is a generator, so streamed replies render token by token. On failure it
rolls the pending user turn back out of history so it is not silently resent.

### Chat history on disk

Saved to `chatbot/history/*.json` (gitignored). Images are written to
`chatbot/history/assets/<name>/` and referenced by relative path, which keeps the JSON small —
a chat with one generated image went from 2.8 MB to under 1 KB. Loading still accepts
inline base64, so pre-existing saved chats keep working with no migration.

RAG working directories live under `rag_chatbot/data/<name>/` by default (override
the parent with `RAG_DATA_DIR`). Each has `sources/`, `chats/`, and `library.sqlite3`.
Every RAG UI callback must receive the active working directory explicitly; never
store a process-global active directory. Services are cached per directory, while
conversation, reference, and active-directory state belongs to the Gradio session.
Opening a different directory cancels pending work and resets session content. Late
results from old jobs must not overwrite the new directory's UI. Qdrant namespaces
and embedded database paths must also be separate for each working directory. The
SQLite catalog is authoritative: only `ready` documents with the current fingerprint
and published vector generation are searchable. Failed uploads and deleted documents
must never appear in retrieval, even if vector cleanup fails. Originals are retained
for saved citations. Do not mix conversation state or persistence between tabs.
Changing embedding/chunk settings requires reimport; never reuse incompatible vectors.

Document preview selection is separate from the retrieval filter. Load previews only
on demand, reset them when the working directory changes, and discard late results.
The viewer and ingestion must share `ingestion.PDFIUM_LOCK` for all PDFium operations.
Office previews run locally with a bounded conversion worker; never close a user's
existing Office application or execute document macros to produce a preview.
LibreOffice must use a disposable profile with macros/active content disabled; only
terminate processes owned by that conversion. Keep preview and indexing support
distinct: legacy DOC/PPT previews do not imply support for indexing those formats.

Source checkboxes define the shared selection for chat and future tools. An empty
selection means no sources; never convert it into the service's `None` (all sources).
Keep selections keyed by document ID, preserve unchecked sources across refresh/indexing,
and reject stale updates after a workspace switch. Editable upload filenames are validated
library metadata; never rename the user's upload or move stable managed originals.

## Conventions

- Log through `from chatbot.log import logger` (or `.log` inside chatbot); no `print()` in app code.
- Catch specific exceptions. Bare `except:` previously hid a real tokenizer bug.
- Prefer adding to a `config.py` table or predicate over an inline model-name check.
- Use pathlib and platform guards for native integrations. Physical filenames must
  fit UTF-8 byte limits as well as character limits; keep Windows-invalid names
  rejected on every platform so data directories remain portable.
- Clean generated artifacts without deleting API keys, histories, sources, vectors,
  or user work. Keep local data and credentials ignored by Git.
