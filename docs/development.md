# Development and maintenance

[Back to README](../Readme.md)

## Project layout

```text
webui.py                 Chat, RAG Chat, and Plugins shell
plugins/                 Bundled tool catalog, enable/disable settings, manager UI
chatbot/
  chatbot.py             Original chatbot layout and callbacks
  api.py, config.py       Shared OpenAI calls and model configuration
  messages.py, tokens.py  Message conversion and context budgeting
  readers.py             Original attachment readers
  history_store.py        Original chat persistence
  log.py                 Shared logging
  history/               Existing chats and image assets (ignored)
  api_key.py             Optional private key file (ignored)
rag_chatbot/
  chatbot.py             RAG layout and callbacks
  service.py             Indexing, retrieval, evidence assembly
  chat_context.py        Complete selected-source context and preflight limits
  source_actions.py      Pasted text and retries through the shared indexer
  ingestion.py           Document/image extraction and chunking
  viewer.py              On-demand PDF, image, Office, and spreadsheet previews
  office_preview.py      Local Office-to-PDF conversion and preview cache
  store.py               SQLite catalog and Qdrant hybrid retrieval
  history.py             RAG conversation persistence
  autosave.py            Completed-reply autosave and latest restore
  research*.py           Search, notes, provenance, and note export
  workspace_*.py         Local descriptions and editor controls
  workspaces.py          Working-directory creation, validation, and settings
  config.py, schemas.py   Settings and data types
  tools/                 Source snapshots, generation, jobs/UI, artifacts, exports
  data/                  Working directories (ignored)
    Default/
      sources/           Managed originals, derived images, extraction caches
      chats/             Autosaved conversations, named snapshots, and citations
      notes/             Research notes, origins, and source references
      workspace.json     Local workspace description
      artifacts/         Saved tool outputs, provenance, and exports
      library.sqlite3    This directory's document catalog and text index
docs/                    RAG, Studio, and development guides
compose.yaml             Local Qdrant server
start_webui.bat          Windows launcher
start_webui.sh           macOS/Linux launcher
requirements.txt        Shared runtime dependencies with Windows-only markers
requirements-dev.txt    Test dependencies
.github/workflows/      Windows/macOS/Linux test matrix and LibreOffice smoke
tests/                   Offline unit and integration tests
```

Existing chat JSON and relative image references need no conversion when moving
`history/` to `chatbot/history/`. The loader also recognizes the legacy root directory
when the new one does not exist. Back up `chatbot/history/`, the entire RAG data
directory, and the Qdrant Docker volume together. Include `.cache/studio-presets.json`
for reusable tool instructions and `.cache/plugins.json` for plugin preferences.
Stop indexing/app writes while creating a consistent backup.

## Running the tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/conftest.py` redirects default histories, workspaces, vector paths, presets,
and plugin settings to temporary storage. It blocks the shared OpenAI client unless
a test explicitly substitutes a fake. Keep regression cases in focused `test_*.py`
modules; the localhost scripts stay opt-in and run separately in CI. `pytest.ini`
limits discovery to `tests/`. No paid API calls are made.

Tiktoken downloads its public tokenizer data on first use and caches
it under `.cache/tiktoken` (override with `TIKTOKEN_CACHE_DIR`).

Run `python tests/manual_rag_smoke.py` for an optional localhost Gradio HTTP/RPC
check of upload, indexing, answers, citation previews, autosave/restore, notes,
source search, descriptions, shared instruction presets, pasted-text indexing,
retry caches, full-document context, and note exports.
It uses temporary storage and fake models, so it makes no paid API requests.
Run `python tests/manual_tools_smoke.py` for the same kind of check of tool
generation, edits, source references, saved outputs, exports, and Plugins controls,
including slide navigation and PPTX export, flashcard reveal/navigation, and local
quiz scoring/restart after workspace switches.

The GitHub Actions workflow installs dependencies, runs `pip check`, the test
suite, and the localhost smoke on Windows/macOS/Linux. Its Linux Python 3.13 job
also installs LibreOffice and checks real Word/PowerPoint PDF conversion, including
generated Presentation exports. To run
that optional check locally on macOS/Linux with LibreOffice installed:

```sh
RUN_LIBREOFFICE_SMOKE=1 python -m pytest -q tests/test_libreoffice_integration.py
```

CI results are available after the workflow runs in GitHub; local verification
does not establish that every platform has passed. The optional
LibreOffice and filesystem-link tests skip when their prerequisites are unavailable.

Implementation references: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings),
[Qdrant local setup](https://qdrant.tech/documentation/quickstart/),
[LangChain recursive splitting](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter),
[python-pptx speaker notes](https://python-pptx.readthedocs.io/en/latest/user/notes.html).
