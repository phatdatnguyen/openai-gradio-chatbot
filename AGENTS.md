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
| `webui.py` | Root Chat, RAG Chat, and Plugins Gradio shell. No API calls or model metadata. |
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
| `rag_chatbot/tools/` | Selected-source snapshots, bounded Studio jobs including Compare/Evidence Check, session UI, artifacts and exports. |
| `plugins/` | Trusted bundled tool catalog, local enable/disable settings, dependency status and Plugins tab. |
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
Identical reuploads repair missing/corrupt managed originals atomically after a hash
check; preserve valid vectors and extraction/embedding caches. A ready catalog row
without a published generation is not searchable and must be republished on retry.
Changing embedding/chunk settings requires reimport; never reuse incompatible vectors.
The Workspace dropdown opens on user `input`, not programmatic `change`, to avoid
reopen loops. Disable selection during the switch; stale opens must not publish
state or finish a newer pending switch.
Gradio dropdown `input` also fires on unchanged blur. Dynamic document, sheet,
history, and reference selections can arrive after their choices have changed.
Primitive-only read callbacks bypass component preprocessing and validate against
current session/catalog membership before reading files; File/Dataframe uploads
and model/export controls retain normal preprocessing. Avoid redundant dropdown
inputs in file callbacks when canonical session state already owns the selection.
Unchanged viewer selections must preserve the current page/sheet. Saved citation
downloads must resolve inside the workspace's sources directory, never its catalog,
chats, or artifacts. History lists must exclude atomic temporary files and links.

RAG chat shares web-search, streaming, and sampling options with the API layer.
Search defaults off; an empty document selection can use the web only when explicitly
enabled. Keep web URL links separate from numbered local references. Preserve the
web-search provenance flag in saved histories and exclude web-derived assistant
context from later document-only answers. Model/effort capability predicates govern
sampling controls, and nonstreaming requests use the blocking shared sender.

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

### Studio tools

Summary, Spreadsheet, Presentation, Graphics, Plot, Mind Map, Flashcards, Quizzes,
Compare, and Evidence Check are implemented. Audio/video are outside the current rollout. Tools share the source
checkboxes and Prompt model/reasoning controls, but never reuse or mutate chat history.
All paid text generation goes through `chatbot.api.generate_structured()` and the shared
`config.supports_structured_output()` predicate. Graphics image requests go through
`chatbot.api.generate_image_asset()`, also used by the original Chat image wrapper.
Treat refusals, malformed JSON, and
incomplete replies as failures; never quietly retry using a weaker output format.

Whole-document tools use ordered catalog snapshots of all selected ready chunks,
including content hashes, fingerprints and published generations. Do not implement
them by asking `RAGService.answer()` to summarize its top-k retrieval window. Validate
citations against the snapshot; check cancellation and snapshot validity before each
model call and before publishing the result. Exceeding an explicit processing budget
must fail before generation rather than silently omit evidence. These checks establish
source membership, not that the model's claim is factually supported.
When initial evidence notes omit source identifiers, carry forward each missing
source's full original passage with only its own ID, preserving source order.
Generated malformed, unknown, or duplicate identifiers must still fail validation.
When consolidated notes omit source identifiers, carry forward the original notes
covering those identifiers with their complete text and citations. Never attach
omitted IDs to unrelated generated text. Keep malformed/unknown citations strict,
preserve nonshrinking batches, and require progress across each consolidation round
under the existing token, round, and request limits.

Spreadsheet keeps one current result under `artifacts/spreadsheet/current.json`,
saved automatically after generation and restored on workspace open. Its explicit
Save button below the editable dataframe updates that same result, including cell
edits and original references; there is no saved-output dropdown, Load button, or
version-creation button. Retain older JSON under `artifacts/<uuid>/` and adopt the
latest only when no current Spreadsheet exists. Keep atomic replacement
and cancellation guards so stale jobs cannot overwrite current outputs.
Tool outputs stay separate from `chats/`. Summary keeps a single current result under
`artifacts/summary/current.json`, replaced atomically after generation or Edit/Save
and restored automatically on workspace open. Summary export uses the chosen portable
filename and downloads directly without creating another history entry. Older Summary
artifacts are retained, with the latest adopted only when no current result exists.
Spreadsheet's direct Export includes current cell edits and writes the chosen portable
filename under `artifacts/spreadsheet/exports/`, without adding a history version.
Both tool layouts put the preview/table on the left and bounded reference excerpts
and vertically stacked filename/format/export controls on the right.
Presentation (`presentation`) appears before Graphics, Plot, and Mind Map in tool tabs and the Plugins
catalog. It uses the whole-source evidence pipeline with strict cited slide data:
title, bullets, notes, and source_ids. `tools/presentation.py` owns the bounds and
validation. The slide-count slider defaults to 10, range 1–20; generation must return
exactly that count, including any cover, and cover every selected document. Keep one
current deck at `artifacts/presentation/current.json`, replaced atomically with
cancellation guards and restored on workspace open. Slide navigation is session-only
and must reject stale tokens/workspace revisions, including after workspace restore.
The left preview/notes and right References/export controls follow other tools.
PPTX is the default export; Markdown and JSON preserve source/provenance snapshots.
Export with lazy `python-pptx`, native editable text, escaped plain data, bounded
text layout, and per-slide references/source details in notes. Do not add unrequested
cover/reference slides or run generated markup/code. No Office installation or remote
renderer is needed. Missing PPTX dependencies must not break startup or text exports.
Graphics (`graphics`) appears before Plot and Mind Map in both tool tabs and the
Plugins catalog. Prepare a bounded cited visual brief using the whole-source pipeline,
with coverage of every selected document, then make one shared Image API request.
Validate image options before paid work, and check cancellation and snapshot validity
before and after the image request. Use shared image tables for both Chat and Graphics;
`gpt-image-2.5-sunburst` is the default, with `gpt-image-2.5-flare` also available.
Model-specific quality/background/fidelity options belong in config, not inline model
checks. Transparent JPEG must fail before generation. Do not silently retry image calls.
Keep one current result at `artifacts/graphics/current.json`, with a separate bounded,
validated raster asset using an immutable name and scoped paths. Do not store base64
images in the manifest. Save the image before atomically publishing its manifest;
failed or cancelled jobs must leave the previous result intact. Restore on workspace
open; guard reference/export callbacks against stale output tokens and workspace
revisions. Use the standard preview/References/export layout and direct downloads.
Graphics and Plot tab selection also reloads only that tab's current result to
refresh previews after hidden-tab mounting. Skip active/stopping jobs and deselection;
reject reads superseded by a workspace switch, generation, or newer tab open. Restore
from disk without API requests, preserve prior output on restore errors, and keep
these callbacks separate from the all-tools workspace reset.
Raster export preserves the generated file; Markdown/JSON retain the brief, source
snapshot, and image settings. Generated rasters are illustrations, not verified plots.
Plot (`plot`) appears immediately before Mind Map in the tool tabs and Plugins catalog.
Use the bounded whole-source pipeline to extract cited numeric data; chart type is
Auto (default), Bar, Line, or Scatter. An explicitly selected type must be honored.
Validate content centrally in `tools/plot.py`: at most six unique series and 10,000
total points (also the per-series limit); finite bounded numeric values (never booleans),
categorical bars versus numeric line/scatter x values, and valid source identifiers.
Each point retains a short verbatim evidence excerpt from a cited source and its
explicit numerical values must occur in that excerpt. These checks establish
occurrence, not correct semantic pairing or units; show the read-only data table
and retain references for review. Do not invent missing values, interpolate,
convert units, or digitize images in this stage. Each selected document must have
plotted data or a cited limitation; no usable values is an actionable failure.
Keep one current plot under `artifacts/plot/current.json`, replaced atomically with
cancellation guards and restored on workspace open. Session tokens and workspace
revisions guard references and exports. Follow the standard left preview/table and
right References/vertical export layout. PNG is the default export; SVG/PDF render
the same data, CSV/JSON retain values and source/provenance snapshots. Escape
untrusted CSV text while preserving validated numeric fields as numbers.
Matplotlib is isolated in `plugins/plot/requirements.txt`, included by the main
requirements file. Import it lazily; missing dependencies must leave startup,
data viewing, and CSV/JSON exports working. Use trusted Figure/Agg rendering under
a lock without pyplot, GUI windows, external LaTeX, or model-produced code/markup.
Render text literally, including dollar signs and TeX-like strings. Do not pad
missing categories with zeros or silently drop invalid points. Rendering and
exports must not make model calls.
For one selected CSV/TSV, `tools/tabular_plot.py` validates a complete hash-checked
local table and requests only column/group selection through generate_structured.
Copy all row coordinates locally, preserving duplicates; never transcribe thousands
of points through model notes. Use separate bounds of 10,000 rows/chunks, 128 columns,
8 MiB original/source text, and 16 MiB saved output. Require explicit indexed numeric
passages, validate the mapping, and recheck source hashes/generations/cancellation
before and after generation. Unsupported filtering/aggregation/calculation requests
fail explicitly. Keep the UI table bounded to the first 200 rows with a clear label;
the rendered chart and CSV/JSON exports retain every point. Dense rendering reduces
tick labels/markers without sampling points or creating unbounded figure dimensions.
Mind Map (`mind_maps`) uses the same two-column layout with a bounded map viewport
and export controls; it intentionally has no References panel. Keep one current map
per workspace under `artifacts/mind_maps/current.json`. Validate a connected tree
with one root, bounded nodes/depth, and source IDs from the selected snapshot.
At the generation boundary, assign app-owned node IDs and remap parent links using
exact original IDs. Model ID formatting (numeric, Unicode, punctuation, or length)
must not reject an otherwise valid tree. Still reject blank/duplicate IDs, unknown
parents, cycles, invalid citations, and excessive size/depth; keep persisted/rendered
IDs strict, and never rewrite source IDs as node IDs.
Generate SVG only from trusted rendering code and escaped data; no model-produced
SVG, HTML, scripts, or remote renderer. Preserve source/provenance data in saved
maps and Markdown/JSON exports. Exclude standalone References and Translation tools.
Flashcards (`flashcards`) and Quizzes (`quizzes`) share the whole-source evidence
pipeline, with cited answers and coverage of every selected document. Validate their
content centrally in `tools/learning.py`: bounded, nonempty cards/questions, unique
prompts, four distinct quiz options, an integer answer index 0–3, and known citations.
Each learning tool has an integer count slider to the right of its instructions,
default 20, bounded to 1–60 Flashcards or 1–40 quiz questions. Validate the count
before paid work, pass it explicitly to generation, and require an exact count in
the generated result without retries or truncation. Record `settings.item_count`.
Keep count validation at generation time so older saved sets remain loadable.
Keep one current output per tool under `artifacts/<tool_id>/current.json`, replaced
atomically after generation and restored on workspace open. Study navigation, revealed
answers, and quiz attempts are session state only. Check answers and calculate scores
locally; count a question once until Restart quiz, and never make another API request
for study interactions. Reset study state on workspace changes or regeneration and
reject stale interactions. Markdown/CSV/JSON exports include complete content and
answer keys, source snapshots, and provenance; escape formula-like CSV values.
Quiz options display A./B./C./D. on full-width rows; their callback values remain
0-3. Quizzes defaults to Aiken (`aiken`, `.txt`) export and also supports Moodle XML
(`moodle_xml`, `.xml`). These formats are quiz-only and require no new dependencies.
Aiken contains only single-line questions/options and `ANSWER: X`, separated by
blank lines; protect parser-ambiguous question prefixes. It cannot retain explanations
or references. Moodle XML uses escaped `plain_text`, single-answer multichoice,
ABCD numbering, and explanation/cited evidence in general feedback. Keep the saved
payload and complete JSON source/provenance snapshot unchanged during exports.
Compare (`compare`) and Evidence Check (`evidence_check`) complete the current rollout
and follow Quizzes in the tool tabs and Plugins catalog. Both use the whole-source
pipeline and shared Prompt model/effort without modifying chat history. Validate
their structured content centrally in `tools/review.py`. Compare requires 2–8
documents and 1–12 unique aspects, with exactly one finding per selected document
per aspect. Found findings cite only their own document; not_found findings have
no citations. Each aspect has found evidence and a synthesis citing its findings.
Evidence Check accepts 1–20 nonempty claims, one per line, at most 1,000 characters
each. Assign app-owned claim IDs and retain exact input text in settings and results;
never let the model silently rewrite or omit a claim. Classify selected-source
support as supported, contradicted, mixed, or not_found. Validate each evidence
quotation against its cited original passage and validate stance/verdict consistency.
Not_found is insufficient selected-source evidence, never proof that a claim is false.
Keep supporting, contradicting, and context evidence distinct, preserve quotations
through notes/consolidation, and explain that model verdicts are not independent
fact verification. Do not force citations to irrelevant documents to satisfy coverage.
Both tools keep one atomic current output at artifacts/<tool_id>/current.json,
restore it on workspace open, and retain the prior output on failure/cancellation.
Use read-only reports/tables with bounded previews, References, and vertical
filename/format/Export controls. Markdown/CSV/JSON retain source/provenance details;
CSV cells must be formula-safe. No additional dependencies or browser installers.
Guard references and exports with session tokens and workspace revisions.

Preserve source snapshots and mark user edits. Scope all paths to the workspace and
reject traversal/symlink escapes. Render
data using trusted templates; never execute generated Python, JavaScript, SVG scripts,
spreadsheet formulas, or shell commands. Escape untrusted spreadsheet values on export.
Every direct exporter must check its cancellation predicate before work and before
atomic replacement. A stale export must preserve any existing download, including
when a workspace, output token, filename, or format changes during serialization.

The Plugins tab manages only the trusted bundled catalog in this stage. Persist app-wide
enable/disable settings under ignored `.cache/plugins.json`; dependency checks and
exporters stay lazy. Optional export failures must not break UI startup or basic text
exports. There is no browser-driven package installation or arbitrary plugin loader.
Only the plugin table's Enabled boolean column is editable. Apply checkbox changes
immediately, validate against trusted catalog rows, and reject attempts to enable
planned tools or modify metadata. Update changed settings only so stale views cannot
overwrite unrelated choices from another session.

## Conventions

- Log through `from chatbot.log import logger` (or `.log` inside chatbot); no `print()` in app code.
- Catch specific exceptions. Bare `except:` previously hid a real tokenizer bug.
- Prefer adding to a `config.py` table or predicate over an inline model-name check.
- Use pathlib and platform guards for native integrations. Physical filenames must
  fit UTF-8 byte limits as well as character limits; keep Windows-invalid names
  rejected on every platform so data directories remain portable.
- Clean generated artifacts without deleting API keys, histories, sources, vectors,
  or user work. Keep local data and credentials ignored by Git.
