# OpenAI Chat and RAG Chat

The root `webui.py` hosts two independent Gradio tabs:

- **Chat** keeps the original chatbot, including document attachments, image tools,
  web search, LaTeX, and saved conversations.
- **RAG Chat** indexes a reusable document library and retrieves relevant passages
  for each question. Answers have numbered references with document locations,
  passage previews, image previews where available, and original-file downloads.

Both tabs start with **`gpt-6-luna`**. The app targets Windows, macOS, and Linux
with Python 3.10 or newer. See [Installation](#installation) for each platform.

The screenshots below show the original Chat interface.

![WebUI1](./chatbot/images/WebUI1.png)
![WebUI2](./chatbot/images/WebUI2.png)

## Features

- Select different OpenAI models
- Streaming responses, with a Stop button; models without streaming return their full reply
- Model-specific reasoning effort controls for the o-series, GPT-5, and GPT-6 models
- Web search
- Image analysis
- Document analysis (PDF, MS Word, MS Excel, MS PowerPoint, HTML, text files)
- Image generation and editing
- Save, load and delete chat history

Failed requests keep the attempted prompt visible for copying and exclude it from
future requests. Saved chats use atomic JSON replacement, and image files stay under
`chatbot/history/assets`. Loading a missing or damaged chat preserves the active conversation.

Image context usage is estimated from dimensions and documented model rules; models
without published sizing rules use a conservative fallback. Generated image bytes are
kept for display and saving, with a short placeholder sent in later text conversations.
The original Chat tab extracts native document text. Use RAG Chat's optional vision
extraction for scanned PDFs and images. Legacy `.xls` files are supported through `xlrd`.

## Installation

Use 64-bit Python 3.10 or newer; the CI matrix covers Python 3.10 and 3.13 on
Windows, macOS, and Linux. An OpenAI API key is needed for generation and indexing;
the interface and local previews can open without one.

Clone the project, then create a virtual environment:

```sh
git clone https://github.com/phatdatnguyen/openai-gradio-chatbot
cd openai-gradio-chatbot
```

Windows PowerShell:

```powershell
python -m venv chatbot-env
.\chatbot-env\Scripts\Activate.ps1
```

If PowerShell activation is restricted, use `chatbot-env\Scripts\python.exe`
instead of `python` in the following commands. In Command Prompt, activation is
`chatbot-env\Scripts\activate.bat`.

macOS/Linux:

```sh
python3 -m venv chatbot-env
. chatbot-env/bin/activate
```

Some Linux distributions package `venv` separately; install your distribution's
Python venv package if environment creation reports it missing.

Install dependencies on any platform:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Gradio is pinned to a tested release. Version 6.13.0 has a browser rendering loop
when Dataframe values update, which can freeze tab switching and history controls.
The upstream fix shipped in [6.16.0](https://github.com/gradio-app/gradio/releases/tag/gradio@6.16.0);
this project uses 6.30.0. Other direct dependencies have tested version floors and
compatibility bounds where appropriate. NumPy and urllib3 are installed transitively
by pandas and requests. `pywin32` is installed only on Windows. Microsoft Office,
LibreOffice, and Docker are optional system applications, not pip dependencies.
After updating dependencies, restart the app and hard-refresh the browser
(`Ctrl+Shift+R`, or `Cmd+Shift+R` on macOS).

Set the API key in the same terminal before launching.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "<your API key>"
```

macOS/Linux:

```sh
export OPENAI_API_KEY="<your API key>"
```

Alternatively, create the ignored file `chatbot/api_key.py`:

```python
API_KEY = "<your API key>"
```

`OPENAI_API_KEY` takes precedence. A legacy root `api_key.py` is still supported.
Never commit key files. Both tabs use the same configured API client.

### Optional Office layout previews

| Platform | Local converter | Without the converter |
|---|---|---|
| Windows | Installed Microsoft Word/PowerPoint and platform-marked `pywin32` | DOCX/PPTX content preview |
| macOS | LibreOffice application | DOCX/PPTX content preview |
| Linux | LibreOffice Writer and Impress | DOCX/PPTX content preview |

Install [LibreOffice](https://www.libreoffice.org/download/download-libreoffice/)
if you want page/slide layout previews on macOS/Linux. The app finds `soffice` or
`libreoffice` on `PATH`, or the standard macOS application location. For a custom
installation, set `LIBREOFFICE_PATH` to its executable, for example:

```sh
export LIBREOFFICE_PATH="/Applications/LibreOffice.app/Contents/MacOS/soffice"
```

Conversions use a temporary profile with macros disabled, have a time limit, and
cache the resulting PDF beside the managed original. Your normal LibreOffice
profile is not used. Fonts and converter versions can affect pagination and layout.
Minimal Linux installations also need fonts, such as the `fonts-dejavu-core` package.
DOC/PPT files can be previewed with a converter but must still be converted to a
supported format before indexing. PDF/image/spreadsheet previews and DOCX/PPTX
text extraction need neither Microsoft Office nor LibreOffice.

## Start web UI

With the virtual environment active, run on any platform:

```sh
python webui.py
```

Or use `start_webui.bat` on Windows and `sh start_webui.sh` on macOS/Linux. These
launchers use `chatbot-env` beside the script and work from another directory.
Open the local URL printed in the terminal (normally `http://127.0.0.1:7860`).
The server listens only on this computer. Stop it with `Ctrl+C` in its terminal.


## RAG library

The first RAG version includes:

- PDF page references, DOCX sections/tables, XLSX/XLS sheet/cell locations, PPTX
  slides/tables/notes, CSV/TSV rows, HTML, Markdown, and plain text.
- Optional OpenAI vision extraction for scanned PDF pages, image files, and embedded
  raster pictures in DOCX/PPTX. Text-bearing PDF pages currently use native text;
  figures on those pages are not automatically interpreted. Legacy `.doc` and `.ppt`
  files must first be converted to `.docx` and `.pptx`.
- Token-aware chunks using LangChain text splitters, OpenAI
  `text-embedding-3-small` embeddings, Qdrant semantic search, and SQLite FTS5 keyword
  search combined with reciprocal rank fusion.
- Separate working directories and document filters, content-hash deduplication, persistent originals,
  resumable extraction/embedding caches, and independent saved RAG conversations.
- Context budgeting, source-only prompting, and validation that generated numbered
  references belong to retrieved sources. These checks do not prove a claim is correct;
  use the source preview to verify important answers.

Indexing sends extracted text to OpenAI for embeddings. Answering sends the question,
recent conversation context, and retrieved passages. Enabling vision also sends image
data for transcription/description; relevant extracted images can accompany answers
when the selected answer model supports vision. API usage is billed to your key.
Library originals, catalogs, caches, vectors, and saved chats remain on your computer
with the default local configuration.

Each working directory is a subfolder of `rag_chatbot/data`, containing its own
`sources/`, `chats/`, and `library.sqlite3`. The active directory controls indexing,
retrieval, chat saving/loading, and source previews. Qdrant indexes are also separated
by working directory. Editing the directory dropdown does not change the active
directory until **Create / Open** is clicked.

The RAG tab has one outer row. Its left column (one third) contains the working
directory, Sources, and Document viewer. The right column (two thirds) contains
two equal columns: the chatbot with Prompt controls below, and saved chat history
with the **References from the latest answer** accordion below. The selected
reference's download, image, and excerpt appear directly below **Open reference**
inside that accordion. A Tools placeholder spans the bottom of
the right column. **Toggle chat history** shows or hides the saved-chat controls
while keeping references available.

The document viewer loads one page, frame, slide, or table segment at a time:

- PDFs have rendered pages with page navigation; images have a local image preview
  and frame navigation for supported multipage images.
- XLS/XLSX workbooks have a sheet selector and a paginated grid with row numbers
  and column letters. XLSX formulas are shown as formulas rather than recalculated.
  Previews are limited to 10,000 rows and 50 columns per sheet.
- Word and PowerPoint use locally installed Microsoft Office on Windows, or
  optional LibreOffice on macOS/Linux, to create cached PDF previews. Originals
  are unchanged. If conversion is unavailable,
  DOCX/PPTX use a content preview with text, tables, and embedded pictures; this
  fallback does not reproduce the original page/slide layout. Legacy DOC/PPT files
  require a local converter or downloading the original to open them.
- Text, HTML, and CSV/TSV sources also have bounded local previews.

Previewing does not call OpenAI or send documents to an online viewer. To preserve
open presentations, a new PowerPoint conversion requires the existing PowerPoint
application to be closed; cached previews remain usable. Preview support is independent
of indexing support: legacy DOC/PPT sources still need conversion to DOCX/PPTX or PDF
before they can be indexed.

### Start the vector database

For a large library, start Docker Desktop (Windows/macOS) or Docker Engine with
the Compose plugin (Linux), then run this from the project directory:

```sh
docker compose up -d
```

Qdrant listens on `127.0.0.1:6333` and persists vectors in a Docker named volume.
The compose file pins Qdrant `v1.19.2`. `docker compose stop` preserves data. Do not use
`docker compose down -v` unless you intend to delete the vector database.
The Chat tab and library catalog still open when Qdrant is unavailable; RAG indexing
checks the connection before calling the embedding API.

For a small trial without Docker, explicitly select embedded Qdrant. Use a
separate trial data directory so an existing server-backed library stays intact.

Windows PowerShell, after activating the virtual environment:

```powershell
$env:QDRANT_LOCAL_PATH = "$PWD\.cache\qdrant-local"
$env:RAG_DATA_DIR = "$PWD\.cache\rag-trial"
python webui.py
```

macOS/Linux:

```sh
QDRANT_LOCAL_PATH="$PWD/.cache/qdrant-local" RAG_DATA_DIR="$PWD/.cache/rag-trial" python webui.py
```

Embedded mode is for one running app process and smaller libraries. Server mode is
recommended as the library grows. Switching vector backends does not migrate vectors;
use a separate `RAG_DATA_DIR` for a trial backend, or migrate the database before switching.
To clear the embedded override in the current shell, run
`Remove-Item Env:QDRANT_LOCAL_PATH, Env:RAG_DATA_DIR`. The macOS/Linux example
sets these variables only for that one command.

`QDRANT_LOCAL_PATH` is a base directory; each working directory uses its own embedded
database below it. Other optional environment variables: `RAG_DATA_DIR` (the working
directory root, default `rag_chatbot/data`),
`QDRANT_URL` (default `http://127.0.0.1:6333`), and `QDRANT_API_KEY`.
Embedding, vision, chunk, and retrieval settings live in `rag_chatbot/config.py`.
Changing embedding/chunk settings creates a separate index fingerprint; reimport
documents to populate it. Old indexed versions remain in the catalog for explicit removal.

### Use RAG Chat

1. Open **RAG Chat**, choose an existing working directory or type a new name, then
   click **Create / Open**. An empty **Default** directory is available on first use.
2. Upload files in **Sources**. The filename textbox starts with the uploaded name;
   edit it before indexing if you want a different name in the library and citations.
   For multiple uploads, choose each file in the upload selector to edit its name.
   Keep the original file extension (it is added if omitted). Enable vision if you
   want scanned pages/images read, then click **Index documents**.
3. Check document statuses. A failed or stopped import can be retried by uploading the
   same file; completed extraction and embedding batches are reused.
   Reimporting an already indexed file with vision enabled upgrades its extraction
   without adding a duplicate. Stop prevents further OCR/embedding calls after the
   current request finishes; use Refresh library to see the final status.
4. Check the sources to use for chat, then ask a question using the Prompt controls.
   Ready sources start checked; **Select all** checks ready sources and **Clear**
   searches nothing. Your choices survive refresh and indexing. Newly indexed sources
   start checked, and failed sources are never searched. Changing the checks stops
   a reply in progress; the next question uses the new selection.
5. Select a document in **Document viewer** to open its preview automatically. This
   selection is independent of the checked sources used for chat retrieval.
6. Select a reference in the history/reference column to inspect its passage and
   original file. Save and load conversations from that directory's history controls.

Save a conversation before switching working directories. Opening another directory
clears the active conversation, uploads, filters, document preview, and references, and stops work from
the previous directory. Files and saved chats in other directories remain available
when you open those directories again.

Source names are library metadata: the file on your computer and the managed original's
stable path are unchanged. Renaming an identical reimport updates its library name
without creating a duplicate or recomputing embeddings. Existing saved chats retain
their citation snapshots. **Remove checked sources** removes the checked documents
from retrieval; failed documents can also be checked for removal.
Previous replies remain visible when the selection changes. If earlier replies cite
unchecked or unavailable sources, their generated text is omitted from the model's
conversation context; previous user questions remain available for follow-ups.

Removing a document excludes it from retrieval. Its managed original and caches are
retained so saved citations remain usable. A changed file is a new library version;
remove the old version explicitly when it should no longer be searched. The initial
version supports files up to 100 MiB and bounded page/image/cell counts, not an
unlimited background ingestion service. Native Office formulas are represented by
cached values where available; the app does not recalculate workbooks. OCR quality,
retrieval quality, and model abstention should be evaluated on your own documents.

This implementation uses `langchain-text-splitters` selectively. The full `langchain`
and `langchain-community` packages are unnecessary for the current readers and direct
SDK integration. Future work can add reranking, full-page figure understanding,
folder watching, and evaluation against a representative research library.

## Project layout

```text
webui.py                 Two-tab application shell
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
  ingestion.py           Document/image extraction and chunking
  viewer.py              On-demand PDF, image, Office, and spreadsheet previews
  office_preview.py      Local Office-to-PDF conversion and preview cache
  store.py               SQLite catalog and Qdrant hybrid retrieval
  history.py             RAG conversation persistence
  workspaces.py          Working-directory creation, validation, and settings
  config.py, schemas.py   Settings and data types
  data/                  Working directories (ignored)
    Default/
      sources/           Managed originals, derived images, extraction caches
      chats/             Saved RAG conversations and citations
      library.sqlite3    This directory's document catalog and text index
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
directory, and the Qdrant Docker volume together. Stop indexing/app writes while
creating a consistent backup.

## Running the tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary libraries, local Qdrant, and fake API responses; no paid OpenAI
calls are made. Tiktoken downloads its public tokenizer data on first use and caches
it under `.cache/tiktoken` (override with `TIKTOKEN_CACHE_DIR`).

Run `python tests/manual_rag_smoke.py` for an optional localhost Gradio HTTP/RPC
check of upload, indexing, answers, citation previews, and saved conversations.
It uses temporary storage and fake models, so it makes no paid API requests.

The GitHub Actions workflow installs dependencies, runs `pip check`, the test
suite, and the localhost smoke on Windows/macOS/Linux. Its Linux Python 3.13 job
also installs LibreOffice and checks real Word/PowerPoint PDF conversion. To run
that optional check locally on macOS/Linux with LibreOffice installed:

```sh
RUN_LIBREOFFICE_SMOKE=1 python -m pytest -q tests/test_libreoffice_integration.py
```

CI results are available after the workflow runs in GitHub; adding the workflow
does not itself establish that every platform has passed.
The current changes were checked with the full suite and localhost smoke on
Windows/Python 3.10, and with real DOCX/PPTX previews on Linux/Python 3.13 using
LibreOffice 25.2. The focused Linux check used distribution-provided test packages;
the complete Linux dependency install/app suite and native macOS checks still need
the CI matrix.

Implementation references: [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings),
[Qdrant local setup](https://qdrant.tech/documentation/quickstart/),
[LangChain recursive splitting](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).
