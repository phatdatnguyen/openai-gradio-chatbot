# RAG Chat guide

[Back to README](../Readme.md) | [Studio tools](studio-tools.md)

## Supported sources

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
- Context budgeting, document-grounded prompting, optional web search, and validation that generated numbered
  references belong to retrieved sources. These checks do not prove a claim is correct;
  use the source preview to verify important answers.

Indexing sends extracted text to OpenAI for embeddings. Answering sends the question,
recent conversation context, and retrieved passages. Enabling vision also sends image
data for transcription/description; relevant extracted images can accompany answers
when the selected answer model supports vision. API usage is billed to your key.
Library originals, catalogs, caches, vectors, and saved chats remain on your computer
with the default local configuration.

## Workspaces and controls

Each working directory is a subfolder of `rag_chatbot/data`, containing its own
`sources/`, `chats/`, `artifacts/` (after saving a tool result), and `library.sqlite3`. The active directory controls indexing,
retrieval, chat saving/loading, and source previews. Qdrant indexes are also separated
by working directory. Selecting a directory in **Workspace** opens it immediately.
To create one, type a new name and press Enter; **Create / Open** remains available.

The RAG tab has one outer row. Its left column (one third) contains the working
directory, Sources, and Document viewer. The right column (two thirds) contains
two equal columns: the chatbot with Prompt controls below, and saved chat history
with the **References from the latest answer** accordion below. The selected
reference's download, image, and excerpt appear directly below **Open reference**
inside that accordion. The Tools area spans the bottom of
the right column. **Toggle chat history** shows or hides the saved-chat controls
while keeping references available.

Prompt also includes **Web search**, **Stream responses**, **Temperature**, and
**Top-p**. Web search is off by default. Enable it to supplement the selected
documents, or to ask a web-only question with no documents checked. Document
references stay numbered; web references appear as clickable links in the answer.
The integration uses OpenAI's [web search tool](https://developers.openai.com/api/docs/guides/tools-web-search).
Turning off streaming waits for the complete answer. Sampling controls are enabled
only when supported by the selected model and reasoning effort; unsupported values
are omitted from requests. These options affect RAG chat, while Studio tools use
their own document-grounded generation settings.

## Document previews

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

## Start the vector database

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

## Use RAG Chat

1. Open **RAG Chat** and select a working directory to open it automatically, or
   type a new name and press Enter. An empty **Default** directory is available on first use.
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
   searches no local documents (explicitly enabled web search can still answer).
   Your choices survive refresh and indexing. Newly indexed sources
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
Replies that used web search are also excluded from subsequent document-only context,
including after saving and reloading, so web-derived text is not reused as local evidence.

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

## Troubleshooting

| Problem | What to try |
|---|---|
| Indexing reports that Qdrant is unavailable | Start Docker and run `docker compose up -d`, or configure embedded mode above. |
| Scanned pages or images have no searchable text | Reupload with vision extraction enabled. This makes additional API requests. |
| An answer has insufficient evidence | Check that the intended sources are ready and checked; try a more specific question. |
| A saved original is missing or damaged | Reupload the same original file. The managed copy is repaired while a valid index is reused. |
| Office layout preview is unavailable | Use the content preview, download the original, or install the local converter described below. |
| The browser shows old controls after an update | Restart the app, then hard-refresh with Ctrl+Shift+R (Cmd+Shift+R on macOS). |

Use **Refresh library** after a stopped import to see its final status. Retrying the
same file reuses completed extraction and embedding batches where possible. Keep
only one app process open when using embedded Qdrant.

## Optional Office layout previews

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
