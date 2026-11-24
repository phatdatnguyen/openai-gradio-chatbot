# Open Notebook comparison

[Back to README](../Readme.md)

Reviewed the local clone of [Open Notebook](https://github.com/lfnovo/open-notebook)
at commit `0c1e7abbd77411057d8feef02fc4588ec9a556d6` on November 22, 2026.
This is a comparison of that checkout, not a claim about later releases. The
reference clone was removed after review; it is not a runtime dependency.

## Adopted ideas

| Open Notebook workflow | Implementation in this app |
|---|---|
| Persistent notebook conversations and notes | Workspace chat autosave, named snapshots, human/AI/excerpt notes with retained references. |
| Direct full-text/vector search | Keyword and Hybrid Search over checked sources, without generating an answer. |
| Reusable transformations | Per-tool instruction presets with explicit Apply, shared across workspaces. |
| Enter Text source | Pasted TXT/Markdown sources using the same indexer, previews, and citations as uploaded files. |
| Retry Processing | Retry checked retained originals, reusing compatible extraction/embedding caches. |
| Full source context | Optional Full selected sources chat mode with explicit limits and complete indexed-text coverage. |

Relevant upstream references: [source intake](https://github.com/lfnovo/open-notebook/blob/0c1e7abbd77411057d8feef02fc4588ec9a556d6/docs/3-USER-GUIDE/adding-sources.md),
[chat context](https://github.com/lfnovo/open-notebook/blob/0c1e7abbd77411057d8feef02fc4588ec9a556d6/open_notebook/utils/context_builder.py),
and [search orchestration](https://github.com/lfnovo/open-notebook/blob/0c1e7abbd77411057d8feef02fc4588ec9a556d6/open_notebook/graphs/ask.py).
These features use this app's existing SQLite/Qdrant and shared OpenAI architecture.
Markdown/JSON note exports are an additional portability improvement.

## Remaining differences and priorities

Open Notebook also has provider configuration, URL ingestion, alternate extraction
engines, durable background jobs, multi-step Ask orchestration, and a dedicated REST
API. Those require separate implementation and testing; they are not included here.

Suggested next priorities are extraction quality on representative scientific PDFs,
then retrieval evaluation and query planning, then durable job recovery for large
libraries. URL imports and other model providers can follow when needed. Keep
audio/video outside the current scope.

This app already has document previews and ten specialized Studio tools with typed
validation, source snapshots, local exports, and cancellation guards. Preserve those
features while adopting useful workflow ideas instead of replacing its architecture.
