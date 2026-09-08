# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Setup and Running

```bash
# Create and activate virtual environment (Windows)
python -m venv chatbot-env
chatbot-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, for the test suite

# Run the app
python webui.py

# Run the tests
pytest -q
```

API key is read from `api_key.py` (`API_KEY = "sk-..."`) with fallback to the
`OPENAI_API_KEY` environment variable. `api_key.py` is gitignored and should never be
committed. A missing key no longer crashes at import — the UI still launches and
`config.get_client()` raises a message explaining what to set.

On Windows, `start_webui.bat` activates the venv and launches `webui.py`.

## Architecture

| Module | Role |
|---|---|
| `webui.py` | Gradio layout and event wiring only. No API calls, no model metadata. |
| `config.py` | All model tables, capability predicates, the shared `OpenAI` client, and `check_model_tables()`. |
| `api.py` | Every OpenAI call. `_build_call()` is the single routing decision. |
| `messages.py` | History ↔ API payload conversion, image encoding, display formatting. |
| `readers.py` | Text extraction from PDF/DOCX/XLSX/PPTX/HTML/TXT. |
| `tokens.py` | Tokenizer resolution, token counting, `trim_history()`. |
| `history_store.py` | Save/load/delete chat history, with image externalization. |
| `log.py` | Shared `logger`. Use this, not `print()`. |
| `tests/test_units.py` | Unit tests for all the pure logic. |

### Adding a new model

1. Add it to `MODEL_CHOICES` and `MODEL_TOKEN_LIMITS` in `config.py`.
2. Add it to whichever of `RESPONSES_API_MODELS`, `WEB_SEARCH_MODELS`,
   `MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH`, `MODEL_MAX_OUTPUT_TOKENS`,
   `VISION_DISABLED_MODELS` apply.
3. Run `pytest -q`. `check_model_tables()` will fail on drift — notably any `*-pro` or
   `*-codex*` model missing from `RESPONSES_API_MODELS`, which is a mistake that has
   been made before.

Capability predicates (`supports_temperature`, `supports_vision`,
`supports_system_message`, `is_reasoning_model`, `reasoning_effort_choices`) are
prefix-driven, so a new `gpt-5.x` or `o-series` model is usually classified correctly
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

Saved to `history/*.json` (gitignored). Images are written to
`history/assets/<name>/` and referenced by relative path, which keeps the JSON small —
a chat with one generated image went from 2.8 MB to under 1 KB. Loading still accepts
inline base64, so pre-existing saved chats keep working with no migration.

## Conventions

- Log through `from log import logger`; there are no `print()` calls in app code.
- Catch specific exceptions. Bare `except:` previously hid a real tokenizer bug.
- Prefer adding to a `config.py` table or predicate over an inline model-name check.
