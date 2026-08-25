"""Unit tests for the pure logic -- the model tables, token accounting, message
conversion and history persistence. No network access.
"""

import base64
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api
import config
import history_store
import messages
import readers
import tokens

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AARAAA//8DAgH/"
    "6h1DkQAAAABJRU5ErkJggg=="
)


# --------------------------------------------------------------------------- tables

def test_model_tables_are_consistent():
    assert config.check_model_tables() is True


def test_every_selectable_model_has_a_context_limit():
    for model in config.MODEL_CHOICES:
        assert config.get_max_context_tokens(model) > 0


def test_pro_and_codex_models_route_to_responses_api():
    """The A3 regression: gpt-5-pro used to fall through to Chat Completions."""
    for model in ("gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro",
                  "o1-pro", "o3-pro", "gpt-5.1-codex", "gpt-5.3-codex"):
        assert config.uses_responses_api(model), model


def test_drift_in_responses_api_table_is_detected():
    original = set(config.RESPONSES_API_MODELS)
    config.RESPONSES_API_MODELS.discard("gpt-5.2-pro")
    try:
        with pytest.raises(ValueError, match="RESPONSES_API_MODELS"):
            config.check_model_tables()
    finally:
        config.RESPONSES_API_MODELS.clear()
        config.RESPONSES_API_MODELS.update(original)


def test_unknown_model_gets_modern_context_default():
    assert config.get_max_context_tokens("gpt-9-future") == config.DEFAULT_CONTEXT_TOKENS


# ----------------------------------------------------------------- capabilities

@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5", "o3", "o1-pro", "gpt-5.1-codex"])
def test_reasoning_models_reject_temperature(model):
    assert config.supports_temperature(model) is False


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo",
                                   "gpt-5-chat-latest", "gpt-5.3-chat"])
def test_chat_models_accept_temperature(model):
    assert config.supports_temperature(model) is True


def test_base_gpt4_is_not_vision_capable():
    """gpt-4 used to be missing from VISION_DISABLED_MODELS."""
    assert config.supports_vision("gpt-4") is False
    assert config.supports_vision("gpt-4o") is True


def test_mini_reasoning_models_reject_system_messages():
    assert config.supports_system_message("o1-mini") is False
    assert config.supports_system_message("o3-mini") is False
    assert config.supports_system_message("gpt-5.5") is True


def test_reasoning_effort_choices_by_family():
    assert config.reasoning_effort_choices("gpt-4o") == []
    assert "minimal" in config.reasoning_effort_choices("gpt-5")
    assert "xhigh" in config.reasoning_effort_choices("gpt-5.1-codex-max")
    assert "minimal" not in config.reasoning_effort_choices("o3")


# ----------------------------------------------------------------------- tokens

def test_gpt5_models_use_o200k_not_cl100k():
    """The C1 regression: encoding_for_model raises KeyError for gpt-5.x."""
    assert tokens.get_encoding("gpt-5.5").name == "o200k_base"
    assert tokens.get_encoding("gpt-5").name == "o200k_base"


def test_older_models_keep_their_own_encoding():
    assert tokens.get_encoding("gpt-4").name == "cl100k_base"
    assert tokens.get_encoding("gpt-4o").name == "o200k_base"


def test_vietnamese_is_not_overcounted():
    """cl100k counted diacritics ~1.78x high, over-trimming history."""
    text = "Xin chao, toi la mot nguoi Viet Nam dang thu nghiem." * 20
    diacritics = "Xin chào, tôi là một người Việt Nam đang thử nghiệm." * 20
    ratio = (tokens.count_text_tokens(diacritics, "gpt-5.5")
             / tokens.count_text_tokens(text, "gpt-5.5"))
    assert ratio < 1.35, f"diacritics inflate token count by {ratio:.2f}x"


def test_trim_history_drops_oldest_first():
    history = [{"role": "user", "content": f"message {i} " + "padding " * 200}
               for i in range(20)]
    trimmed = tokens.trim_history(history, "gpt-3.5-turbo")
    assert 0 < len(trimmed) < len(history)
    assert trimmed[-1] == history[-1], "the newest message must survive"


def test_trim_history_raises_when_newest_message_cannot_fit():
    """The B2 regression: this used to return [] and produce an opaque 400."""
    history = [{"role": "user", "content": "word " * 200000}]
    with pytest.raises(tokens.InputTooLargeError):
        tokens.trim_history(history, "gpt-3.5-turbo")


def test_trim_history_charges_the_system_prompt():
    # Sized so the history only just fits gpt-3.5-turbo's budget, meaning a long
    # system prompt has to push at least one message out.
    history = [{"role": "user", "content": "padding " * 300} for _ in range(10)]
    without = tokens.trim_history(history, "gpt-3.5-turbo")
    with_prompt = tokens.trim_history(history, "gpt-3.5-turbo",
                                      system_prompt="instructions " * 200)
    assert len(without) == len(history), "baseline should fit without the system prompt"
    assert len(with_prompt) < len(without)


def test_output_reserve_scales_with_the_model():
    assert tokens.reserve_for_output("gpt-5.5") > tokens.reserve_for_output("gpt-3.5-turbo")


# --------------------------------------------------------------------- messages

def test_prepare_chat_messages_wraps_images():
    history = [{"role": "user", "content": "what is this", "image_url": "data:image/png;base64,AAA"}]
    payload = messages.prepare_chat_messages(history)
    assert payload[0]["content"][0] == {"type": "text", "text": "what is this"}
    assert payload[0]["content"][1]["image_url"]["url"].startswith("data:image/png")


def test_prepare_responses_input_wraps_images():
    history = [{"role": "user", "content": "hi", "image_url": "data:image/png;base64,AAA"}]
    payload = messages.prepare_responses_input(history)
    assert payload[0]["content"][0]["type"] == "input_text"
    assert payload[0]["content"][1]["type"] == "input_image"


def test_replace_history_content_collapses_blocks_and_shows_images():
    history = [
        {"role": "user",
         "content": messages.build_document_block("report.pdf", "lots of text\nmore text")},
        {"role": "user", "content": messages.build_link_block("http://x.test", "page body")},
        {"role": "user", "content": "look", "image_url": "data:image/png;base64,AAA"},
    ]
    replaced = messages.replace_history_content(history)
    assert replaced[0]["content"] == "[File: report.pdf]"
    assert replaced[1]["content"] == "[URL: http://x.test]"
    assert replaced[2]["content"].endswith("![Image](data:image/png;base64,AAA)")
    assert all("image_url" not in message for message in replaced)


def test_is_data_url():
    assert messages.is_data_url("data:image/png;base64,AAA")
    assert messages.is_data_url("data:image/webp;base64,AAA")
    assert not messages.is_data_url("assets/chat/0001.png")
    assert not messages.is_data_url(None)


# ------------------------------------------------------------------- api wiring

def _call(model, **kwargs):
    kwargs.setdefault("history", [{"role": "user", "content": "hello"}])
    kwargs.setdefault("web_search", config.WEB_SEARCH_OFF)
    kwargs.setdefault("temperature", 0.5)
    kwargs.setdefault("top_p", 0.9)
    kwargs.setdefault("system_prompt", "")
    kwargs.setdefault("reasoning_effort", None)
    return api._build_call(model, **kwargs)


def test_temperature_is_omitted_for_reasoning_models():
    _, kwargs = _call("gpt-5.5")
    assert "temperature" not in kwargs and "top_p" not in kwargs


def test_temperature_is_sent_for_chat_models():
    _, kwargs = _call("gpt-4o")
    assert kwargs["temperature"] == 0.5 and kwargs["top_p"] == 0.9


def test_image_upload_on_a_codex_model_uses_the_responses_api():
    """The A2 regression: process_image always called chat.completions."""
    surface, kwargs = _call(
        "gpt-5.1-codex",
        history=[{"role": "user", "content": "what is this",
                  "image_url": "data:image/png;base64,AAA"}],
    )
    assert surface == api.RESPONSES_SURFACE
    assert kwargs["input"][0]["content"][1]["type"] == "input_image"


def test_web_search_uses_the_ga_tool_name():
    surface, kwargs = _call("gpt-5.5", web_search="high")
    assert surface == api.RESPONSES_SURFACE
    assert kwargs["tools"][0]["type"] == "web_search"
    assert kwargs["tools"][0]["search_context_size"] == "high"


def test_web_search_is_dropped_for_models_that_cannot_search():
    surface, kwargs = _call("gpt-4", web_search="high")
    assert surface == api.CHAT_SURFACE
    assert "tools" not in kwargs


def test_system_prompt_becomes_instructions_on_the_responses_api():
    _, kwargs = _call("gpt-5.5-pro", system_prompt="be terse")
    assert kwargs["instructions"] == "be terse"


def test_system_prompt_is_folded_in_for_models_that_reject_it():
    _, kwargs = _call("o1-mini", system_prompt="be terse")
    assert "instructions" not in kwargs
    assert not any(m["role"] == "system" for m in kwargs["messages"])
    assert kwargs["messages"][0]["content"].startswith("be terse")


def test_reasoning_effort_shape_per_surface():
    _, chat_kwargs = _call("gpt-5.5", reasoning_effort="high")
    assert chat_kwargs["reasoning_effort"] == "high"
    _, responses_kwargs = _call("gpt-5.5-pro", reasoning_effort="high")
    assert responses_kwargs["reasoning"] == {"effort": "high"}


def test_reasoning_effort_auto_is_not_sent():
    _, kwargs = _call("gpt-5.5", reasoning_effort=config.NO_REASONING_EFFORT)
    assert "reasoning_effort" not in kwargs and "reasoning" not in kwargs


# --------------------------------------------------------------------- readers

def test_read_excel_covers_every_sheet(tmp_path):
    """This whole path used to fail because openpyxl was not installed."""
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"b": [3]}).to_excel(writer, sheet_name="Second", index=False)

    text = readers.read_document(str(path))
    assert "First" in text and "Second" in text and "3" in text


def test_read_text_file_handles_utf8_and_cp1252(tmp_path):
    utf8_path = tmp_path / "a.txt"
    utf8_path.write_text("xin chào", encoding="utf-8")
    assert "chào" in readers.read_document(str(utf8_path))

    cp1252_path = tmp_path / "b.txt"
    cp1252_path.write_bytes("caf\xe9".encode("cp1252"))
    assert "caf" in readers.read_document(str(cp1252_path))


def test_empty_document_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(readers.EmptyDocumentError):
        readers.read_document(str(path))


def test_html_is_converted_to_text(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<h1>Title</h1><p>Body text</p>", encoding="utf-8")
    text = readers.read_document(str(path))
    assert "Title" in text and "Body text" in text


# --------------------------------------------------------------- history store

@pytest.fixture
def history_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "history").mkdir()
    return tmp_path / "history"


def test_sanitize_history_file_name():
    assert history_store.sanitize_history_file_name("a/b:c") == "a_b_c"
    assert history_store.sanitize_history_file_name("") == history_store.DEFAULT_HISTORY_NAME
    assert history_store.sanitize_history_file_name("..") == history_store.DEFAULT_HISTORY_NAME
    assert len(history_store.sanitize_history_file_name("x" * 300)) == 100


def test_path_traversal_is_blocked():
    for attempt in ("../../etc/passwd", "..\\..\\windows\\system32"):
        safe = history_store.sanitize_history_file_name(attempt)
        assert "/" not in safe and "\\" not in safe
        assert os.path.dirname(history_store.history_file_path(safe)) == history_store.HISTORY_DIR


def test_save_externalizes_images_and_load_restores_them(history_dir):
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    history = [
        {"role": "user", "content": "a dog", "image_url": inline},
        {"role": "assistant", "content": "here you go"},
    ]

    history_store.save_history(history, "cute dog")

    saved_path = history_dir / "cute dog.json"
    stored = json.loads(saved_path.read_text(encoding="utf8"))
    assert stored[0]["image_url"] == "assets/cute dog/0000.png"
    assert (history_dir / "assets" / "cute dog" / "0000.png").read_bytes() == PNG_1PX
    assert saved_path.stat().st_size < 500, "JSON should no longer carry the image"

    loaded, _ = history_store.load_history("cute dog")
    assert loaded[0]["image_url"] == inline
    assert loaded[1]["content"] == "here you go"


def test_save_externalizes_generated_images_embedded_in_markdown(history_dir):
    """Generated images live in the assistant's markdown, not image_url -- which is
    where essentially all of a saved chat's size comes from."""
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    history = [
        {"role": "user", "content": "draw a dog"},
        {"role": "assistant", "content": f"Generated image.\n\n![Image]({inline})"},
    ]

    history_store.save_history(history, "drawn")

    stored = json.loads((history_dir / "drawn.json").read_text(encoding="utf8"))
    assert "data:image/" not in stored[1]["content"]
    assert "![Image](assets/drawn/0001_0.png)" in stored[1]["content"]
    assert (history_dir / "assets" / "drawn" / "0001_0.png").read_bytes() == PNG_1PX

    loaded, _ = history_store.load_history("drawn")
    assert loaded[1]["content"] == history[1]["content"]


def test_multiple_embedded_images_in_one_message_round_trip(history_dir):
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    history = [{"role": "assistant", "content": f"![a]({inline}) and ![b]({inline})"}]

    history_store.save_history(history, "two")
    stored = json.loads((history_dir / "two.json").read_text(encoding="utf8"))
    assert "0000_0.png" in stored[0]["content"] and "0000_1.png" in stored[0]["content"]

    loaded, _ = history_store.load_history("two")
    assert loaded[0]["content"] == history[0]["content"]


def test_external_image_urls_are_left_alone(history_dir):
    history = [{"role": "assistant", "content": "![x](https://example.test/pic.png)"}]
    history_store.save_history(history, "remote")
    loaded, _ = history_store.load_history("remote")
    assert loaded[0]["content"] == history[0]["content"]


def test_legacy_inline_base64_history_still_loads(history_dir):
    """Existing saved chats must keep working with no migration step."""
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    legacy = [{"role": "user", "content": "old chat", "image_url": inline}]
    (history_dir / "legacy.json").write_text(json.dumps(legacy), encoding="utf8")

    loaded, status = history_store.load_history("legacy")
    assert loaded[0]["image_url"] == inline
    assert "legacy" in status


def test_missing_asset_degrades_gracefully(history_dir):
    orphan = [{"role": "user", "content": "gone", "image_url": "assets/x/0000.png"}]
    (history_dir / "orphan.json").write_text(json.dumps(orphan), encoding="utf8")

    loaded, _ = history_store.load_history("orphan")
    assert "image_url" not in loaded[0]
    assert loaded[0]["content"] == "gone"


def test_delete_removes_json_and_assets(history_dir):
    inline = "data:image/png;base64," + base64.b64encode(PNG_1PX).decode()
    history_store.save_history([{"role": "user", "content": "x", "image_url": inline}], "doomed")
    assert (history_dir / "doomed.json").exists()

    history_store.delete_history("doomed")
    assert not (history_dir / "doomed.json").exists()
    assert not (history_dir / "assets" / "doomed").exists()


def test_delete_missing_history_raises(history_dir):
    with pytest.raises(FileNotFoundError):
        history_store.delete_history("never existed")


def test_saving_empty_history_raises(history_dir):
    with pytest.raises(ValueError):
        history_store.save_history([], "nothing")


def test_select_history_file_reads_the_rendered_rows():
    frame = pd.DataFrame(["alpha", "beta", "gamma"], columns=["File name"])
    assert history_store.select_history_file(frame, 1) == "beta"
    assert history_store.select_history_file(frame, 99) == ""
    assert history_store.select_history_file(frame, None) == ""


def test_scanned_pdf_is_reported_as_scanned(tmp_path):
    """An image-only PDF used to yield the generic 'appears to be empty' warning."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "scanned.pdf"
    with open(path, "wb") as handle:
        writer.write(handle)

    with pytest.raises(readers.ScannedPdfError, match="scanned"):
        readers.read_document(str(path))
