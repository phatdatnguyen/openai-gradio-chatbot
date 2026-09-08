"""Regression coverage for token counting and conversational history trimming."""

import base64
import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tokens
import messages


@pytest.mark.parametrize("model", ["gpt-4", "gpt-4o", "gpt-5.5"])
def test_special_token_spelling_is_counted_as_ordinary_user_text(model):
    text = "Explain the string <|endoftext|> and <|fim_prefix|>."
    encoding = tokens.get_encoding(model)

    assert tokens.count_text_tokens(text, model) == len(encoding.encode_ordinary(text))


@pytest.mark.parametrize("content", [
    "Explain <|endoftext|>.",
    [{"type": "text", "text": "Explain <|endoftext|>."}],
])
def test_special_token_spelling_does_not_break_message_counting(content):
    message = {"role": "user", "content": content}

    assert tokens.count_message_tokens(message, "gpt-4o") > tokens.TOKENS_PER_MESSAGE
    assert tokens.trim_history([message], "gpt-4o", system_prompt="Discuss <|endoftext|>.") == [message]


def test_trim_history_drops_answer_whose_question_no_longer_fits(monkeypatch):
    model = "gpt-4o"
    history = [
        {"role": "user", "content": "old question " * 100},
        {"role": "assistant", "content": "An old answer."},
        {"role": "user", "content": "A new question."},
    ]
    context = tokens.count_tokens(history[1:], model)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)

    assert tokens.trim_history(history, model, reserved_tokens=0) == history[2:]
    assert len(history) == 3


def test_trim_history_keeps_complete_recent_exchange(monkeypatch):
    model = "gpt-4o"
    history = [
        {"role": "user", "content": "old question " * 100},
        {"role": "assistant", "content": "An old answer."},
        {"role": "user", "content": "A recent question."},
        {"role": "assistant", "content": "A recent answer."},
        {"role": "user", "content": "A follow-up question."},
    ]
    context = tokens.count_tokens(history[1:], model)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)

    assert tokens.trim_history(history, model, reserved_tokens=0) == history[2:]


def test_trim_history_keeps_entire_history_when_it_fits(monkeypatch):
    model = "gpt-4o"
    history = [
        {"role": "user", "content": "A question."},
        {"role": "assistant", "content": "An answer."},
        {"role": "user", "content": "A follow-up question."},
    ]
    context = tokens.count_tokens(history, model)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)

    assert tokens.trim_history(history, model, reserved_tokens=0) == history


@pytest.mark.parametrize("prepare", [
    lambda history: history,
    messages.prepare_chat_messages,
    messages.prepare_responses_input,
])
def test_image_transport_size_does_not_become_text_tokens(prepare):
    small_image = "data:image/png;base64," + "A" * 100
    large_image = "data:image/png;base64," + "A" * 2_000_000
    small = prepare([{"role": "user", "content": "Describe it.", "image_url": small_image}])
    large = prepare([{"role": "user", "content": "Describe it.", "image_url": large_image}])

    assert tokens.count_tokens(small, "gpt-4o") == tokens.count_tokens(large, "gpt-4o")
    assert 1400 < tokens.count_tokens(large, "gpt-4o") < 1600
    assert tokens.trim_history(large, "gpt-4o") == large


@pytest.mark.parametrize("part", [
    {"type": "image_url", "image_url": {"url": "https://example.test/image.png", "detail": "low"}},
    {"type": "input_image", "image_url": "https://example.test/image.png", "detail": "low"},
])
def test_image_detail_and_multiple_images_are_charged(part):
    single = {"role": "user", "content": [part]}
    double = {"role": "user", "content": [part, part]}

    assert tokens.count_message_tokens(double, "gpt-4o") - tokens.count_message_tokens(single, "gpt-4o") == 85


@pytest.mark.parametrize(("model", "detail", "expected"), [
    ("gpt-4o-mini", "auto", 2833 + 8 * 5667),
    ("gpt-4o-mini", "low", 2833),
    ("gpt-5.4", "auto", 3000),
    ("gpt-5.4", "original", 12000),
    ("gpt-5.5", "auto", 12000),
    ("gpt-5.6-luna", "auto", 36000),
    ("gpt-5.6-luna", "low", 308),
])
def test_image_budgets_follow_documented_model_detail_limits(model, detail, expected):
    assert tokens.config.image_token_budget(model, detail) == expected


def test_undocumented_image_model_uses_explicit_estimate():
    assert tokens.config.image_token_budget("unlisted-vision-model") == tokens.config.DEFAULT_IMAGE_TOKEN_BUDGET


@pytest.mark.parametrize(("model", "dimensions", "detail", "expected"), [
    ("gpt-4o", (1024, 1024), "high", 765),
    ("gpt-4o", (2048, 4096), "high", 1105),
    ("gpt-4o-mini", (256, 256), "auto", 8500),
    ("gpt-4o-mini", (4096, 4096), "low", 2833),
    ("gpt-5.4", (1024, 1024), "high", 1229),
    ("gpt-5.4", (2048, 2048), "high", 3000),
    ("gpt-5.6-luna", (256, 256), "auto", 77),
])
def test_inline_dimensions_reduce_image_estimates(model, dimensions, detail, expected):
    assert tokens.config.image_token_budget(model, detail, dimensions) == expected


@pytest.mark.parametrize("prepare", [messages.prepare_chat_messages, messages.prepare_responses_input])
def test_image_compression_does_not_affect_dimension_based_counting(prepare):
    image = Image.new("RGB", (256, 256), "white")
    histories = []
    for compression in (0, 9):
        data = io.BytesIO()
        image.save(data, format="PNG", compress_level=compression)
        image_url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
        histories.append(prepare([{"role": "user", "content": "Describe it.", "image_url": image_url}]))

    assert tokens.count_tokens(histories[0], "gpt-4o-mini") == tokens.count_tokens(histories[1], "gpt-4o-mini")
    assert 8500 < tokens.count_tokens(histories[0], "gpt-4o-mini") < 8600
    assert 77 < tokens.count_tokens(histories[0], "gpt-5.6-luna") < 150


def test_base64_inside_plain_user_text_is_still_counted_as_text():
    # Only actual image attachments receive vision budgeting. Text containing a
    # data URL is still sent as text and must not bypass the context limit.
    text = "data:image/png;base64," + "A" * 100
    with_text = {"role": "user", "content": text}
    without_text = {"role": "user", "content": ""}

    assert (tokens.count_message_tokens(with_text, "gpt-4o")
            - tokens.count_message_tokens(without_text, "gpt-4o")) == tokens.count_text_tokens(text, "gpt-4o")
