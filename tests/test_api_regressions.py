"""Exercise request failures and stream cleanup without paid API calls."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api
import config


HISTORY = [{"role": "user", "content": "hello"}]


def build_call(model, **kwargs):
    return api._build_call(model, HISTORY, kwargs.pop("web_search", "None"),
                           1, 1, "", kwargs.pop("effort", "auto"), **kwargs)


@pytest.mark.parametrize("web_search", ["None", "high"])
def test_astra_omits_sampling_parameters_and_passes_reasoning(web_search):
    surface, kwargs = build_call("gpt-6-astra", web_search=web_search, effort="max")
    assert "temperature" not in kwargs and "top_p" not in kwargs
    if surface == api.RESPONSES_SURFACE:
        assert kwargs["reasoning"] == {"effort": "max"}
    else:
        assert kwargs["reasoning_effort"] == "max"
    assert "none" not in config.reasoning_effort_choices("gpt-6-astra")


def test_invalid_reasoning_effort_fails_before_request():
    with pytest.raises(ValueError, match="does not support reasoning effort"):
        build_call("gpt-6-astra", effort="none")


@pytest.mark.parametrize("model,effort", [
    ("gpt-5-pro", "minimal"), ("gpt-5.2-pro", "none"),
    ("gpt-5.4-pro", "low"), ("gpt-5.5-pro", "none"),
    ("gpt-5.2-codex", "none"), ("gpt-5.3-codex", "none"),
])
def test_unsupported_pro_and_codex_efforts_are_not_offered_or_sent(model, effort):
    assert effort not in config.reasoning_effort_choices(model)
    with pytest.raises(ValueError, match="does not support reasoning effort"):
        build_call(model, effort=effort)


@pytest.mark.parametrize("model", ["gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5.2-pro",
                                    "gpt-5.4-pro", "gpt-5.5-pro", "gpt-5.3-codex"])
def test_documented_xhigh_effort_can_be_requested(model):
    surface, kwargs = build_call(model, effort="xhigh")
    if surface == api.RESPONSES_SURFACE:
        assert kwargs["reasoning"] == {"effort": "xhigh"}
    else:
        assert kwargs["reasoning_effort"] == "xhigh"


def test_switching_to_text_model_detects_images_in_older_turns():
    history = [{"role": "user", "content": "Describe", "image_url": "data:image/png;base64,AAA"},
               {"role": "assistant", "content": "A picture."}, HISTORY[0]]
    with pytest.raises(ValueError, match="images already in this conversation"):
        api._build_call("gpt-4", history, "None", 1, 1, "", "auto")


@pytest.mark.parametrize("model", sorted(config.DEEP_RESEARCH_MODELS))
def test_deep_research_always_has_its_required_search_tool(model):
    surface, kwargs = build_call(model)
    assert surface == api.RESPONSES_SURFACE
    assert kwargs["tools"] == [{"type": "web_search", "search_context_size": "medium"}]


def test_folding_instructions_does_not_mutate_multimodal_payload():
    content = [{"type": "input_text", "text": "hello"}]
    payload = [{"role": "user", "content": content}]
    folded = api._fold_system_prompt_into_first_user_message(payload, "be concise", "input_text")
    assert content == [{"type": "input_text", "text": "hello"}]
    assert folded[0]["content"][0] == {"type": "input_text", "text": "be concise"}


class FakeStream:
    def __init__(self, events, final=None):
        self.events = events
        self.final = final
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        yield from self.events

    def get_final_response(self):
        if self.final is None:
            raise RuntimeError("Didn't receive a `response.completed` event.")
        return self.final


def response(text="ok", status="completed", refusal=False):
    part = NS(type="refusal", refusal=text) if refusal else NS(type="output_text", text=text)
    return NS(status=status, output=[NS(type="message", content=[part])],
              error=NS(message="server failure"), incomplete_details=NS(reason="max_output_tokens"))


def install_client(monkeypatch, surface, result):
    def create(**kwargs):
        return result
    client = NS(responses=NS(create=create, stream=create),
                chat=NS(completions=NS(create=create)))
    monkeypatch.setattr(config, "get_client", lambda: client)
    return "gpt-5.2-pro" if surface == "responses" else "gpt-4o"


def chat_chunk(text=None, refusal=None, finish=None):
    return NS(choices=[NS(finish_reason=finish, delta=NS(content=text, refusal=refusal))])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_responses_failure_is_not_a_successful_partial_answer(monkeypatch, status, streaming):
    final = response("partial answer", status=status)
    result = FakeStream([NS(type="response.output_text.delta", delta="partial answer"),
                         NS(type=f"response.{status}", response=final)]) if streaming else final
    model = install_client(monkeypatch, "responses", result)
    with pytest.raises(RuntimeError, match="server failure|incomplete"):
        if streaming:
            list(api.stream_chat(model, HISTORY))
        else:
            api.send_chat(model, HISTORY)
    if streaming:
        assert result.closed


@pytest.mark.parametrize("surface", ["responses", "chat"])
@pytest.mark.parametrize("streaming", [False, True])
def test_refusal_text_is_displayable(monkeypatch, surface, streaming):
    refusal = "I cannot help with that request."
    if surface == "responses":
        final = response(refusal, refusal=True)
        result = FakeStream([NS(type="response.refusal.delta", delta=refusal)], final) if streaming else final
    elif streaming:
        result = FakeStream([chat_chunk(refusal=refusal), chat_chunk(finish="stop")])
    else:
        result = NS(choices=[NS(message=NS(content=None, refusal=refusal), finish_reason="stop")])
    model = install_client(monkeypatch, surface, result)
    actual = "".join(api.stream_chat(model, HISTORY)) if streaming else api.send_chat(model, HISTORY)
    assert actual == refusal
    if streaming:
        assert result.closed


@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_stream_cleanup_when_user_stops(monkeypatch, surface):
    events = [NS(type="response.output_text.delta", delta="hello")] if surface == "responses" else [chat_chunk("hello")]
    stream = FakeStream(events, response())
    model = install_client(monkeypatch, surface, stream)
    generator = api.stream_chat(model, HISTORY)
    assert next(generator) == "hello"
    generator.close()
    assert stream.closed


@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_truncated_stream_is_reported(monkeypatch, surface):
    events = [NS(type="response.output_text.delta", delta="hello")] if surface == "responses" else [chat_chunk("hello")]
    stream = FakeStream(events)
    model = install_client(monkeypatch, surface, stream)
    with pytest.raises(RuntimeError, match="completed|ended before"):
        list(api.stream_chat(model, HISTORY))
    assert stream.closed


def test_responses_error_event_is_reported(monkeypatch):
    stream = FakeStream([NS(type="error", message="stream error")])
    model = install_client(monkeypatch, "responses", stream)
    with pytest.raises(RuntimeError, match="stream error"):
        list(api.stream_chat(model, HISTORY))
    assert stream.closed


@pytest.mark.parametrize("surface", ["responses", "chat"])
@pytest.mark.parametrize("streaming", [False, True])
def test_empty_response_is_reported(monkeypatch, surface, streaming):
    if surface == "responses":
        result = FakeStream([], response("")) if streaming else response("")
    elif streaming:
        result = FakeStream([chat_chunk(finish="stop")])
    else:
        result = NS(choices=[])
    model = install_client(monkeypatch, surface, result)
    with pytest.raises(RuntimeError, match="empty response"):
        if streaming:
            list(api.stream_chat(model, HISTORY))
        else:
            api.send_chat(model, HISTORY)


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_chat_incomplete_finish_is_reported(monkeypatch, finish):
    stream = FakeStream([chat_chunk("partial"), chat_chunk(finish=finish)])
    model = install_client(monkeypatch, "chat", stream)
    with pytest.raises(RuntimeError, match="incomplete"):
        list(api.stream_chat(model, HISTORY))
    assert stream.closed


def test_page_download_is_bounded_and_closed(monkeypatch):
    closed = []

    @contextmanager
    def get(*args, **kwargs):
        assert kwargs["stream"] is True
        try:
            yield NS(raise_for_status=lambda: None, iter_content=lambda **kw: iter([b"1234", b"5678"]))
        finally:
            closed.append(True)

    monkeypatch.setattr(api.requests, "get", get)
    monkeypatch.setattr(api, "MAX_URL_BYTES", 5)
    with pytest.raises(ValueError, match="download limit"):
        api.fetch_url_text("https://example.test")
    assert closed == [True]


def test_html_without_charset_preserves_utf8(monkeypatch):
    @contextmanager
    def get(*args, **kwargs):
        yield NS(raise_for_status=lambda: None,
                 iter_content=lambda **kw: iter(["<p>Xin chào Việt Nam</p>".encode()]),
                 headers={"Content-Type": "text/html"}, encoding="ISO-8859-1")
    monkeypatch.setattr(api.requests, "get", get)
    assert "Xin chào Việt Nam" in api.fetch_url_text("https://example.test")


@pytest.mark.parametrize("header,encoding,body,expected", [
    ("text/html; charset=unknown-charset", "unknown-charset", b"<p>Hello</p>", "Hello"),
    ("text/html; charset = windows-1252", "windows-1252", b"<p>caf\xe9</p>", "café"),
    ("text/html", "ISO-8859-1", b"<p>caf\xe9</p>", "café"),
])
def test_html_charset_handling(monkeypatch, header, encoding, body, expected):
    @contextmanager
    def get(*args, **kwargs):
        yield NS(raise_for_status=lambda: None, iter_content=lambda **kw: iter([body]),
                 headers={"Content-Type": header}, encoding=encoding)
    monkeypatch.setattr(api.requests, "get", get)
    assert expected in api.fetch_url_text("https://example.test")


@pytest.mark.parametrize("refusal", [False, True])
def test_responses_stream_with_installed_sdk(monkeypatch, refusal):
    """Replay real SSE shapes through the SDK, including its final accumulator."""
    import json

    import httpx
    from openai import OpenAI

    text = "Unable to comply." if refusal else "Hello."
    part_type = "refusal" if refusal else "output_text"
    field = "refusal" if refusal else "text"
    part = {"type": part_type, field: text}
    if not refusal:
        part["annotations"] = []
        part["logprobs"] = []
    item = {"id": "msg_test", "type": "message", "role": "assistant",
            "status": "completed", "content": [part]}
    final = {"id": "resp_test", "object": "response", "created_at": 1,
             "model": "gpt-5.2-pro", "status": "completed", "output": [item]}
    empty_part = {**part, field: ""}
    events = [
        {"type": "response.created", "response": {**final, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {**item, "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", "output_index": 0, "content_index": 0,
         "item_id": "msg_test", "part": empty_part},
        {"type": f"response.{part_type}.delta", "output_index": 0,
         "content_index": 0, "item_id": "msg_test", "delta": text, "logprobs": []},
        {"type": "response.completed", "response": final},
    ]
    payload = "".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n"
        for index, event in enumerate(events)
    )
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=payload)

    with OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        monkeypatch.setattr(config, "get_client", lambda: client)
        assert "".join(api.stream_chat("gpt-5.2-pro", HISTORY)) == text
    assert len(calls) == 1


@pytest.mark.parametrize("model", ["o1-pro", "o3-pro", "gpt-5.5-pro"])
def test_models_without_streaming_return_a_blocking_reply(monkeypatch, model):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        assert "stream" not in kwargs
        return response("Full reply")

    def forbidden(**kwargs):
        pytest.fail("This model must never receive a streaming request")

    client = NS(responses=NS(create=create, stream=forbidden))
    monkeypatch.setattr(config, "get_client", lambda: client)
    assert list(api.stream_chat(model, HISTORY)) == ["Full reply"]
    assert calls[0]["model"] == model
