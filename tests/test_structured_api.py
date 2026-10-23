"""Strict Studio requests exercise both API surfaces without paid requests."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatbot import api, config


SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}},
          "required": ["title"], "additionalProperties": False}
HISTORY = [{"role": "user", "content": "Summarize this evidence."}]


def install(monkeypatch, model, text='{"title":"Result"}', *, refusal=False,
            status="completed", finish="stop"):
    calls = []
    part = (NS(type="refusal", refusal="Declined") if refusal
            else NS(type="output_text", text=text))
    response = NS(status=status, output=[NS(type="message", content=[part])],
                  error=NS(message="server failed"),
                  incomplete_details=NS(reason="max_output_tokens"))
    chat = NS(choices=[NS(finish_reason=finish, message=NS(
        content=text, refusal="Declined" if refusal else None))])

    def create(**kwargs):
        calls.append(kwargs)
        return response if config.uses_responses_api(model) else chat

    monkeypatch.setattr(config, "get_client", lambda: NS(
        responses=NS(create=create), chat=NS(completions=NS(create=create))))
    return calls


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gpt-4o"])
def test_strict_generation_routes_and_preserves_model_settings(monkeypatch, model):
    calls = install(monkeypatch, model)
    result = api.generate_structured(model, HISTORY, SCHEMA, system_prompt="Use the sources.",
                                     schema_name="summary", max_output_tokens=4000)
    assert result == {"title": "Result"}
    request = calls[0]
    assert len(calls) == 1
    assert request["model"] == model
    assert "tools" not in request
    if config.uses_responses_api(model):
        spec = request["text"]["format"]
        assert spec["type"] == "json_schema"
        assert request["reasoning"] == {"effort": "medium"}
        assert request["max_output_tokens"] == 4000
        assert request["instructions"] == "Use the sources."
        assert "temperature" not in request
    else:
        spec = request["response_format"]["json_schema"]
        assert request["max_completion_tokens"] == 4000
        assert request["messages"][0] == {"role": "system", "content": "Use the sources."}
    assert spec["strict"] is True and spec["schema"] == SCHEMA
    assert spec["name"] == "summary"


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-4o"])
def test_refusal_is_never_saved_as_a_tool_result(monkeypatch, model):
    calls = install(monkeypatch, model, refusal=True)
    with pytest.raises(RuntimeError, match="declined"):
        api.generate_structured(model, HISTORY, SCHEMA)
    assert len(calls) == 1


@pytest.mark.parametrize("model,kwargs", [
    ("gpt-6-luna", {"status": "incomplete"}),
    ("gpt-6-luna", {"status": "failed"}),
    ("gpt-4o", {"finish": "length"}),
    ("gpt-4o", {"finish": "content_filter"}),
])
def test_truncation_is_not_success_even_when_json_is_parseable(monkeypatch, model, kwargs):
    calls = install(monkeypatch, model, **kwargs)
    with pytest.raises(RuntimeError, match="incomplete|failed"):
        api.generate_structured(model, HISTORY, SCHEMA)
    assert len(calls) == 1  # No silent retry or weaker-format fallback.


@pytest.mark.parametrize("text", ["not JSON", '```json\n{"title":"x"}\n```',
                                 '{"title":"x","title":"y"}', '{"title":NaN}',
                                 '{"title":Infinity}', '{"title":1e400}', '[]', 'null'])
def test_malformed_json_is_rejected(monkeypatch, text):
    install(monkeypatch, "gpt-6-luna", text)
    with pytest.raises(RuntimeError, match="JSON|object"):
        api.generate_structured("gpt-6-luna", HISTORY, SCHEMA)


@pytest.mark.parametrize("model", ["unknown", "gpt-4", "gpt-3.5-turbo",
                                  "gpt-5.1-codex", "o3-deep-research"])
def test_unsupported_models_fail_before_accessing_client(monkeypatch, model):
    def unexpected():
        pytest.fail("Unsupported tool must not access the client")
    monkeypatch.setattr(config, "get_client", unexpected)
    with pytest.raises(ValueError, match="not enabled"):
        api.generate_structured(model, HISTORY, SCHEMA)


def test_source_input_cannot_be_silently_trimmed(monkeypatch):
    calls = install(monkeypatch, "gpt-6-luna")
    monkeypatch.setattr(config, "get_max_context_tokens", lambda *a: 1000)
    history = [{"role": "user", "content": "Older source " * 1500}, *HISTORY]
    with pytest.raises(api.InputTooLargeError):
        api.generate_structured("gpt-6-luna", history, SCHEMA)
    assert calls == []


@pytest.mark.parametrize("status", ["cancelled", "queued", "in_progress"])
def test_nonterminal_or_cancelled_result_cannot_become_an_artifact(monkeypatch, status):
    install(monkeypatch, "gpt-6-luna", status=status)
    with pytest.raises(RuntimeError, match="did not complete"):
        api.generate_structured("gpt-6-luna", HISTORY, SCHEMA)


@pytest.mark.parametrize("kwargs", [{"schema_name": "bad name"}, {"max_output_tokens": 0},
                                    {"max_output_tokens": True}, {"max_output_tokens": 50000}])
def test_invalid_options_fail_without_request(monkeypatch, kwargs):
    calls = install(monkeypatch, "gpt-6-luna")
    with pytest.raises(ValueError):
        api.generate_structured("gpt-6-luna", HISTORY, SCHEMA, **kwargs)
    assert not calls
