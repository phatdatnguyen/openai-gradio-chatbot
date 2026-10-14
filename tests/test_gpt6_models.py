"""GPT-6 request contracts from the official model pages; no paid API calls."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import api, config, tokens


MODELS = ("gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol")
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
HISTORY = [{"role": "user", "content": "Explain how retrieval works."}]


def build_call(model, *, effort="auto", search="None", history=None):
    return api._build_call(
        model, HISTORY if history is None else history,
        search, 0.4, 0.8, "Be concise.", effort,
    )


@pytest.mark.parametrize("model", MODELS)
def test_gpt6_models_have_documented_shared_capabilities(model):
    assert model in config.MODEL_CHOICES
    assert config.get_max_context_tokens(model) == 1_050_000
    assert config.get_max_context_tokens(model, web_search=True) == 128_000
    assert config.max_output_tokens(model) == 128_000
    assert config.supports_vision(model)
    assert config.supports_streaming(model)
    assert config.supports_web_search(model)
    assert config.supports_system_message(model)
    assert config.is_reasoning_model(model)
    assert tokens.get_encoding(model).name == "o200k_base"


def test_model_tables_remain_consistent_and_default_is_gpt6_luna():
    assert config.check_model_tables()
    assert config.DEFAULT_MODEL == "gpt-6-luna"


def test_gpt6_models_are_selectable_in_both_built_tabs_without_opening_storage(monkeypatch):
    import importlib

    from chatbot import history_store

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(history_store, "get_history_file_list", lambda: [])
    rag_ui = importlib.import_module("rag_chatbot.chatbot")

    def unexpected_service(*args, **kwargs):
        pytest.fail("Constructing model controls must not open a workspace or database")

    monkeypatch.setattr(rag_ui, "get_service", unexpected_service)
    app = importlib.import_module("webui").create_app()
    for dropdown in (app.chat_tab.llm_model, app.rag_tab.model):
        assert set(MODELS) <= {value for label, value in dropdown.choices}
        assert dropdown.value == config.DEFAULT_MODEL
    for dropdown in (app.chat_tab.reasoning_effort, app.rag_tab.reasoning_effort):
        assert [value for label, value in dropdown.choices] == config.reasoning_effort_choices(config.DEFAULT_MODEL)
        assert dropdown.value in {value for label, value in dropdown.choices}
        assert dropdown.interactive


@pytest.mark.parametrize("model", MODELS)
def test_gpt6_effort_choices_do_not_offer_unsupported_values(model):
    expected = ["auto", *REASONING_EFFORTS]
    if model in ("gpt-6-luna", "gpt-6-sol"):
        expected.insert(1, "none")
    assert config.reasoning_effort_choices(model) == expected


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("search", ["None", "high"])
def test_gpt6_routes_through_responses_with_search_and_instructions(model, search):
    surface, kwargs = build_call(model, effort="max", search=search)
    assert surface == api.RESPONSES_SURFACE
    assert kwargs["model"] == model
    assert kwargs["input"] == HISTORY
    assert kwargs["instructions"] == "Be concise."
    assert kwargs["reasoning"] == {"effort": "max"}
    assert "messages" not in kwargs and "reasoning_effort" not in kwargs
    if search == "None":
        assert "tools" not in kwargs
    else:
        assert kwargs["tools"] == [{"type": "web_search", "search_context_size": "high"}]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", [None, "auto", *REASONING_EFFORTS])
def test_reasoning_and_default_efforts_omit_sampling(model, effort):
    _, kwargs = build_call(model, effort=effort)
    assert "temperature" not in kwargs and "top_p" not in kwargs
    if effort in (None, "auto"):
        assert "reasoning" not in kwargs
    else:
        assert kwargs["reasoning"] == {"effort": effort}


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6-sol"])
@pytest.mark.parametrize("search", ["None", "high"])
def test_luna_and_sol_accept_sampling_only_with_explicit_none(model, search):
    assert not config.supports_temperature(model)
    assert config.supports_temperature(model, "none")
    _, kwargs = build_call(model, effort="none", search=search)
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["temperature"] == 0.4
    assert kwargs["top_p"] == 0.8


@pytest.mark.parametrize("model,effort", [
    ("gpt-6-luna", "minimal"),
    ("gpt-6-sol", "minimal"),
    ("gpt-6.1-sol", "none"),
    ("gpt-6.1-sol", "minimal"),
])
def test_unsupported_efforts_are_rejected_before_request(model, effort):
    with pytest.raises(ValueError, match="does not support reasoning effort"):
        build_call(model, effort=effort)


@pytest.mark.parametrize("model", MODELS)
def test_gpt6_preserves_image_input_and_uses_conservative_unpublished_budget(model):
    # The vision sizing guide does not yet publish rules for these three variants.
    # Keep its existing conservative fallback rather than assuming Astra's rate.
    image_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AARAAA//8DAgH/"
        "6h1DkQAAAABJRU5ErkJggg=="
    )
    assert config.image_token_budget(model) == config.DEFAULT_IMAGE_TOKEN_BUDGET
    surface, kwargs = build_call(model, history=[{
        "role": "user", "content": "Describe this image.", "image_url": image_url,
    }])
    assert surface == api.RESPONSES_SURFACE
    assert kwargs["input"] == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Describe this image."},
            {"type": "input_image", "image_url": image_url},
        ],
    }]
