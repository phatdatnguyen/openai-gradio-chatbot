"""Shared Chat/Graphics image requests, with mocked clients and no paid calls."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import api, config


MODELS_25 = ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare")
IMAGE_DATA = "aW1hZ2U="


def options(**overrides):
    result = {
        "image_model": config.DEFAULT_IMAGE_MODEL,
        "image_size": "auto",
        "image_quality": "auto",
        "image_background": "auto",
        "image_output_format": "png",
        "image_output_compression": 100,
        "image_moderation": "auto",
        "image_input_fidelity": "high",
    }
    result.update(overrides)
    return result


def client(monkeypatch, response=None):
    calls = []
    if response is None:
        response = NS(data=[NS(b64_json=IMAGE_DATA, revised_prompt=None)])

    def generate(**kwargs):
        calls.append(("generate", kwargs))
        return response

    def edit(**kwargs):
        image = kwargs.pop("image")
        assert not image.closed
        calls.append(("edit", kwargs, image.read(), image))
        return response

    monkeypatch.setattr(config, "get_client", lambda: NS(images=NS(generate=generate, edit=edit)))
    return calls


@pytest.mark.parametrize("model", MODELS_25)
@pytest.mark.parametrize("quality", ["xhigh", "max"])
def test_new_models_generate_with_documented_controls(monkeypatch, model, quality):
    calls = client(monkeypatch)
    result = api.generate_image_asset("Create a diagram", None, **options(
        image_model=model, image_size="3840x2160", image_quality=quality,
        image_background="transparent", image_output_format="webp",
        image_output_compression=85.0, image_moderation="low",
    ))
    assert result == {"b64_json": IMAGE_DATA, "output_format": "webp", "revised_prompt": None}
    assert calls == [("generate", {
        "model": model, "prompt": "Create a diagram", "size": "3840x2160",
        "quality": quality, "background": "transparent", "output_format": "webp",
        "output_compression": 85, "moderation": "low",
    })]


@pytest.mark.parametrize("model", [*MODELS_25, "gpt-image-2", "gpt-image-1.5"])
def test_edit_routes_fidelity_by_capability_and_closes_input(monkeypatch, tmp_path, model):
    calls = client(monkeypatch)
    path = tmp_path / "input.png"
    path.write_bytes(b"input image")
    result = api.generate_image_asset("Edit this diagram", path, **options(image_model=model))
    assert result["output_format"] == "png"
    assert calls[0][0] == "edit"
    kwargs = calls[0][1]
    assert ("input_fidelity" in kwargs) is (model == "gpt-image-1.5")
    assert "output_compression" not in kwargs
    assert "moderation" not in kwargs
    assert calls[0][2] == b"input image"
    assert calls[0][3].closed


@pytest.mark.parametrize("changes,match", [
    ({"image_model": "unknown"}, "supported image model"),
    ({"image_model": []}, "supported image model"),
    ({"image_size": "999999x999999"}, "size"),
    ({"image_model": "gpt-image-1.5", "image_size": "3840x2160"}, "size"),
    ({"image_model": "gpt-image-2", "image_quality": "max"}, "quality"),
    ({"image_quality": "invalid"}, "quality"),
    ({"image_background": "invalid"}, "background"),
    ({"image_model": "gpt-image-2", "image_background": "transparent"}, "background"),
    ({"image_background": "transparent", "image_output_format": "jpeg"}, "PNG or WebP"),
    ({"image_output_format": "svg"}, "output format"),
    ({"image_moderation": "off"}, "moderation"),
    ({"image_input_fidelity": "low"}, "input fidelity"),
    ({"image_output_compression": -1}, "compression"),
    ({"image_output_compression": 101}, "compression"),
    ({"image_output_compression": 84.2}, "compression"),
    ({"image_output_compression": float("nan")}, "compression"),
    ({"image_output_compression": float("inf")}, "compression"),
    ({"image_output_compression": True}, "compression"),
    ({"image_output_compression": "80"}, "compression"),
])
def test_invalid_options_fail_before_client_or_paid_call(monkeypatch, changes, match):
    monkeypatch.setattr(config, "get_client", lambda: pytest.fail("Invalid settings must not obtain a client"))
    with pytest.raises(ValueError, match=match):
        api.generate_image_asset("Create a diagram", None, **options(**changes))


@pytest.mark.parametrize("prompt", [None, "", "  ", "x" * 32001])
def test_invalid_prompt_fails_before_paid_call(monkeypatch, prompt):
    monkeypatch.setattr(config, "get_client", lambda: pytest.fail("Invalid prompt must not obtain a client"))
    with pytest.raises(ValueError, match="prompt"):
        api.generate_image_asset(prompt, None, **options())


@pytest.mark.parametrize("image_data", [None, "", "  ", 7])
def test_missing_image_data_fails_without_retry(monkeypatch, image_data):
    calls = client(monkeypatch, NS(data=[NS(b64_json=image_data)]))
    with pytest.raises(RuntimeError, match="returned no image"):
        api.generate_image_asset("Create a diagram", None, **options())
    assert len(calls) == 1


def test_empty_image_response_fails(monkeypatch):
    calls = client(monkeypatch, NS(data=[]))
    with pytest.raises(RuntimeError, match="returned no image"):
        api.generate_image_asset("Create a diagram", None, **options())
    assert len(calls) == 1


def test_server_format_is_used_in_chat_and_revised_prompt_is_preserved(monkeypatch):
    calls = client(monkeypatch, NS(data=[NS(b64_json=IMAGE_DATA, revised_prompt="Detailed diagram")],
                                   output_format="webp"))
    result = api.generate_image("Create a diagram", None, **options())
    assert result == (
        "Generated image with `gpt-image-2.5-sunburst`.\n\n"
        "Revised prompt:\nDetailed diagram\n\n"
        f"![Image](data:image/webp;base64,{IMAGE_DATA})"
    )
    assert len(calls) == 1


@pytest.mark.parametrize("model", MODELS_25)
@pytest.mark.parametrize("quality", ["xhigh", "max"])
def test_installed_sdk_serializes_new_options_without_network(monkeypatch, model, quality):
    import httpx
    from openai import OpenAI

    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "created": 0, "output_format": "webp",
            "data": [{"b64_json": IMAGE_DATA}],
        })

    with OpenAI(api_key="test-only-no-network", http_client=httpx.Client(
            transport=httpx.MockTransport(handle))) as sdk:
        monkeypatch.setattr(config, "get_client", lambda: sdk)
        result = api.generate_image_asset("Create a diagram", None, **options(
            image_model=model, image_quality=quality, image_size="3840x2160",
            image_background="transparent", image_output_format="webp",
        ))
    assert len(calls) == 1
    assert calls[0]["model"] == model
    assert calls[0]["quality"] == quality
    assert calls[0]["size"] == "3840x2160"
    assert calls[0]["background"] == "transparent"
    assert result["b64_json"] == IMAGE_DATA
    assert result["output_format"] == "webp"


def test_chat_controls_default_to_sunburst_and_update_new_options(monkeypatch):
    from chatbot import chatbot, history_store

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(history_store, "get_history_file_list", lambda: [])
    monkeypatch.setattr(config, "get_client", lambda: pytest.fail("Building image controls must be offline"))
    with chatbot.gr.Blocks() as demo:
        chatbot.build_tab()
    model = next(component for component in demo.config["components"]
                 if component["props"].get("label") == "Image model")
    assert model["props"]["value"] == "gpt-image-2.5-sunburst"
    assert set(MODELS_25) <= {value for label, value in model["props"]["choices"]}
    for name in MODELS_25:
        size, quality, background, fidelity = chatbot.build_image_option_updates(name)
        assert "3840x2160" in {value for label, value in size.choices}
        assert {"xhigh", "max"} <= {value for label, value in quality.choices}
        assert "transparent" in {value for label, value in background.choices}
        assert not fidelity.interactive
