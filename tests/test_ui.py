"""Offline regression tests for UI state and event wiring."""

import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api
import history_store


@pytest.fixture
def ui(monkeypatch):
    # Build the UI without inspecting the user's saved conversations or sending
    # Gradio analytics. OpenAI calls are stubbed individually below.
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(history_store, "get_history_file_list", lambda: [])
    return importlib.import_module("webui")


def _send(ui, history, **overrides):
    values = {
        "llm_model": "gpt-4o", "web_search": "off", "temperature": 1,
        "top_p": 1, "text": "Please help with this draft", "image": None,
        "document": None, "url": None, "history": history,
        "generate_image": False, "system_prompt": "", "reasoning_effort": None,
        "stream_output": False, "image_model": "gpt-image-1", "image_size": "auto",
        "image_quality": "auto", "image_background": "auto",
        "image_output_format": "png", "image_output_compression": 100,
        "image_moderation": "auto", "image_input_fidelity": None,
    }
    values.update(overrides)
    return ui.on_user_input(**values)


@pytest.mark.parametrize("stream_output", [False, True])
def test_failed_send_preserves_prompt_for_copying_without_resending(ui, monkeypatch,
                                                                  stream_output):
    previous = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    warnings = []
    monkeypatch.setattr(ui.gr, "Warning", warnings.append)

    def fail(*args, **kwargs):
        raise RuntimeError("Test request failed")

    def fail_mid_stream(*args, **kwargs):
        yield "Partial response"
        fail()

    monkeypatch.setattr(api, "send_chat", fail)
    monkeypatch.setattr(api, "stream_chat", fail_mid_stream)

    final = list(_send(ui, previous, stream_output=stream_output))[-1]
    state, display, *inputs = final
    assert state == previous
    assert len(previous) == 2
    assert display[:-2] == previous
    assert display[-2] == {"role": "user", "content": "Please help with this draft"}
    assert "Request failed" in display[-1]["content"]
    assert "not added to the conversation" in display[-1]["content"]
    assert all(value == ui.gr.skip() for value in inputs)
    assert warnings == ["Test request failed"]


def test_failed_load_keeps_active_chat(ui, monkeypatch):
    def fail(name):
        raise FileNotFoundError("Saved file is missing")

    monkeypatch.setattr(history_store, "load_history", fail)
    state, display, status = ui.on_load_history("missing")
    assert state == ui.gr.skip()
    assert display == ui.gr.skip()
    assert "Saved file is missing" in status


def test_successful_load_replaces_active_chat(ui, monkeypatch):
    saved = [{"role": "user", "content": "Saved question"}]
    monkeypatch.setattr(history_store, "load_history", lambda name: (saved, "Loaded"))
    assert ui.on_load_history("saved") == (saved, saved, "Loaded")


def test_send_triggers_share_pending_event_and_chat_changes_cancel_it(ui):
    dependencies = ui.demo.config["dependencies"]
    send = next(item for item in dependencies if item["id"] == ui.send_event["id"])
    assert set(map(tuple, send["targets"])) == {
        (ui.text_input._id, "submit"), (ui.send_button._id, "click"),
    }
    assert send["trigger_mode"] == "once"
    for button in (ui.stop_button, ui.new_chat_button, ui.load_button):
        assert any(
            (button._id, "click") in map(tuple, item["targets"])
            and send["id"] in item["cancels"]
            for item in dependencies
        )
