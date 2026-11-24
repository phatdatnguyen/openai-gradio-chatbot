"""Queued browser chat actions must keep their original workspace scope."""

import asyncio
from types import SimpleNamespace

import gradio as gr
from gradio.state_holder import SessionState
import pytest

from rag_chatbot import chatbot as ui


@pytest.mark.parametrize("callback", [ui.on_answer, ui.on_save_chat, ui.on_load_chat,
                                     ui.on_new_chat, ui.on_stop_chat])
def test_old_click_cannot_act_on_new_workspace_states(monkeypatch, callback):
    monkeypatch.setattr(ui, "get_service", lambda *args: pytest.fail("Stale clicks must not access storage or models"))
    with gr.Blocks() as app:
        controls = ui.build_tab()
    state = SessionState(app)
    state[controls.workspace_state._id] = "Beta"
    operation = {"revision": 6, "switching": False, "chat_workspace_epoch": "beta-epoch"}
    state[controls.operation_state._id] = operation
    history = [{"role": "assistant", "content": "Beta's existing answer"}]
    state[controls.history_state._id] = history
    function = next(item for item in app.fns.values() if item.fn is callback)
    values = {controls.chat_workspace_epoch._id: "alpha-epoch", controls.question._id: "Old question",
              controls.saved_chat._id: "Same name", controls.chat_name._id: "Old snapshot"}

    async def invoke():
        result = await app.process_api(function,
            [values.get(component._id, component.value) if not component.stateful else None
             for component in function.inputs], state=state)
        if result.get("iterator") is not None:
            await result["iterator"].aclose()
        return result

    result = asyncio.run(invoke())
    assert all(value is None if component.stateful else value in (
        gr.skip(), {"__type__": "update", "props": gr.skip()})
        for component, value in zip(function.outputs, result["data"]))
    assert state[controls.history_state._id] == history
    assert operation == {"revision": 6, "switching": False, "chat_workspace_epoch": "beta-epoch"}


def test_full_context_ui_forwards_mode_and_cooperative_cancellation(monkeypatch, tmp_path):
    operation = {"revision": 0, "chat_workspace_epoch": "visible-epoch"}
    calls = []

    def answer(*args, **kwargs):
        calls.append(kwargs)
        assert not kwargs["is_cancelled"]()
        yield [{"role": "assistant", "content": "Complete answer", "sources": []}], []

    monkeypatch.setattr(ui, "get_service", lambda workspace:
        SimpleNamespace(answer=answer, settings=SimpleNamespace(data_dir=tmp_path)))
    result = list(ui.on_answer("Read everything", [], "Default", ["d1"], "gpt-4o", "auto",
        "Alpha", operation, context_mode="full", workspace_epoch="visible-epoch"))
    assert "autosaved" in result[-1][6]
    assert calls[0]["context_mode"] == "full"
    operation["revision"] += 1
    assert calls[0]["is_cancelled"]()


def test_full_context_rejects_empty_selection_even_with_web_search(monkeypatch):
    monkeypatch.setattr(ui, "get_service", lambda *args: pytest.fail("Empty full context must not incur charges"))
    result = list(ui.on_answer("Read everything", [], "Default", [], "gpt-4o", "auto",
        "Alpha", {"revision": 0}, web_search="medium", context_mode="full"))
    assert "No sources selected" in result[-1][6]
