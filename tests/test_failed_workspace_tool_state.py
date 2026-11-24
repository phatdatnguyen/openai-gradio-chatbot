"""A failed workspace switch retains usable results, never in-flight work."""

from copy import deepcopy

import pytest

from rag_chatbot.tools import graphics_ui, learning_ui, plot_ui, presentation_ui, review_ui, ui
from rag_chatbot.workspaces import create_workspace, workspace_path


TOOL_IDS = ("presentation", "graphics", "plot", "flashcards", "quizzes", "compare", "evidence_check")


@pytest.fixture
def retained():
    create_workspace("Research")
    path = workspace_path("Research") / "sources" / "report.txt"
    path.write_text("Evidence here.", encoding="utf-8")
    source = {"id": "c1", "document_id": "d1", "filename": "report.txt", "locator": "Section 1",
              "text": "Evidence here.", "source_path": "sources/report.txt"}
    operation = {"revision": 4, "answer_revision": 2, "switching": False,
                 "summary_editing": True, "selected_ids": ["d1"]}
    envelopes = {}
    for tool_id in TOOL_IDS:
        payload = {"tool_id": tool_id, "title": "Saved result", "sources": [source], "documents": [{"id": "d1"}]}
        if tool_id == "presentation":
            payload["content"] = {"slides": [
                {"title": f"Slide {index}", "bullets": ["Evidence"], "notes": "Speaker notes", "source_ids": ["c1"]}
                for index in range(1, 4)]}
        elif tool_id == "flashcards":
            payload["content"] = {"cards": [{"front": f"Question {index}", "back": "Answer", "source_ids": ["c1"]}
                                            for index in range(1, 4)]}
        elif tool_id == "quizzes":
            payload["content"] = {"questions": [
                {"question": f"Question {index}", "options": ["One", "Two", "Three", "Four"],
                 "correct_option": 1, "explanation": "Evidence", "source_ids": ["c1"]} for index in range(1, 4)]}
        envelope = {"workspace": "Research", "payload": payload, "token": tool_id + "-token", "view_revision": 0}
        publisher = (learning_ui if tool_id in ("flashcards", "quizzes") else
                     review_ui if tool_id in review_ui.LABELS else
                     {"graphics": graphics_ui, "plot": plot_ui, "presentation": presentation_ui}[tool_id])
        publisher.publish(envelope, operation)
        envelopes[tool_id] = envelope
    return operation, envelopes, path


def _failed_switch(operation):
    before = operation["revision"]
    ui.cancel_tool_job(operation)
    operation.update(revision=before + 2, switching=True, workspace_open_pending=False)
    rebound = ui.rebind_retained_workspace("Research", operation, before, before + 3)
    operation.update(revision=before + 3, switching=False)
    return rebound


def test_rebind_preserves_outputs_edits_study_progress_and_tokens(retained):
    operation, envelopes, _ = retained
    slides = presentation_ui.on_navigate("next", None, envelopes["presentation"], "Research", operation)
    cards = learning_ui.on_flashcard("reveal", envelopes["flashcards"], "Research", operation)
    quiz = learning_ui.on_quiz("check", "1", envelopes["quizzes"], "Research", operation)
    before = {key: deepcopy(operation[key]) for key in ("presentation", "graphics", "plot", "learning", "compare", "evidence_check")}
    assert _failed_switch(operation) == 7
    assert operation["summary_editing"] is True
    for tool_id in TOOL_IDS:
        state = operation["learning"][tool_id] if tool_id in ("flashcards", "quizzes") else operation[tool_id]
        old = before["learning"][tool_id] if tool_id in ("flashcards", "quizzes") else before[tool_id]
        assert state == {**old, "workspace_revision": 7}
    assert presentation_ui.on_navigate("next", None, slides[0], "Research", operation)[4] == "Slide 3 of 3"
    assert learning_ui.on_flashcard("next", cards[0], "Research", operation)[4] == "Card 2 of 3"
    assert "1 / 1 answered" in learning_ui.on_quiz("next", None, quiz[0], "Research", operation)[6]


@pytest.mark.parametrize("tool_id", TOOL_IDS)
def test_reference_controls_work_again_after_retaining_workspace(retained, tool_id):
    operation, envelopes, path = retained
    _failed_switch(operation)
    envelope = envelopes[tool_id]
    if tool_id in review_ui.LABELS:
        result = review_ui.on_reference(tool_id, "0", envelope, "Research", operation)
    else:
        view = learning_ui if tool_id in ("flashcards", "quizzes") else {
            "presentation": presentation_ui, "graphics": graphics_ui, "plot": plot_ui}[tool_id]
        result = view.on_reference("0", envelope, "Research", operation)
    assert result == ("Evidence here.", str(path))


def test_rebind_ignores_old_other_and_unpublished_states(retained):
    operation, _, _ = retained
    operation["graphics"]["workspace"] = "Other"
    operation["plot"]["workspace_revision"] = 1
    operation["compare"].pop("token")
    operation["tool_job"] = {"id": "stopped", "workspace": "Research", "cancelled": True}
    untouched = {key: deepcopy(operation[key]) for key in ("graphics", "plot", "compare", "tool_job")}
    assert _failed_switch(operation) == 4
    for key, value in untouched.items():
        assert operation[key] == value


@pytest.mark.parametrize("changes", [{"revision": 99}, {"switching": False}, {"workspace_open_pending": True}])
def test_stale_failed_open_cannot_rebind_newer_state(retained, changes):
    operation, _, _ = retained
    operation.update(revision=6, switching=True, workspace_open_pending=False)
    operation.update(changes)
    before = deepcopy(operation)
    assert ui.rebind_retained_workspace("Research", operation, 4, 7) == 0
    assert operation == before


@pytest.mark.parametrize("tool_id", ["presentation", "flashcards", "quizzes"])
def test_navigation_started_before_failed_switch_does_not_publish_old_state(retained, monkeypatch, tool_id):
    operation, envelopes, _ = retained
    module = presentation_ui if tool_id == "presentation" else learning_ui
    original = module._view

    def interrupted(*args, **kwargs):
        output = original(*args, **kwargs)
        assert _failed_switch(operation) == 7
        return output

    monkeypatch.setattr(module, "_view", interrupted)
    envelope = envelopes[tool_id]
    if tool_id == "presentation":
        result = module.on_navigate("next", None, envelope, "Research", operation)
        state = operation[tool_id]
    elif tool_id == "flashcards":
        result = module.on_flashcard("next", envelope, "Research", operation)
        state = operation["learning"][tool_id]
    else:
        result = module.on_quiz("check", "1", envelope, "Research", operation)
        state = operation["learning"][tool_id]
    assert result == (ui.gr.skip(),) * ui.OUTPUT_COUNTS[tool_id]
    assert state["workspace_revision"] == 7 and state["index"] == 0 and state["view_revision"] == 0
    assert not state.get("attempts")


@pytest.mark.parametrize("tool_id", TOOL_IDS)
def test_export_started_before_failed_switch_stays_cancelled(retained, monkeypatch, tool_id):
    operation, envelopes, _ = retained
    seen = []

    def interrupted(*args, is_cancelled, **kwargs):
        assert not is_cancelled()
        _failed_switch(operation)
        seen.append(is_cancelled())
        raise RuntimeError("Cancelled export")

    exporter = ("export_learning" if tool_id in ("flashcards", "quizzes") else
                "export_review" if tool_id in review_ui.LABELS else
                "export_current_graphics" if tool_id == "graphics" else "export_" + tool_id)
    monkeypatch.setattr(ui._artifacts(), exporter, interrupted)
    envelope = envelopes[tool_id]
    if tool_id in ("flashcards", "quizzes"):
        result = learning_ui.on_export(tool_id, envelope, "Saved", "json", "Research", operation)
    elif tool_id in review_ui.LABELS:
        result = review_ui.on_export(tool_id, envelope, "Saved", "json", "Research", operation)
    else:
        module = {"presentation": presentation_ui, "graphics": graphics_ui, "plot": plot_ui}[tool_id]
        result = module.on_export(envelope, "Saved", "json", "Research", operation)
    assert seen == [True] and result == (ui.gr.skip(), None)


def test_generation_captured_before_failed_switch_cannot_replace_retained_output(retained, monkeypatch):
    operation, _, _ = retained
    # Summary editing is independently preserved by the helper; use Plot to
    # exercise a running job without changing that unsaved editor.
    monkeypatch.setattr(ui, "_enabled", lambda _: True)
    checked = []

    def backend(*args, is_cancelled, **kwargs):
        assert not is_cancelled()
        yield {"status": "Waiting", "result": None}
        checked.append(is_cancelled())
        yield {"result": {"tool_id": "plot"}}

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui, "_render", lambda *a, **k: pytest.fail("Cancelled model result must not publish"))
    generation = ui.on_generate("plot", "Instructions", "", "Research", ["d1"], "Science",
                                "gpt-6-luna", "medium", operation)
    next(generation)
    next(generation)
    job = operation["tool_job"]
    assert _failed_switch(operation) == 7
    assert job["cancelled"] is True
    assert list(generation) == [] and checked == [True]
    assert operation["plot"]["token"] == "plot-token"
    assert operation["plot"]["workspace_revision"] == 7


@pytest.mark.parametrize("tool_id", TOOL_IDS)
def test_reference_started_before_failed_switch_stays_cancelled(retained, monkeypatch, tool_id):
    operation, envelopes, _ = retained
    original = ui.on_reference

    def interrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        _failed_switch(operation)
        return result

    monkeypatch.setattr(ui, "on_reference", interrupted)
    envelope = envelopes[tool_id]
    if tool_id in review_ui.LABELS:
        result = review_ui.on_reference(tool_id, "0", envelope, "Research", operation)
    else:
        module = learning_ui if tool_id in ("flashcards", "quizzes") else {
            "presentation": presentation_ui, "graphics": graphics_ui, "plot": plot_ui}[tool_id]
        result = module.on_reference("0", envelope, "Research", operation)
    assert result == (ui.gr.skip(), ui.gr.skip())
