"""Session-only study actions, stale-result guards and lightweight Gradio wiring."""

from copy import deepcopy

import pytest

from rag_chatbot.tools import learning_ui, ui


@pytest.fixture
def operation():
    return {"revision": 0, "answer_revision": 0, "switching": False, "selected_ids": ["doc1"]}


@pytest.fixture
def payloads():
    base = {"title": "Study", "sources": [{"id": "c1", "document_id": "doc1", "filename": "Paper.pdf",
        "locator": "Page 2", "text": "Evidence", "source_path": "sources/doc1.pdf"}],
        "documents": [{"id": "doc1"}], "coverage": {"documents": 1, "chunks": 1, "processed_chunks": 1}}
    cards = [{"front": "First card?", "back": "First hidden answer", "source_ids": ["c1"]},
             {"front": "Second card?", "back": "Second hidden answer", "source_ids": ["c1"]}]
    questions = [{"question": "First question?", "options": ["One", "Two", "Three", "Four"],
                  "correct_option": 1, "explanation": "First hidden explanation", "source_ids": ["c1"]},
                 {"question": "Second question?", "options": ["A", "B", "C", "D"],
                  "correct_option": 2, "explanation": "Second hidden explanation", "source_ids": ["c1"]}]
    return {"flashcards": dict(deepcopy(base), tool_id="flashcards", content={"cards": cards}),
            "quizzes": dict(deepcopy(base), tool_id="quizzes", content={"questions": questions})}


@pytest.fixture(autouse=True)
def fake_services(monkeypatch):
    monkeypatch.setattr(ui, "_enabled", lambda tool_id: True)
    store = ui._artifacts()
    monkeypatch.setattr(store, "save_current_learning", lambda workspace, payload, **kwargs: deepcopy(payload))
    monkeypatch.setattr(store, "load_current_learning", lambda workspace, tool_id: None)
    monkeypatch.setattr(store, "load_current_summary", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_spreadsheet", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_mind_map", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_presentation", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_graphics", lambda workspace: None)
    monkeypatch.setattr(store, "load_current_plot", lambda workspace: None)


def start(payload, operation, workspace="Alpha"):
    output = learning_ui.render_output(workspace, payload, "Ready")
    learning_ui.publish(output[0], operation)
    return output


def generate(tool_id, operation):
    return list(ui.on_generate(tool_id, "Instructions", "", "Alpha", ["doc1"], "Research",
                               "gpt-6-luna", "medium", operation))


def test_flashcards_reveal_and_navigation_are_local(payloads, operation, monkeypatch):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Study actions cannot call the API"))
    output = start(payloads["flashcards"], operation)
    assert output[2] == "First card?"
    assert output[3] == ui.gr.update(value="", visible=False)
    assert output[4] == "Card 1 of 2"
    assert not output[6]["interactive"] and output[7]["interactive"]
    revealed = learning_ui.on_flashcard("reveal", output[0], "Alpha", operation)
    assert "First hidden answer" in revealed[3]["value"] and "[1]" in revealed[3]["value"]
    assert revealed[3]["visible"] and not revealed[5]["interactive"]
    following = learning_ui.on_flashcard("next", revealed[0], "Alpha", operation)
    assert following[2] == "Second card?" and following[3]["value"] == ""
    assert not following[3]["visible"] and not following[7]["interactive"]
    previous = learning_ui.on_flashcard("previous", following[0], "Alpha", operation)
    assert previous[2] == "First card?" and previous[3]["value"] == ""


def test_quiz_answers_hidden_until_checked_and_score_once(payloads, operation, monkeypatch):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Quiz actions cannot call the API"))
    output = start(payloads["quizzes"], operation)
    assert output[4] == "" and output[3]["value"] is None
    assert output[3]["choices"] == [("A. One", "0"), ("B. Two", "1"), ("C. Three", "2"), ("D. Four", "3")]
    assert "0 / 0 answered" in output[6]
    assert "explanation" not in str(output[2:])
    unchecked = learning_ui.on_quiz("check", None, output[0], "Alpha", operation)
    assert "Choose an answer" in unchecked[1] and operation["learning"]["quizzes"]["attempts"] == {}
    wrong = learning_ui.on_quiz("check", "0", output[0], "Alpha", operation)
    assert "Not quite" in wrong[4] and "First hidden explanation" in wrong[4] and "[1]" in wrong[4]
    assert r"Answer: B\. Two" in wrong[4]
    assert "0 / 1 answered" in wrong[6] and not wrong[3]["interactive"] and not wrong[7]["interactive"]
    repeat = learning_ui.on_quiz("check", "1", wrong[0], "Alpha", operation)
    assert repeat[3]["value"] == "0" and "0 / 1 answered" in repeat[6]
    assert operation["learning"]["quizzes"]["attempts"] == {0: 0}
    following = learning_ui.on_quiz("next", "0", repeat[0], "Alpha", operation)
    assert following[4] == "" and following[3]["value"] is None and following[7]["interactive"]
    assert following[3]["choices"] == [("A. A", "0"), ("B. B", "1"), ("C. C", "2"), ("D. D", "3")]
    correct = learning_ui.on_quiz("check", "2", following[0], "Alpha", operation)
    assert "**Correct.**" in correct[4] and "1 / 2 answered" in correct[6]
    assert r"Answer: C\. C" in correct[4]
    previous = learning_ui.on_quiz("previous", "2", correct[0], "Alpha", operation)
    assert "Not quite" in previous[4] and previous[3]["value"] == "0"
    restarted = learning_ui.on_quiz("restart", "0", previous[0], "Alpha", operation)
    assert restarted[4] == "" and "0 / 0 answered" in restarted[6]
    assert restarted[5] == "Question 1 of 2" and operation["learning"]["quizzes"]["attempts"] == {}


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
@pytest.mark.parametrize("change", ["workspace", "switching", "output", "navigation"])
def test_stale_study_actions_are_discarded(payloads, operation, tool_id, change):
    old = start(payloads[tool_id], operation)[0]
    if change == "workspace":
        operation["revision"] += 1
    elif change == "switching":
        operation["switching"] = True
    elif change == "output":
        start(payloads[tool_id], operation)
    elif tool_id == "flashcards":
        learning_ui.on_flashcard("next", old, "Alpha", operation)
    else:
        learning_ui.on_quiz("next", None, old, "Alpha", operation)
    before = deepcopy(operation)
    result = (learning_ui.on_flashcard("reveal", old, "Alpha", operation) if tool_id == "flashcards"
              else learning_ui.on_quiz("check", "1", old, "Alpha", operation))
    assert result == (ui.gr.skip(),) * ui.OUTPUT_COUNTS[tool_id]
    assert operation == before


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_interaction_rechecks_after_rendering(payloads, operation, monkeypatch, tool_id):
    old = start(payloads[tool_id], operation)[0]
    original = learning_ui._view

    def view(*args, **kwargs):
        result = original(*args, **kwargs)
        operation["revision"] += 1
        return result

    monkeypatch.setattr(learning_ui, "_view", view)
    result = (learning_ui.on_flashcard("reveal", old, "Alpha", operation) if tool_id == "flashcards"
              else learning_ui.on_quiz("check", "1", old, "Alpha", operation))
    assert result == (ui.gr.skip(),) * ui.OUTPUT_COUNTS[tool_id]
    assert not operation["learning"][tool_id]["revealed"]
    assert operation["learning"][tool_id]["attempts"] == {}


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_regeneration_saves_one_output_and_resets_study(payloads, operation, monkeypatch, tool_id):
    old = start(payloads[tool_id], operation)[0]
    operation["learning"][tool_id].update(index=1, revealed=True, attempts={0: 0})
    calls = []

    def save(workspace, payload, *, is_cancelled):
        assert not is_cancelled()
        calls.append((workspace, payload))
        return dict(payload, id="current")

    monkeypatch.setattr(ui._artifacts(), "save_current_learning", save)
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"result": payloads[tool_id]}]))
    result = generate(tool_id, operation)[-1]
    assert len(result) == ui.OUTPUT_COUNTS[tool_id]
    assert calls == [("Alpha", payloads[tool_id])]
    assert result[0]["payload"]["id"] == "current" and result[0]["token"] != old["token"]
    state = operation["learning"][tool_id]
    assert state["index"] == 0 and state["attempts"] == {} and not state["revealed"]
    assert "saved for this workspace" in result[1]


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
@pytest.mark.parametrize("item_count", [1, 20, 30.0])
def test_selected_count_reaches_generation(payloads, operation, monkeypatch, tool_id, item_count):
    captured = []

    def backend(*args, **kwargs):
        captured.append(kwargs)
        yield {"result": payloads[tool_id]}

    monkeypatch.setattr(ui, "_generate", backend)
    result = list(ui.on_generate(tool_id, "Focus on measurements", "", "Alpha", ["doc1"], "Research",
                                "gpt-6-luna", "medium", operation, item_count))
    assert "Completed" in result[-1][1]
    assert captured[0]["item_count"] == item_count
    assert type(captured[0]["item_count"]) is int
    assert captured[0]["instructions"] == "Focus on measurements"


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_invalid_count_does_not_start_generation(payloads, operation, monkeypatch, tool_id):
    old = start(payloads[tool_id], operation)[0]
    before = deepcopy(operation)
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Invalid counts cannot generate"))
    result = list(ui.on_generate(tool_id, "Instructions", "", "Alpha", ["doc1"], "Research",
                                "gpt-6-luna", "medium", operation, 0))
    assert len(result) == 1 and result[0][0] == ui.gr.skip()
    assert result[0][1] and operation == before
    assert operation["learning"][tool_id]["token"] == old["token"]


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_cancelled_generation_does_not_publish_study_state(payloads, operation, monkeypatch, tool_id):
    start(payloads[tool_id], operation)
    before = deepcopy(operation["learning"])
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"result": payloads[tool_id]}]))
    original = learning_ui.publish

    def publish(*args, **kwargs):
        operation["revision"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(learning_ui, "publish", publish)
    result = generate(tool_id, operation)
    assert len(result) == 1 and result[0][0] == ui.gr.skip()
    assert operation["learning"] == before


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_workspace_restore_and_empty_reset_clear_attempts(payloads, operation, monkeypatch, tool_id):
    old = start(payloads[tool_id], operation)[0]
    operation["learning"][tool_id].update(index=1, revealed=True, attempts={0: 1})
    monkeypatch.setattr(ui._artifacts(), "load_current_learning",
        lambda workspace, selected: payloads[selected] if workspace == "Alpha" else None)
    restored = ui.on_reset("Alpha", operation)
    offset = sum(count for name, count in list(ui.OUTPUT_COUNTS.items())[:list(ui.OUTPUT_COUNTS).index(tool_id)])
    envelope = restored[offset]
    assert envelope["token"] != old["token"]
    assert operation["learning"][tool_id]["attempts"] == {} and operation["learning"][tool_id]["index"] == 0
    operation["revision"] += 1
    empty = ui.on_reset("Beta", operation)
    assert empty[offset] is None and tool_id not in operation["learning"]
    assert learning_ui.on_export(tool_id, old, "Study", "json", "Beta", operation) == (ui.gr.skip(), None)


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_restored_learning_accepts_final_workspace_commit(payloads, operation, monkeypatch, tool_id):
    from rag_chatbot.chatbot import on_finish_workspace_switch
    operation.update(revision=2, switching=True, workspace_open_pending=False)
    monkeypatch.setattr(ui._artifacts(), "load_current_learning", lambda workspace, selected: payloads[selected])
    restored = ui.on_workspace_reset("Alpha", operation)
    offset = sum(count for name, count in list(ui.OUTPUT_COUNTS.items())[:list(ui.OUTPUT_COUNTS).index(tool_id)])
    envelope = restored[offset]
    assert not learning_ui._matches(envelope, "Alpha", operation, tool_id)
    on_finish_workspace_switch(operation)
    assert operation["revision"] == 3 and not operation["switching"]
    assert learning_ui._matches(envelope, "Alpha", operation, tool_id)
    result = (learning_ui.on_flashcard("reveal", envelope, "Alpha", operation) if tool_id == "flashcards"
              else learning_ui.on_quiz("check", "1", envelope, "Alpha", operation))
    assert result[0]["token"] == envelope["token"]
    operation.update(switching=True, revision=4, workspace_open_pending=True)
    on_finish_workspace_switch(operation)
    assert operation["switching"] and not learning_ui._matches(result[0], "Alpha", operation, tool_id)


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
def test_plugin_disabled_blocks_generation(payloads, operation, monkeypatch, tool_id):
    monkeypatch.setattr(ui, "_enabled", lambda tool_id: False)
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Disabled generation cannot call API"))
    result = generate(tool_id, operation)
    assert len(result) == 1 and "disabled" in result[0][1]


@pytest.mark.parametrize("tool_id", ["flashcards", "quizzes"])
@pytest.mark.parametrize("change", [None, "workspace", "output", "filename"])
def test_exports_use_complete_content_and_discard_stale_download(payloads, operation, monkeypatch, tmp_path, tool_id, change):
    envelope = start(payloads[tool_id], operation)[0]
    calls = []

    def export(workspace, payload, output_format, filename, *, is_cancelled):
        assert not is_cancelled()
        calls.append((workspace, payload, output_format, filename))
        if change == "workspace":
            operation["revision"] += 1
        elif change == "output":
            start(payloads[tool_id], operation)
        elif change == "filename":
            learning_ui.on_export_options(tool_id, operation)
        assert is_cancelled() == bool(change)
        return tmp_path / "Study.json"

    monkeypatch.setattr(ui._artifacts(), "export_learning", export)
    result = learning_ui.on_export(tool_id, envelope, "Study", "json", "Alpha", operation)
    assert calls == [("Alpha", payloads[tool_id], "json", "Study")]
    if change:
        assert result == (ui.gr.skip(), None)
    else:
        assert result[1] == str(tmp_path / "Study.json")


def test_cards_escape_markup_and_keep_numbered_references(payloads, operation):
    payload = payloads["flashcards"]
    payload["content"]["cards"][0]["front"] = '<script>alert(1)</script> ![x](https://example.org/image)'
    payload["content"]["cards"][0]["back"] = '<img src=x onerror="bad()">'
    result = start(payload, operation)
    assert "<script>" not in result[2] and "![x]" not in result[2]
    revealed = learning_ui.on_flashcard("reveal", result[0], "Alpha", operation)
    assert '<img' not in revealed[3]["value"].replace('\\<', '')
    assert "References: [1]" in revealed[3]["value"]


def test_learning_math_keeps_delimiters_without_model_markup():
    text = r"Use \(E = mc^{2}\), \[a < b\], and $$x_1$$. <img src=x> [link](https://example.com)"
    rendered = learning_ui._text(text)
    assert r"\(E = mc^{2}\)" in rendered and r"\[a \lt  b\]" in rendered and "$$x_1$$" in rendered
    assert "<img" not in rendered.replace(r"\<", "") and "[link]" not in rendered
    assert learning_ui._text(r"\(\href{javascript:bad()}{x}\)").startswith(r"\\\(")


def test_study_state_is_session_local(payloads, operation):
    second = deepcopy(operation)
    first_output = start(payloads["quizzes"], operation)
    second_output = start(payloads["quizzes"], second)
    learning_ui.on_quiz("check", "1", first_output[0], "Alpha", operation)
    assert second["learning"]["quizzes"]["attempts"] == {}
    assert learning_ui.on_quiz("check", "1", first_output[0], "Alpha", second) == (ui.gr.skip(),) * 17
    assert learning_ui.on_quiz("next", None, second_output[0], "Alpha", second)[5] == "Question 2 of 2"


def test_learning_ui_controls_are_bounded_and_wired_without_storage(monkeypatch):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("UI construction must stay lazy"))
    with ui.gr.Blocks() as app:
        workspace, selected, collection = ui.gr.State("Alpha"), ui.gr.State(["doc1"]), ui.gr.State("Research")
        model, reasoning, operation = ui.gr.State("gpt-6-luna"), ui.gr.State("medium"), ui.gr.State({"revision": 0})
        controls = ui.build_tools(workspace, selected, collection, model, reasoning, operation)
    for tool_id in ("flashcards", "quizzes"):
        tool = controls.controls[tool_id]
        assert tool.instructions.value == learning_ui.DEFAULT_INSTRUCTIONS[tool_id]
        assert tool.item_count.value == 20 and tool.item_count.minimum == 1 and tool.item_count.step == 1
        assert tool.item_count.maximum == (60 if tool_id == "flashcards" else 40)
        assert tool.instructions.scale == 3 and tool.item_count.scale == 1
        assert tool.filename.value == ui.TOOL_LABELS[tool_id]
        if tool_id == "quizzes":
            assert tool.output_format.choices == [("Aiken (.txt)", "aiken"), ("Moodle XML (.xml)", "moodle_xml"),
                                                  ("Markdown", "markdown"), ("CSV", "csv"), ("JSON", "json")]
            assert tool.output_format.value == "aiken"
            assert tool.options.elem_classes == ["rag-quiz-options"]
        else:
            assert tool.output_format.choices == [(item, item) for item in ("markdown", "csv", "json")]
            assert tool.output_format.value == "markdown"
        assert tool.main_column.scale == 3 and tool.sidebar.scale == 1
        assert tool.excerpt.lines == tool.excerpt.max_lines == 6 and tool.preview.max_height == 300
        assert tool.preview.latex_delimiters == ui.LATEX_DELIMITERS
        assert len(tool.outputs) == ui.OUTPUT_COUNTS[tool_id]
        assert not hasattr(tool, "saved") and not hasattr(tool, "save_button")
        assert app.fns[tool.generation["id"]].outputs == tool.outputs
        assert app.fns[tool.generation["id"]].inputs == [tool.instructions, tool.columns, workspace, selected,
                                                        collection, model, reasoning, operation, tool.item_count]
        assert app.fns[tool.export_event["id"]].inputs == [tool.payload_state, tool.filename, tool.output_format, workspace, operation]
        assert tool.download_event["trigger_only_on_success"]
        assert tool.download_event["trigger_after"] == tool.export_event["id"]
    flashcards, quizzes = controls.controls["flashcards"], controls.controls["quizzes"]
    assert not flashcards.answer.visible and flashcards.answer.max_height == 300
    assert flashcards.answer.latex_delimiters == quizzes.feedback.latex_delimiters == ui.LATEX_DELIMITERS
    assert app.fns[flashcards.reveal_event["id"]].fn.func is learning_ui.on_flashcard
    assert app.fns[quizzes.check_event["id"]].fn.func is learning_ui.on_quiz
    assert app.fns[quizzes.check_event["id"]].inputs == [quizzes.options, quizzes.payload_state, workspace, operation]
    assert app.fns[quizzes.restart_event["id"]].fn.args == ("restart",)
