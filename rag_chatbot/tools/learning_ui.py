"""Local study interactions; generated answers never require another model call."""

from copy import deepcopy
from functools import partial
import re
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from chatbot.log import logger
from .learning import DEFAULT_LEARNING_COUNT, MAX_FLASHCARDS, MAX_QUIZ_QUESTIONS


DEFAULT_INSTRUCTIONS = {
    "flashcards": "Please help me create flashcards from the documents",
    "quizzes": "Please help me create a quiz from the documents",
}


def _ui():
    from . import ui
    return ui


def _text(value):
    # Keep prose, links and HTML literal while letting the shared Gradio math
    # renderer handle equations. Never opt into KaTeX's trusted HTML/URL mode.
    def literal(text):
        return re.sub(r"([\\`*_{}\[\]<>()#+.!|~$-])", r"\\\1", text)

    parts = re.split(r"(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$)", str(value))
    result = []
    for index, part in enumerate(parts):
        if index % 2 and not re.search(r"\\(?:href|url|includegraphics|html\w*)\b", part):
            # Inequalities remain equations, and raw HTML delimiters cannot
            # enter Markdown even if the equation itself is malformed.
            result.append(part.replace("<", r"\lt ").replace(">", r"\gt "))
        else:
            result.append(literal(part))
    return "".join(result)


def _citations(payload, item):
    numbers = {source["id"]: index + 1 for index, source in enumerate(payload.get("sources", []))}
    return "References: " + " ".join(f"[{numbers[source_id]}]" for source_id in item["source_ids"])


def _initial(envelope):
    return {"token": envelope["token"], "workspace": envelope["workspace"], "view_revision": 0,
            "index": 0, "revealed": False, "attempts": {}}


def publish(envelope, operation_state, *, is_cancelled=None):
    """Called only after generation/storage or workspace restore remains current."""
    ui = _ui()
    tool_id = envelope["payload"]["tool_id"]
    with ui._JOB_LOCK:
        if is_cancelled is not None and is_cancelled():
            return False
        state = _initial(envelope)
        # Workspace restoration runs before on_finish_workspace_switch applies
        # its final revision increment. Study callbacks must wait for that
        # commit and accept its resulting revision, not the intermediate one.
        state["workspace_revision"] = operation_state.get("revision", 0) + bool(operation_state.get("switching"))
        operation_state.setdefault("learning", {})[tool_id] = state
        on_export_options(tool_id, operation_state)
    return True


def clear(tool_id, operation_state):
    operation_state.setdefault("learning", {}).pop(tool_id, None)
    on_export_options(tool_id, operation_state)


def _view(envelope, state, message, *, initial=False):
    ui = _ui()
    payload = envelope["payload"]
    tool_id = payload["tool_id"]
    items = payload["content"]["cards" if tool_id == "flashcards" else "questions"]
    index = state["index"]
    item = items[index]
    envelope = {**envelope, "view_revision": state["view_revision"]}
    previous = gr.update(interactive=index > 0)
    following = gr.update(interactive=index + 1 < len(items))
    # References/export controls stay unchanged during study navigation.
    tail = ([ui._coverage(payload), gr.update(choices=ui._reference_choices(payload), value=None),
             "", None, None, gr.skip()] if initial else [gr.skip()] * 6)
    if tool_id == "flashcards":
        answer = (_text(item["back"]) + "\n\n" + _citations(payload, item)) if state["revealed"] else ""
        return (envelope, message, _text(item["front"]), gr.update(value=answer, visible=state["revealed"]),
                f"Card {index + 1} of {len(items)}", gr.update(interactive=not state["revealed"]),
                previous, following, *tail)
    attempts = state["attempts"]
    attempted = index in attempts
    selected = attempts.get(index)
    feedback = ""
    if attempted:
        correct = selected == item["correct_option"]
        answer = f"{'ABCD'[item['correct_option']]}. {item['options'][item['correct_option']]}"
        feedback = ("**Correct.**" if correct else "**Not quite.**") + "\n\n"
        feedback += "Answer: " + _text(answer) + "\n\n" + _text(item["explanation"])
        feedback += "\n\n" + _citations(payload, item)
    score = sum(choice == items[number]["correct_option"] for number, choice in attempts.items())
    options = gr.update(choices=[(f"{'ABCD'[number]}. {option}", str(number))
                                 for number, option in enumerate(item["options"])],
                        value=str(selected) if attempted else None, interactive=not attempted)
    return (envelope, message, _text(item["question"]), options, feedback,
            f"Question {index + 1} of {len(items)}", f"Score: {score} / {len(attempts)} answered ({len(items)} total)",
            gr.update(interactive=not attempted), previous, following, gr.update(interactive=True), *tail)


def render_output(workspace, payload, message):
    envelope = {"workspace": workspace, "payload": payload, "token": uuid4().hex, "view_revision": 0}
    return _view(envelope, _initial(envelope), message, initial=True)


def empty_output(tool_id, message=""):
    tail = ["", gr.update(choices=[], value=None), "", None, None, _ui().TOOL_LABELS[tool_id]]
    disabled = lambda: gr.update(interactive=False)
    if tool_id == "flashcards":
        return (None, message, "", gr.update(value="", visible=False), "", disabled(), disabled(), disabled(), *tail)
    return (None, message, "", gr.update(choices=[], value=None, interactive=False), "", "", "",
            disabled(), disabled(), disabled(), disabled(), *tail)


def _matches(envelope, workspace, operation_state, tool_id, *, view=True):
    state = operation_state.get("learning", {}).get(tool_id)
    return (bool(envelope) and bool(state) and envelope.get("workspace") == workspace
            and state["workspace"] == workspace and envelope.get("payload", {}).get("tool_id") == tool_id
            and envelope.get("token") == state["token"]
            and _ui()._current(operation_state, state["workspace_revision"])
            and (not view or envelope.get("view_revision") == state["view_revision"]))


def _interact(tool_id, action, selection, envelope, workspace, operation_state):
    ui = _ui()
    skip = (gr.skip(),) * ui.OUTPUT_COUNTS[tool_id]
    with ui._JOB_LOCK:
        if not _matches(envelope, workspace, operation_state, tool_id):
            return skip
        state = deepcopy(operation_state["learning"][tool_id])
    items = envelope["payload"]["content"]["cards" if tool_id == "flashcards" else "questions"]
    if action in {"next", "previous"}:
        state["index"] = max(0, min(len(items) - 1, state["index"] + (1 if action == "next" else -1)))
        state["revealed"] = False
    elif tool_id == "flashcards" and action == "reveal":
        state["revealed"] = True
    elif tool_id == "quizzes" and action == "restart":
        state.update(index=0, attempts={})
    elif tool_id == "quizzes" and action == "check":
        if state["index"] not in state["attempts"]:
            if not isinstance(selection, str) or selection not in {"0", "1", "2", "3"}:
                return ui._status("Choose an answer before checking it.", tool_id)
            state["attempts"][state["index"]] = int(selection)
    else:
        return ui._status("Unknown study action.", tool_id)
    state["view_revision"] += 1
    result = _view(envelope, state, gr.skip())
    with ui._JOB_LOCK:
        if not _matches(envelope, workspace, operation_state, tool_id):
            return skip
        operation_state["learning"][tool_id] = state
    return result


def on_flashcard(action, envelope, workspace, operation_state):
    return _interact("flashcards", action, None, envelope, workspace, operation_state)


def on_quiz(action, selection, envelope, workspace, operation_state):
    return _interact("quizzes", action, selection, envelope, workspace, operation_state)


def on_export_options(tool_id, operation_state):
    key = tool_id + "_export_revision"
    operation_state[key] = operation_state.get(key, 0) + 1
    return None


def on_export(tool_id, envelope, filename, output_format, workspace, operation_state):
    ui = _ui()
    key = tool_id + "_export_revision"
    export_revision = operation_state.get(key, 0)

    def current():
        return (_matches(envelope, workspace, operation_state, tool_id, view=False)
                and operation_state.get(key, 0) == export_revision)

    try:
        if not current():
            return gr.skip(), None
        path = ui._artifacts().export_learning(workspace, envelope["payload"], output_format, filename=filename,
                                               is_cancelled=lambda: not current())
        return (f"{ui.TOOL_LABELS[tool_id]} exported.", str(path)) if current() else (gr.skip(), None)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as exc:
        logger.exception("Exporting learning output failed")
        return (f"Could not export {ui.TOOL_LABELS[tool_id]}: {exc}", None) if current() else (gr.skip(), None)


def on_reference(selection, envelope, workspace, operation_state):
    tool_id = envelope.get("payload", {}).get("tool_id") if envelope else None
    if tool_id not in DEFAULT_INSTRUCTIONS or not _matches(envelope, workspace, operation_state, tool_id, view=False):
        return gr.skip(), gr.skip()
    result = _ui().on_reference(selection, envelope, workspace, operation_state)
    return result if _matches(envelope, workspace, operation_state, tool_id, view=False) else (gr.skip(), gr.skip())


def build(tool_id, workspace, selected_ids, collection, model, reasoning_effort, operation_state):
    ui = _ui()
    label = ui.TOOL_LABELS[tool_id]
    payload_state = gr.State(None)
    columns = gr.State("")
    with gr.Row(equal_height=True) as instruction_row:
        instructions = gr.Textbox(label=f"{label} instructions", value=DEFAULT_INSTRUCTIONS[tool_id],
                                  lines=2, scale=3)
        item_count = gr.Slider(label="Number of flashcards" if tool_id == "flashcards" else "Number of questions",
                               minimum=1, maximum=MAX_FLASHCARDS if tool_id == "flashcards" else MAX_QUIZ_QUESTIONS,
                               step=1, precision=0, value=DEFAULT_LEARNING_COUNT, scale=1, min_width=180)
    with gr.Row():
        generate_button = gr.Button(f"Generate {label}", variant="primary")
        stop_button = gr.Button("Stop tool generation", variant="stop")
    status = gr.Markdown()
    coverage = gr.Markdown()
    with gr.Row(equal_height=True) as body:
        with gr.Column(scale=3, min_width=300) as main_column:
            progress = gr.Markdown()
            preview = gr.Markdown(height=200, max_height=300, container=True, padding=True,
                                  latex_delimiters=ui.LATEX_DELIMITERS)
            if tool_id == "flashcards":
                reveal_button = gr.Button("Show answer", interactive=False)
                answer = gr.Markdown(height=200, max_height=300, visible=False, container=True, padding=True,
                                     latex_delimiters=ui.LATEX_DELIMITERS)
                with gr.Row():
                    previous_button = gr.Button("Previous", interactive=False)
                    next_button = gr.Button("Next", interactive=False)
            else:
                options = gr.Radio(label="Your answer", choices=[], interactive=False,
                                   elem_classes=["rag-quiz-options"])
                check_button = gr.Button("Check answer", interactive=False)
                feedback = gr.Markdown(max_height=300, container=True, padding=True,
                                       latex_delimiters=ui.LATEX_DELIMITERS)
                score = gr.Markdown()
                with gr.Row():
                    previous_button = gr.Button("Previous", interactive=False)
                    next_button = gr.Button("Next", interactive=False)
                restart_button = gr.Button("Restart quiz", interactive=False)
            gr.Markdown("Saved automatically for this workspace. Study progress resets when you reopen it or generate again.")
        with gr.Column(scale=1, min_width=260, elem_classes=["rag-summary-sidebar"]) as sidebar:
            with gr.Accordion("References", open=True) as references_group:
                reference = gr.Dropdown(label="Open reference", choices=[])
                excerpt = gr.Textbox(label="Source excerpt", lines=6, max_lines=6, interactive=False)
                original = gr.File(label="Original source", interactive=False)
            with gr.Group() as export_group:
                filename = gr.Textbox(label="Filename", value=label)
                formats = ([("Aiken (.txt)", "aiken"), ("Moodle XML (.xml)", "moodle_xml"),
                            ("Markdown", "markdown"), ("CSV", "csv"), ("JSON", "json")]
                           if tool_id == "quizzes" else ["markdown", "csv", "json"])
                output_format = gr.Dropdown(label="Export format", choices=formats,
                                           value="aiken" if tool_id == "quizzes" else "markdown")
                export_button = gr.Button("Export")
                gr.Markdown("Aiken includes questions and answer keys. Other formats also include explanations and references."
                            if tool_id == "quizzes" else "Exports include answers and references.")
    download = gr.File(visible=False, interactive=False)
    tail = [coverage, reference, excerpt, original, download, filename]
    if tool_id == "flashcards":
        outputs = [payload_state, status, preview, answer, progress, reveal_button, previous_button, next_button, *tail]
        inputs = [payload_state, workspace, operation_state]
        reveal_event = reveal_button.click(partial(on_flashcard, "reveal"), inputs, outputs, queue=False)
        callback = on_flashcard
    else:
        outputs = [payload_state, status, preview, options, feedback, progress, score, check_button,
                   previous_button, next_button, restart_button, *tail]
        inputs = [options, payload_state, workspace, operation_state]
        check_event = check_button.click(partial(on_quiz, "check"), inputs, outputs, queue=False)
        restart_event = restart_button.click(partial(on_quiz, "restart"), inputs, outputs, queue=False)
        callback = on_quiz
    previous_event = previous_button.click(partial(callback, "previous"), inputs, outputs, queue=False)
    next_event = next_button.click(partial(callback, "next"), inputs, outputs, queue=False)
    generation = generate_button.click(partial(ui.on_generate, tool_id),
        [instructions, columns, workspace, selected_ids, collection, model, reasoning_effort, operation_state, item_count],
        outputs, concurrency_limit=None, trigger_mode="once")
    stop_button.click(ui.on_stop, operation_state, status, queue=False)
    export_event = export_button.click(partial(on_export, tool_id),
        [payload_state, filename, output_format, workspace, operation_state], [status, download],
        concurrency_id="rag-tool-storage", concurrency_limit=1)
    download_event = export_event.success(fn=None, inputs=download, outputs=None,
        js=ui.SUMMARY_DOWNLOAD_JS.replace('"Summary"', f'"{label}"'), queue=False)
    filename.input(partial(on_export_options, tool_id), operation_state, download, queue=False)
    output_format.input(partial(on_export_options, tool_id), operation_state, download, queue=False)
    reference.input(on_reference, [reference, payload_state, workspace, operation_state], [excerpt, original], queue=False,
                    preprocess=False)
    controls = dict(payload_state=payload_state, instructions=instructions, columns=columns,
        item_count=item_count, instruction_row=instruction_row,
        generate_button=generate_button, stop_button=stop_button, status=status, coverage=coverage,
        preview=preview, progress=progress, previous_button=previous_button, next_button=next_button,
        previous_event=previous_event, next_event=next_event, reference=reference, excerpt=excerpt,
        original=original, filename=filename, output_format=output_format, export_button=export_button,
        download=download, generation=generation, outputs=outputs, export_event=export_event,
        download_event=download_event, body=body, main_column=main_column, sidebar=sidebar,
        references_group=references_group, export_group=export_group)
    if tool_id == "flashcards":
        controls.update(answer=answer, reveal_button=reveal_button, reveal_event=reveal_event)
    else:
        controls.update(options=options, feedback=feedback, score=score, check_button=check_button,
                        check_event=check_event, restart_button=restart_button, restart_event=restart_event)
    return SimpleNamespace(**controls)
