"""Graphics settings, saved previews, and stale session interactions."""

from copy import deepcopy
import pandas as pd
import pytest

from chatbot import config
from rag_chatbot.tools import graphics_ui, ui


@pytest.fixture
def operation():
    return {"revision": 0, "answer_revision": 0, "switching": False, "selected_ids": ["doc1"]}


@pytest.fixture
def payload():
    return {"tool_id": "graphics", "title": "Research graphic", "content": {
        "prompt": "Show a labelled overview of the evidence.", "caption": "Evidence overview <plain text>",
        "source_ids": ["c1"], "asset_path": "artifacts/graphics/assets/image.png", "output_format": "png"},
        "sources": [{"id": "c1", "document_id": "doc1", "filename": "Paper.pdf", "locator": "Page 2",
                     "text": "Evidence here", "source_path": "sources/doc1.pdf"}],
        "documents": [{"id": "doc1"}], "coverage": {"documents": 1, "chunks": 3, "processed_chunks": 3}}


@pytest.fixture(autouse=True)
def fake_services(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "_enabled", lambda _: True)
    store = ui._artifacts()
    monkeypatch.setattr(store, "save_current_graphics", lambda workspace, payload, **kwargs: deepcopy(payload))
    monkeypatch.setattr(store, "graphics_image_path", lambda workspace, payload: tmp_path / "image.png")
    for name in ("summary", "spreadsheet", "mind_map", "presentation", "graphics", "plot"):
        monkeypatch.setattr(store, "load_current_" + name, lambda workspace: None)
    monkeypatch.setattr(store, "load_current_learning", lambda workspace, tool_id: None)


def start(payload, operation):
    result = graphics_ui.render_output("Alpha", payload, "Ready")
    graphics_ui.publish(result[0], operation)
    return result


def generate(operation, **overrides):
    defaults = config.get_image_model_config(config.DEFAULT_IMAGE_MODEL)
    args = {"instructions": "Focus on findings", "image_model": config.DEFAULT_IMAGE_MODEL,
            "image_size": defaults["default_size"], "image_quality": defaults["default_quality"],
            "image_background": defaults["default_background"], "image_output_format": "png",
            "image_output_compression": 100, "image_moderation": "auto", "workspace": "Alpha",
            "selected_ids": ["doc1"], "collection": "Research", "model": "gpt-6-luna",
            "reasoning_effort": "medium", "operation_state": operation}
    args.update(overrides)
    return list(graphics_ui.on_generate(**args))


def test_generation_saves_validated_asset_before_rendering(payload, operation, monkeypatch):
    old = start(payload, operation)[0]
    events = []

    def backend(*args, **kwargs):
        assert args == ("Alpha", "graphics", ["doc1"])
        assert kwargs["instructions"] == "Focus on findings"
        assert kwargs["image_options"]["image_model"] == config.DEFAULT_IMAGE_MODEL
        assert kwargs["image_options"]["image_output_format"] == "webp"
        assert kwargs["image_options"]["image_output_compression"] == 80
        assert "slide_count" not in kwargs and "item_count" not in kwargs
        yield {"result": dict(payload, image_data={"b64_json": "transient"})}

    def save(workspace, generated, *, is_cancelled):
        assert workspace == "Alpha" and not is_cancelled() and "image_data" in generated
        events.append("save")
        return dict(payload, id="saved")

    render = graphics_ui.render_output

    def checked_render(workspace, generated, status):
        assert generated["id"] == "saved" and "image_data" not in generated and events == ["save"]
        events.append("render")
        return render(workspace, generated, status)

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui._artifacts(), "save_current_graphics", save)
    monkeypatch.setattr(graphics_ui, "render_output", checked_render)
    result = generate(operation, image_output_format="webp", image_output_compression=80)[-1]
    assert len(result) == ui.OUTPUT_COUNTS["graphics"] == 11
    assert result[0]["payload"]["id"] == "saved" and result[0]["token"] != old["token"]
    assert result[3] == payload["content"]["caption"] and result[4] == payload["content"]["prompt"]
    assert "saved for this workspace" in result[1] and events == ["save", "render"]
    assert operation["graphics"]["token"] == result[0]["token"] and "tool_job" not in operation


@pytest.mark.parametrize("options", [{"image_model": "unknown"}, {"image_size": "7x7"},
                                    {"image_output_format": "svg"}, {"image_output_compression": 101}])
def test_invalid_image_options_prevent_generation(operation, monkeypatch, options):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Invalid settings cannot generate"))
    result = generate(operation, **options)
    assert len(result) == 1 and result[0][1] and "tool_job" not in operation


@pytest.mark.parametrize("change", ["disabled", "no_sources", "switching"])
def test_preflight_prevents_generation(operation, monkeypatch, change):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Invalid requests cannot generate"))
    options = {}
    if change == "disabled":
        monkeypatch.setattr(ui, "_enabled", lambda _: False)
    elif change == "switching":
        operation["switching"] = True
    else:
        options["selected_ids"] = []
    assert len(generate(operation, **options)) == 1 and "tool_job" not in operation


@pytest.mark.parametrize("change", ["workspace", "selection", "stop"])
def test_cancelled_result_does_not_save(payload, operation, monkeypatch, change):
    old = start(payload, operation)[0]

    def backend(*args, **kwargs):
        if change == "workspace":
            operation["revision"] += 1
        elif change == "selection":
            operation["selected_ids"] = []
        else:
            ui.on_stop(operation)
        assert kwargs["is_cancelled"]()
        yield {"result": payload}

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui._artifacts(), "save_current_graphics", lambda *args, **kwargs: pytest.fail("No cancelled save"))
    assert len(generate(operation)) == 1
    assert operation["graphics"]["token"] == old["token"] and "tool_job" not in operation


def test_cancelled_render_does_not_publish(payload, operation, monkeypatch):
    old = start(payload, operation)[0]
    render = graphics_ui.render_output

    def delayed(*args):
        result = render(*args)
        operation["revision"] += 1
        return result

    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"result": payload}]))
    monkeypatch.setattr(graphics_ui, "render_output", delayed)
    assert len(generate(operation)) == 1
    assert operation["graphics"]["token"] == old["token"]


def test_workspace_restore_survives_switch_commit_and_resets_empty_workspace(payload, operation, monkeypatch):
    from rag_chatbot.chatbot import on_finish_workspace_switch
    old = start(payload, operation)[0]
    monkeypatch.setattr(ui._artifacts(), "load_current_graphics", lambda workspace: payload if workspace == "Alpha" else None)
    operation.update(revision=2, switching=True, workspace_open_pending=False)
    restored = ui.on_workspace_reset("Alpha", operation)
    offset = sum(ui.OUTPUT_COUNTS[key] for key in ("summary", "spreadsheet", "presentation"))
    envelope = restored[offset]
    assert envelope["token"] != old["token"] and restored[offset + 10] == "Graphics"
    assert not graphics_ui._matches(envelope, "Alpha", operation)
    on_finish_workspace_switch(operation)
    assert graphics_ui._matches(envelope, "Alpha", operation)
    operation["revision"] += 1
    empty = ui.on_reset("Beta", operation)
    assert empty[offset] is None and "graphics" not in operation
    assert graphics_ui.on_export(envelope, "Image", "image", "Beta", operation) == (ui.gr.skip(), None)


def test_late_workspace_restore_cannot_publish(payload, operation, monkeypatch):
    def load(workspace):
        operation["revision"] += 1
        return payload

    monkeypatch.setattr(ui._artifacts(), "load_current_graphics", load)
    assert ui.on_reset("Alpha", operation) == (ui.gr.skip(),) * ui.TOTAL_OUTPUT_COUNT
    assert "graphics" not in operation


@pytest.mark.parametrize("change", [None, "workspace", "output", "filename", "format"])
def test_export_original_with_cancellation_and_stale_download_guard(payload, operation, monkeypatch, tmp_path, change):
    envelope = start(payload, operation)[0]

    def export(workspace, value, output_format, *, filename, is_cancelled):
        assert (workspace, value, output_format, filename) == ("Alpha", payload, "image", "Image")
        assert not is_cancelled()
        if change == "workspace":
            operation["revision"] += 1
        elif change == "output":
            start(payload, operation)
        elif change:
            graphics_ui.on_export_options(operation)
        assert is_cancelled() == bool(change)
        return tmp_path / "Image.png"

    monkeypatch.setattr(ui._artifacts(), "export_current_graphics", export)
    result = graphics_ui.on_export(envelope, "Image", "image", "Alpha", operation)
    assert result == ((ui.gr.skip(), None) if change else ("Graphics exported.", str(tmp_path / "Image.png")))


def test_older_reference_cannot_overwrite_newer_selection(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]
    latest = []

    def reference(selection, *args):
        if selection == "0":
            latest.append(graphics_ui.on_reference("1", envelope, "Alpha", operation))
            return "Older source", "older.pdf"
        return "Latest source", "latest.pdf"

    monkeypatch.setattr(ui, "on_reference", reference)
    assert graphics_ui.on_reference("0", envelope, "Alpha", operation) == (ui.gr.skip(), ui.gr.skip())
    assert latest == [("Latest source", "latest.pdf")]


def test_reference_from_replaced_output_is_discarded(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]

    def reference(*args):
        start(payload, operation)
        return "Old source", None

    monkeypatch.setattr(ui, "on_reference", reference)
    assert graphics_ui.on_reference("0", envelope, "Alpha", operation) == (ui.gr.skip(), ui.gr.skip())


def test_output_tokens_are_session_scoped(payload, operation):
    other = deepcopy(operation)
    first = start(payload, operation)[0]
    second = start(payload, other)[0]
    assert graphics_ui._matches(first, "Alpha", operation)
    assert not graphics_ui._matches(first, "Alpha", other)
    assert graphics_ui._matches(second, "Alpha", other)
    assert not graphics_ui.publish(second, operation, is_cancelled=lambda: True)
    assert operation["graphics"]["token"] == first["token"]


def test_lazy_layout_and_shared_normal_chat_image_models(monkeypatch):
    from chatbot import chatbot
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("UI construction must not access tool storage"))
    monkeypatch.setattr(ui, "_enabled", lambda _: pytest.fail("UI construction must not access settings"))
    monkeypatch.setattr(chatbot, "history_dataframe", lambda: pd.DataFrame(columns=["File name"]))
    with ui.gr.Blocks() as app:
        chat = chatbot.build_tab()
        workspace, selected, collection = ui.gr.State("Alpha"), ui.gr.State(["doc1"]), ui.gr.State("Research")
        model, reasoning, operation = ui.gr.State("gpt-6-luna"), ui.gr.State("medium"), ui.gr.State({"revision": 0})
        tool = ui.build_tools(workspace, selected, collection, model, reasoning, operation).controls["graphics"]
    assert tool.instructions.value == graphics_ui.DEFAULT_INSTRUCTIONS
    assert tool.image_model.value == chat.image_model.value == config.DEFAULT_IMAGE_MODEL == "gpt-image-2.5-sunburst"
    assert tool.image_model.choices == chat.image_model.choices
    for name in config.IMAGE_MODEL_CHOICES:
        graphics_options = graphics_ui.on_image_model_change(name)
        chat_options = chatbot.build_image_option_updates(name)
        for graphics, normal in zip(graphics_options, chat_options):
            assert graphics["value"] == normal.value
            assert graphics["choices"] == [value for _, value in normal.choices]
    assert graphics_ui.on_image_format_change("png")["interactive"] is False
    assert graphics_ui.on_image_format_change("webp")["interactive"] is True
    assert tool.preview.height == 480 and not tool.preview.interactive
    assert tool.brief.lines == tool.brief.max_lines == tool.excerpt.lines == tool.excerpt.max_lines == 6
    assert tool.main_column.scale == 3 and tool.sidebar.scale == 1
    assert tool.output_format.value == "image" and tool.filename.value == "Graphics"
    assert [value for _, value in tool.output_format.choices] == ["image", "markdown", "json"]
    assert len(tool.outputs) == ui.OUTPUT_COUNTS["graphics"]
    assert not hasattr(tool, "saved") and not hasattr(tool, "save_button")
    assert app.fns[tool.generation["id"]].fn is graphics_ui.on_generate
    assert app.fns[tool.generation["id"]].inputs == [tool.instructions, tool.image_model, tool.image_size,
        tool.image_quality, tool.image_background, tool.image_output_format, tool.image_output_compression,
        tool.image_moderation, workspace, selected, collection, model, reasoning, operation]
    assert app.fns[tool.generation["id"]].outputs == tool.outputs
    assert app.fns[tool.export_event["id"]].inputs == [tool.payload_state, tool.filename, tool.output_format, workspace, operation]
    assert tool.download_event["trigger_only_on_success"] and tool.download_event["trigger_after"] == tool.export_event["id"]
    children = {}

    def walk(node):
        children[node["id"]] = [child["id"] for child in node.get("children", [])]
        for child in node.get("children", []):
            walk(child)

    walk(app.config["layout"])
    assert children[tool.body._id] == [tool.main_column._id, tool.sidebar._id]
    assert children[tool.sidebar._id] == [tool.references_group._id, tool.export_group._id]
