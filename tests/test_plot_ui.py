"""Plot control wiring, single-output persistence, and stale session guards."""

from copy import deepcopy

import pytest

from rag_chatbot.tools import plot_ui, ui


@pytest.fixture
def operation():
    return {"revision": 0, "answer_revision": 0, "switching": False, "selected_ids": ["doc1"]}


@pytest.fixture
def payload():
    return {"tool_id": "plot", "title": "Measured values", "content": {
        "chart_type": "bar", "x_label": "Sample", "y_label": "Pressure", "x_unit": "", "y_unit": "Pa",
        "series": [{"name": "Measured", "points": [
            {"label": "A", "x": None, "y": 12.5, "source_ids": ["c1"], "evidence": "A: 12.5 Pa"},
            {"label": "B", "x": None, "y": 14, "source_ids": ["c1"], "evidence": "B: 14 Pa"}]}],
        "limitations": [{"text": "Only two samples were reported.", "source_ids": ["c1"]}]},
        "sources": [{"id": "c1", "document_id": "doc1", "filename": "Paper.pdf", "locator": "Page 2",
                     "text": "A: 12.5 Pa; B: 14 Pa", "source_path": "sources/doc1.pdf"}],
        "documents": [{"id": "doc1"}], "coverage": {"documents": 1, "chunks": 3, "processed_chunks": 3}}


@pytest.fixture(autouse=True)
def fake_services(monkeypatch):
    monkeypatch.setattr(ui, "_enabled", lambda _: True)
    monkeypatch.setattr(plot_ui, "_preview", lambda payload: "<div>Local plot preview</div>")
    store = ui._artifacts()
    monkeypatch.setattr(store, "save_current_plot", lambda workspace, payload, **kwargs: deepcopy(payload))
    for name in ("summary", "spreadsheet", "mind_map", "presentation", "graphics", "plot"):
        monkeypatch.setattr(store, "load_current_" + name, lambda workspace: None)
    monkeypatch.setattr(store, "load_current_learning", lambda workspace, tool_id: None)


def start(payload, operation):
    result = plot_ui.render_output("Alpha", payload, "Ready")
    plot_ui.publish(result[0], operation)
    return result


def generate(operation, **overrides):
    args = {"instructions": "Focus on findings", "chart_type": "auto", "workspace": "Alpha",
            "selected_ids": ["doc1"], "collection": "Research", "model": "gpt-6-luna",
            "reasoning_effort": "medium", "operation_state": operation}
    args.update(overrides)
    return list(plot_ui.on_generate(**args))


def test_render_lists_exact_points_units_evidence_and_numbered_citations(payload, operation):
    result = start(payload, operation)
    assert result[3]["headers"] == ["Series", "Label", "X: Sample", "Y: Pressure (Pa)", "References", "Evidence"]
    assert result[3]["value"] == [["Measured", "A", "", 12.5, "[1]", "A: 12.5 Pa"],
                                  ["Measured", "B", "", 14, "[1]", "B: 14 Pa"]]
    assert result[4]["visible"] and result[4]["value"] == "Only two samples were reported. [1]"
    assert result[6]["choices"][0][1] == "0"
    payload["content"]["limitations"] = []
    assert plot_ui.render_output("Alpha", payload, "Ready")[4] == ui.gr.update(value="", visible=False)


def test_large_plot_table_preview_is_bounded_without_changing_saved_or_export_data(payload, operation, monkeypatch):
    point = payload["content"]["series"][0]["points"][0]
    payload["content"]["series"][0]["points"] = [{**point, "label": f"Sample {index}"}
                                                   for index in range(10_000)]
    before = deepcopy(payload)
    result = start(payload, operation)
    assert len(result[3]["value"]) == plot_ui.MAX_PREVIEW_ROWS == 200
    assert "first 200 of 10,000" in result[3]["label"]
    assert "CSV/JSON" in result[3]["label"]
    assert len(result[0]["payload"]["content"]["series"][0]["points"]) == 10_000
    assert payload == before

    def export(workspace, value, output_format, **kwargs):
        assert len(value["content"]["series"][0]["points"]) == 10_000
        return "Plot.csv"

    monkeypatch.setattr(ui._artifacts(), "export_plot", export)
    assert plot_ui.on_export(result[0], "Plot", "csv", "Alpha", operation) == ("Plot exported.", "Plot.csv")


def test_generation_saves_one_current_plot_and_publishes_new_token(payload, operation, monkeypatch):
    old = start(payload, operation)[0]
    saved = []

    def backend(*args, **kwargs):
        assert args == ("Alpha", "plot", ["doc1"])
        assert kwargs["instructions"] == "Focus on findings" and kwargs["chart_type"] == "scatter"
        assert kwargs["model"] == "gpt-6-luna" and kwargs["reasoning_effort"] == "medium"
        assert "image_options" not in kwargs and "slide_count" not in kwargs and "item_count" not in kwargs
        yield {"result": payload}

    def save(workspace, generated, *, is_cancelled):
        assert workspace == "Alpha" and generated == payload and not is_cancelled()
        saved.append(generated)
        return dict(payload, id="saved")

    monkeypatch.setattr(ui, "_generate", backend)
    monkeypatch.setattr(ui._artifacts(), "save_current_plot", save)
    result = generate(operation, chart_type="scatter")[-1]
    assert len(result) == ui.OUTPUT_COUNTS["plot"] == 11
    assert result[0]["payload"]["id"] == "saved" and result[0]["token"] != old["token"]
    assert "saved for this workspace" in result[1] and len(saved) == 1
    assert operation["plot"]["token"] == result[0]["token"] and "tool_job" not in operation


@pytest.mark.parametrize("chart_type", ["pie", "Auto", None, 2, True])
def test_invalid_chart_type_prevents_generation(operation, monkeypatch, chart_type):
    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: pytest.fail("Invalid settings cannot generate"))
    result = generate(operation, chart_type=chart_type)
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
    monkeypatch.setattr(ui._artifacts(), "save_current_plot", lambda *args, **kwargs: pytest.fail("No cancelled save"))
    assert len(generate(operation)) == 1
    assert operation["plot"]["token"] == old["token"] and "tool_job" not in operation


def test_workspace_change_during_render_prevents_save_and_publish(payload, operation, monkeypatch):
    old = start(payload, operation)[0]

    def delayed(payload):
        operation["revision"] += 1
        return "<div>Stale image</div>"

    monkeypatch.setattr(ui, "_generate", lambda *args, **kwargs: (event for event in [{"result": payload}]))
    monkeypatch.setattr(plot_ui, "_preview", delayed)
    monkeypatch.setattr(ui._artifacts(), "save_current_plot", lambda *args, **kwargs: pytest.fail("No cancelled save"))
    assert len(generate(operation)) == 1
    assert operation["plot"]["token"] == old["token"]


def test_guarded_publish_leaves_previous_plot(payload, operation):
    old = start(payload, operation)[0]
    replacement = plot_ui.render_output("Alpha", payload, "Ready")[0]
    assert not plot_ui.publish(replacement, operation, is_cancelled=lambda: True)
    assert operation["plot"]["token"] == old["token"]


def test_workspace_restore_survives_switch_commit_and_clears_empty_workspace(payload, operation, monkeypatch):
    from rag_chatbot.chatbot import on_finish_workspace_switch
    old = start(payload, operation)[0]
    monkeypatch.setattr(ui._artifacts(), "load_current_plot", lambda workspace: payload if workspace == "Alpha" else None)
    operation.update(revision=2, switching=True, workspace_open_pending=False)
    restored = ui.on_workspace_reset("Alpha", operation)
    offset = sum(ui.OUTPUT_COUNTS[key] for key in ("summary", "spreadsheet", "presentation", "graphics"))
    envelope = restored[offset]
    assert envelope["token"] != old["token"] and restored[offset + 10] == "Plot"
    assert not plot_ui._matches(envelope, "Alpha", operation)
    on_finish_workspace_switch(operation)
    assert plot_ui._matches(envelope, "Alpha", operation)
    operation["revision"] += 1
    empty = ui.on_reset("Beta", operation)
    assert empty[offset] is None and "plot" not in operation
    assert plot_ui.on_export(envelope, "Plot", "png", "Beta", operation) == (ui.gr.skip(), None)


def test_preview_dependency_hint_preserves_restored_data(payload, operation, monkeypatch):
    hint = "<div>Install python -m pip install -r plugins/plot/requirements.txt</div>"
    monkeypatch.setattr(plot_ui, "_preview", lambda _: hint)
    monkeypatch.setattr(ui._artifacts(), "load_current_plot", lambda workspace: payload)
    restored = ui.on_reset("Alpha", operation)
    offset = sum(ui.OUTPUT_COUNTS[key] for key in ("summary", "spreadsheet", "presentation", "graphics"))
    assert restored[offset + 2] == hint
    assert restored[offset + 3]["value"][0][3] == 12.5
    assert plot_ui._matches(restored[offset], "Alpha", operation)


def test_late_workspace_restore_cannot_publish(payload, operation, monkeypatch):
    def load(workspace):
        operation["revision"] += 1
        return payload

    monkeypatch.setattr(ui._artifacts(), "load_current_plot", load)
    assert ui.on_reset("Alpha", operation) == (ui.gr.skip(),) * ui.TOTAL_OUTPUT_COUNT
    assert "plot" not in operation


@pytest.mark.parametrize("change", [None, "workspace", "output", "filename", "format"])
def test_export_and_stale_download_guard(payload, operation, monkeypatch, tmp_path, change):
    envelope = start(payload, operation)[0]

    def export(workspace, value, output_format, *, filename, is_cancelled):
        assert (workspace, value, output_format, filename) == ("Alpha", payload, "png", "Values")
        assert not is_cancelled()
        if change == "workspace":
            operation["revision"] += 1
        elif change == "output":
            start(payload, operation)
        elif change:
            plot_ui.on_export_options(operation)
        assert is_cancelled() == bool(change)
        return tmp_path / "Values.png"

    monkeypatch.setattr(ui._artifacts(), "export_plot", export)
    result = plot_ui.on_export(envelope, "Values", "png", "Alpha", operation)
    assert result == ((ui.gr.skip(), None) if change else ("Plot exported.", str(tmp_path / "Values.png")))


def test_export_reports_missing_optional_dependency_without_losing_output(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]

    def export(*args, **kwargs):
        raise RuntimeError("Install python -m pip install -r plugins/plot/requirements.txt")

    monkeypatch.setattr(ui._artifacts(), "export_plot", export)
    message, download = plot_ui.on_export(envelope, "Values", "png", "Alpha", operation)
    assert "plugins/plot/requirements.txt" in message and download is None
    assert plot_ui._matches(envelope, "Alpha", operation)


def test_reference_rejects_outdated_output_and_workspace(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]
    monkeypatch.setattr(ui, "on_reference", lambda *args: pytest.fail("Stale reference cannot read source"))
    assert plot_ui.on_reference("0", envelope, "Beta", operation) == (ui.gr.skip(), ui.gr.skip())
    start(payload, operation)
    assert plot_ui.on_reference("0", envelope, "Alpha", operation) == (ui.gr.skip(), ui.gr.skip())


def test_out_of_order_reference_selection_does_not_publish_old_excerpt(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]
    latest = []

    def reference(selection, *args):
        if selection == "0":
            latest.append(plot_ui.on_reference("1", envelope, "Alpha", operation))
            return "Old excerpt", "old.pdf"
        return "Newest excerpt", "new.pdf"

    monkeypatch.setattr(ui, "on_reference", reference)
    assert plot_ui.on_reference("0", envelope, "Alpha", operation) == (ui.gr.skip(), ui.gr.skip())
    assert latest == [("Newest excerpt", "new.pdf")]


def test_sessions_cannot_export_each_others_plot(payload, operation, monkeypatch):
    envelope = start(payload, operation)[0]
    other = {"revision": 0}
    start(payload, other)
    monkeypatch.setattr(ui._artifacts(), "export_plot", lambda *args, **kwargs: pytest.fail("Wrong session cannot export"))
    assert plot_ui.on_export(envelope, "Values", "json", "Alpha", other) == (ui.gr.skip(), None)


def test_ui_wiring_is_lazy_and_uses_standard_layout(monkeypatch):
    monkeypatch.setattr(ui, "_artifacts", lambda: pytest.fail("UI construction must not access tool storage"))
    monkeypatch.setattr(ui, "_enabled", lambda _: pytest.fail("UI construction must not access settings"))
    monkeypatch.setattr(plot_ui, "_preview", lambda _: pytest.fail("UI construction must not render a plot"))
    with ui.gr.Blocks() as app:
        workspace, selected, collection = ui.gr.State("Alpha"), ui.gr.State(["doc1"]), ui.gr.State("Research")
        model, reasoning, operation = ui.gr.State("gpt-6-luna"), ui.gr.State("medium"), ui.gr.State({"revision": 0})
        tools = ui.build_tools(workspace, selected, collection, model, reasoning, operation)
        tool = tools.controls["plot"]
    assert list(tools.controls)[3:6] == ["graphics", "plot", "mind_maps"]
    assert tool.instructions.value == plot_ui.DEFAULT_INSTRUCTIONS
    assert tool.chart_type.value == "auto" and tool.chart_type.choices == plot_ui.CHART_CHOICES
    assert not tool.table.interactive and tool.table.max_height == 320
    assert not tool.limitations.interactive and tool.limitations.lines == tool.limitations.max_lines == 4
    assert tool.excerpt.lines == tool.excerpt.max_lines == 6
    assert tool.main_column.scale == 3 and tool.sidebar.scale == 1
    assert tool.output_format.value == "png" and tool.filename.value == "Plot"
    assert [value for _, value in tool.output_format.choices] == ["png", "svg", "pdf", "csv", "json"]
    assert len(tool.outputs) == ui.OUTPUT_COUNTS["plot"] == 11
    assert not hasattr(tool, "saved") and not hasattr(tool, "save_button")
    assert app.fns[tool.generation["id"]].fn is plot_ui.on_generate
    assert app.fns[tool.generation["id"]].inputs == [tool.instructions, tool.chart_type, workspace,
                                                    selected, collection, model, reasoning, operation]
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
