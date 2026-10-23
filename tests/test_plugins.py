"""Local tool settings stay offline, lazy, and independent of optional exports."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from plugins import manager


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    path = tmp_path / ".cache" / "plugins.json"
    monkeypatch.setattr(manager, "SETTINGS_PATH", path)
    return path


def test_default_catalog_is_read_only_until_preference_saved(settings):
    assert manager.is_enabled("summary") is True
    assert manager.is_enabled("spreadsheet") is True
    assert manager.is_enabled("presentation") is True
    assert manager.is_enabled("mind_maps") is True
    assert manager.is_enabled("flashcards") is True
    assert manager.is_enabled("quizzes") is True
    assert manager.is_enabled("graphics") is True
    assert manager.is_enabled("plot") is True
    assert manager.is_enabled("compare") is True
    assert manager.is_enabled("evidence_check") is True
    assert not settings.exists()
    assert manager.get_status("compare")["status"] == "Enabled"
    assert not settings.parent.exists()


def test_preferences_persist_and_apply_immediately(settings):
    assert manager.set_enabled("summary", False)["status"] == "Disabled"
    assert not manager.is_enabled("summary")
    assert manager.is_enabled("spreadsheet")
    assert json.loads(settings.read_text(encoding="utf-8")) == {"version": 1, "enabled": {"summary": False}}
    assert manager.set_enabled("summary", True)["enabled"]


def test_unknown_planned_or_invalid_updates_are_rejected(settings, monkeypatch):
    monkeypatch.setattr(manager, "CATALOG", (*manager.CATALOG,
        {"id": "future", "name": "Future tool", "available": False}))
    for tool, value in (("unknown", True), ("future", True), ("summary", "false")):
        with pytest.raises(ValueError):
            manager.set_enabled(tool, value)
    assert not settings.exists()


def test_missing_exporter_does_not_disable_core_tool(monkeypatch):
    def missing(distribution):
        raise manager.importlib.metadata.PackageNotFoundError(distribution)
    monkeypatch.setattr(manager.importlib.metadata, "version", missing)
    status = manager.get_status("summary")
    assert status["enabled"]
    assert status["dependencies"] == [{"export": "DOCX", "package": "python-docx", "installed": False, "version": None}]
    assert manager.get_status("spreadsheet")["enabled"]


@pytest.mark.parametrize("content", ["not json", '{"version":2,"enabled":{}}',
                                    '{"version":1,"enabled":{"summary":"false"}}',
                                    '{"version":1,"enabled":{"unknown":true}}'])
def test_corrupt_settings_are_reported_without_overwrite(settings, content):
    settings.parent.mkdir()
    settings.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        manager.set_enabled("summary", False)
    assert settings.read_text(encoding="utf-8") == content


def test_deeply_nested_settings_report_error_instead_of_crashing_ui(settings):
    from plugins import ui
    settings.parent.mkdir()
    content = "[" * 2000 + "0" + "]" * 2000
    settings.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        manager.is_enabled("summary")
    _, status, _ = ui.on_refresh()
    assert "Could not read plugin settings" in status
    assert settings.read_text(encoding="utf-8") == content


def test_linked_settings_rejected(settings, tmp_path):
    target = tmp_path / "outside.json"
    target.write_text('{"version":1,"enabled":{}}', encoding="utf-8")
    settings.parent.mkdir()
    try:
        settings.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    with pytest.raises(ValueError, match="linked"):
        manager.set_enabled("summary", False)
    assert target.read_text(encoding="utf-8") == '{"version":1,"enabled":{}}'


def test_plugin_ui_constructs_without_reading_preferences(monkeypatch):
    import gradio as gr
    from plugins import ui
    def forbidden():
        raise AssertionError("Settings must remain lazy during UI construction")
    monkeypatch.setattr(manager, "_read_settings", forbidden)
    with gr.Blocks() as app:
        components = ui.build_tab()
    assert "selection" not in components and "enabled" not in components and "apply" not in components
    assert components["catalog"].headers == ["Tool", "Enabled", "Version", "Optional exporters", "Description"]
    assert components["catalog"].interactive
    assert components["catalog"].static_columns == [0, 2, 3, 4]
    assert components["catalog"].datatype == ["str", "bool", "str", "str", "str"]
    assert components["catalog"].value["data"] == []
    assert components["catalog"].row_count == (len(manager.CATALOG), "fixed")
    assert components["catalog"].column_count == (5, "fixed")
    callback = app.fns[components["edit_event"]["id"]]
    assert callback.fn is ui.on_catalog_input
    assert callback.trigger_mode == "multiple"
    assert callback.concurrency_limit == 1
    assert components["snapshot"] in callback.inputs


def plugin_table():
    from plugins import ui
    rows, _, snapshot = ui.on_refresh()
    return pd.DataFrame(rows, columns=ui.HEADERS), snapshot


def test_checkbox_toggle_persists_without_replacing_table(settings):
    from plugins import ui
    table, snapshot = plugin_table()
    table.at[0, "Enabled"] = False
    rendered, message, saved_snapshot = ui.on_catalog_input(table, snapshot)
    assert rendered == ui.gr.skip()
    assert "Summary disabled" in message
    assert saved_snapshot["rows"][0][1] is False
    assert not manager.is_enabled("summary")
    assert manager.is_enabled("spreadsheet")
    assert json.loads(settings.read_text(encoding="utf-8"))["enabled"] == {"summary": False}


def test_stale_session_toggle_preserves_other_sessions_changes():
    from plugins import ui
    first, first_snapshot = plugin_table()
    second, second_snapshot = plugin_table()
    first.at[0, "Enabled"] = False
    ui.on_catalog_input(first, first_snapshot)
    second.at[1, "Enabled"] = False  # Its stale Summary checkbox is still True.
    ui.on_catalog_input(second, second_snapshot)
    assert not manager.is_enabled("summary")
    assert not manager.is_enabled("spreadsheet")
    assert second_snapshot["rows"][0][1] is True  # Display snapshot, not global preferences.


def test_rapid_toggles_are_all_applied_in_order():
    from plugins import ui
    table, snapshot = plugin_table()
    frames = []
    for row, value in [(0, False), (1, False), (0, True), (1, True), (0, False)]:
        table.at[row, "Enabled"] = value
        frames.append(table.copy(deep=True))
    for frame in frames:
        result = ui.on_catalog_input(frame, snapshot)
        assert result[0] == ui.gr.skip()
    assert not manager.is_enabled("summary")
    assert manager.is_enabled("spreadsheet")


def test_planned_checkbox_edit_restored_without_partial_write(settings, monkeypatch):
    from plugins import ui
    monkeypatch.setattr(manager, "CATALOG", (*manager.CATALOG,
        {"id": "future", "name": "Future tool", "description": "Planned", "available": False}))
    monkeypatch.setattr(ui, "CATALOG", manager.CATALOG)
    table, snapshot = plugin_table()
    baseline = deepcopy(snapshot)
    table.at[0, "Enabled"] = False
    planned_row = next(index for index, entry in enumerate(manager.CATALOG) if not entry["available"])
    table.at[planned_row, "Enabled"] = True
    restored, message, _ = ui.on_catalog_input(table, snapshot)
    assert "planned" in message
    assert restored == baseline["rows"]
    assert restored[planned_row][1] is False
    assert snapshot == baseline
    assert not settings.exists()


@pytest.mark.parametrize("column", ["Tool", "Version", "Optional exporters", "Description"])
def test_metadata_edits_rejected_and_checkboxes_not_saved(settings, column):
    from plugins import ui
    table, snapshot = plugin_table()
    baseline = deepcopy(snapshot)
    table.at[0, "Enabled"] = False
    table.at[0, column] = "tampered"
    restored, message, _ = ui.on_catalog_input(table, snapshot)
    assert "read-only" in message
    assert restored == baseline["rows"]
    assert not settings.exists()
    assert snapshot == baseline


@pytest.mark.parametrize("malformed", ["headers", "row_removed", "reordered", "non_boolean", "null_metadata", "not_dataframe"])
def test_malformed_table_update_leaves_preferences_unchanged(settings, malformed):
    from plugins import ui
    table, snapshot = plugin_table()
    manager.set_enabled("spreadsheet", False)
    before = settings.read_bytes()
    if malformed == "headers":
        table.columns = ["Changed", *ui.HEADERS[1:]]
    elif malformed == "row_removed":
        table = table.iloc[:-1]
    elif malformed == "reordered":
        table = table.iloc[::-1]
    elif malformed == "non_boolean":
        table = table.astype(object)
        table.at[0, "Enabled"] = "false"
    elif malformed == "null_metadata":
        table.at[0, "Tool"] = pd.NA
    else:
        table = table.values.tolist()
    result = ui.on_catalog_input(table, snapshot)
    assert "Could not change" in result[1]
    assert settings.read_bytes() == before


def test_valid_multirow_checkbox_edit_is_atomic(settings):
    from plugins import ui
    table, snapshot = plugin_table()
    table.at[0, "Enabled"] = False
    table.at[1, "Enabled"] = False
    _, message, _ = ui.on_catalog_input(table, snapshot)
    assert "Summary disabled" in message and "Spreadsheet disabled" in message
    assert json.loads(settings.read_text(encoding="utf-8"))["enabled"] == {
        "summary": False, "spreadsheet": False}


@pytest.mark.parametrize("tool_id,label", [("graphics", "Graphics"), ("mind_maps", "Mind Map"), ("flashcards", "Flashcards"), ("quizzes", "Quizzes"), ("compare", "Compare"), ("evidence_check", "Evidence Check")])
def test_available_tool_checkbox_without_optional_packages(settings, tool_id, label):
    from plugins import ui
    table, snapshot = plugin_table()
    row = next(index for index, entry in enumerate(manager.CATALOG) if entry["id"] == tool_id)
    assert table.at[row, "Tool"] == label and table.at[row, "Enabled"]
    assert manager.get_status(tool_id)["dependencies"] == []
    table.at[row, "Enabled"] = False
    _, message, _ = ui.on_catalog_input(table, snapshot)
    assert f"{label} disabled" in message and not manager.is_enabled(tool_id)
    table.at[row, "Enabled"] = True
    ui.on_catalog_input(table, snapshot)
    assert manager.is_enabled(tool_id)


def test_completed_catalog_matches_requested_tools():
    planned = {entry["name"] for entry in manager.CATALOG if not entry["available"]}
    assert planned == set()
    assert [entry["id"] for entry in manager.CATALOG][-2:] == ["compare", "evidence_check"]
    assert not {"References", "Translation"}.intersection(entry["name"] for entry in manager.CATALOG)


def test_presentation_precedes_mind_map_and_can_toggle_without_exporter(settings, monkeypatch):
    from plugins import ui

    def missing(distribution):
        raise manager.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(manager.importlib.metadata, "version", missing)
    table, snapshot = plugin_table()
    names = list(table["Tool"])
    assert names.index("Presentation") + 1 == names.index("Graphics")
    assert names.index("Graphics") + 1 == names.index("Plot")
    assert names.index("Plot") + 1 == names.index("Mind Map")
    row = names.index("Presentation")
    status = manager.get_status("presentation")
    assert status["enabled"] and status["dependencies"] == [
        {"export": "PPTX", "package": "python-pptx", "installed": False, "version": None}]
    table.at[row, "Enabled"] = False
    _, message, _ = ui.on_catalog_input(table, snapshot)
    assert "Presentation disabled" in message and not manager.is_enabled("presentation")
    table.at[row, "Enabled"] = True
    ui.on_catalog_input(table, snapshot)
    assert manager.is_enabled("presentation")


def test_plot_precedes_mind_map_and_can_toggle_without_renderer(monkeypatch):
    from plugins import ui

    def missing(distribution):
        raise manager.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(manager.importlib.metadata, "version", missing)
    table, snapshot = plugin_table()
    names = list(table["Tool"])
    row = names.index("Plot")
    assert row + 1 == names.index("Mind Map")
    status = manager.get_status("plot")
    assert status["enabled"] and status["dependencies"] == [
        {"export": "Plot previews / PNG / SVG / PDF", "package": "matplotlib", "installed": False, "version": None}]
    table.at[row, "Enabled"] = False
    _, message, _ = ui.on_catalog_input(table, snapshot)
    assert "Plot disabled" in message and not manager.is_enabled("plot")
    table.at[row, "Enabled"] = True
    ui.on_catalog_input(table, snapshot)
    assert manager.is_enabled("plot")
