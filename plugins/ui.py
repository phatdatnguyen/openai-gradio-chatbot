"""Editable enable checkboxes for the trusted local app tool catalog."""

from copy import deepcopy
import threading

import gradio as gr
import pandas as pd

from .manager import CATALOG, list_statuses, set_enabled_many


HEADERS = ["Tool", "Enabled", "Version", "Optional exporters", "Description"]
STATIC_COLUMNS = [0, 2, 3, 4]
_EDIT_LOCK = threading.RLock()


def _rows():
    rows = []
    for status in list_statuses():
        dependencies = "; ".join(
            f"{item['export']}: {item['package']} {item['version']}" if item["installed"]
            else f"{item['export']}: {item['package']} missing (optional)"
            for item in status["dependencies"])
        description = status["description"]
        if not status["available"]:
            description = "Planned. " + description
        rows.append([status["name"], status["enabled"], status["version"], dependencies or "—", description])
    return rows


def _snapshot(rows):
    return {"ids": [entry["id"] for entry in CATALOG], "rows": deepcopy(rows)}


def on_refresh():
    try:
        with _EDIT_LOCK:
            rows = _rows()
            return rows, "Local tool status refreshed. Click an Enabled checkbox to save its setting.", _snapshot(rows)
    except (ValueError, OSError) as exc:
        return gr.skip(), f"Could not read plugin settings: {exc}", gr.skip()


def _validate_table(table, snapshot):
    if (not isinstance(snapshot, dict)
            or snapshot.get("ids") != [entry["id"] for entry in CATALOG]
            or not isinstance(snapshot.get("rows"), list)
            or len(snapshot["rows"]) != len(CATALOG)):
        raise ValueError("Refresh the plugin list before editing.")
    if not isinstance(table, pd.DataFrame) or list(table.columns) != HEADERS:
        raise ValueError("Keep the plugin table's column headings unchanged.")
    rows = table.values.tolist()
    if len(rows) != len(snapshot["rows"]):
        raise ValueError("Plugin rows cannot be added, removed, or reordered.")
    for row, trusted in zip(rows, snapshot["rows"]):
        if len(row) != len(HEADERS) or any(type(row[index]) is not str or row[index] != trusted[index]
                                           for index in STATIC_COLUMNS):
            raise ValueError("Only the Enabled checkboxes can be edited; plugin details are read-only.")
        if type(row[1]) is not bool:
            raise ValueError("Enabled values must be checkboxes (true or false).")
    return rows


def on_catalog_input(table, snapshot):
    """Save only checkbox changes relative to this session's displayed table.

    Other sessions can change preferences while this table is open. Comparing
    against the trusted display snapshot, rather than today's global settings,
    prevents a stale unchanged checkbox from overwriting another session's edit.
    Successful edits leave the browser's table in place so rapid subsequent
    clicks are not replaced by an earlier callback's full-table response.
    """
    with _EDIT_LOCK:
        try:
            rows = _validate_table(table, snapshot)
            changes = {identifier: row[1] for identifier, row, previous in
                       zip(snapshot["ids"], rows, snapshot["rows"]) if row[1] != previous[1]}
            if not changes:
                return gr.skip(), gr.skip(), gr.skip()
            set_enabled_many(changes)
            snapshot["rows"] = deepcopy(rows)
            labels = [f"{row[0]} {'enabled' if row[1] else 'disabled'}"
                      for identifier, row in zip(snapshot["ids"], rows) if identifier in changes]
            return (gr.skip(), "; ".join(labels) + ". Applies to new generations in all sessions; "
                    "saved outputs remain available.", snapshot)
        except (ValueError, OSError) as exc:
            # Restore this session's known table, without making any settings
            # changes. Metadata edits and planned-tool toggles fail closed.
            restored = deepcopy(snapshot["rows"]) if isinstance(snapshot, dict) and "rows" in snapshot else gr.skip()
            return restored, f"Could not change plugin setting: {exc}", gr.skip()


def build_tab():
    gr.Markdown("### Local tools and plugins\n"
                "Click an **Enabled** checkbox to turn a bundled tool on or off. "
                "Other columns are read-only. Planned tools stay disabled until implemented. "
                "Export libraries are checked locally; this panel does not install packages or connect external services. "
                "Tools use the existing OpenAI connection when generating outputs. "
                "Reviewing flashcards and checking quiz answers run locally.")
    status = gr.Textbox(label="Plugin status", interactive=False)
    snapshot = gr.State(None)
    catalog = gr.Dataframe(headers=HEADERS, value=[], interactive=True, type="pandas",
                           datatype=["str", "bool", "str", "str", "str"],
                           static_columns=STATIC_COLUMNS, row_count=(len(CATALOG), "fixed"),
                           column_count=(len(HEADERS), "fixed"), wrap=True, max_height=420,
                           label="Bundled tools")
    refresh = gr.Button("Refresh status")
    outputs = [catalog, status, snapshot]
    queue = {"concurrency_id": "plugin-preferences", "concurrency_limit": 1}
    # Gradio 6.30's boolean cells emit input; edit is only emitted for text-cell
    # blur. Preserve every rapid checkbox action in this serialized queue.
    edit_event = catalog.input(on_catalog_input, [catalog, snapshot], outputs,
                               trigger_mode="multiple", **queue)
    refresh.click(on_refresh, outputs=outputs, **queue)
    # The root shell registers loading, keeping construction free of file I/O.
    return {"catalog": catalog, "status": status, "snapshot": snapshot, "refresh": refresh,
            "refresh_callback": on_refresh, "edit_callback": on_catalog_input,
            "edit_event": edit_event, "load_outputs": outputs}
