"""RAG UI callback and shell tests without a database or paid API requests."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    return importlib.import_module("rag_chatbot.chatbot")


@pytest.fixture
def service(ui, monkeypatch, tmp_path):
    documents = [
        {"id": "ready-1", "name": "Report.pdf", "collection": "Research", "status": "ready",
         "chunk_count": 3, "error": ""},
        {"id": "failed-2", "name": "Scan.pdf", "collection": "Research", "status": "failed",
         "chunk_count": 0, "error": "Enable vision to read this scan."},
    ]
    instance = SimpleNamespace(
        settings=SimpleNamespace(data_dir=tmp_path, fingerprint=lambda: "current-v1"),
        store=SimpleNamespace(collections=lambda: ["Research"],
                              list_documents=lambda collection=None: list(documents)),
    )
    monkeypatch.setattr(ui, "get_service", lambda workspace: instance)
    return instance


def _source(tmp_path):
    original = tmp_path / "report.txt"
    original.write_text("The launch date is June 1.", encoding="utf8")
    return {"number": 1, "document_id": "ready-1", "filename": "report.txt",
            "locator": "paragraph 1", "text": "The launch date is June 1.",
            "source_path": str(original), "image_path": None}


def test_root_tabs_build_without_initializing_rag(ui, monkeypatch):
    from chatbot import history_store

    def unexpected(*args):
        pytest.fail("Building the UI must not initialize the library or connect to an API")

    monkeypatch.setattr(ui, "get_service", unexpected)
    monkeypatch.setattr(history_store, "get_history_file_list", lambda: [])
    host = importlib.import_module("webui")
    app = host.create_app()
    tabs = [component["props"]["label"] for component in app.config["components"]
            if component["type"] == "tabitem"]
    assert tabs == ["Chat", "RAG Chat"]
    assert app.chat_tab.state is not app.rag_tab.history_state
    assert app.rag_tab.history_state is not app.rag_tab.sources_state
    assert app.chat_tab.send_event["id"] != app.rag_tab.send_event["id"]
    dependencies = app.config["dependencies"]
    for button, event in [(app.chat_tab.stop_button, app.chat_tab.send_event),
                          (app.rag_tab.stop_button, app.rag_tab.send_event),
                          (app.rag_tab.index_stop_button, app.rag_tab.index_event)]:
        cancellations = [item["cancels"] for item in dependencies
                         if (button._id, "click") in map(tuple, item["targets"]) and item["cancels"]]
        assert cancellations == [[event["id"]]]
    index_stop = next(item for item in dependencies
                      if app.fns[item["id"]].fn is ui.on_stop_index)
    assert index_stop["queue"] is False


def test_library_refresh_preserves_valid_scope_and_shows_failed_documents(ui, service):
    frame, collection, documents, status = ui.on_refresh("Default", "Research", ["ready-1", "removed"])[:4]
    assert frame["Status"].tolist() == ["ready", "failed"]
    assert "vision" in frame.iloc[1]["Error"]
    assert collection["value"] == "Research"
    assert documents["value"] == ["ready-1"]
    assert {choice[1] for choice in documents["choices"]} == {"ready-1", "failed-2"}
    assert "refreshed" in status


def test_answer_preserves_source_metadata_but_only_renders_chat_content(ui, service, tmp_path):
    source = _source(tmp_path)
    calls = []
    completed = [{"role": "user", "content": "When?"},
                 {"role": "assistant", "content": "June 1 [1].", "sources": [source]}]

    def answer(question, history, collection, **kwargs):
        calls.append((question, history, collection, kwargs))
        yield completed, [source]

    service.answer = answer
    updates = list(ui.on_answer(" When? ", [], "Research", ["ready-1"], "gpt-4o", "auto", "Default", {"revision": 0}))
    state, display, sources, frame, selector, question, status, *preview = updates[-1]
    assert state == completed
    assert "sources" not in display[-1]
    assert sources == [source]
    assert frame.iloc[0]["Reference"] == "[1]"
    assert calls[0][3]["document_ids"] == ["ready-1"]
    assert question == ui.gr.skip()
    assert preview == [None, None, "", None]


def test_failed_answer_rolls_back_state_without_overwriting_new_draft(ui, service):
    previous = [{"role": "user", "content": "Prior question"},
                {"role": "assistant", "content": "Prior answer"}]

    def answer(*args, **kwargs):
        yield previous + [{"role": "assistant", "content": "partial"}], []
        raise RuntimeError("Connection interrupted")

    service.answer = answer
    final = list(ui.on_answer("New question", previous, "Research", ["ready-1"], "gpt-4o", "auto", "Default", {"revision": 0}))[-1]
    assert final[0] == previous
    assert final[1][-2]["content"] == "New question"
    assert "Connection interrupted" in final[1][-1]["content"]
    assert final[2] == []
    assert final[5] == ui.gr.skip()


def test_stopping_answer_closes_service_generator(ui, service):
    closed = []

    def answer(*args, **kwargs):
        try:
            yield [{"role": "assistant", "content": "partial"}], []
            yield [{"role": "assistant", "content": "complete"}], []
        finally:
            closed.append(True)

    service.answer = answer
    stream = ui.on_answer("Question", [], "Research", ["ready-1"], "gpt-4o", "auto", "Default", {"revision": 0})
    next(stream)
    next(stream)
    stream.close()
    assert closed == [True]


def test_stop_discards_pending_turn_but_keeps_question_visible(ui):
    previous = [{"role": "assistant", "content": "Prior answer"}]
    partial = previous + [{"role": "user", "content": "New question"},
                          {"role": "assistant", "content": "Partial response"}]
    result = ui.on_stop_chat(partial, {"history": previous, "question": "New question"})
    assert result[0] == previous
    assert result[1][-2]["content"] == "New question"
    assert "Generation stopped" in result[1][-1]["content"]
    assert result[2] == [] and result[-1] is None
    assert result[5] == ui.gr.skip()
    assert ui.on_stop_chat(previous, None)[0] == ui.gr.skip()


def test_stale_index_is_marked_for_reindexing(ui, service):
    service.store.list_documents = lambda collection=None: [
        {"id": "stale", "name": "Old.pdf", "collection": "Research", "status": "ready",
         "fingerprint": "old-v1", "chunk_count": 2},
    ]
    frame, _, documents, _ = ui.on_refresh("Default", "Research")[:4]
    assert frame.iloc[0]["Status"] == "reindex needed"
    assert documents["choices"] == [("Old.pdf (reindex needed)", "stale")]


def test_index_progress_and_error_are_visible(ui, service):
    def ingest(paths, collection, use_vision=False, display_names=None):
        assert paths == ["upload.pdf"] and collection == "Research" and use_vision
        yield "Reading page 1"
        raise RuntimeError("Could not read page 2")

    service.ingest = ingest
    updates = list(ui.on_index(["upload.pdf"], "Research", True, "Default", {"revision": 0}))
    assert updates[0][0] == "Reading page 1"
    assert "Could not read page 2" in updates[-1][0]
    assert updates[-1][1]["Status"].tolist() == ["ready", "failed"]


def test_stop_indexing_signals_service_immediately(ui, service):
    stopped = []
    service.cancel_ingestion = lambda: stopped.append(True)
    assert "Stopping after the current request" in ui.on_stop_index("Default")
    assert stopped == [True]


def test_reference_previews_are_limited_to_the_library(ui, service, tmp_path):
    source = _source(tmp_path)
    original, image, excerpt = ui.on_select_source("0", [source], "Default")
    assert original == source["source_path"] and image is None
    assert excerpt == source["text"]
    unsafe = dict(source, source_path=str(tmp_path.parent / "private.txt"))
    original, image, excerpt = ui.on_select_source("0", [unsafe], "Default")
    assert original is None and image is None
    assert "outside the document library" in excerpt
    assert ui.on_select_source("-1", [source], "Default")[:2] == (None, None)


def test_saved_chat_restores_sources_and_failed_load_keeps_current_chat(ui, service, monkeypatch, tmp_path):
    source = _source(tmp_path)
    saved = [{"role": "assistant", "content": "June 1 [1].", "sources": [source]}]

    def load(name, *, data_dir):
        assert data_dir == service.settings.data_dir
        if name == "missing":
            raise FileNotFoundError("Missing chat")
        return saved

    monkeypatch.setattr(ui, "_history_store", lambda: SimpleNamespace(load_chat=load))
    restored = ui.on_load_chat("Saved chat", [], None, "Default")
    assert restored[0] == saved and restored[2] == [source]
    failed = ui.on_load_chat("missing", [], None, "Default")
    assert all(value == ui.gr.skip() for value in failed[:6])
    assert "Missing chat" in failed[6]
    canceled = ui.on_load_chat("missing", saved + [{"role": "assistant", "content": "partial"}],
                              {"history": saved, "question": "Pending question"}, "Default")
    assert canceled[0] == saved and canceled[-1] is None
    assert "Pending question" == canceled[1][-2]["content"]


def test_delete_rejects_documents_outside_selected_collection(ui, service):
    deleted = []
    service.store.delete_document = deleted.append
    result = ui.on_delete_documents(["other-collection"], "Research", "Default")
    assert "do not belong" in result[3]
    assert deleted == []
    result = ui.on_delete_documents(["failed-2"], "Research", "Default")
    assert deleted == ["failed-2"]
    assert "removed from search" in result[3]


def test_workspace_layout_and_event_scope(ui, monkeypatch):
    with ui.gr.Blocks() as app:
        controls = ui.build_tab()
    assert controls.sources_column.scale == 1 and controls.right_column.scale == 2
    assert controls.chat_column.scale == controls.details_column.scale == 1
    assert controls.top_row.equal_height is True
    parents = {}
    children = {}
    def walk(node):
        children[node["id"]] = [child["id"] for child in node.get("children", [])]
        for child in node.get("children", []):
            parents[child["id"]] = node["id"]
            walk(child)
    walk(app.config["layout"])
    assert children[controls.outer_row._id] == [controls.sources_column._id, controls.right_column._id]
    assert children[controls.right_column._id] == [controls.top_row._id, controls.bottom_row._id]
    assert children[controls.top_row._id] == [controls.chat_column._id, controls.details_column._id]
    assert parents[controls.prompt_group._id] == controls.chat_column._id
    assert parents[controls.viewer_group._id] == controls.sources_column._id
    assert children[controls.details_column._id] == [controls.history_column._id,
        controls.references_group._id]
    assert not hasattr(controls, "selected_reference_group")
    def descendants(component_id):
        for child_id in children[component_id]:
            yield child_id
            yield from descendants(child_id)
    reference_controls = [controls.sources_table._id, controls.source_selector._id,
        controls.original_file._id, controls.source_image._id, controls.source_text._id]
    assert [component_id for component_id in descendants(controls.references_group._id)
            if component_id in reference_controls] == reference_controls
    assert parents[controls.tools_column._id] == controls.bottom_row._id
    assert parents[controls.history_buttons._id] == controls.history_column._id
    assert children[controls.history_buttons._id] == [controls.load_button._id, controls.save_button._id]
    assert isinstance(controls.document_filter, ui.gr.CheckboxGroup)
    assert not hasattr(controls, "review_button")
    assert controls.review_event["trigger_after"] == controls.review_selection_event["id"]
    assert controls.review_event["trigger_mode"] == "always_last"
    assert controls.review_selection_event["targets"] == [(controls.review_document._id, "input")]
    assert controls.load_event["targets"] == [(app._id, "load")]
    assert controls.workspace_name.allow_custom_value
    assert controls.workspace_state is not controls.workspace_name
    assert controls.workspace_state.value is None
    dependencies = app.config["dependencies"]
    review_cancels = [event["cancels"] for event in dependencies
                     if (controls.review_document._id, "input") in map(tuple, event["targets"])
                     and event["cancels"]]
    assert review_cancels == [[event["id"] for event in controls.viewer_events[1:]]]
    cancellation = next(event for event in dependencies if event["cancels"] == [
        controls.send_event["id"], controls.index_event["id"],
        *(event["id"] for event in controls.viewer_events)])
    assert cancellation["targets"] == [(controls.workspace_button._id, "click")]
    assert controls.workspace_open_event["trigger_after"] == controls.workspace_cancel_event["id"]
    assert controls.workspace_ready_event["trigger_after"] == controls.workspace_open_event["id"]
    functions = {fn.fn: fn for fn in app.fns.values() if fn.fn is not None}
    for callback in (ui.on_answer, ui.on_index, ui.on_refresh, ui.on_select_source,
                     ui.on_save_chat, ui.on_load_chat, ui.on_delete_documents, ui.on_stop_index,
                     ui.on_review_document, ui.on_preview_document):
        assert controls.workspace_state in functions[callback].inputs
        assert controls.workspace_name not in functions[callback].inputs
    for component in (controls.history_state, controls.sources_state, controls.pending_state,
                      controls.question, controls.upload, controls.document_filter, controls.chat_name):
        assert component in functions[ui.on_open_workspace].outputs
    assert controls.viewer_state in functions[ui.on_open_workspace].outputs
    assert functions[ui.on_review_document].concurrency_id != functions[ui.on_open_workspace].concurrency_id
    assert controls.viewer_image not in functions[ui.on_answer].outputs
    assert controls.upload_filename in functions[ui.on_index].inputs
    assert controls.upload_editor_path in functions[ui.on_index].inputs
    assert controls.upload_names_state in functions[ui.on_open_workspace].outputs


def test_workspace_open_clears_state_and_errors_keep_active(ui, monkeypatch, tmp_path):
    services = {}
    for name in ("Alpha", "Beta"):
        path = tmp_path / name
        path.mkdir()
        services[name] = SimpleNamespace(settings=SimpleNamespace(data_dir=path, fingerprint=lambda: "v1"),
            store=SimpleNamespace(collections=lambda: [], list_documents=lambda collection=None: []))
    opened = []
    def open_workspace(name, create=False):
        if name not in services:
            raise ValueError("Invalid workspace name")
        opened.append(name)
        return services[name].settings.data_dir
    monkeypatch.setattr(ui, "get_service", services.__getitem__)
    monkeypatch.setattr(ui, "_workspaces", lambda: SimpleNamespace(workspace_path=open_workspace,
        list_workspaces=lambda: list(services), default_workspace=lambda: "Alpha"))
    monkeypatch.setattr(ui, "_history_store", lambda: SimpleNamespace(
        list_chats=lambda *, data_dir: [data_dir.name + " chat"]))
    revision = {"revision": 1, "switching": True}
    result = ui.on_open_workspace("Beta", "Alpha", operation_state=revision)
    assert revision == {"revision": 2, "switching": True, "viewer_revision": 1, "review_document_id": None, "selected_ids": [], "known_ready_ids": []}
    ready, button = ui.on_finish_workspace_switch(revision)
    assert ready is revision and revision == {"revision": 3, "switching": False, "viewer_revision": 1,
                                              "review_document_id": None, "selected_ids": [], "known_ready_ids": []}
    assert button["interactive"] is True
    assert len(result) == 24 + ui.REVIEW_OUTPUT_COUNT + ui.UPLOAD_OUTPUT_COUNT
    assert result[0] == "Beta" and result[1]["value"] == "Beta"
    assert result[5]["value"] == []  # document selection
    assert result[7]["choices"] == ["Beta chat"] and result[7]["value"] is None
    assert result[8] is None and result[11] == "RAG chat"  # upload/name
    assert result[13] == [] and result[15] == [] and result[18] == ""
    assert result[20:24] == (None, None, "", None)
    assert result[24]["value"] is None and result[25] is None
    failed = ui.on_open_workspace("../bad", "Alpha")
    assert failed[0] == ui.gr.skip() and "Invalid workspace" in failed[2]
    assert all(value == ui.gr.skip() for value in failed[3:])
    assert ui.on_initial_load()[0] == "Alpha"


def test_work_submitted_during_workspace_switch_does_not_start(ui, monkeypatch):
    def unexpected(workspace):
        pytest.fail("A workspace switch is pending; no service request may start")
    monkeypatch.setattr(ui, "get_service", unexpected)
    revision = {"revision": 1, "switching": True}
    answer = list(ui.on_answer("Question", [], "Default", ["ready-1"], "gpt-4o", "auto", "Alpha", revision))
    assert len(answer) == 1 and "Wait for the workspace" in answer[0][6]
    indexed = list(ui.on_index(["report.txt"], "Default", False, "Alpha", revision))
    assert len(indexed) == 1 and "Wait for the workspace" in indexed[0][0]


def test_switch_signals_old_index_and_discards_late_answer(ui, monkeypatch, tmp_path):
    revision = {"revision": 0}
    calls = []
    def answer(*args, **kwargs):
        try:
            calls.append("answer Alpha")
            yield [{"role": "assistant", "content": "partial"}], []
            yield [{"role": "assistant", "content": "late"}], []
        finally:
            calls.append("closed")
    service = SimpleNamespace(answer=answer, cancel_ingestion=lambda: calls.append("cancel Alpha"))
    monkeypatch.setattr(ui, "get_service", lambda workspace: service if workspace == "Alpha" else None)
    stream = ui.on_answer("Question", [], "Default", ["ready-1"], "gpt-4o", "auto", "Alpha", revision)
    next(stream)
    next(stream)
    switched, _, button = ui.on_prepare_workspace_switch("Alpha", revision)
    assert switched is revision and revision["revision"] == 1
    assert button["interactive"] is False
    assert list(stream) == []
    assert calls == ["answer Alpha", "cancel Alpha", "closed"]


def test_switch_discards_late_index_results(ui, monkeypatch):
    revision = {"revision": 0}
    calls = []
    def ingest(*args, **kwargs):
        try:
            yield "First document"
            yield "Old workspace complete"
        finally:
            calls.append("closed")
    service = SimpleNamespace(ingest=ingest, cancel_ingestion=lambda: calls.append("cancel"))
    monkeypatch.setattr(ui, "get_service", lambda workspace: service)
    stream = ui.on_index(["report.txt"], "Default", False, "Alpha", revision)
    assert next(stream)[0] == "First document"
    ui.on_prepare_workspace_switch("Alpha", revision)
    assert list(stream) == [] and calls == ["cancel", "closed"]


def test_history_and_reference_are_bound_to_active_workspace(ui, monkeypatch, tmp_path):
    services = {}
    for name in ("Alpha", "Beta"):
        path = tmp_path / name
        path.mkdir()
        services[name] = SimpleNamespace(settings=SimpleNamespace(data_dir=path))
    monkeypatch.setattr(ui, "get_service", services.__getitem__)
    from rag_chatbot import history
    monkeypatch.setattr(ui, "_history_store", lambda: history)
    source = _source(services["Alpha"].settings.data_dir)
    saved = [{"role": "assistant", "content": "Workspace Alpha answer", "sources": [source]}]
    assert ui.on_save_chat(saved, "same-name", "Alpha")[0].startswith("Saved")
    assert not history.list_chats(data_dir=services["Beta"].settings.data_dir)
    assert ui.on_load_chat("same-name", [], None, "Beta")[0] == ui.gr.skip()
    assert ui.on_load_chat("same-name", [], None, "Alpha")[0] == saved
    rejected = ui.on_select_source("0", [source], "Beta")
    assert rejected[:2] == (None, None) and "outside the document library" in rejected[2]
    assert ui.on_select_source("0", [source], "Alpha")[0] == source["source_path"]


def _preview(**changes):
    values = dict(image=None, html="", table=[], headers=[], text="Original document text",
                  unit_label="Page", unit_count=3, unit=1, sheets=[], sheet=None, status="Preview ready.")
    values.update(changes)
    return SimpleNamespace(**values)


def test_document_review_resolves_active_id_and_navigation(ui, service, monkeypatch, tmp_path):
    original = _source(tmp_path)["source_path"]
    service.store.get_document = lambda document_id: {"id": document_id, "path": original, "status": "ready"}
    calls = []
    def preview(path, *, data_dir, unit, sheet):
        calls.append((path, data_dir, unit, sheet))
        if sheet:
            return _preview(text="", table=[["Budget", 42]], headers=["Item", "Amount"],
                            unit_label="Rows page", sheets=["Budget", "Summary"], sheet=sheet, unit=unit)
        return _preview(unit=unit)
    monkeypatch.setattr(ui, "preview_document", preview)
    operation = {"revision": 0}
    opened = ui.on_review_document("ready-1", "Default", operation)
    assert opened[0] == {"workspace": "Default", "document_id": "ready-1", "unit": 1, "sheet": None}
    assert opened[1] == original and opened[5]["visible"] is True
    assert opened[2]["visible"] is False and opened[3]["visible"] is False
    assert calls == [(original, tmp_path, 1, None)]
    next_page = ui.on_preview_next("ready-1", 1, None, "Default", operation)
    assert next_page[6]["value"] == 2 and next_page[7] == "Page 2 of 3"
    first_page = ui.on_preview_previous("ready-1", 1, None, "Default", operation)
    assert first_page[6]["value"] == 1
    sheet = ui.on_preview_sheet("ready-1", "Budget", "Default", operation)
    assert calls[-1][2:] == (1, "Budget")
    assert sheet[4]["visible"] is True and sheet[4]["value"].iloc[0]["Amount"] == 42
    assert sheet[5]["visible"] is False
    assert sheet[8]["choices"] == ["Budget", "Summary"] and sheet[8]["value"] == "Budget"


def test_document_review_rejects_foreign_deleted_and_outside_paths(ui, service, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid source must never reach a document renderer")
    monkeypatch.setattr(ui, "preview_document", unexpected)
    documents = {
        "deleted": {"status": "deleted", "path": str(tmp_path / "old.txt")},
        "outside": {"status": "ready", "path": str(tmp_path.parent / "sibling" / "source.txt")},
    }
    service.store.get_document = documents.get
    for document_id in ("foreign-id", "deleted", "outside"):
        result = ui.on_review_document(document_id, "Default", {"revision": 0})
        assert result[0] is None and result[1] is None
        assert result[9].startswith("Could not review document")


@pytest.mark.parametrize("invalidate", ["workspace", "selection", "delete"])
def test_document_review_discards_late_results(ui, service, monkeypatch, tmp_path, invalidate):
    document = {"id": "ready-1", "path": _source(tmp_path)["source_path"], "status": "ready"}
    service.store.get_document = lambda document_id: document
    operation = {"revision": 0}
    def preview(*args, **kwargs):
        if invalidate == "workspace":
            operation["revision"] += 1
        elif invalidate == "selection":
            ui.on_review_selection("another-source", operation)
        else:
            document["status"] = "deleted"
        return _preview()
    monkeypatch.setattr(ui, "preview_document", preview)
    result = ui.on_review_document("ready-1", "Default", operation)
    if invalidate == "delete":
        assert result[0] is None and result[1] is None and "removed" in result[9]
    else:
        assert result == (ui.gr.skip(),) * ui.VIEWER_OUTPUT_COUNT


def test_review_choices_refresh_without_rendering_and_preserve_selection(ui, service, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Refreshing a library must not render a source")
    monkeypatch.setattr(ui, "preview_document", unexpected)
    operation = {"revision": 0, "viewer_revision": 4}
    refreshed = ui.on_refresh("Default", "Research", [], "failed-2", operation)
    assert refreshed[4]["value"] == "failed-2"
    assert {choice[1] for choice in refreshed[4]["choices"]} == {"ready-1", "failed-2"}
    assert refreshed[5:] == (ui.gr.skip(),) * ui.VIEWER_OUTPUT_COUNT
    assert operation["viewer_revision"] == 4
    service.store.list_documents = lambda collection=None: []
    removed = ui.on_refresh("Default", "Research", [], "failed-2", operation)
    assert removed[4]["value"] is None and removed[5] is None and removed[6] is None
    assert operation["viewer_revision"] == 5


def test_deleting_reviewed_document_clears_viewer(ui, service):
    documents = service.store.list_documents()
    service.store.list_documents = lambda collection=None: list(documents)
    service.store.delete_document = lambda document_id: documents.__setitem__(slice(None),
        [document for document in documents if document["id"] != document_id])
    operation = {"revision": 0, "viewer_revision": 2}
    removed = ui.on_delete_documents(["ready-1"], "Research", "Default", "ready-1", operation)
    assert removed[4]["value"] is None and removed[5] is None
    assert operation["viewer_revision"] == 3


def test_index_completion_preserves_a_newer_review_selection(ui, service):
    service.ingest = lambda *args, **kwargs: iter(["Indexing"])
    operation = {"revision": 0, "review_document_id": "ready-1"}
    stream = ui.on_index(["report.txt"], "Research", False, "Default", operation, "ready-1")
    next(stream)
    ui.on_review_selection("failed-2", operation)
    final = list(stream)[-1]
    assert final[4]["value"] == "failed-2"
    assert final[5:] == (ui.gr.skip(),) * ui.VIEWER_OUTPUT_COUNT


def test_queued_preview_for_an_old_selection_never_renders(ui, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A superseded preview should not read storage or run a converter")
    monkeypatch.setattr(ui, "get_service", unexpected)
    operation = {"revision": 0, "review_document_id": "new-selection"}
    assert ui.on_review_document("old-selection", "Default", operation) == (
        ui.gr.skip(),) * ui.VIEWER_OUTPUT_COUNT


def test_upload_names_are_per_path_and_live_index_edit_is_captured(ui, service):
    first, second = "/one/report.pdf", "/two/report.pdf"
    state, filename, selector, owner = ui.on_upload_change([first, second], "", None, "", {})
    assert filename["value"] == "report.pdf" and owner == first
    assert selector["visible"] is True
    assert ui._upload_display_names([first, second], "report.pdf", first, state, owner) == [None, None]
    state, filename, owner = ui.on_upload_selection(second, "Alpha report.pdf", owner, state)
    assert filename == "report.pdf" and owner == second
    ui.on_upload_name("Beta report.pdf", owner, state)
    state, filename, owner = ui.on_upload_selection(first, "Beta report.pdf", owner, state)
    assert filename == "Alpha report.pdf" and owner == first
    captured = []
    service.ingest = lambda paths, collection, **kwargs: captured.append(kwargs["display_names"]) or iter([])
    list(ui.on_index([first, second], "Research", False, "Default", {"revision": 0},
                     filename="Latest alpha.pdf", upload_names=state, selected_upload=first, editor_path=owner))
    assert captured == [["Latest alpha.pdf", "Beta report.pdf"]]
    # The generator captures a copy; it does not overwrite edits for a later run.
    assert state["names"][first]["name"] == "Alpha report.pdf"
    state, filename, selector, owner = ui.on_upload_change([second], "Latest alpha.pdf", first, owner, state)
    assert list(state["names"]) == [second] and filename["value"] == "Beta report.pdf"
    state, filename, selector, owner = ui.on_upload_change([], "Beta report.pdf", second, owner, state)
    assert state["names"] == {} and filename["value"] == "" and owner == ""


def test_index_waits_for_matching_upload_selector_and_editor(ui, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An inconsistent upload snapshot must not start storage or ingestion")
    monkeypatch.setattr(ui, "get_service", unexpected)
    state = {"names": {"/one/report.pdf": {"name": "Alpha.pdf", "dirty": True},
                       "/two/report.pdf": {"name": "Beta.pdf", "dirty": True}}, "selected": "/two/report.pdf"}
    # Even if the selector callback already changed server State, the browser
    # still submits the old editor owner until its textbox update arrives.
    results = list(ui.on_index(list(state["names"]), "Default", False, "Default", {"revision": 0},
        filename="Alpha.pdf", upload_names=state, selected_upload="/two/report.pdf", editor_path="/one/report.pdf"))
    assert len(results) == 1 and "Wait for the selected upload" in results[0][0]


def test_explicit_edit_back_to_original_name_stays_an_override(ui):
    state, _, _, owner = ui.on_upload_change(["/one/report.pdf"], "", None, "", {})
    ui.on_upload_name("Custom.pdf", owner, state)
    ui.on_upload_name("report.pdf", owner, state)
    assert ui._upload_display_names([owner], "report.pdf", owner, state, owner) == ["report.pdf"]


def test_source_checks_default_ready_preserve_unchecks_and_add_new_ready(ui, service):
    documents = service.store.list_documents()
    service.store.list_documents = lambda collection=None: list(documents)
    operation = {"revision": 0}
    initial = ui._library_updates("Default", "Research", operation_state=operation, reset=True)
    assert initial[2]["value"] == ["ready-1"]
    ui.on_source_selection([], operation)
    assert ui.on_refresh("Default", "Research", ["ready-1"], operation_state=operation)[2]["value"] == []
    documents.append({"id": "ready-3", "name": "New.txt", "status": "ready"})
    refreshed = ui.on_refresh("Default", "Research", [], operation_state=operation)
    assert refreshed[2]["value"] == ["ready-3"]
    all_ready = ui.on_select_all_sources("Default", "Research", operation)
    assert all_ready[0]["value"] == ["ready-1", "ready-3"]
    assert ui.on_clear_sources(operation)[0]["value"] == []
    assert operation["selected_ids"] == []


def test_index_completion_keeps_latest_checks_and_does_not_reselect_unchecked(ui, service):
    documents = service.store.list_documents()
    service.store.list_documents = lambda collection=None: list(documents)
    operation = {"revision": 0}
    ui._library_updates("Default", "Research", operation_state=operation, reset=True)
    def ingest(*args, **kwargs):
        yield "Indexing"
        documents.append({"id": "new", "name": "New.txt", "status": "ready"})
    service.ingest = ingest
    stream = ui.on_index(["new.txt"], "Research", False, "Default", operation)
    next(stream)
    ui.on_source_selection([], operation)
    final = list(stream)[-1]
    assert final[3]["value"] == ["new"] and operation["revision"] == 0


def test_empty_source_selection_makes_no_service_call(ui, monkeypatch):
    monkeypatch.setattr(ui, "get_service", lambda *args: pytest.fail("No sources means no API or service call"))
    previous = [{"role": "assistant", "content": "Earlier answer"}]
    results = list(ui.on_answer("Question", previous, "Default", [], "gpt-4o", "auto", "Default", {"revision": 0}))
    assert len(results) == 1 and "No sources selected" in results[0][6]
    assert results[0][:6] == (ui.gr.skip(),) * 6


def test_changing_checks_cancels_answer_without_canceling_index_revision(ui, service):
    operation = {"revision": 4}
    closed = []
    def answer(*args, **kwargs):
        try:
            yield [{"role": "assistant", "content": "Partial"}], []
            yield [{"role": "assistant", "content": "Late"}], []
        finally:
            closed.append(True)
    service.answer = answer
    stream = ui.on_answer("Question", [], "Research", ["ready-1"], "gpt-4o", "auto", "Default", operation)
    pending = next(stream)[-1]
    next(stream)
    reset = ui.on_source_selection([], operation, [], pending)
    assert reset[0] == [] and reset[-1] is None and "stopped" in reset[6]
    assert list(stream) == [] and closed == [True]
    assert operation["revision"] == 4 and operation["selected_ids"] == []


def test_new_review_selection_waits_for_old_render_and_then_displays(ui, service, monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    original = _source(tmp_path)["source_path"]
    service.store.get_document = lambda document_id: {"status": "ready", "path": original}
    started, release = Event(), Event()
    calls = []
    def render(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=3)
        return _preview()
    monkeypatch.setattr(ui, "preview_document", render)
    operation = {"revision": 0}
    ui.on_review_selection("first", operation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        old = pool.submit(ui.on_review_document, "first", "Default", operation)
        assert started.wait(timeout=3)
        cleared = ui.on_review_selection("second", operation)
        assert cleared[0] is None and cleared[1] is None
        release.set()
        assert old.result(timeout=3) == (ui.gr.skip(),) * ui.VIEWER_OUTPUT_COUNT
        # The always_last chained event invokes the newest selection once the
        # separate viewer queue is free; it is not canceled by its own input.
        fresh = pool.submit(ui.on_review_document, "second", "Default", operation).result(timeout=3)
    assert fresh[0]["document_id"] == "second" and fresh[1] == original
    assert len(calls) == 2
