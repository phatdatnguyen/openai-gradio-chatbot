"""Optional local HTTP/RPC smoke; no paid APIs or production data.

Run explicitly: python tests/manual_rag_smoke.py
This is intentionally outside pytest's test_*.py discovery: it starts a temporary
localhost server and uses in-memory Qdrant with deterministic model substitutes.
"""

import json
from functools import partial
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix="rag-rpc-smoke-") as directory:
        root = Path(directory)
        os.environ["RAG_DATA_DIR"] = str(root / "data")
        os.environ["GRADIO_TEMP_DIR"] = str(root / "gradio")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["OPENAI_API_KEY"] = "smoke-test-not-a-real-key"

        from gradio_client import Client, handle_file
        from qdrant_client import QdrantClient
        from chatbot import config, history_store
        from rag_chatbot import chatbot as ui, workspaces, research, research_ui, workspace_ui
        from rag_chatbot import service as service_module
        from rag_chatbot import source_actions
        from rag_chatbot.tools import presets, presets_ui
        from rag_chatbot.config import Settings
        from rag_chatbot.service import RAGService
        from rag_chatbot.store import LibraryStore

        def prohibit_paid_client():
            raise AssertionError("The smoke test must never invoke the OpenAI client.")

        config.get_client = prohibit_paid_client
        history_store.HISTORY_DIR = str(root / "original-chat-history")
        presets.SETTINGS_PATH = root / "studio-presets.json"
        calls = {"embeddings": 0, "answers": 0}

        def embed(texts):
            calls["embeddings"] += 1
            return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

        def answer(model, history, **kwargs):
            calls["answers"] += 1
            date = "July 2" if "July 2" in history[-1]["content"] else "June 1"
            assert date in history[-1]["content"]
            yield "The launch date is "
            yield f"{date} [1]."

        def send_answer(model, history, **kwargs):
            assert kwargs["web_search"] == "medium"
            assert kwargs["temperature"] == 0.4 and kwargs["top_p"] == 0.8
            assert "Document evidence" not in history[-1]["content"] or "[]" in history[-1]["content"]
            calls["answers"] += 1
            return "Web-only reply from [OpenAI](https://openai.com/)."

        services = {}

        def get_service(workspace):
            if workspace not in services:
                settings = Settings(data_dir=workspaces.workspace_path(workspace),
                                    embedding_dimensions=8, chunk_size=80, chunk_overlap=10)
                store = LibraryStore(settings, vector_client=QdrantClient(":memory:"))
                services[workspace] = RAGService(settings, store=store, embedder=embed,
                                                  streamer=answer, sender=send_answer)
            return services[workspace]

        ui.get_service = get_service
        service_module.get_service = get_service

        import webui
        app = webui.demo
        names = {function.fn: function.api_name for function in app.fns.values()
                 if function.fn is not None}
        client = None
        try:
            app.queue().launch(server_name="127.0.0.1", prevent_thread_lock=True,
                               quiet=True, inbrowser=False, share=False, show_error=True, css=webui.APP_CSS)
            client = Client(app.local_url, verbose=False, download_files=root / "downloads")

            client_values = {}

            def call(function, *args, tool_id=None):
                block = next(item for item in app.fns.values() if (
                    isinstance(item.fn, partial) and item.fn.func is function and item.fn.args[0] == tool_id
                    if tool_id is not None else item.fn is function
                    or isinstance(item.fn, partial) and item.fn.func is function))
                submitted = list(args)
                inputs = [component for component in block.inputs if not component.skip_api]
                for component in inputs[len(submitted):]:
                    submitted.append(client_values.get(component._id, component.value))
                job = client.submit(*submitted, api_name="/" + block.api_name)
                job.result()
                outputs = [component for component in block.outputs if not component.skip_api]
                merged = {}
                for result in job.outputs():
                    values = result if len(outputs) != 1 else (result,)
                    for component, value in zip(outputs, values):
                        if value == {"__type__": "update", "props": {"__type__": "update"}}:
                            continue
                        if isinstance(value, dict) and value.get("__type__") == "update":
                            if isinstance(merged.get(component._id), dict):
                                merged[component._id] = {**merged[component._id], **value}
                            elif len(value) > 1:
                                merged[component._id] = value
                        else:
                            merged[component._id] = value
                for component_id, value in merged.items():
                    if isinstance(value, dict) and value.get("__type__") == "update":
                        if "value" in value:
                            client_values[component_id] = value["value"]
                        elif isinstance(value.get("props"), dict) and "value" in value["props"]:
                            client_values[component_id] = value["props"]["value"]
                    else:
                        client_values[component_id] = value
                return merged

            controls = app.rag_tab
            call(ui.on_initial_load)
            call(ui.on_workspace_extras)
            saved_sources = {}
            for workspace, date in [("Alpha", "June 1"), ("Beta", "July 2")]:
                preparing = call(ui.on_prepare_workspace_switch)
                assert "Opening workspace" in preparing[controls.workspace_status._id]
                opened = call(ui.on_open_workspace, workspace)
                calls_before_ready = dict(calls)
                blocked = call(ui.on_answer, "This must wait for workspace readiness",
                               "Default", [], "gpt-4o", "auto")
                assert "Wait for the workspace" in blocked[controls.status._id]
                assert calls == calls_before_ready
                call(ui.on_workspace_extras)
                call(ui.on_finish_workspace_switch)
                assert opened[controls.workspace_name._id]["value"] == workspace
                assert opened[controls.chatbot._id] == []
                assert opened[controls.sources_table._id]["data"] == []
                assert opened[controls.original_file._id] is None
                assert opened[controls.upload._id] is None
                assert opened[controls.document_filter._id]["value"] == []
                assert opened[controls.chat_name._id] == "RAG chat"
                assert opened[controls.question._id] == ""
                assert opened[controls.review_document._id]["value"] is None
                assert opened[controls.viewer_original._id] is None
                assert opened[controls.upload_filename._id]["value"] == ""
                assert opened[controls.upload_editor_path._id] == ""
                described = call(workspace_ui.on_save, f"Research project {workspace}")
                assert "saved" in described[controls.workspace_details.status._id]

                document = root / f"{workspace}-launch-notes.txt"
                document.write_text(f"Project {workspace} launches on {date}. The owner is Morgan.\n",
                                    encoding="utf-8")
                decoy = root / f"{workspace}-other-notes.txt"
                other_date = "July 2" if date == "June 1" else "June 1"
                decoy.write_text(f"A different project launches on {other_date}.\n", encoding="utf-8")
                files = [handle_file(str(document)), handle_file(str(decoy))]
                uploads = call(ui.on_upload_files_change, files, "", "")
                assert uploads[controls.upload_filename._id]["value"] == document.name
                choices = uploads[controls.upload_selector._id]["choices"]
                owner, other_owner = choices[0][1], choices[1][1]
                renamed = f"{workspace} approved launch.txt"
                switched_upload = call(ui.on_upload_selection, other_owner, renamed, owner)
                assert switched_upload[controls.upload_filename._id] == decoy.name
                call(ui.on_upload_name, "Unselected project.txt", other_owner)
                switched_back = call(ui.on_upload_selection, owner, "Unselected project.txt", other_owner)
                assert switched_back[controls.upload_filename._id] == renamed
                # Index must read this last textbox edit even without a separate
                # filename-input RPC first.
                renamed = f"{workspace} final launch.txt"
                indexed = call(ui.on_index_uploads, files, "Default", False, renamed, owner)
                rows = indexed[controls.library_table._id]["data"]
                assert len(rows) == 2 and all(row[1] == "ready" for row in rows), rows
                assert {row[0] for row in rows} == {renamed, "Unselected project.txt"}
                review_choices = indexed[controls.review_document._id]["choices"]
                document_id = next(choice[1] for choice in review_choices if choice[0] == renamed)
                assert set(indexed[controls.document_filter._id]["value"]) == {choice[1] for choice in review_choices}
                call(ui.on_clear_sources)
                before_empty = dict(calls)
                empty_answer = call(ui.on_answer, "No sources means no request", "Default", [], "gpt-4o", "auto")
                assert "No sources selected" in empty_answer[controls.status._id] and calls == before_empty
                checked = call(ui.on_select_all_sources, "Default")
                assert len(checked[controls.document_filter._id]["value"]) == 2
                call(ui.on_source_selection, [document_id])
                call(ui.on_review_selection, document_id)
                reviewed = call(ui.on_review_document, document_id)
                assert date in reviewed[controls.viewer_text._id]["value"]
                assert reviewed[controls.viewer_text._id]["visible"] is True
                assert Path(reviewed[controls.viewer_original._id]).is_file()
                assert reviewed[controls.viewer_unit._id]["value"] == 1
                refreshed = call(ui.on_refresh, "Default", [document_id], document_id)
                assert refreshed[controls.review_document._id]["value"] == document_id
                assert refreshed[controls.document_filter._id]["value"] == [document_id]

                answered = call(ui.on_answer, f"When does Project {workspace} launch?",
                                "Default", [document_id], "gpt-4o", "auto")
                assert f"{date} [1]" in str(answered[controls.chatbot._id])
                references = answered[controls.sources_table._id]["data"]
                assert references and references[0][0] == "[1]", references
                assert {reference[1] for reference in references} == {renamed}
                preview = call(ui.on_select_source, "0")
                assert date in preview[controls.source_text._id]
                original = Path(preview[controls.original_file._id])
                assert original.is_file() and date in original.read_text(encoding="utf-8")

                # Autosave and research interactions use the real HTTP/RPC
                # session, while all generation/embedding remains substituted.
                assert "autosaved" in answered[controls.status._id]
                assert len(list(services[workspace].settings.data_dir.glob("chats/Autosaved *.json"))) == 1
                before_full = calls["embeddings"]
                full_answer = call(ui.on_answer, f"Read all indexed text about Project {workspace}",
                    "Default", [document_id], "gpt-4o", "auto", config.WEB_SEARCH_OFF, True, 1.0, 1.0, "full")
                assert f"{date} [1]" in str(full_answer[controls.chatbot._id])
                assert calls["embeddings"] == before_full
                research_controls = controls.research
                before_research = dict(calls)
                searched = call(research_ui.on_search, "launches", "keyword", "Default", [document_id])
                assert date in str(searched[research_controls.results._id]), searched
                passage = call(research_ui.on_search_reference, "0")
                assert date in passage[research_controls.search_excerpt._id]
                excerpt_note = call(research_ui.on_capture_excerpt, "0")
                assert "Source excerpt" in excerpt_note[research_controls.provenance._id]
                captured = call(research_ui.on_capture_answer)
                assert "AI-derived" in captured[research_controls.provenance._id]
                edited = call(research_ui.on_save, f"{workspace} findings", f"Reviewed: {date}")
                assert "Edited by the user" in edited[research_controls.provenance._id]
                assert len(research.list_notes(workspace)) == 2
                filtered = call(research_ui.on_filter, "Reviewed")
                assert len(filtered[research_controls.note_selector._id]["choices"]) == 1
                assert calls == before_research
                for export_format in ("md", "json"):
                    call(research_ui.on_note_export_options, f"{workspace} findings", f"Reviewed: {date}",
                         "Research note", export_format)
                    exported = call(research_ui.on_note_export, f"{workspace} findings", f"Reviewed: {date}",
                                    "Research note", export_format)
                    exported_path = Path(exported[research_controls.note_download._id])
                    assert exported_path.suffix == "." + export_format
                    assert date in exported_path.read_text(encoding="utf-8")
                    assert calls == before_research

                summary = controls.tools.controls["summary"]
                if workspace == "Alpha":
                    preset_saved = call(presets_ui.on_save, "Methods", "Summarize the methods and limitations.",
                                        tool_id="summary")
                    assert "saved" in preset_saved[summary.presets.status._id]
                else:
                    refreshed_presets = call(presets_ui.on_refresh, tool_id="summary")
                    choices = refreshed_presets[summary.presets.selector._id]["choices"]
                    assert choices[0][0] == "Methods"
                    call(presets_ui.on_select, choices[0][1], tool_id="summary")
                applied = call(presets_ui.on_apply, tool_id="summary")
                assert applied[summary.instructions._id] == "Summarize the methods and limitations."
                assert calls == before_research

                saved = call(ui.on_save_chat, "RPC smoke chat")
                assert "Saved RPC smoke chat" in saved[controls.save_status._id]
                saved_file = services[workspace].settings.data_dir / "chats" / "RPC smoke chat.json"
                payload = json.loads(saved_file.read_text(encoding="utf-8"))
                assert payload["messages"][-1]["context_mode"] == "full"
                saved_sources[workspace] = payload["messages"][-1]["sources"]
                assert saved_sources[workspace][0]["filename"] == renamed
                assert {source["document_id"] for source in saved_sources[workspace]} == {document_id}

                call(ui.on_new_chat)
                loaded = call(ui.on_load_chat, "RPC smoke chat")
                assert f"{date} [1]" in str(loaded[controls.chatbot._id])
                assert loaded[controls.sources_table._id]["data"] == references
                restored = call(ui.on_select_source, "0")
                assert date in restored[controls.source_text._id]

                text_result = call(source_actions.on_text, "Pasted source.md",
                                   f"# Project {workspace}\nReviewed launch: {date}.", "Default")
                assert "1 indexed" in text_result[controls.source_actions.status._id]
                text_refresh = call(ui.on_source_action_refresh, "Default")
                text_id = next(choice[1] for choice in text_refresh[controls.review_document._id]["choices"]
                               if choice[0] == "Pasted source.md")
                assert set(text_refresh[controls.document_filter._id]["value"]) == {document_id, text_id}
                text_document = services[workspace].store.get_document(text_id)
                services[workspace].store.stage_document(text_document["path"], "Default",
                    services[workspace].settings.fingerprint(), force=True, display_name="Pasted source.md")
                services[workspace].store.fail_document(text_id, "Simulated failure for retry coverage")
                call(ui.on_refresh, "Default", [document_id, text_id], document_id)
                call(ui.on_source_selection, [text_id])
                before_retry = calls["embeddings"]
                retried = call(source_actions.on_retry, "Default", False)
                assert "1 indexed" in retried[controls.source_actions.status._id], retried
                assert calls["embeddings"] == before_retry  # Reuse the verified cached vectors.
                call(ui.on_source_action_refresh, "Default")
                assert services[workspace].store.get_document(text_id)["status"] == "ready"
                call(ui.on_source_selection, [document_id])

            # Rendering also works for originals that have no searchable text.
            # Stage local fixtures directly so this preview check needs no OCR.
            from pypdf import PdfWriter
            from openpyxl import Workbook

            pdf_path = root / "two-pages.pdf"
            pdf = PdfWriter()
            pdf.add_blank_page(width=200, height=200)
            pdf.add_blank_page(width=200, height=200)
            pdf.write(pdf_path)
            workbook_path = root / "two-sheets.xlsx"
            workbook = Workbook()
            workbook.active.title = "First"
            workbook.active.append(["Item", "Count"])
            workbook.active.append(["First sheet item", 1])
            second_sheet = workbook.create_sheet("Second")
            second_sheet.append(["Item", "Count"])
            second_sheet.append(["Second sheet item", 2])
            workbook.save(workbook_path)
            workspace_service = services["Beta"]
            pdf_document = workspace_service.store.stage_document(
                pdf_path, "Default", workspace_service.settings.fingerprint())
            workbook_document = workspace_service.store.stage_document(
                workbook_path, "Default", workspace_service.settings.fingerprint())
            call(ui.on_refresh, "Default", [], None)
            call(ui.on_review_selection, pdf_document["id"])
            pdf_preview = call(ui.on_review_document, pdf_document["id"])
            assert "1 of 2" in pdf_preview[controls.viewer_unit_info._id]
            assert Path(pdf_preview[controls.viewer_image._id]["value"]).is_file()
            second_page = call(ui.on_preview_next, pdf_document["id"], 1, None)
            assert "2 of 2" in second_page[controls.viewer_unit_info._id]
            call(ui.on_review_selection, workbook_document["id"])
            sheet_preview = call(ui.on_review_document, workbook_document["id"])
            assert "First sheet item" in str(sheet_preview[controls.viewer_table._id]["value"])
            other_sheet = call(ui.on_preview_sheet, workbook_document["id"], "Second")
            assert "Second sheet item" in str(other_sheet[controls.viewer_table._id]["value"])
            removed = call(ui.on_delete_documents, [workbook_document["id"]], "Default", workbook_document["id"])
            assert removed[controls.review_document._id]["value"] is None
            assert removed[controls.viewer_original._id] is None

            # Server-side active State, rather than an editable workspace draft,
            # is the only workspace input exposed to save/load/index endpoints.
            save_function = next(fn for fn in app.fns.values() if fn.fn is ui.on_save_chat)
            assert controls.workspace_name not in save_function.inputs
            assert controls.workspace_state in save_function.inputs
            rejected = ui.on_select_source("0", saved_sources["Alpha"], "Beta")
            assert rejected[:2] == (None, None) and "outside the document library" in rejected[2]
            switched = call(ui.on_workspace_selection, "Alpha")
            assert switched[controls.workspace_name._id]["value"] == "Alpha"
            assert switched[controls.workspace_name._id]["interactive"] is True
            assert "June 1 [1]" in str(switched[controls.chatbot._id])
            assert "July 2" not in str(switched[controls.chatbot._id])
            assert switched[controls.workspace_details.description._id] == "Research project Alpha"
            assert len(switched[controls.research.note_selector._id]["choices"]) == 2
            # A delayed blur from the old workspace must reach our canonical
            # selection guard, not fail in Gradio's stale choice preprocessor.
            stale_selection = call(ui.on_review_selection, workbook_document["id"])
            assert not stale_selection, stale_selection
            assert not call(ui.on_review_document, workbook_document["id"])
            assert not call(ui.on_preview_sheet, workbook_document["id"], "Second")
            assert not call(ui.on_upload_selection, "File", "old name.xlsx", "old-upload")
            assert not call(ui.on_select_source, "File")
            loaded = call(ui.on_load_chat, "RPC smoke chat")
            assert "June 1 [1]" in str(loaded[controls.chatbot._id])
            assert "July 2" not in str(loaded[controls.chatbot._id])
            unchanged = call(ui.on_workspace_selection, "Alpha")
            assert controls.chatbot._id not in unchanged
            before_web = calls["embeddings"]
            web_answer = call(ui.on_answer, "Search the web", "Default", [], "gpt-4o", "auto",
                              "medium", False, 0.4, 0.8)
            assert "Web-only reply" in str(web_answer[controls.chatbot._id])
            assert web_answer[controls.sources_table._id]["data"] == []
            assert calls["embeddings"] == before_web
            assert calls["embeddings"] >= 4 and calls["answers"] == 5, calls
            print("PASS: one-session HTTP upload/index/document review/answer/citation/save/load in two isolated workspaces")
            print("PASS: completed chats autosave and restore; notes and descriptions remain workspace-scoped")
            print("PASS: local source search, cited notes/edits, and shared Studio presets make no model calls")
            print("PASS: pasted text indexes, failed sources retry from cache, full context avoids query embeddings, notes export")
            print("PASS: automatic workspace selection opens immediately; unchanged selection leaves chat intact")
            print("PASS: delayed file/sheet/reference selections after a workspace switch are safely ignored")
            print("PASS: original PDF page images and spreadsheet sheet switching render through RPC")
            print("PASS: per-upload names reach citations; checked sources scope answers; empty checks make no API calls")
            print("PASS: optional web-only reply uses blocking generation and the selected sampling values")
            print("Fake model calls:", json.dumps(calls), "Paid API calls: 0")
        finally:
            if client is not None:
                client.close()
            app.close()
            for service in services.values():
                service.store.close()


if __name__ == "__main__":
    main()
