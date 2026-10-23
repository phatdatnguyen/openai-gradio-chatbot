"""A superseded export must not replace a previously prepared download."""

import pytest

from rag_chatbot.tools import artifacts
from rag_chatbot.workspaces import create_workspace


@pytest.fixture(params=["summary", "spreadsheet", "mind_maps", "presentation", "flashcards", "quizzes"])
def tool(request, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    tool_id = request.param
    content = {
        "summary": {"sections": [{"heading": "Finding", "paragraphs": [{"text": "Measured.", "source_ids": ["s1"]}]}]},
        "spreadsheet": {"columns": ["Finding"], "rows": [{"document_id": "d1", "filename": "study.txt",
            "cells": [{"column": "Finding", "value": "Measured.", "source_ids": ["s1"]}]}]},
        "mind_maps": {"nodes": [{"id": "root", "parent_id": "", "label": "Finding",
                                "description": "Measured.", "source_ids": ["s1"]}]},
        "presentation": {"slides": [{"title": "Finding", "bullets": ["Measured."], "notes": "Measured.",
                                    "source_ids": ["s1"]}]},
        "flashcards": {"cards": [{"front": "What?", "back": "Measured.", "source_ids": ["s1"]}]},
        "quizzes": {"questions": [{"question": "What?", "options": ["Measured", "Modelled", "Not measured", "Unknown"],
                                    "correct_option": 0, "explanation": "Measured.", "source_ids": ["s1"]}]},
    }[tool_id]
    payload = {"tool_id": tool_id, "title": "Original", "model": "gpt-6-luna", "content": content,
        "sources": [{"id": "s1", "document_id": "d1", "filename": "study.txt", "locator": "Text",
                     "text": "Measured.", "source_path": "sources/study.txt"}],
        "documents": [{"id": "d1"}], "coverage": {}, "settings": {}}
    exporter = {"summary": artifacts.export_summary, "spreadsheet": artifacts.export_spreadsheet,
                "mind_maps": artifacts.export_mind_map, "presentation": artifacts.export_presentation,
                "flashcards": artifacts.export_learning, "quizzes": artifacts.export_learning}[tool_id]
    return payload, exporter


@pytest.mark.parametrize("stop_at", [1, 2])
def test_cancelled_export_keeps_existing_download_and_removes_temporary_file(tool, stop_at):
    payload, export = tool
    path = export("Research", payload, "json", "Download")
    original, checks = path.read_bytes(), []

    def cancelled():
        checks.append(True)
        return len(checks) == stop_at

    with pytest.raises(RuntimeError, match="export cancelled"):
        export("Research", {**payload, "title": "Stale replacement"}, "json", "Download", is_cancelled=cancelled)
    assert len(checks) == stop_at
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".pending-*"))


def test_invalid_export_guard_fails_before_creating_download(tool):
    payload, export = tool
    with pytest.raises(ValueError, match="guard must be callable"):
        export("Research", payload, "json", "Download", is_cancelled=True)
