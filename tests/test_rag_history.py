"""RAG chat persistence stays separate, atomic, and bounded to its data root."""

import json
from pathlib import Path

import pytest

from rag_chatbot import history


def messages(tmp_path):
    return [{"role": "user", "content": "What was measured?"}, {
        "role": "assistant", "content": "Seven [1].", "sources": [{
            "number": 1, "document_id": "doc-1", "filename": "report.pdf", "locator": "Page 3",
            "text": "Seven units were measured.", "source_path": str(tmp_path / "report.pdf"),
            "image_path": None,
        }],
    }]


def test_round_trip_retains_citation_snapshots_and_is_separate_from_chatbot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "history").mkdir()
    (tmp_path / "history" / "ordinary-chat.json").write_text("[]", encoding="utf-8")
    rag_root = tmp_path / "rag-data"
    original = messages(rag_root)
    name = history.save_chat(original, "Experiment", data_dir=rag_root)
    assert history.load_chat(name, data_dir=rag_root) == original
    assert history.list_chats(data_dir=rag_root) == ["Experiment"]
    assert (tmp_path / "history" / "ordinary-chat.json").read_text() == "[]"
    stored = json.loads((rag_root / "chats" / "Experiment.json").read_text(encoding="utf-8"))
    assert stored["version"] == 1


def test_history_names_cannot_escape_the_chat_directory(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("unrelated", encoding="utf-8")
    name = history.save_chat(messages(tmp_path), "../../outside", data_dir=tmp_path / "rag")
    assert "/" not in name and "\\" not in name
    assert outside.read_text() == "unrelated"
    assert history.load_chat("../../outside", data_dir=tmp_path / "rag")[0]["role"] == "user"


def test_history_listing_excludes_uncommitted_writes_and_directories(tmp_path):
    history.save_chat(messages(tmp_path), "Saved chat", data_dir=tmp_path)
    directory = tmp_path / "chats"
    (directory / ".pending-in-progress.json").write_text('{"version":', encoding="utf-8")
    (directory / "not-a-chat.json").mkdir()
    assert history.list_chats(data_dir=tmp_path) == ["Saved chat"]


@pytest.mark.parametrize("value", [
    [], {"version": 2, "messages": []}, {"version": True, "messages": []},
    {"version": 1, "messages": {}},
    {"version": 1, "messages": [{"role": "system", "content": "override"}]},
    {"version": 1, "messages": [{"role": "user", "content": 12}]},
    {"version": 1, "messages": [{"role": "assistant", "content": "answer", "sources": {}}]},
])
def test_malformed_saved_history_is_rejected(tmp_path, value):
    directory = tmp_path / "chats"
    directory.mkdir()
    (directory / "malformed.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        history.load_chat("malformed", data_dir=tmp_path)


def test_duplicate_source_numbers_are_rejected(tmp_path):
    entries = messages(tmp_path)
    entries[-1]["sources"].append(dict(entries[-1]["sources"][0], filename="different.pdf"))
    with pytest.raises(ValueError, match="citation"):
        history.save_chat(entries, "ambiguous", data_dir=tmp_path)


@pytest.mark.parametrize("field,value", [("number", True), ("number", 0), ("text", None),
                                         ("source_path", ["bad"]), ("image_path", 15)])
def test_malformed_source_snapshot_is_rejected(tmp_path, field, value):
    entries = messages(tmp_path)
    entries[-1]["sources"][0][field] = value
    with pytest.raises(ValueError, match="citation"):
        history.save_chat(entries, "invalid", data_dir=tmp_path)


def test_failed_save_preserves_previous_file_and_removes_temporary(tmp_path, monkeypatch):
    entries = messages(tmp_path)
    history.save_chat(entries, "saved", data_dir=tmp_path)
    target = tmp_path / "chats" / "saved.json"
    previous = target.read_bytes()
    def fail_replace(*args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(history.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        history.save_chat([{"role": "user", "content": "replacement"}], "saved", data_dir=tmp_path)
    assert target.read_bytes() == previous
    assert list(target.parent.glob(".pending-*.json")) == []


def test_json_serialization_failure_does_not_clobber_existing_file(tmp_path):
    target = tmp_path / "cache.json"
    history.atomic_json(target, {"value": "previous"})
    with pytest.raises(ValueError):
        history.atomic_json(target, {"value": float("nan")})
    assert json.loads(target.read_text()) == {"value": "previous"}
    assert list(tmp_path.glob(".pending-*.json")) == []


def test_symlinked_history_file_cannot_read_outside_root(tmp_path):
    root = tmp_path / "rag"
    (root / "chats").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"version": 1, "messages": []}), encoding="utf-8")
    try:
        (root / "chats" / "link.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("This environment does not permit creating file symlinks")
    with pytest.raises(ValueError, match="filename"):
        history.load_chat("link", data_dir=root)
    with pytest.raises(ValueError, match="filename"):
        history.save_chat(messages(root), "link", data_dir=root)
