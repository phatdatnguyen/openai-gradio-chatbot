"""Completed RAG replies survive reloads without persisting partial or stale work."""

import copy
import os
from types import SimpleNamespace

import pytest

from rag_chatbot import autosave, chatbot as ui, history, workspaces


def messages(root, text="Measured 7 [1]."):
    return [{"role": "user", "content": "What was measured?"},
            {"role": "assistant", "content": text, "web_search": True, "sources": [{
                "number": 1, "document_id": "doc1", "filename": "measurements.txt",
                "locator": "Text", "text": "Measured 7.",
                "source_path": str(root / "sources" / "measurements.txt"), "image_path": None}]}]


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    values = {}
    for workspace in ("Alpha", "Beta"):
        workspaces.create_workspace(workspace)
        root = workspaces.workspace_path(workspace)
        (root / "sources" / "measurements.txt").write_text("Measured 7.", encoding="utf-8")
        instance = SimpleNamespace(settings=SimpleNamespace(data_dir=root, fingerprint=lambda: "v1"),
            store=SimpleNamespace(collections=lambda: ["Default"], list_documents=lambda collection=None: []))

        def answer(question, previous, collection, *, _root=root, **kwargs):
            yield previous + [{"role": "user", "content": question}, messages(_root)[-1]], messages(_root)[-1]["sources"]

        instance.answer = answer
        values[workspace] = instance
    monkeypatch.setattr(ui, "get_service", values.__getitem__)
    return values


def send(state, *, workspace="Alpha", previous=None):
    return ui.on_answer("What was measured?", previous or [], "Default", ["doc1"],
                        "gpt-4o", "auto", workspace, state)


def test_completed_answer_saves_once_then_updates_same_file_with_all_citations(services):
    state = {"revision": 0}
    root = services["Alpha"].settings.data_dir
    stream = send(state)
    next(stream)  # Pending user message.
    next(stream)  # Streamed content is still not a committed answer.
    assert history.list_chats(data_dir=root) == []
    final = list(stream)[-1]
    assert final[6] == "Conversation autosaved."
    name = state["autosave"]["name"]
    assert history.load_chat(name, data_dir=root) == final[0]
    assert history.load_chat(name, data_dir=root)[-1]["web_search"] is True
    second = list(send(state, previous=final[0]))[-1]
    assert state["autosave"]["name"] == name
    assert history.list_chats(data_dir=root) == [name]
    assert history.load_chat(name, data_dir=root) == second[0]
    assert ui.on_refresh_saved_chats("Alpha", state)["value"] == name


def test_workspace_open_restores_latest_and_forks_identity_for_independent_browser(services):
    first_state = {"revision": 0}
    final = list(send(first_state))[-1]
    first_id = first_state["autosave"]["id"]
    second_state = {"revision": 0}
    restored = ui.on_initial_load(second_state)
    assert restored[0] == "Alpha"
    assert restored[13] == final[0] and restored[15] == final[2]
    assert "Restored" in restored[19]
    assert second_state["autosave"]["id"] != first_id
    second = list(send(second_state, previous=restored[13]))[-1]
    root = services["Alpha"].settings.data_dir
    assert len(history.list_chats(data_dir=root)) == 2
    assert history.load_chat(first_state["autosave"]["name"], data_dir=root) == final[0]
    assert autosave.latest(data_dir=root)["messages"] == second[0]
    assert autosave.latest(data_dir=services["Beta"].settings.data_dir) is None


def test_new_chat_preserves_previous_autosave_and_creates_another_identity(services):
    state = {"revision": 0}
    previous = list(send(state))[-1][0]
    name = state["autosave"]["name"]
    assert ui.on_new_chat("Alpha", state)[0] == []
    assert state["autosave"]["name"] != name
    fresh = list(send(state))[-1][0]
    root = services["Alpha"].settings.data_dir
    assert len(history.list_chats(data_dir=root)) == 2
    assert history.load_chat(name, data_dir=root) == previous
    assert len(fresh) == 2


def test_loading_named_snapshot_forks_autosave_without_modifying_snapshot(services):
    root = services["Alpha"].settings.data_dir
    original = messages(root)
    history.save_chat(original, "Manual snapshot", data_dir=root)
    state = {"revision": 0}
    loaded = ui.on_load_chat("Manual snapshot", [], None, "Alpha", state)
    final = list(send(state, previous=loaded[0]))[-1]
    assert history.load_chat("Manual snapshot", data_dir=root) == original
    assert len(final[0]) == 4
    assert autosave.latest(data_dir=root)["messages"] == final[0]


def test_slow_load_cannot_overwrite_a_new_chat_in_the_same_workspace(services, monkeypatch):
    state = {"revision": 0}

    def interrupted(*args, **kwargs):
        ui.on_new_chat("Alpha", state)
        return messages(services["Alpha"].settings.data_dir)

    monkeypatch.setattr(history, "load_chat", interrupted)
    result = ui.on_load_chat("Old conversation", [], None, "Alpha", state)
    assert all(value == ui.gr.skip() for value in result)
    assert state["autosave"]["saved"] is False


@pytest.mark.parametrize("action", ["failure", "stop", "new", "switch", "selection"])
def test_incomplete_and_superseded_answers_never_replace_autosave(services, action):
    state = {"revision": 0}
    original = list(send(state))[-1][0]
    root = services["Alpha"].settings.data_dir
    snapshot = autosave.latest(data_dir=root)

    def answer(question, previous, collection, **kwargs):
        yield previous + [{"role": "assistant", "content": "Partial response"}], []
        if action == "failure":
            raise RuntimeError("Request interrupted")
        yield previous + [{"role": "assistant", "content": "Late response"}], []

    services["Alpha"].answer = answer
    stream = send(state, previous=original)
    pending = next(stream)[-1]
    next(stream)
    if action == "stop":
        ui.on_stop_chat(original, pending, state)
    elif action == "new":
        ui.on_new_chat("Alpha", state)
    elif action == "switch":
        state["revision"] += 1
    elif action == "selection":
        ui.on_source_selection([], state, original, pending)
    list(stream)
    assert autosave.latest(data_dir=root) == snapshot
    assert history.list_chats(data_dir=root) == [snapshot["name"]]


def test_failed_autosave_keeps_successful_answer_visible(services, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(autosave, "save", fail)
    state = {"revision": 0}
    final = list(send(state))[-1]
    assert final[0][-1]["content"] == "Measured 7 [1]."
    assert final[-1] is None
    assert "autosave failed" in final[6] and "Disk full" in final[6]
    assert not state["autosave"]["saved"]


@pytest.mark.parametrize("stop_at", [1, 2, 3])
def test_atomic_autosave_cancellation_preserves_previous_bytes(tmp_path, stop_at):
    session = autosave.new_session("Alpha")
    name = autosave.save(messages(tmp_path), session, data_dir=tmp_path)
    path = tmp_path / "chats" / f"{name}.json"
    before = path.read_bytes()
    calls = []

    def cancelled():
        calls.append(True)
        return len(calls) == stop_at

    with pytest.raises(autosave.AutosaveCancelled):
        autosave.save(messages(tmp_path, "Replacement"), session, data_dir=tmp_path, is_cancelled=cancelled)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".pending-*"))


def test_latest_skips_corrupt_newer_files_and_ignores_manual_chats(tmp_path):
    session = autosave.new_session("Alpha")
    name = autosave.save(messages(tmp_path), session, data_dir=tmp_path)
    broken = autosave.new_session("Alpha")
    (tmp_path / "chats" / f"{broken['name']}.json").write_text("[]", encoding="utf-8")
    history.save_chat(messages(tmp_path, "Manual"), "Newest manual", data_dir=tmp_path)
    assert autosave.latest(data_dir=tmp_path) == {"name": name, "messages": messages(tmp_path)}


def test_latest_reads_only_newest_valid_history(tmp_path, monkeypatch):
    sessions = [autosave.new_session("Alpha") for _ in range(3)]
    for index, session in enumerate(sessions):
        name = autosave.save(messages(tmp_path, f"Answer {index}"), session, data_dir=tmp_path)
        os.utime(tmp_path / "chats" / f"{name}.json", (1000 + index, 1000 + index))
    reads = []
    original = autosave.json.loads

    def count(raw):
        reads.append(True)
        return original(raw)

    monkeypatch.setattr(autosave.json, "loads", count)
    assert autosave.latest(data_dir=tmp_path)["messages"][-1]["content"] == "Answer 2"
    assert len(reads) == 1


def test_autosave_choices_have_short_labels_with_full_unique_values():
    session = autosave.new_session("Alpha")
    choices = ui._chat_choices(["Manual snapshot", session["name"]])
    assert choices[0] == "Manual snapshot"
    assert choices[1][1] == session["name"]
    assert "UTC" in choices[1][0] and session["id"] not in choices[1][0]


def test_autosave_size_limit_preserves_previous_output(tmp_path, monkeypatch):
    session = autosave.new_session("Alpha")
    name = autosave.save(messages(tmp_path), session, data_dir=tmp_path)
    monkeypatch.setattr(autosave, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="16 MiB"):
        autosave.save(messages(tmp_path, "New"), session, data_dir=tmp_path)
    assert history.load_chat(name, data_dir=tmp_path) == messages(tmp_path)


@pytest.mark.parametrize("target", ["directory", "file"])
def test_linked_autosave_paths_are_rejected(tmp_path, target):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    session = autosave.new_session("Alpha")
    try:
        if target == "directory":
            (root / "chats").symlink_to(outside, target_is_directory=True)
        else:
            (root / "chats").mkdir()
            original = outside / "private.json"
            original.write_text("Private", encoding="utf-8")
            (root / "chats" / f"{session['name']}.json").symlink_to(original)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is unavailable")
    with pytest.raises(ValueError, match="linked|own chats"):
        autosave.save(messages(root), session, data_dir=root)


def test_stale_history_refresh_cannot_publish_new_workspace_choices(services, monkeypatch):
    state = {"revision": 0}
    list(send(state))
    original = history.list_chats

    def changed(**kwargs):
        state["revision"] += 1
        return original(**kwargs)

    monkeypatch.setattr(history, "list_chats", changed)
    assert ui.on_refresh_saved_chats("Alpha", state) == ui.gr.skip()


def test_workspace_switch_during_autosave_serialization_discards_pending_result(services, monkeypatch):
    state = {"revision": 0}
    original = autosave.json.dumps

    def changed(value, *args, **kwargs):
        encoded = original(value, *args, **kwargs)
        state["revision"] += 1
        return encoded

    monkeypatch.setattr(autosave.json, "dumps", changed)
    updates = list(send(state))
    assert updates[-1][-1] is not None  # No final publication from the old job.
    assert history.list_chats(data_dir=services["Alpha"].settings.data_dir) == []


def test_workspace_open_discarded_by_revision_does_not_replace_active_chat_identity(services, monkeypatch):
    state = {"revision": 0}
    list(send(state))
    previous = copy.deepcopy(state["autosave"])
    original = autosave.latest

    def changed(**kwargs):
        state["revision"] += 1
        return original(**kwargs)

    monkeypatch.setattr(autosave, "latest", changed)
    result = ui.on_open_workspace("Beta", "Alpha", operation_state=state)
    assert all(item == ui.gr.skip() for item in result)
    assert state["autosave"] == previous
