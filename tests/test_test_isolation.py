"""The suite's default environment must never use personal data or paid clients."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from chatbot import api, config, history_store
from plugins import manager
from rag_chatbot.config import Settings
from rag_chatbot.tools import presets


def test_unmocked_paid_request_fails_before_obtaining_a_real_client():
    with pytest.raises(pytest.fail.Exception, match="inject a fake OpenAI client"):
        api.embed_texts(["Must stay offline"], dimensions=3)


def test_api_contract_test_can_inject_a_client(monkeypatch):
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0])])))
    monkeypatch.setattr(config, "get_client", lambda: client)
    assert api.embed_texts(["Offline embedding"], dimensions=3) == [[1.0, 0.0, 0.0]]


def test_default_storage_paths_stay_in_this_tests_temporary_directory(tmp_path):
    settings = Settings()
    for value in (settings.data_dir, settings.qdrant_local_path, history_store.HISTORY_DIR,
                  manager.SETTINGS_PATH, presets.SETTINGS_PATH):
        assert Path(value).is_relative_to(tmp_path)

