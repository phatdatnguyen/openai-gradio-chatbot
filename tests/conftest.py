"""Default isolation; individual tests may inject clients or scoped stores."""

import os

import pytest


# Set these before test collection imports the eagerly constructed SDK client
# or Gradio. Never load a user's real environment key into a test client.
os.environ["OPENAI_API_KEY"] = "offline-test-placeholder"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


@pytest.fixture(autouse=True)
def isolated_app_defaults(tmp_path, monkeypatch):
    from chatbot import config, history_store
    from plugins import manager
    from rag_chatbot.tools import presets

    def prohibit_paid_client():
        pytest.fail("Tests must inject a fake OpenAI client or response before making model requests.")

    monkeypatch.setattr(config, "get_client", prohibit_paid_client)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "rag-data"))
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setattr(history_store, "HISTORY_DIR", str(tmp_path / "chat-history"))
    monkeypatch.setattr(manager, "SETTINGS_PATH", tmp_path / "preferences" / "plugins.json")
    monkeypatch.setattr(presets, "SETTINGS_PATH", tmp_path / "preferences" / "studio-presets.json")

