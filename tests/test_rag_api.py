from types import SimpleNamespace

import pytest

from chatbot import api


def test_embedding_response_is_reordered_and_parameters_passed(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[0., 1.]),
                                     SimpleNamespace(index=0, embedding=[1., 0.])])

    monkeypatch.setattr(api.config, "get_client", lambda: SimpleNamespace(
        embeddings=SimpleNamespace(create=create)))
    assert api.embed_texts(["first", "second"], "test-embedding", 2) == [[1., 0.], [0., 1.]]
    assert calls == [{"model": "test-embedding", "input": ["first", "second"],
                      "dimensions": 2, "encoding_format": "float"}]


@pytest.mark.parametrize("data", [
    [], [SimpleNamespace(index=1, embedding=[1., 0.])],
    [SimpleNamespace(index=0, embedding=[1.])],
    [SimpleNamespace(index=0, embedding=[float("nan"), 1.])],
    [SimpleNamespace(index=0, embedding=[0., 0.])],
])
def test_bad_embedding_responses_are_rejected(monkeypatch, data):
    monkeypatch.setattr(api.config, "get_client", lambda: SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=data))))
    with pytest.raises(RuntimeError):
        api.embed_texts(["one"], dimensions=2)


@pytest.mark.parametrize("texts", [[], [""], [" "], [None]])
def test_empty_embedding_input_does_not_call_api(monkeypatch, texts):
    def fail():
        pytest.fail("No API call should occur for empty input")
    monkeypatch.setattr(api.config, "get_client", fail)
    with pytest.raises(ValueError):
        api.embed_texts(texts)


def test_vision_uses_shared_chat_routing_and_local_image(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "image_file_to_data_url", lambda path: "data:image/png;base64,AA==")
    monkeypatch.setattr(api, "send_chat", lambda *args, **kwargs: calls.append((args, kwargs)) or "Table: 42")
    assert api.describe_document_image("page.png") == "Table: 42"
    args, kwargs = calls[0]
    assert args[0] == "gpt-4.1-mini"
    assert args[1][0]["image_url"].startswith("data:image/")
    assert "Do not invent" in args[1][0]["content"]
    assert kwargs["system_prompt"]
