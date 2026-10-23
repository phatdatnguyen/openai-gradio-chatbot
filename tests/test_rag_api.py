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


def web_response(text, annotations):
    return SimpleNamespace(status="completed", output=[SimpleNamespace(type="message", content=[
        SimpleNamespace(type="output_text", text=text, annotations=annotations)])])


def citation(text, marker, *, title="Research paper", url="https://example.org/paper"):
    return SimpleNamespace(type="url_citation", title=title, url=url,
        start_index=text.index(marker), end_index=text.index(marker) + len(marker))


def test_web_annotations_replace_exact_spans_and_keep_local_numeric_citations():
    text = "Local [1]. Web evidence α 🔬 [web-a] then β [web-b]."
    annotations = [citation(text, "[web-a]", title="First source", url="https://example.org/a"),
                   citation(text, "[web-b]", title="Second source", url="https://example.org/b")]
    output = api._response_text(web_response(text, annotations))
    assert output.startswith("Local [1]. Web evidence α 🔬 [First source](https://example.org/a) "
                             "then β [Second source](https://example.org/b).")
    assert "Web sources:" in output
    assert "[web-a]" not in output and "[web-b]" not in output


def test_web_citation_markdown_is_escaped_and_unsafe_urls_ignored():
    text = "Claim [web]."
    annotation = citation(text, "[web]", title="A [title](bad)", url="https://example.org/paper(1)")
    unsafe = SimpleNamespace(type="url_citation", url="javascript:alert(1)", title="Bad",
                             start_index=0, end_index=5)
    output = api._response_text(web_response(text, [annotation, unsafe, annotation]))
    assert r"[A \[title\]\(bad\)](https://example.org/paper%281%29)" in output
    assert "javascript:" not in output
    assert output.count("- [") == 1


def test_bad_web_annotation_offsets_do_not_corrupt_text():
    text = "Keep this original statement."
    annotation = SimpleNamespace(type="url_citation", url="https://example.org", title="Source",
                                 start_index=-1, end_index=5000)
    output = api._response_text(web_response(text, [annotation]))
    assert output.startswith(text)
    assert "[Source](https://example.org)" in output


def test_stream_keeps_string_deltas_and_supplies_final_annotated_text(monkeypatch):
    raw = "Local [1], web [99]."
    response = web_response(raw, [citation(raw, "[99]")])
    calls, final, closed = [], [], []
    class Stream:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
        def __iter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta=raw)
        def get_final_response(self):
            return response
    def stream(**kwargs):
        calls.append(kwargs)
        return Stream()
    monkeypatch.setattr(api.config, "get_client", lambda: SimpleNamespace(responses=SimpleNamespace(stream=stream)))
    pieces = list(api.stream_chat("gpt-4o", [{"role": "user", "content": "Question"}],
                  web_search="medium", final_text_callback=final.append))
    assert pieces[0] == raw
    assert "Web sources:" in pieces[-1]
    assert all(isinstance(piece, str) for piece in pieces)
    assert final[0].startswith("Local [1], web [Research paper](https://example.org/paper).")
    assert calls[0]["tools"] == [{"type": "web_search", "search_context_size": "medium"}]
    assert closed == [True]


def test_blocking_web_request_preserves_clickable_annotation_links(monkeypatch):
    raw = "Finding [web]."
    response = web_response(raw, [citation(raw, "[web]")])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response
    monkeypatch.setattr(api.config, "get_client", lambda: SimpleNamespace(responses=SimpleNamespace(create=create)))
    text = api.send_chat("gpt-4o", [{"role": "user", "content": "Question"}], web_search="low")
    assert "[Research paper](https://example.org/paper)" in text
    assert calls[0]["tools"][0]["search_context_size"] == "low"
