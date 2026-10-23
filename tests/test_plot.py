"""Numeric Plot validation, source grounding, persistence and local export."""

from copy import deepcopy
import csv
from decimal import Decimal
from io import StringIO
import json
from types import SimpleNamespace

import pytest

from chatbot import config
from rag_chatbot.config import Settings
from rag_chatbot.store import LibraryStore
from rag_chatbot.tools import artifacts, plot, service
from rag_chatbot.tools.exports import export_bytes, plot_export_filename
from rag_chatbot.workspaces import create_workspace, workspace_path


def content(kind="bar"):
    return {"chart_type": kind, "x_label": "Time" if kind != "bar" else "Sample", "y_label": "Value",
            "x_unit": "s" if kind != "bar" else "", "y_unit": "nm", "series": [{"name": "Measured",
                "points": [{"label": "Sample A", "x": None if kind == "bar" else 1, "y": 2,
                            "source_ids": ["chunk-1"], "evidence": "At 1 s, sample A measured 2 nm."}]}],
            "limitations": []}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    return "Research"


@pytest.fixture
def payload(workspace):
    return {"tool_id": "plot", "title": "Reported measurements", "model": config.DEFAULT_MODEL,
            "content": content(), "sources": [{"id": "chunk-1", "document_id": "doc-1", "filename": "study.txt",
                "locator": "Page 2", "text": "Measured samples. At 1 s, sample A measured 2 nm.",
                "source_path": str(workspace_path(workspace) / "sources" / "study.txt")}],
            "documents": [{"id": "doc-1", "name": "study.txt", "fingerprint": "v1", "generation": "g1"}],
            "coverage": {"documents": 1, "chunks": 1, "processed_chunks": 1},
            "settings": {"chart_type": "auto"}}


@pytest.mark.parametrize("kind", ["bar", "line", "scatter"])
def test_valid_numeric_charts_are_not_mutated(kind):
    value = content(kind)
    before = deepcopy(value)
    assert plot.validate_plot(value, sources=[{"id": "chunk-1", "text": "At 1 s, sample A measured 2 nm."}]) is value
    assert value == before


@pytest.mark.parametrize("value", ["auto", "bar", "line", "scatter"])
def test_chart_options(value):
    assert plot.validate_chart_type(value) == value


@pytest.mark.parametrize("value", [None, True, [], {}, "pie", "Bar", ""])
def test_bad_option_fails_before_sources_or_api(monkeypatch, value):
    monkeypatch.setattr(service, "get_service", lambda _: pytest.fail("sources accessed"))
    monkeypatch.setattr(service.api, "generate_structured", lambda *a, **k: pytest.fail("paid API"))
    with pytest.raises(ValueError, match="chart type"):
        list(service.generate_tool("Research", "plot", ["doc"], chart_type=value))


@pytest.mark.parametrize("value", [True, False, "2", None, [], {}, float("nan"), float("inf"),
                                  float("-inf"), 10 ** 1000, 1.1e100])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_coordinates_are_finite_real_numbers(value, axis):
    data = content("scatter")
    data["series"][0]["points"][0][axis] = value
    with pytest.raises(ValueError, match="finite numbers"):
        plot.validate_plot(data)


@pytest.mark.parametrize("text,expected", [("Values -2.50, +4 and .25", {Decimal("-2.5"), Decimal(4), Decimal(".25")}),
    ("Value 1,250.5 nm; -2e-3 V.", {Decimal("1250.5"), Decimal("-.002")}),
    ("Value 1.2 × 10⁻³ A.", {Decimal(".0012")}),
    ("Value 2.5 × 10^−2 A.", {Decimal(".025")}),
    ("Value 4.2e−3 nm.", {Decimal(".0042")}),
    ("Value 10⁻³ A.", {Decimal(".001")}),
    ("Not a scalar 1 × 100 or 2³.", set()),
    ("Value 10%.", {Decimal(10)}),
    ("x=2.5, y=4.5.", {Decimal("2.5"), Decimal("4.5")}),
    ("Values 1, 5 and 1.2e-3.", {Decimal(1), Decimal(5), Decimal(".0012")}),
    ("Ambiguous 1,5 or 1.2.3", set()),
    ("Ambiguous 1,25 and model A10B", set())])
def test_safe_explicit_number_parser(text, expected):
    assert plot.explicit_numbers(text) == expected


@pytest.mark.parametrize("damage", ["invented", "converted", "wrong_source", "rounded", "edited_quote"])
def test_point_values_must_be_in_original_cited_excerpts(damage):
    data = content()
    point = data["series"][0]["points"][0]
    sources = [{"id": "chunk-1", "text": point["evidence"]}, {"id": "chunk-2", "text": "Different evidence."}]
    if damage == "invented":
        point["y"] = 5
    elif damage == "converted":
        point["y"] = 2e-9
    elif damage == "rounded":
        point["evidence"] = sources[0]["text"] = "Value 2.12 nm."
    elif damage == "wrong_source":
        point["source_ids"] = ["chunk-2"]
    else:
        point["evidence"] = "At 1 s, sample A measured 3 nm."
        point["y"] = 3
    with pytest.raises(ValueError, match="evidence|quote"):
        plot.validate_plot(data, sources=sources)


def test_whitespace_only_excerpt_normalization():
    data = content()
    plot.validate_plot(data, sources=[{"id": "chunk-1", "text": "At 1 s,\nsample A measured\t2 nm."}])


def test_superscripts_cannot_be_rewritten_into_different_numeric_evidence():
    data = content()
    point = data["series"][0]["points"][0]
    point.update(y=23, evidence="Value 23 units.")
    with pytest.raises(ValueError, match="quote"):
        plot.validate_plot(data, sources=[{"id": "chunk-1", "text": "Value 2³ units."}])


@pytest.mark.parametrize("damage", ["category_duplicate", "series_duplicate", "line_duplicate", "bar_x", "unknown_id",
                                   "no_ids", "duplicate_id", "unknown_field", "code_field", "no_points", "no_series",
                                   "too_many_series", "too_many_points", "too_many_total", "long_evidence", "control"])
def test_structure_bounds_and_citations(damage):
    data = content("line" if damage == "line_duplicate" else "bar")
    series = data["series"][0]
    point = series["points"][0]
    if damage in {"category_duplicate", "line_duplicate"}:
        series["points"].append(deepcopy(point))
    elif damage == "series_duplicate":
        data["series"].append(deepcopy(series))
    elif damage == "bar_x":
        point["x"] = 1
    elif damage == "unknown_id":
        point["source_ids"] = ["missing"]
    elif damage == "no_ids":
        point["source_ids"] = []
    elif damage == "duplicate_id":
        point["source_ids"] *= 2
    elif damage == "unknown_field":
        data["url"] = "https://example.test"
    elif damage == "code_field":
        point["python"] = "bad()"
    elif damage == "no_points":
        series["points"] = []
    elif damage == "no_series":
        data["series"] = []
    elif damage == "too_many_series":
        data["series"] = [{**deepcopy(series), "name": str(i)} for i in range(plot.MAX_SERIES + 1)]
    elif damage == "too_many_points":
        series["points"] = [{**point, "label": str(i)} for i in range(plot.MAX_POINTS_PER_SERIES + 1)]
    elif damage == "too_many_total":
        series["points"] = [{**point, "label": str(i)} for i in range(plot.MAX_POINTS // 2 + 1)]
        data["series"] = [{**deepcopy(series), "name": str(i)} for i in range(2)]
    elif damage == "long_evidence":
        point["evidence"] += "x" * 600
    else:
        point["label"] = "bad\x00value"
    with pytest.raises(ValueError):
        plot.validate_plot(data, allowed_source_ids={"chunk-1"})


def test_scatter_duplicate_x_and_unsorted_line_are_accepted():
    data = content("scatter")
    data["series"][0]["points"].append(deepcopy(data["series"][0]["points"][0]))
    plot.validate_plot(data)
    data["chart_type"] = "line"
    data["series"][0]["points"][0]["x"] = 2
    plot.validate_plot(data)  # trusted renderer owns sorting; data is not changed


@pytest.mark.parametrize("series_count", [1, 3])
def test_ten_thousand_scatter_points_are_valid_with_original_evidence(series_count):
    data = content("scatter")
    point = data["series"][0]["points"][0]
    data["series"] = [{"name": f"Species {index}", "points": [deepcopy(point)
        for _ in range(10_000 // series_count + (index < 10_000 % series_count))]}
        for index in range(series_count)]
    before = deepcopy(data)
    assert plot.validate_plot(data, sources=[{"id": "chunk-1", "text": point["evidence"]}]) is data
    assert sum(len(series["points"]) for series in data["series"]) == 10_000
    assert data == before


def test_current_plot_is_single_atomic_scoped_result(workspace, payload):
    assert artifacts.load_current_plot(workspace) is None
    assert not (workspace_path(workspace) / "artifacts").exists()
    before = deepcopy(payload)
    first = artifacts.save_current_plot(workspace, payload)
    second = artifacts.save_current_plot(workspace, {**payload, "title": "Replacement"})
    assert payload == before and first["id"] == second["id"]
    assert first["created_at"] == second["created_at"]
    assert artifacts.load_current_plot(workspace) == second
    assert second["sources"][0]["source_path"] == "sources/study.txt"
    assert artifacts.list_artifacts(workspace) == []
    create_workspace("Other")
    assert artifacts.load_current_plot("Other") is None


@pytest.mark.parametrize("stop_at", [1, 2])
def test_cancelled_save_preserves_old_plot(workspace, payload, stop_at):
    first = artifacts.save_current_plot(workspace, payload)
    current = workspace_path(workspace) / "artifacts/plot/current.json"
    before, calls = current.read_bytes(), []

    def cancelled():
        calls.append(True)
        return len(calls) == stop_at

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_plot(workspace, {**payload, "title": "Stale"}, is_cancelled=cancelled)
    assert current.read_bytes() == before and artifacts.load_current_plot(workspace) == first
    assert not list(current.parent.glob(".pending-*"))


def test_invalid_or_ungrounded_plot_does_not_replace_saved(workspace, payload):
    first = artifacts.save_current_plot(workspace, payload)
    payload["content"]["series"][0]["points"][0]["y"] = 999
    with pytest.raises(ValueError, match="explicitly"):
        artifacts.save_current_plot(workspace, payload)
    assert artifacts.load_current_plot(workspace) == first


def test_persisted_plot_requires_each_document_point_or_limitation(workspace, payload):
    payload["documents"].append({"id": "doc-2"})
    payload["sources"].append({**payload["sources"][0], "id": "chunk-2", "document_id": "doc-2",
                               "text": "A qualitative interpretation without measurements."})
    with pytest.raises(ValueError, match="every selected"):
        artifacts.save_current_plot(workspace, payload)
    payload["content"]["limitations"].append({"text": "The second source has no measurements.", "source_ids": ["chunk-2"]})
    assert artifacts.save_current_plot(workspace, payload)["content"]["limitations"]


@pytest.mark.parametrize("filename", ["../Plot", "sub\\Plot", "NUL", "CON.png", " Plot", "Plot.", "x" * 101,
                                      "界" * 85, "\ud800"])
def test_export_filenames_are_portable(filename):
    with pytest.raises(ValueError, match="filename"):
        plot_export_filename(filename, "png")


def test_csv_json_export_keep_full_evidence_and_safe_text_without_renderer(workspace, payload, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def no_renderer(name, *args, **kwargs):
        if name.startswith("matplotlib") or name.endswith("plot_render"):
            raise ImportError("renderer unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_renderer)
    point = payload["content"]["series"][0]["points"][0]
    point.update(label="=HYPERLINK(bad)", y=-2, evidence="Reported -2 nm.")
    payload["sources"][0]["text"] = point["evidence"]
    payload["content"]["series"][0]["name"] = "@bad"
    saved = artifacts.save_current_plot(workspace, payload)
    current = workspace_path(workspace) / "artifacts/plot/current.json"
    before = current.read_bytes()
    csv_path = artifacts.export_plot(workspace, saved, "csv", "Figure.png")
    rows = list(csv.reader(StringIO(csv_path.read_text(encoding="utf-8-sig"))))
    assert rows[1][0] == "'@bad" and rows[1][1] == "'=HYPERLINK(bad)"
    assert rows[1][3] == "-2" and rows[1][8] == "[1]"
    assert "Sources" in csv_path.read_text(encoding="utf-8-sig")
    assert "Documents:" in csv_path.read_text(encoding="utf-8-sig")
    json_path = artifacts.export_plot(workspace, saved, "json", "Figure.csv")
    assert json.loads(json_path.read_text(encoding="utf-8")) == saved
    assert artifacts.load_current_plot(workspace) == saved and current.read_bytes() == before


@pytest.mark.parametrize("stop_at", [1, 2, 3])
def test_cancelled_export_never_overwrites_download(workspace, payload, stop_at):
    saved = artifacts.save_current_plot(workspace, payload)
    path = artifacts.export_plot(workspace, saved, "json")
    original, calls = path.read_bytes(), []

    def cancelled():
        calls.append(True)
        return len(calls) == stop_at

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.export_plot(workspace, {**saved, "title": "Late"}, "json", is_cancelled=cancelled)
    assert path.read_bytes() == original and not list(path.parent.glob(".pending-*"))


def test_linked_current_plot_cannot_escape_workspace(workspace, payload, tmp_path):
    artifacts.save_current_plot(workspace, payload)
    current = workspace_path(workspace) / "artifacts/plot/current.json"
    external = tmp_path / "outside.json"
    external.write_bytes(current.read_bytes())
    current.unlink()
    try:
        current.symlink_to(external)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    with pytest.raises(ValueError, match="linked"):
        artifacts.load_current_plot(workspace)


@pytest.fixture
def catalog(tmp_path):
    store = LibraryStore(Settings(data_dir=tmp_path / "catalog"))
    documents = []
    for number, texts in [(1, [f"At {i} s, value {i + 2} nm." for i in range(1, 13)]),
                          (2, ["This source discusses qualitative mechanisms only."])]:
        path = tmp_path / f"source-{number}.txt"
        path.write_text("\n".join(texts), encoding="utf-8")
        document = store.stage_document(path, "science", store.settings.fingerprint())
        with store._db:
            for index, text in enumerate(texts):
                store._db.execute("INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
                    (f"doc{number}-chunk{index}", document["id"], text, f"page {index + 1}", "text", index))
            store._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=? WHERE id=?",
                              (len(texts), document["id"]))
        documents.append(document["id"])
    yield store, documents
    store.close()


def fake_generation(monkeypatch, catalog, *, transform=None, after_final=None):
    store, documents = catalog
    calls = []
    monkeypatch.setattr(service, "get_service", lambda _: SimpleNamespace(store=store))

    def generate(model, history, schema, **options):
        prompt = json.loads(history[-1]["content"])
        name = options["schema_name"]
        calls.append((name, prompt))
        if name == "tool_evidence_notes":
            return {"notes": [{"source_id": row["source_id"], "text": row["text"]} for row in prompt["evidence"]]}
        assert name == "tool_plot"
        data = content("line" if prompt["requested_chart_type"] == "auto" else prompt["requested_chart_type"])
        data["series"][0]["points"] = [{"label": f"Sample {index}", "x": None if data["chart_type"] == "bar" else index,
            "y": index + 2, "source_ids": [f"s{index}"], "evidence": f"At {index} s, value {index + 2} nm."}
            for index in range(1, 13)]
        data["limitations"] = [{"text": "The second source contains no numeric measurements.", "source_ids": ["s13"]}]
        result = {"title": "Reported values", "status": "complete", **data}
        if after_final:
            after_final()
        return transform(result) if transform else result

    monkeypatch.setattr(service.api, "generate_structured", generate)
    monkeypatch.setattr(service.api, "generate_image_asset", lambda *a, **k: pytest.fail("Plot must not call image API"))
    return calls, documents


def test_generation_processes_all_selected_chunks_and_maps_citations(catalog, monkeypatch):
    calls, documents = fake_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", "plot", documents))[-1]["result"]
    assert result["coverage"] == {"documents": 2, "chunks": 13, "processed_chunks": 13}
    assert result["settings"]["chart_type"] == "auto" and result["settings"]["model_requests"] == 3
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", "tool_plot"]
    assert "verbatim" in calls[0][1]["task"].lower()
    assert result["content"]["series"][0]["points"][0]["source_ids"] == ["doc1-chunk0"]
    assert result["content"]["limitations"][0]["source_ids"] == ["doc2-chunk0"]


@pytest.mark.parametrize("keep_notes", [False, True])
def test_bar_plot_keeps_omitted_numeric_passages_without_retry(catalog, monkeypatch, keep_notes):
    calls, documents = fake_generation(monkeypatch, catalog)
    fake = service.api.generate_structured

    def omit_notes(*args, **kwargs):
        result = fake(*args, **kwargs)
        if kwargs["schema_name"] == "tool_evidence_notes":
            result["notes"] = result["notes"][:1] if keep_notes else []
        return result

    monkeypatch.setattr(service.api, "generate_structured", omit_notes)
    result = list(service.generate_tool("work", "plot", documents, chart_type="bar"))[-1]["result"]
    expected = [{"text": source["text"], "source_ids": [source["source_id"]]}
                for name, prompt in calls if name == "tool_evidence_notes" for source in prompt["evidence"]]
    assert calls[-1][1]["evidence"]["notes"] == expected
    assert result["content"]["chart_type"] == "bar"
    assert len(result["content"]["series"][0]["points"]) == 12
    assert result["coverage"]["processed_chunks"] == 13
    assert result["settings"]["model_requests"] == 3


@pytest.mark.parametrize("kind", ["bar", "line", "scatter"])
def test_generation_respects_requested_chart_type(catalog, monkeypatch, kind):
    calls, documents = fake_generation(monkeypatch, catalog)
    result = list(service.generate_tool("work", "plot", documents, chart_type=kind))[-1]["result"]
    assert result["content"]["chart_type"] == kind
    assert all(prompt["requested_chart_type"] == kind for _, prompt in calls)


@pytest.mark.parametrize("damage,message", [("no_numeric_data", "No compatible"), ("budget_exceeded", "exceeds"),
    ("unknown_id", "citations"), ("no_coverage", "omitted"), ("invented", "explicitly"),
    ("rewritten_excerpt", "quote"), ("wrong_type", "requested chart")])
def test_invalid_generation_fails_without_retry(catalog, monkeypatch, damage, message):
    def corrupt(result):
        if damage in {"no_numeric_data", "budget_exceeded"}:
            result["status"], result["series"] = damage, []
        elif damage == "unknown_id":
            result["series"][0]["points"][0]["source_ids"] = ["missing"]
        elif damage == "no_coverage":
            result["limitations"] = []
        elif damage == "invented":
            result["series"][0]["points"][0]["y"] = 999
        elif damage == "rewritten_excerpt":
            result["series"][0]["points"][0]["evidence"] = "A rewritten value 3 at time 1."
        else:
            result["chart_type"] = "scatter"
        return result

    calls, documents = fake_generation(monkeypatch, catalog, transform=corrupt)
    with pytest.raises(ValueError, match=message):
        list(service.generate_tool("work", "plot", documents, chart_type="line"))
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["cancel", "snapshot"])
def test_inflight_final_result_is_rejected_after_cancellation_or_source_change(catalog, monkeypatch, change):
    stopped = False

    def invalidate():
        nonlocal stopped
        if change == "cancel":
            stopped = True
        else:
            with catalog[0]._db:
                catalog[0]._db.execute("UPDATE documents SET generation='new'")

    calls, documents = fake_generation(monkeypatch, catalog, after_final=invalidate)
    events = []
    with pytest.raises((service.ToolCancelled, ValueError)):
        for event in service.generate_tool("work", "plot", documents, is_cancelled=lambda: stopped):
            events.append(event)
    assert all(event["result"] is None for event in events) and len(calls) == 3


def test_request_budget_fails_before_paid_work(catalog, monkeypatch):
    calls, documents = fake_generation(monkeypatch, catalog)
    monkeypatch.setattr(service, "MAX_MODEL_CALLS", 2)
    with pytest.raises(ValueError, match="too many"):
        list(service.generate_tool("work", "plot", documents))
    assert calls == []
