"""Real indexed CSV plots use one mapping call and retain every local data row."""

from collections import Counter
import csv
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_chatbot.config import Settings
from rag_chatbot.ingestion import chunk_elements, extract_elements
from rag_chatbot.store import LibraryStore
from rag_chatbot.tools import artifacts, service
from rag_chatbot.workspaces import create_workspace, workspace_path


INSTRUCTIONS = ("Please help me make a scatter plot for this dataset, x-axis is sepal length "
                "and y-axis is sepal width, colored by species")
HEADERS = ["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


@pytest.fixture
def indexed_csv(tmp_path, monkeypatch):
    """Publish ingestion output in SQLite without embedding or Qdrant requests."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    settings = Settings(data_dir=workspace_path("Research"))
    store = LibraryStore(settings)
    monkeypatch.setattr(service, "get_service", lambda _: SimpleNamespace(store=store))
    monkeypatch.setattr(service.api, "generate_image_asset", lambda *a, **k: pytest.fail("No image API"))

    def stage(count):
        path = tmp_path / "IrisFlower.csv"
        records = []
        # Synthetic Iris-like observations include exact repeated coordinates;
        # their multiplicity must survive plotting, saving, and exporting.
        for index in range(count):
            species_index = index % 3
            variation = (index // 3) % 5
            records.append([str(index + 1), f"{[4.9, 5.9, 6.7][species_index] + variation / 10:.1f}",
                f"{[3.3, 2.7, 3.0][species_index] + variation / 10:.1f}",
                ["1.4", "4.2", "5.6"][species_index], ["0.2", "1.3", "2.1"][species_index],
                SPECIES[species_index]])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADERS)
            writer.writerows(records)
        document = store.stage_document(path, "flowers", settings.fingerprint())
        managed_path = Path(document["path"])
        if not managed_path.is_absolute():
            managed_path = store.data_dir / managed_path
        chunks = chunk_elements(extract_elements(managed_path), document["id"],
                                settings.chunk_size, settings.chunk_overlap)
        with store._db:
            store._db.executemany(
                "INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
                [(chunk.id, document["id"], chunk.text, chunk.locator, chunk.kind, index)
                 for index, chunk in enumerate(chunks)])
            store._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=? WHERE id=?",
                              (len(chunks), document["id"]))
        return SimpleNamespace(store=store, document_id=document["id"], path=managed_path,
                               records=records, chunks=chunks)

    yield stage
    store.close()


def mock_mapping(monkeypatch, *, after_call=None):
    calls = []

    def generate(model, history, schema, **kwargs):
        assert kwargs["schema_name"] == "tool_plot_columns", "Numeric rows must never pass through model notes"
        prompt = json.loads(history[-1]["content"])
        calls.append(prompt)
        assert prompt["user_instructions"] == INSTRUCTIONS
        assert kwargs["max_output_tokens"] == 2000
        if after_call:
            after_call()
        return {"status": "complete", "title": "Iris sepal measurements", "chart_type": "scatter",
                "x_column": 1, "y_column": 2, "group_column": 5, "reason": ""}

    monkeypatch.setattr(service.api, "generate_structured", generate)
    return calls


def generate(data, **kwargs):
    return service.generate_tool("Research", "plot", [data.document_id], INSTRUCTIONS,
                                 collection="flowers", **kwargs)


def coordinate_counts(payload):
    return Counter((series["name"], point["x"], point["y"])
                   for series in payload["content"]["series"] for point in series["points"])


@pytest.mark.parametrize("count", [150, 10_000])
def test_complete_iris_scatter_uses_one_mapping_request_and_preserves_every_row(indexed_csv, monkeypatch, count):
    data = indexed_csv(count)
    calls = mock_mapping(monkeypatch)
    result = list(generate(data))[-1]["result"]
    assert len(calls) == result["settings"]["model_requests"] == 1
    assert calls[0]["table"]["row_count"] == count
    assert calls[0]["table"]["columns"] == [{"index": index, "name": name} for index, name in enumerate(HEADERS)]
    assert len(calls[0]["table"]["sample_rows"]) == 3
    assert result["settings"]["data_mode"] == "direct_table"
    assert result["settings"]["row_count"] == count
    assert result["content"]["chart_type"] == "scatter"
    assert result["content"]["x_label"] == "SepalLengthCm"
    assert result["content"]["y_label"] == "SepalWidthCm"
    assert [series["name"] for series in result["content"]["series"]] == SPECIES
    assert sum(len(series["points"]) for series in result["content"]["series"]) == count
    expected = Counter((row[5], float(row[1]), float(row[2])) for row in data.records)
    assert coordinate_counts(result) == expected
    assert all(value > 1 for value in expected.values())
    assert result["coverage"] == {"documents": 1, "chunks": len(data.chunks), "processed_chunks": len(data.chunks)}
    assert {source["id"] for source in result["sources"]} == {chunk.id for chunk in data.chunks}
    sources = {source["id"]: source["text"] for source in result["sources"]}
    for series in result["content"]["series"]:
        for point in series["points"]:
            assert len(point["source_ids"]) == 1
            assert point["evidence"] in sources[point["source_ids"][0]]
    if count == 10_000:
        assert len(data.chunks) > 512  # This path must bypass the LLM whole-source notes budget.

        saved = artifacts.save_current_plot("Research", result)
        restored = artifacts.load_current_plot("Research")
        assert restored == saved and coordinate_counts(restored) == expected
        csv_path = artifacts.export_plot("Research", restored, "csv", "Iris scatter")
        exported = list(csv.reader(StringIO(csv_path.read_text(encoding="utf-8-sig"))))
        assert exported[0][:4] == ["Series", "Label", "X", "Y"]
        point_rows = exported[1:count + 1]
        assert len(point_rows) == count and exported[count + 1] == []
        assert Counter((row[0], float(row[2]), float(row[3])) for row in point_rows) == expected
        assert all(row[8] and row[9] for row in point_rows)
        json_path = artifacts.export_plot("Research", restored, "json", "Iris scatter")
        assert json.loads(json_path.read_text(encoding="utf-8")) == saved
        assert artifacts.load_current_plot("Research") == saved


@pytest.mark.parametrize("when", ["before_request", "after_request"])
@pytest.mark.parametrize("change", ["cancel", "catalog", "original"])
def test_cancelled_or_changed_tables_never_publish_or_make_another_request(indexed_csv, monkeypatch, when, change):
    data = indexed_csv(150)
    stopped = False

    def invalidate():
        nonlocal stopped
        if change == "cancel":
            stopped = True
        elif change == "catalog":
            with data.store._db:
                data.store._db.execute("UPDATE documents SET generation='replacement' WHERE id=?", (data.document_id,))
        else:
            with data.path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

    calls = mock_mapping(monkeypatch, after_call=invalidate if when == "after_request" else None)
    events = generate(data, is_cancelled=lambda: stopped)
    first = next(events)
    assert first["result"] is None
    if when == "before_request":
        invalidate()
    with pytest.raises((service.ToolCancelled, ValueError), match="stopped|changed"):
        list(events)
    assert len(calls) == int(when == "after_request")
    assert artifacts.load_current_plot("Research") is None


def test_too_many_csv_rows_fail_before_mapping_request(indexed_csv, monkeypatch):
    data = indexed_csv(10_001)
    monkeypatch.setattr(service.api, "generate_structured", lambda *a, **k: pytest.fail("No paid request for excess data"))
    with pytest.raises(ValueError, match="10,000"):
        list(generate(data))
