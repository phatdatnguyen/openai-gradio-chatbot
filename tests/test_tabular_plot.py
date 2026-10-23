"""Direct CSV Plot covers every row without model transcription or paid calls."""

from copy import deepcopy
import csv
import hashlib
import io
from pathlib import Path

import pytest

from rag_chatbot.ingestion import _table_elements
from rag_chatbot.tools import tabular_plot


class TableStore:
    def __init__(self, root, rows, *, suffix=".csv", encoding="utf-8-sig"):
        self.data_dir = root
        (root / "sources").mkdir(parents=True)
        self.path = root / "sources" / ("table" + suffix)
        output = io.StringIO(newline="")
        csv.writer(output, delimiter="\t" if suffix == ".tsv" else ",").writerows(rows)
        self.path.write_bytes(output.getvalue().encode(encoding))
        self.document = {"id": "doc", "name": "Renamed table" + suffix, "path": str(self.path),
            "content_hash": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "fingerprint": "fingerprint", "generation": "generation", "collection": "Default"}
        self.sources = [{"id": f"chunk-{index}", "document_id": "doc", "filename": self.document["name"],
                         "locator": element.locator, "text": element.text, "source_path": str(self.path)}
                        for index, element in enumerate(_table_elements(enumerate(rows, 1), self.path.name))]
        self.document["chunk_count"] = len(self.sources)
        self.valid = True
        self.checks = 0

    def get_document(self, identifier):
        return deepcopy(self.document) if identifier == "doc" else None

    def snapshot_sources(self, selected, collection, **bounds):
        assert selected == ["doc"]
        assert bounds["max_chunks"] >= len(self.sources)
        assert bounds["max_characters"] >= sum(len(source["text"]) for source in self.sources)
        return {"documents": [deepcopy(self.document)], "sources": deepcopy(self.sources)}

    def validate_source_snapshot(self, documents):
        self.checks += 1
        if not self.valid:
            raise ValueError("Source changed during generation")
        assert documents[0]["generation"] == self.document["generation"]


def mapping(**changes):
    return {"status": "complete", "title": "Sepal measurements", "chart_type": "scatter",
            "x_column": 0, "y_column": 1, "group_column": 2, "reason": "", **changes}


def iris_rows(count):
    return [["sepal length", "sepal width", "species"],
            *[["5.1", "3.5", ("setosa", "versicolor", "virginica")[index % 3]] for index in range(count)]]


@pytest.mark.parametrize("count", [150, 10_000])
def test_all_csv_observations_and_duplicate_coordinates_are_preserved(tmp_path, count):
    store = TableStore(tmp_path, iris_rows(count))
    table = tabular_plot.prepare_tabular_plot(store, ["doc"])
    title, content, settings = table.build(mapping())
    assert title == "Sepal measurements"
    assert [series["name"] for series in content["series"]] == ["setosa", "versicolor", "virginica"]
    points = [point for series in content["series"] for point in series["points"]]
    assert len(points) == count == settings["row_count"] == table.prompt["row_count"]
    assert {(point["x"], point["y"]) for point in points} == {(5.1, 3.5)}
    assert len({point["label"] for point in points}) == count
    assert len(table.prompt["sample_rows"]) == 3
    assert settings["data_mode"] == "direct_table"
    assert store.checks >= 4
    texts = {source["id"]: source["text"] for source in table.snapshot["sources"]}
    assert all(point["evidence"] in texts[point["source_ids"][0]] for point in points)


@pytest.mark.parametrize("suffix,encoding", [(".csv", "utf-8-sig"), (".tsv", "cp1252")])
def test_managed_extension_and_index_encoding_are_respected(tmp_path, suffix, encoding):
    store = TableStore(tmp_path, [["longueur", "largeur", "espèce"], ["5.1", "3.5", "café"]],
                       suffix=suffix, encoding=encoding)
    table = tabular_plot.prepare_tabular_plot(store, ["doc"])
    _, content, _ = table.build(mapping())
    assert content["series"][0]["name"] == "café"
    assert content["x_label"] == "longueur"


def test_non_tabular_or_multiple_sources_keep_generic_pipeline(tmp_path):
    store = TableStore(tmp_path, iris_rows(1))
    assert tabular_plot.prepare_tabular_plot(store, ["doc", "doc2"]) is None
    assert tabular_plot.prepare_tabular_plot(store, []) is None
    assert tabular_plot.prepare_tabular_plot(store, ["missing"]) is None
    store.document["path"] = str(tmp_path / "sources/paper.pdf")
    assert tabular_plot.prepare_tabular_plot(store, ["doc"]) is None


def test_ten_thousand_and_one_rows_fail_during_preflight(tmp_path):
    store = TableStore(tmp_path, iris_rows(10_001))
    with pytest.raises(ValueError, match="10,000.*No rows were dropped"):
        tabular_plot.prepare_tabular_plot(store, ["doc"])


@pytest.mark.parametrize("after_prepare", [False, True])
def test_original_hash_is_checked_before_and_after_mapping(tmp_path, after_prepare):
    store = TableStore(tmp_path, iris_rows(3))
    table = tabular_plot.prepare_tabular_plot(store, ["doc"]) if after_prepare else None
    store.path.write_text("different original", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after indexing"):
        if table:
            table.build(mapping())
        else:
            tabular_plot.prepare_tabular_plot(store, ["doc"])


def test_changed_published_snapshot_is_rejected(tmp_path):
    store = TableStore(tmp_path, iris_rows(3))
    table = tabular_plot.prepare_tabular_plot(store, ["doc"])
    store.valid = False
    with pytest.raises(ValueError, match="Source changed"):
        table.validate_current()


def test_cancellation_is_checked_during_local_row_processing(tmp_path):
    store = TableStore(tmp_path, iris_rows(150))
    table = tabular_plot.prepare_tabular_plot(store, ["doc"])
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        if calls == 20:
            raise RuntimeError("Cancelled")

    with pytest.raises(RuntimeError, match="Cancelled"):
        table.build(mapping(), check_cancelled=stop)
    assert calls == 20


@pytest.mark.parametrize("change", [{"x_column": True}, {"y_column": -1}, {"group_column": 12},
                                    {"group_column": 0}, {"chart_type": "pie"}, {"status": "whatever"}])
def test_malformed_column_mapping_is_rejected(tmp_path, change):
    store = TableStore(tmp_path, iris_rows(3))
    table = tabular_plot.prepare_tabular_plot(store, ["doc"])
    with pytest.raises(ValueError):
        table.build(mapping(**change))


def test_explicit_chart_choice_must_be_honored(tmp_path):
    table = tabular_plot.prepare_tabular_plot(TableStore(tmp_path, iris_rows(3)), ["doc"])
    with pytest.raises(ValueError, match="honor"):
        table.build(mapping(), requested_chart_type="bar")


def test_unsupported_operations_are_reported_without_dropping_rows(tmp_path):
    table = tabular_plot.prepare_tabular_plot(TableStore(tmp_path, iris_rows(3)), ["doc"])
    with pytest.raises(ValueError, match="unsupported.*aggregation"):
        table.build(mapping(status="unsupported", reason="This request needs aggregation."))


@pytest.mark.parametrize("bad", ["", "NaN", "inf", "=1+2", "1/2", "1e999999", "0.1234567890123456789"])
def test_missing_or_unsupported_numeric_cells_never_disappear(tmp_path, bad):
    rows = iris_rows(3)
    rows[2][0] = bad
    table = tabular_plot.prepare_tabular_plot(TableStore(tmp_path, rows), ["doc"])
    with pytest.raises(ValueError, match="Row 3"):
        table.build(mapping())


def test_empty_unselected_cells_match_ingested_passage(tmp_path):
    rows = [["x", "empty", "y", "group"], ["1", "", "2", "a"]]
    table = tabular_plot.prepare_tabular_plot(TableStore(tmp_path, rows), ["doc"])
    _, content, _ = table.build(mapping(y_column=2, group_column=3))
    assert content["series"][0]["points"][0]["evidence"] == "A2: 1 | C2: 2"


def test_grouping_limit_does_not_silently_merge_or_drop_series(tmp_path):
    rows = [["x", "y", "group"], *[["1", "2", str(i)] for i in range(7)]]
    table = tabular_plot.prepare_tabular_plot(TableStore(tmp_path, rows), ["doc"])
    with pytest.raises(ValueError, match="6 series.*No rows"):
        table.build(mapping())


def test_missing_indexed_row_fails_before_mapping(tmp_path):
    store = TableStore(tmp_path, iris_rows(3))
    store.sources[0]["text"] = store.sources[0]["text"].replace("A3: 5.1 | B3: 3.5 | C3: versicolor", "")
    with pytest.raises(ValueError, match="missing data rows"):
        tabular_plot.prepare_tabular_plot(store, ["doc"])


def test_two_coordinates_must_share_one_original_indexed_passage(tmp_path):
    store = TableStore(tmp_path, iris_rows(1))
    store.sources[0]["text"] = "A2: 5.1\nOther text B2: 3.5"
    table = tabular_plot.prepare_tabular_plot(store, ["doc"])
    with pytest.raises(ValueError, match="not together in an indexed passage"):
        table.build(mapping())


def test_outside_source_path_is_rejected(tmp_path):
    store = TableStore(tmp_path / "workspace", iris_rows(1))
    outside = tmp_path / "outside.csv"
    outside.write_bytes(store.path.read_bytes())
    store.sources[0]["source_path"] = str(outside)
    with pytest.raises(ValueError, match="inside the workspace sources"):
        tabular_plot.prepare_tabular_plot(store, ["doc"])


def test_symlink_source_is_rejected(tmp_path):
    store = TableStore(tmp_path, iris_rows(1))
    link = tmp_path / "sources/link.csv"
    try:
        link.symlink_to(store.path)
    except OSError:
        pytest.skip("Symlinks are not available")
    store.sources[0]["source_path"] = str(link)
    with pytest.raises(ValueError, match="linked paths"):
        tabular_plot.prepare_tabular_plot(store, ["doc"])


@pytest.mark.parametrize("rows", [[], [["x", "y"]], [["x", "x"], ["1", "2"]],
                                  [["x", "y"], ["1", "2", "3"]]])
def test_invalid_or_empty_tables_fail_during_preflight(tmp_path, rows):
    store = TableStore(tmp_path, rows)
    with pytest.raises(ValueError):
        tabular_plot.prepare_tabular_plot(store, ["doc"])
