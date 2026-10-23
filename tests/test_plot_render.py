"""Trusted chart rendering: exact data, plain labels, and optional dependencies."""

import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
import math
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from rag_chatbot.tools import plot_render


def plot_payload(chart_type="bar"):
    points = [
        {"label": label, "x": None if chart_type == "bar" else index,
         "y": value, "source_ids": ["s1"],
         "evidence": f"At {index} s, Sample {label} measured {value} units."}
        for index, (label, value) in enumerate((("A", 2), ("B", 4), ("C", 7)), 1)
    ]
    return {"tool_id": "plot", "title": "Measured sample response",
            "sources": [{"id": "s1", "text": " ".join(point["evidence"] for point in points)}],
            "content": {"chart_type": chart_type, "x_label": "Sample" if chart_type == "bar" else "Time",
                        "x_unit": "" if chart_type == "bar" else "s", "y_label": "Response",
                        "y_unit": "units", "series": [{"name": "Measurements", "points": points}],
                        "limitations": []}}


def capture_figure(monkeypatch, inspect):
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    def savefig(figure, output, **kwargs):
        inspect(figure)
        output.write(b"inspected")

    monkeypatch.setattr(Figure, "savefig", savefig)
    return matplotlib


def test_import_does_not_load_matplotlib():
    code = "import sys; import rag_chatbot.tools.plot_render; assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("format,signature", [("png", b"\x89PNG\r\n\x1a\n"),
                                                ("svg", b"<?xml"), ("pdf", b"%PDF-")])
def test_export_formats(format, signature):
    pytest.importorskip("matplotlib")
    data = plot_render.plot_bytes(plot_payload(), format)
    assert data.startswith(signature)
    if format == "png":
        with Image.open(BytesIO(data)) as image:
            assert 700 < image.width < 5000
            assert 500 < image.height < 3000
            image.verify()
    elif format == "svg":
        root = ET.fromstring(data)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert "Measured sample response" in data.decode()


def test_line_points_sorted_without_mutating_payload(monkeypatch):
    payload = plot_payload("line")
    payload["content"]["series"][0]["points"].reverse()
    original = deepcopy(payload)

    def inspect(figure):
        line = figure.axes[0].lines[0]
        assert list(line.get_xdata()) == [1, 2, 3]
        assert list(line.get_ydata()) == [2, 4, 7]

    capture_figure(monkeypatch, inspect)
    plot_render.plot_bytes(payload, "png")
    assert payload == original


def test_missing_bar_categories_are_gaps_not_zero(monkeypatch):
    payload = plot_payload()
    points = payload["content"]["series"][0]["points"]
    payload["content"]["series"] = [{"name": "First", "points": [points[0], points[2]]},
                                      {"name": "Second", "points": [points[1]]}]

    def inspect(figure):
        axis = figure.axes[0]
        assert [patch.get_height() for patch in axis.patches] == [2, 7, 4]
        assert [label.get_text() for label in axis.get_xticklabels()] == ["A", "C", "B"]
        assert axis.get_ylim()[0] == 0
        assert axis.patches[2].get_x() > axis.patches[1].get_x()
        assert figure.legends

    capture_figure(monkeypatch, inspect)
    plot_render.plot_bytes(payload, "png")


def test_scatter_has_no_connecting_lines(monkeypatch):
    def inspect(figure):
        axis = figure.axes[0]
        assert not axis.lines
        assert len(axis.collections) == 1
        assert list(axis.collections[0].get_offsets()[:, 1]) == [2, 4, 7]

    capture_figure(monkeypatch, inspect)
    plot_render.plot_bytes(plot_payload("scatter"), "png")


def dense_payload(chart_type):
    payload = plot_payload(chart_type)
    payload["sources"][0]["text"] = "At 1 s, measured 2 units."
    points = [{"label": f"Sample {index}", "x": None if chart_type == "bar" else 1,
               "y": 2, "source_ids": ["s1"], "evidence": "At 1 s, measured 2 units."}
              for index in range(10_000)]
    if chart_type == "scatter":
        payload["content"]["series"] = [{"name": f"Species {index + 1}", "points": points[index::3]}
                                           for index in range(3)]
    else:
        payload["content"]["series"][0]["points"] = points
    return payload


def test_ten_thousand_scatter_points_keep_all_values_and_species_colors(monkeypatch):
    payload = dense_payload("scatter")

    def inspect(figure):
        axis = figure.axes[0]
        assert not axis.lines and len(axis.collections) == 3
        assert sum(len(collection.get_offsets()) for collection in axis.collections) == 10_000
        colors = {tuple(collection.get_facecolors()[0]) for collection in axis.collections}
        assert len(colors) == 3
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Species 1", "Species 2", "Species 3"]
        assert len(axis.texts) == 0
        assert tuple(figure.get_size_inches()) == (9.6, 6.4)

    capture_figure(monkeypatch, inspect)
    plot_render.plot_bytes(payload, "png")


def test_ten_thousand_bars_keep_all_values_with_bounded_ticks_and_canvas(monkeypatch):
    payload = dense_payload("bar")

    def inspect(figure):
        axis = figure.axes[0]
        assert len(axis.collections) == 1 and not axis.patches
        assert len(axis.collections[0].get_paths()) == 10_000
        assert all(path.vertices[:, 1].max() == 2 for path in axis.collections[0].get_paths())
        assert len(axis.get_xticks()) == plot_render.MAX_CATEGORY_TICKS
        labels = axis.get_xticklabels()
        assert labels[0].get_text() == "Sample 0" and labels[-1].get_text() == "Sample 9999"
        assert figure.get_size_inches()[0] <= 30 and figure.get_size_inches()[1] <= 8.2

    capture_figure(monkeypatch, inspect)
    plot_render.plot_bytes(payload, "png")


@pytest.mark.parametrize("chart_type", ["bar", "scatter"])
def test_ten_thousand_point_png_has_bounded_dimensions(chart_type):
    pytest.importorskip("matplotlib")
    data = plot_render.plot_bytes(dense_payload(chart_type), "png")
    with Image.open(BytesIO(data)) as image:
        assert image.width < 5000 and image.height < 3000
        image.verify()


@pytest.mark.parametrize("chart_type,value", [("bar", 0), ("bar", -2), ("line", 1e100),
                                                ("scatter", 1e-200)])
def test_constant_and_extreme_values_have_finite_axes(monkeypatch, chart_type, value):
    payload = plot_payload(chart_type)
    for point in payload["content"]["series"][0]["points"]:
        point["y"] = value
        point["evidence"] += f" The result was {value} units."
    payload["sources"][0]["text"] = " ".join(point["evidence"]
                                              for point in payload["content"]["series"][0]["points"])

    def inspect(figure):
        axis = figure.axes[0]
        assert all(math.isfinite(bound) for bound in (*axis.get_xlim(), *axis.get_ylim()))
        if chart_type == "bar":
            assert 0 in axis.get_ylim()

    capture_figure(monkeypatch, inspect)
    plot_render.plot_bytes(payload, "png")


def test_untrusted_text_is_plain_and_escaped_even_in_svg(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    payload = plot_payload()
    payload["title"] = '<script>alert("x")</script> $\\frac{unclosed'
    payload["content"]["y_label"] = r"$\input{secret}$"
    with matplotlib.rc_context({"text.usetex": True, "text.parse_math": True}):
        svg = plot_render.plot_bytes(payload, "svg").decode()
        assert matplotlib.rcParams["text.usetex"] is True
        assert matplotlib.rcParams["text.parse_math"] is True
    root = ET.fromstring(svg)
    assert not any(element.tag.rsplit("}", 1)[-1] == "script" for element in root.iter())
    assert "&lt;script&gt;" in svg
    assert "\\input{secret}" in svg
    assert "\\frac{unclosed" in svg


def test_preview_is_bounded_png_with_escaped_alt():
    pytest.importorskip("matplotlib")
    payload = plot_payload()
    payload["title"] = '<img src=x onerror="alert(1)">'
    preview = plot_render.render_plot_html(payload)
    assert "max-height:480px" in preview and "overflow:auto" in preview
    assert '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;' in preview
    assert preview.count("<img ") == 1
    encoded = re.search(r'data:image/png;base64,([^\"]+)', preview).group(1)
    assert base64.b64decode(encoded).startswith(b"\x89PNG")


def test_invalid_data_rejected_before_importing_plotting(monkeypatch):
    payload = plot_payload()
    payload["content"]["series"][0]["points"][0]["source_ids"] = ["invented"]
    monkeypatch.setattr(plot_render, "_load_matplotlib", lambda: pytest.fail("Must validate before plotting"))
    with pytest.raises(ValueError):
        plot_render.plot_bytes(payload, "png")
    with pytest.raises(ValueError):
        plot_render.plot_bytes(plot_payload(), "html")


def test_missing_dependency_retains_data_export_guidance(monkeypatch):
    def missing():
        raise plot_render.PlotDependencyError(plot_render._INSTALL_MESSAGE)

    monkeypatch.setattr(plot_render, "_load_matplotlib", missing)
    with pytest.raises(plot_render.PlotDependencyError, match="plugins/plot/requirements.txt"):
        plot_render.plot_bytes(plot_payload(), "png")
    preview = plot_render.render_plot_html(plot_payload())
    assert "plugins/plot/requirements.txt" in preview
    assert "CSV, and JSON exports remain available" in preview
    assert "data:image" not in preview


def test_plotting_calls_are_serialized(monkeypatch):
    pytest.importorskip("matplotlib")
    original_load = plot_render._load_matplotlib
    active, maximum = 0, 0

    def tracked_load():
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        time.sleep(0.02)
        active -= 1
        return original_load()

    capture_figure(monkeypatch, lambda figure: None)
    monkeypatch.setattr(plot_render, "_load_matplotlib", tracked_load)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: plot_render.plot_bytes(plot_payload(), "png"), range(3)))
    assert results == [b"inspected"] * 3
    assert maximum == 1
