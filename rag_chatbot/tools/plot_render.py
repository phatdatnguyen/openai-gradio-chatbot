"""Local, trusted rendering of validated chart data without generated code."""

import base64
import html
from io import BytesIO
import textwrap
import threading


_MATPLOTLIB_LOCK = threading.RLock()
_COLORS = ("#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626", "#0891b2")
MAX_CATEGORY_TICKS = 40
_COLLECTION_BAR_THRESHOLD = 200
_INSTALL_MESSAGE = (
    "Plot previews and PNG, SVG, or PDF exports require Matplotlib. "
    "Install it with: python -m pip install -r plugins/plot/requirements.txt. "
    "The data table, CSV, and JSON exports remain available."
)


class PlotDependencyError(RuntimeError):
    """The optional local plotting dependency is unavailable."""


def _validated_content(payload):
    from .evidence import require_text
    from .plot import validate_plot

    if not isinstance(payload, dict) or payload.get("tool_id") != "plot":
        raise ValueError("Choose a Plot output to render.")
    require_text(payload.get("title"), limit=200)
    sources = payload.get("sources")
    if (not isinstance(sources, list) or not sources
            or any(not isinstance(source, dict) or not isinstance(source.get("id"), str)
                   for source in sources)):
        raise ValueError("Plot sources must have identifiers.")
    return validate_plot(payload.get("content"),
                         allowed_source_ids={source["id"] for source in sources}, sources=sources)


def _load_matplotlib():
    # The caller owns _MATPLOTLIB_LOCK, including while importing and changing
    # rcParams. Figure/Agg avoids a GUI backend and pyplot's global figure list.
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.text import Text
    except ImportError as exc:
        raise PlotDependencyError(_INSTALL_MESSAGE) from exc
    return matplotlib, Figure, FigureCanvasAgg, Text


def _display_text(value, *, width=60, lines=3):
    normalized = " ".join(value.split())
    wrapped = textwrap.wrap(normalized, width=width,
                            break_long_words=True, break_on_hyphens=False) or [""]
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        # Preserve the end of long labels, which frequently holds the sample
        # identifier; full original labels always remain in the data payload.
        tail_length = min(10, max(4, width // 3))
        wrapped[-1] = wrapped[-1][:width - tail_length - 1].rstrip(" .") + "\u2026" + normalized[-tail_length:]
    return "\n".join(wrapped)


def _axis_label(label, unit):
    # Up to 160 label characters plus 80 unit characters fit without truncating
    # a unit. Units are part of the meaning of the values being plotted.
    return _display_text(f"{label} ({unit})" if unit else label, width=65, lines=5)


def _categories(content):
    return list(dict.fromkeys(point["label"] for series in content["series"]
                              for point in series["points"]))


def _figure_size(content):
    if content["chart_type"] == "bar":
        count = len(_categories(content))
        return max(9.6, min(30.0, 3.0 + count * 0.55)), (8.2 if count > 20 else 7.2 if count > 8 else 6.4)
    return 9.6, 6.4


def _draw_plot(figure, content, title):
    axis = figure.subplots()
    chart_type = content["chart_type"]
    categories = _categories(content) if chart_type == "bar" else []
    positions = {label: index for index, label in enumerate(categories)}
    series_count = len(content["series"])
    for index, series in enumerate(content["series"]):
        color = _COLORS[index]
        label = _display_text(series["name"], width=26, lines=2)
        points = series["points"]
        if chart_type == "bar":
            width = 0.8 / series_count
            offset = (index - (series_count - 1) / 2) * width
            # Render only supplied points. Missing categories are gaps, never
            # invented zero values, and all series share stable category order.
            xs = [positions[point["label"]] + offset for point in points]
            ys = [point["y"] for point in points]
            if len(points) > _COLLECTION_BAR_THRESHOLD:
                # One collection renders every supplied bar without creating
                # thousands of separate Artist objects or text labels.
                from matplotlib.collections import PolyCollection

                half_width = width * 0.47
                vertices = [[(x - half_width, 0), (x - half_width, y),
                             (x + half_width, y), (x + half_width, 0)]
                            for x, y in zip(xs, ys)]
                axis.add_collection(PolyCollection(vertices, facecolor=color,
                    edgecolor="none", label=label, zorder=3))
                axis.autoscale_view()
            else:
                axis.bar(xs, ys, width=width * 0.94, label=label, color=color, zorder=3)
        elif chart_type == "line":
            ordered = sorted(points, key=lambda point: point["x"])
            axis.plot([point["x"] for point in ordered], [point["y"] for point in ordered],
                      color=color, label=label, linewidth=2,
                      marker="o" if len(points) <= 200 else None, markersize=4,
                      zorder=3)
        else:
            axis.scatter([point["x"] for point in points], [point["y"] for point in points],
                         color=color, label=label, s=16 if len(points) > 500 else 40,
                         alpha=0.7 if len(points) > 500 else 0.85, zorder=3)

    if chart_type == "bar":
        dense = len(categories) > 20
        rotation = 90 if dense else 35 if len(categories) > 8 else 0
        # Tick labels are a view aid; data is never sampled. Always retain the
        # first and last label, with evenly spaced labels between them.
        tick_positions = (list(range(len(categories))) if len(categories) <= MAX_CATEGORY_TICKS else
                          [round(index * (len(categories) - 1) / (MAX_CATEGORY_TICKS - 1))
                           for index in range(MAX_CATEGORY_TICKS)])
        axis.set_xticks(tick_positions)
        axis.set_xticklabels([_display_text(label, width=24 if dense else 18, lines=1 if dense else 3)
                              for label in (categories[index] for index in tick_positions)], rotation=rotation,
                             ha="right" if rotation == 35 else "center", fontsize=9 if dense else 11)
        values = [point["y"] for series in content["series"] for point in series["points"]]
        if all(value == 0 for value in values):
            axis.set_ylim(0, 1)
        elif min(values) >= 0:
            axis.set_ylim(bottom=0)
        elif max(values) <= 0:
            axis.set_ylim(top=0)
        axis.axhline(0, color="#64748b", linewidth=0.8, zorder=2)
        axis.margins(x=0.025)
    else:
        axis.margins(x=0.07, y=0.12)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-4, 5),
                              useOffset=False, useMathText=False)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, 5),
                          useOffset=False, useMathText=False)
    axis.set_title(_display_text(title, width=76, lines=3), fontsize=17,
                   fontweight="bold", loc="left", pad=20, color="#172554")
    axis.set_xlabel(_axis_label(content["x_label"], content["x_unit"]), labelpad=12)
    axis.set_ylabel(_axis_label(content["y_label"], content["y_unit"]), labelpad=12)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for edge in ("top", "right"):
        axis.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        axis.spines[edge].set_color("#cbd5e1")
    axis.tick_params(axis="both", color="#cbd5e1", labelcolor="#334155", length=3, pad=7)
    if series_count > 1:
        # Figure's outside legend participates in constrained layout, so even
        # long rotated category labels cannot collide with the legend.
        figure.legend(*axis.get_legend_handles_labels(), loc="outside lower left",
                      ncol=min(series_count, 3), frameon=False, fontsize=9,
                      borderaxespad=0.8, handlelength=1.6, columnspacing=1.8)
    return axis


def plot_bytes(payload, format):
    """Export a validated chart using trusted plotting calls and native fonts."""
    if not isinstance(format, str) or format not in {"png", "svg", "pdf"}:
        raise ValueError("Choose PNG, SVG, or PDF for a Plot image export.")
    content = _validated_content(payload)
    with _MATPLOTLIB_LOCK:
        matplotlib, Figure, FigureCanvasAgg, Text = _load_matplotlib()
        with matplotlib.rc_context({
            "text.usetex": False, "text.parse_math": False,
            "font.family": "DejaVu Sans", "font.size": 11,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "axes.formatter.use_mathtext": False, "svg.fonttype": "none",
            "svg.hashsalt": "rag-plot", "pdf.fonttype": 42,
        }):
            figure = Figure(figsize=_figure_size(content), dpi=140, layout="constrained")
            FigureCanvasAgg(figure)
            try:
                _draw_plot(figure, content, payload["title"])
                for text in figure.findobj(match=Text):
                    text.set_usetex(False)
                    text.set_parse_math(False)
                output = BytesIO()
                figure.savefig(output, format=format, dpi=140, facecolor="white",
                               bbox_inches="tight", pad_inches=0.2)
                return output.getvalue()
            finally:
                figure.clear()


def render_plot_html(payload):
    """Return an escaped raster preview; optional dependency failure is local."""
    try:
        data = plot_bytes(payload, "png")
    except PlotDependencyError as exc:
        return ('<div class="rag-plot-preview" style="padding:16px;overflow-wrap:anywhere">'
                + html.escape(str(exc)) + "</div>")
    content = payload["content"]
    minimum_width = round(_figure_size(content)[0] * 75)
    encoded = base64.b64encode(data).decode("ascii")
    title = html.escape(payload["title"], quote=True)
    return (
        '<div class="rag-plot-preview" style="max-height:480px;overflow:auto;'
        'padding:4px;border:1px solid #e2e8f0;border-radius:8px;background:white">'
        f'<img alt="{title}" src="data:image/png;base64,{encoded}" '
        f'style="display:block;width:100%;min-width:{minimum_width}px;height:auto" /></div>'
    )
