"""Figure extras drawn on already prepared curves: a zoom inset, hand-written peak marks
(vertical guide lines with labels) and shaded bands.

The inset helpers are the ones XRD Maker uses (plot_annotations.py there), with the unit
text changed to cm⁻¹. Nothing here normalises or offsets intensities: the app prepares the
curves first and passes them in.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import html
import math
import re
from typing import Sequence

import numpy as np


_TAG = re.compile(r"</?(?:i|b|sub|sup|br)\s*/?>", re.I)
UNIT_MPL = "cm$^{\\mathrm{{-}1}}$"      # what rich_text.rich("cm^{-1}") gives: an upright −1
UNIT_HTML = "cm<sup>−1</sup>"


@dataclass(frozen=True)
class PreparedCurve:
    name: str
    x: Sequence[float]
    y: Sequence[float]
    color: str = "#000000"
    linewidth: float = 1.2


@dataclass(frozen=True)
class PeakMark:
    x: float
    label: str = ""


@dataclass(frozen=True)
class Band:
    low: float
    high: float
    label: str = ""


def _hover_text(name) -> str:
    """Names may carry Plotly's <i>/<sub> markup for the legend; hover text shows them plainly."""
    return html.escape(html.unescape(_TAG.sub("", str(name))))


def _curve_arrays(curve: PreparedCurve):
    x, y = np.asarray(curve.x, dtype=float), np.asarray(curve.y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Prepared curves need equal-length one-dimensional x/y arrays.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Prepared curves must contain finite x/y values.")
    order = np.argsort(x, kind="stable")
    return x[order], y[order], order


# ------------------------------------------------------------------------------------ marks
_NUMBER = r"[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?"
_RANGE = re.compile(rf"^\s*({_NUMBER})\s*(?:-|–|—|〜|～|~|to)\s*({_NUMBER})(?:[\s,;\t:]+(.*))?$")
_SINGLE = re.compile(rf"^\s*({_NUMBER})(?:[\s,;\t:]+(.*))?$")


def parse_marks(text: str) -> tuple[list[PeakMark], list[Band]]:
    """One mark per line: ``796``, ``796 SiC (TO)``, ``796, SiC``; a band: ``770-822 SiC window``.

    Lines that do not start with a number are skipped; ``#`` starts a comment.
    """
    marks, bands = [], []
    for line in str(text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _RANGE.match(line)
        if m:
            low, high = sorted((float(m.group(1)), float(m.group(2))))
            if math.isfinite(low) and math.isfinite(high) and high > low:
                bands.append(Band(low, high, (m.group(3) or "").strip()))
            continue
        m = _SINGLE.match(line)
        if m:
            value = float(m.group(1))
            if math.isfinite(value):
                marks.append(PeakMark(value, (m.group(2) or "").strip()))
    return marks, bands


def mark_text(mark: PeakMark, show_value: bool, decimals: int) -> str:
    """The label of a mark: its label, its value, or both ('796 SiC (TO)')."""
    value = f"{mark.x:.{max(0, int(decimals))}f}"
    if mark.label and show_value:
        return f"{value} {mark.label}"
    return mark.label or value


def top_label_rows(items, x_range, axes_width_points, font_size, gap_points=3.0, max_rows=3):
    """Rows (0 = nearest the axes) for labels above the axes so that neighbours do not overlap.

    ``items`` are (x, plain text, rotated) tuples. Widths are estimated from the text length
    (0.55 × font size per character; rotated text is one line high), so the layout is the same
    in the matplotlib figure and the Plotly preview.
    """
    span = float(x_range[1] - x_range[0])
    if span <= 0 or axes_width_points <= 0:
        return [0] * len(items)
    per_unit = axes_width_points / span
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    edges: list[float] = []           # right edge (points) of the last label in each row
    rows = [0] * len(items)
    for i in order:
        x, text, rotated = items[i]
        width = font_size * 1.2 if rotated else 0.55 * font_size * max(1, len(text))
        centre = (x - x_range[0]) * per_unit
        left, right = centre - width / 2, centre + width / 2
        row = next((r for r, edge in enumerate(edges) if left > edge + gap_points), None)
        if row is None:
            if len(edges) < max_rows:
                edges.append(right)
                row = len(edges) - 1
            else:
                row = min(range(len(edges)), key=lambda r: edges[r])
                edges[row] = right
        else:
            edges[row] = right
        rows[i] = row
    return rows


def _top_labels(marks, bands, x_range, label_mode, show_value, decimals, show_band_labels, plain,
                axes_width_points, font_size):
    """(x, text, rotated, kind) for every label above the axes and its row: marks first, then band labels."""
    items = []
    if label_mode != "none":
        for mark in marks:
            if x_range is not None and not (x_range[0] <= mark.x <= x_range[1]):
                continue
            text = mark_text(mark, show_value, decimals)
            if text:
                items.append((mark.x, text, label_mode == "rotated", "mark"))
    if show_band_labels:
        for band in bands:
            if x_range is not None and (band.high < x_range[0] or band.low > x_range[1]):
                continue
            if band.label:
                items.append(((band.low + band.high) / 2, band.label, False, "band"))
    if x_range is None:
        xs = [x for x, _, _, _ in items]
        x_range = (min(xs), max(xs) + 1e-9) if xs else (0, 1)
    rows = top_label_rows([(x, plain(text), rotated) for x, text, rotated, _ in items], x_range,
                          axes_width_points, font_size)
    return items, rows


def top_labels_height(marks, bands, *, x_range, label_mode="top", show_value=True, decimals=0,
                      show_band_labels=True, plain=None, axes_width_points=500.0, font_size=7.0) -> float:
    """Height in points that the labels above the axes need (0 when there are none)."""
    plain = plain or (lambda s: s)
    items, rows = _top_labels(marks, bands, x_range, label_mode, show_value, decimals, show_band_labels, plain,
                              float(axes_width_points), float(font_size))
    if not items:
        return 0.0
    heights = []
    for (x, text, rotated, _), row in zip(items, rows):
        extent = 0.55 * font_size * max(1, len(plain(text))) if rotated else font_size * 1.1
        heights.append(2 + row * font_size * 1.35 + extent)
    return max(heights)


def add_mpl_bands(ax, bands: Sequence[Band], *, color="#BBBBBB", alpha=0.3, x_range=None):
    """Shaded x ranges behind the curves (labels are drawn by add_mpl_marks)."""
    for band in bands:
        if x_range is not None and (band.high < x_range[0] or band.low > x_range[1]):
            continue
        ax.axvspan(band.low, band.high, facecolor=color, alpha=alpha, edgecolor="none", zorder=0)


def add_mpl_marks(ax, marks: Sequence[PeakMark], bands: Sequence[Band] = (), *, color="#555555", linestyle="--",
                  linewidth=0.8, label_mode="top", show_value=True, decimals=0, font_size=7, rich=None, plain=None,
                  x_range=None, show_band_labels=True, axes_width_points=500.0):
    """Vertical guide lines at the marks, and the mark and band labels above the axes in non-overlapping rows."""
    rich = rich or (lambda s: s)
    plain = plain or (lambda s: s)
    for mark in marks:
        if x_range is not None and not (x_range[0] <= mark.x <= x_range[1]):
            continue
        if linestyle != "none":
            ax.axvline(mark.x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8, zorder=0.5)
    items, rows = _top_labels(marks, bands, x_range, label_mode, show_value, decimals, show_band_labels, plain,
                              float(axes_width_points), float(font_size))
    for (x, text, rotated, kind), row in zip(items, rows):
        ax.annotate(rich(text), xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 2 + row * font_size * 1.35), textcoords="offset points",
                    ha="center", va="bottom", rotation=90 if rotated else 0, fontsize=font_size,
                    color=color if kind == "mark" else "#333333", annotation_clip=False)


def add_plotly_bands(fig, bands: Sequence[Band], *, color="#BBBBBB", alpha=0.3, x_range=None):
    for band in bands:
        if x_range is not None and (band.high < x_range[0] or band.low > x_range[1]):
            continue
        fig.add_shape(type="rect", x0=band.low, x1=band.high, y0=0, y1=1, xref="x", yref="y domain",
                      fillcolor=color, opacity=alpha, line=dict(width=0), layer="below")


def add_plotly_marks(fig, marks: Sequence[PeakMark], bands: Sequence[Band] = (), *, color="#555555", linestyle="dash",
                     linewidth=0.8, label_mode="top", show_value=True, decimals=0, font_size=7, to_html=None, plain=None,
                     x_range=None, show_band_labels=True, axes_width_points=500.0, font_family="Arial"):
    to_html = to_html or (lambda s: html.escape(str(s)))
    plain = plain or (lambda s: s)
    for mark in marks:
        if x_range is not None and not (x_range[0] <= mark.x <= x_range[1]):
            continue
        if linestyle != "none":
            fig.add_shape(type="line", x0=mark.x, x1=mark.x, y0=0, y1=1, xref="x", yref="y domain",
                          line=dict(color=color, width=linewidth * 4 / 3, dash=linestyle), opacity=0.8,
                          layer="below")
    items, rows = _top_labels(marks, bands, x_range, label_mode, show_value, decimals, show_band_labels, plain,
                              float(axes_width_points), float(font_size))
    for (x, text, rotated, kind), row in zip(items, rows):
        fig.add_annotation(x=x, y=1.0, xref="x", yref="y domain", text=to_html(text), showarrow=False,
                           textangle=-90 if rotated else 0, xanchor="center", yanchor="bottom",
                           yshift=(2 + row * font_size * 1.35) * 4 / 3,
                           font=dict(size=font_size, color=color if kind == "mark" else "#333333", family=font_family))


# ------------------------------------------------------------------------------------ inset
@contextmanager
def _preserve_mpl_limits(ax):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    data_lim = ax.dataLim.frozen()
    auto_x, auto_y = ax.get_autoscalex_on(), ax.get_autoscaley_on()
    try:
        yield
    finally:
        ax.dataLim.set(data_lim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_autoscalex_on(auto_x)
        ax.set_autoscaley_on(auto_y)


def _zoom_geometry(x_range, location, width, height, pad):
    if len(x_range) != 2 or not np.isfinite(x_range).all() or x_range[0] >= x_range[1]:
        raise ValueError("Zoom interval must contain two finite, increasing coordinates.")
    if location not in ("upper left", "upper right", "lower left", "lower right"):
        raise ValueError("Inset location must select an upper/lower left/right corner.")
    if not all(math.isfinite(v) for v in (width, height, pad)) or width <= 0 or height <= 0 or pad < 0 or max(width, height) + 2 * pad > 1:
        raise ValueError("Inset width/height and padding must fit inside the main axes.")
    left = pad if location.endswith("left") else 1 - pad - width
    bottom = 1 - pad - height if location.startswith("upper") else pad
    return (float(x_range[0]), float(x_range[1])), (left, bottom, width, height)


def _zoom_curves(curves, interval, y_padding):
    if not math.isfinite(y_padding) or y_padding < 0:
        raise ValueError("Inset intensity padding must be finite and nonnegative.")
    low, high = interval
    clipped, limit_values = [], []
    for curve in curves:
        x, y, _ = _curve_arrays(curve)
        if not len(x) or high < x[0] or low > x[-1]:
            continue
        start = max(0, int(np.searchsorted(x, low, side="left")) - 1)
        stop = min(len(x), int(np.searchsorted(x, high, side="right")) + 1)
        clipped.append((curve, x[start:stop], y[start:stop]))
        inside = (x >= low) & (x <= high)
        limit_values.extend(y[inside].tolist())
        for edge in (low, high):
            if x[0] <= edge <= x[-1]:
                limit_values.append(float(np.interp(edge, x, y)))
    if not clipped or not limit_values:
        return [], None
    minimum, maximum = min(limit_values), max(limit_values)
    span = maximum - minimum
    margin = y_padding * (span if span > 0 else max(abs(maximum), 1.0))
    if margin == 0 and span == 0:
        margin = max(abs(maximum), 1.0) * 0.05
    return clipped, (minimum - margin, maximum + margin)


def add_mpl_zoom_inset(ax, curves: Sequence[PreparedCurve], x_range, *, location="upper left",
                       width: float = 0.42, height: float = 0.35, pad: float = 0.05,
                       font_size: float | None = None, show_indicator: bool = True,
                       show_yticklabels: bool = False, y_padding: float = 0.08, reverse_x: bool = False):
    """A fixed-position zoom of the same prepared curves; None if the interval is empty.
    ``reverse_x`` runs the inset from high to low x, as the main axes of an FT-IR figure do."""
    from matplotlib.patches import Rectangle
    interval, bounds = _zoom_geometry(x_range, location, width, height, pad)
    clipped, y_range = _zoom_curves(curves, interval, y_padding)
    if not clipped:
        return None
    size = float(font_size) if font_size is not None else max(7, ax.xaxis.label.get_fontsize() * 0.72)
    with _preserve_mpl_limits(ax):
        inset = ax.inset_axes(bounds, transform=ax.transAxes, zorder=10)
        inset.set_in_layout(False)
        inset.set_facecolor("white")
        for curve, x, y in clipped:
            inset.plot(x, y, color=curve.color, linewidth=curve.linewidth, label=curve.name)
        inset.set_xlim(interval[::-1] if reverse_x else interval)
        inset.set_ylim(y_range)
        parent_ticks = ax.xaxis.get_major_ticks()
        tick_width = parent_ticks[0].tick1line.get_markeredgewidth() if parent_ticks else 0.8
        inset.tick_params(axis="both", direction="in", labelsize=size, length=3, width=tick_width)
        inset.tick_params(axis="y", labelleft=show_yticklabels)
        inset.text(.97, .96, f"{interval[0]:g}–{interval[1]:g} {UNIT_MPL}", transform=inset.transAxes,
                   ha="right", va="top", fontsize=size,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1))
        for spine in inset.spines.values():
            spine.set_color(ax.spines["left"].get_edgecolor())
            spine.set_linewidth(ax.spines["left"].get_linewidth())
        if show_indicator:
            indicator = Rectangle((interval[0], 0), interval[1] - interval[0], 1,
                transform=ax.get_xaxis_transform(), facecolor="none", edgecolor="#777777",
                linewidth=0.8, linestyle="--", zorder=3, clip_on=True)
            ax.add_patch(indicator)
    return inset


def _plotly_axis_key(reference: str, letter: str):
    if not re.fullmatch(letter + r"(?:[2-9]|[1-9]\d+)?", reference):
        raise ValueError("Plotly axis references must use x/y, x2/y2, and so on.")
    return letter + "axis" + reference[1:]


def add_plotly_zoom_inset(fig, curves: Sequence[PreparedCurve], x_range, *, location="upper left",
                          width: float = 0.42, height: float = 0.35, pad: float = 0.05,
                          font_size: float | None = None, show_indicator: bool = True,
                          show_yticklabels: bool = False, y_padding: float = 0.08,
                          parent_xaxis: str = "x", parent_yaxis: str = "y", reverse_x: bool = False):
    """Independent inset axes without changing existing axes or trace data; returns (xref, yref) or None."""
    import plotly.graph_objects as go
    interval, bounds = _zoom_geometry(x_range, location, width, height, pad)
    clipped, y_range = _zoom_curves(curves, interval, y_padding)
    if not clipped:
        return None
    xkey, ykey = _plotly_axis_key(parent_xaxis, "x"), _plotly_axis_key(parent_yaxis, "y")
    parent_x, parent_y = getattr(fig.layout, xkey), getattr(fig.layout, ykey)
    x_domain, y_domain = list(parent_x.domain or (0, 1)), list(parent_y.domain or (0, 1))
    left, bottom, iw, ih = bounds
    domain_x = [x_domain[0] + value * (x_domain[1] - x_domain[0]) for value in (left, left + iw)]
    domain_y = [y_domain[0] + value * (y_domain[1] - y_domain[0]) for value in (bottom, bottom + ih)]
    used = {1}
    for key in fig.layout:
        match = re.fullmatch(r"[xy]axis(\d*)", key)
        if match:
            used.add(int(match[1] or "1"))
    for trace in fig.data:
        for letter in ("x", "y"):
            reference = getattr(trace, letter + "axis", None)
            if reference:
                used.add(int(reference[1:] or "1"))
    number = max(used) + 1
    xref, yref = f"x{number}", f"y{number}"
    size = float(font_size) if font_size is not None else max(7, float(fig.layout.font.size or 14) * 0.72)
    font = dict(family=fig.layout.font.family or "Arial", color=fig.layout.font.color or "black", size=size)
    linecolor = parent_x.linecolor or "black"
    frame_width = parent_x.linewidth or 0.8
    fig.update_layout({
        f"xaxis{number}": dict(domain=domain_x, anchor=yref, range=list(interval)[::-1] if reverse_x else list(interval), autorange=False,
            showline=True, mirror=True, linecolor=linecolor, linewidth=frame_width, ticks="inside",
            tickwidth=parent_x.tickwidth or frame_width,
            tickfont=font, nticks=4, showgrid=False, zeroline=False, automargin=False),
        f"yaxis{number}": dict(domain=domain_y, anchor=xref, range=list(y_range), autorange=False,
            showline=True, mirror=True, linecolor=parent_y.linecolor or linecolor, linewidth=parent_y.linewidth or frame_width,
            tickwidth=parent_y.tickwidth or parent_y.linewidth or frame_width,
            ticks="inside", tickfont=font, showticklabels=show_yticklabels, nticks=3,
            showgrid=False, zeroline=False, automargin=False),
    })
    fig.add_trace(go.Scatter(x=[interval[0], interval[1], interval[1], interval[0], interval[0]],
        y=[y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]], xaxis=xref, yaxis=yref,
        mode="lines", line=dict(width=0, color="white"), fill="toself", fillcolor="white",
        showlegend=False, hoverinfo="skip", cliponaxis=True, name="_zoom_background"))
    for curve, x, y in clipped:
        fig.add_trace(go.Scatter(x=x, y=y, xaxis=xref, yaxis=yref, mode="lines", name=curve.name,
            line=dict(color=curve.color, width=curve.linewidth * 4 / 3), showlegend=False, cliponaxis=True,
            hovertemplate=_hover_text(curve.name) + "<br>%{x:.1f} " + UNIT_HTML + "<br>I=%{y:.4g}<extra></extra>"))
    fig.add_annotation(x=domain_x[1] - .03 * (domain_x[1] - domain_x[0]),
        y=domain_y[1] - .04 * (domain_y[1] - domain_y[0]), xref="paper", yref="paper",
        text=f"{interval[0]:g}–{interval[1]:g} {UNIT_HTML}", showarrow=False, xanchor="right", yanchor="top",
        bgcolor="white", borderpad=1, font=font)
    if show_indicator:
        all_min = min(float(np.min(x)) for _, x, _ in clipped)
        all_max = max(float(np.max(x)) for _, x, _ in clipped)
        fig.add_shape(type="rect", x0=max(interval[0], all_min), x1=min(interval[1], all_max),
            y0=0, y1=1, xref=parent_xaxis, yref=parent_yaxis + " domain", fillcolor="rgba(0,0,0,0)",
            line=dict(color="#777777", width=0.8, dash="dash"), layer="below")
    return xref, yref
