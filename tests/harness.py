"""Run the real Raman Maker UI headlessly with Streamlit's AppTest.

The app source is loaded as text, a capture block is appended that stores the figure,
the preview data and the project JSON in session state, and ``format_func`` is removed
from selectboxes (Streamlit 1.32's AppTest serialises formatted options incorrectly;
the real browser shows the labels). Nothing scientific is mocked.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "tests" / "data"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

CAPTURE = '''
if active:
    st.session_state["_report"] = {
        "names": [f.name for f in active],
        "labels": list(labels), "colors": list(colors_sel), "orders": list(orders), "visibles": list(visibles),
        "curves": [(plain(r["label"]), r["color"], float(r["level"]), float(np.min(r["y_plot"])), float(np.max(r["y_plot"])),
                    int(len(r["x"])), float(r["x"][0]), float(r["x"][-1])) for r in curves],
        "mpl_lines": [(l.get_label(), l.get_color(), len(l.get_xdata())) for l in fig.axes[0].get_lines()],
        "mpl_legend": [t.get_text() for t in fig.axes[0].get_legend().get_texts()] if fig.axes[0].get_legend() else [],
        "mpl_axes": len(fig.axes) + len(fig.axes[0].child_axes), "mpl_texts": [t.get_text() for t in fig.axes[0].texts],
        "xlabel": fig.axes[0].get_xlabel(), "ylabel": fig.axes[0].get_ylabel(), "yticks": list(fig.axes[0].get_yticks()),
        "xlim": tuple(float(v) for v in fig.axes[0].get_xlim()), "ylim": tuple(float(v) for v in fig.axes[0].get_ylim()),
        "preview_xrange": list(pfig.layout.xaxis.range) if pfig.layout.xaxis.range else None,
        "peak_annotations": [(a.x, a.y, a.text, a.yanchor) for a in pfig.layout.annotations if a.yref in (None, "y")],
        "preview": [(t.name, t.line.color, len(t.x), t.legendrank) for t in pfig.data if t.mode == "lines" and (t.xaxis in (None, "x"))],
        "preview_annotations": [a.text for a in pfig.layout.annotations],
        "preview_shapes": len(pfig.layout.shapes),
        "png": png_bytes, "pdf_header": pdf_bytes[:5], "svg": svg_bytes, "tiff_header": (tiff_bytes or b"")[:4],
        "png_export": png_export_bytes, "project": build_session_json(),
        "style": encode_style_template(dict(st.session_state)),
        "peaks": peak_rows() if show_peaks else [],
    }
'''


def test_source() -> str:
    tree = ast.parse((APP_DIR / "app.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "selectbox":
            node.keywords = [kw for kw in node.keywords if kw.arg != "format_func"]
    tree = ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n" + CAPTURE


def data_files(*names) -> dict[str, bytes]:
    return {name: (DATA_DIR / name).read_bytes() for name in names}


def run_app(files: dict[str, bytes] | None = None, state: dict | None = None, timeout: int = 180):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(test_source(), default_timeout=timeout)
    if files is not None:
        at.session_state["_spectrum_bytes"] = dict(files)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def rerun(at, state: dict | None = None):
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def report(at) -> dict:
    assert not at.exception, [str(e.value) for e in at.exception]
    return at.session_state["_report"]
