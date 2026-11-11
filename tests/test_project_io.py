"""Projects, style templates and the figure extras (marks, bands, inset)."""
import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from figure_style import STYLE_DEFAULTS
from plot_extras import (PeakMark, PreparedCurve, add_mpl_bands, add_mpl_marks, add_mpl_zoom_inset, add_plotly_bands,
                         add_plotly_marks, add_plotly_zoom_inset, mark_text, parse_marks, top_label_rows, top_labels_height)
from project_io import GLOBAL_KEYS, PATTERN_KEY, ProjectError, clean_settings, decode_project, encode_project, project_fingerprint
from style_templates import TEMPLATE_KEYS, decode_style_template, encode_style_template

SETTINGS = {
    "spec_mode": "ftir", "ir_display": "transmittance", "x_reverse": True, "yunit_0": "transmittance_percent",
    "xrange": (200.0, 1000.0), "normalize_mode": "window", "norm_window": (770.0, 822.0), "stack": True,
    "global_offset": 1.1, "show_legend": True, "legend_loc": "upper left", "show_yticks": False,
    "major_tick": 100.0, "show_minor": True, "minor_tick": 50.0, "x_label": "Raman shift (cm^{-1})", "y_label": "Intensity (a.u.)",
    "spike_remove": True, "spike_threshold": 6.0, "smooth_on": False, "smooth_window": 7, "smooth_order": 2,
    "baseline_method": "als", "baseline_order": 3, "baseline_lam": 5.0, "baseline_p": 0.01, "multi_mode": "mean",
    "show_peaks": True, "peak_prom": 0.1, "peak_distance": 5.0, "peak_decimals": 1, "peak_label_rotate": True,
    "peak_fit": True, "peak_model": "lorentzian", "peak_half_window": 15.0,
    "marks_text": "796 SiC (TO)\n770-822 window", "marks_style": "dashed", "marks_color": "#555555",
    "marks_label_mode": "rotated", "marks_show_value": True, "marks_decimals": 0, "marks_fontsize": 6,
    "bands_color": "#BBBBBB", "bands_alpha": 0.3, "bands_label": True,
    "show_zoom": True, "zoom_range": (900.0, 980.0), "zoom_location": "upper left", "annotation_font_size": 6,
    "show_side_labels": True, "label_side": "右", "label_fontsize": 7, "label_offset_x": 20.0, "label_offset_y": 0.1,
    "fig_width": 88 / 25.4, "fig_height": 70 / 25.4, "dpi_export": 300, "font_size": 7.0, "tick_font_size": 6.5,
    "legend_font_size": 6.5, "show_panel": True, "lang_radio": "日本語",
    "palette": "blues", "palette_reverse": False, "palette_steps": 0, "palette_from": 2, "palette_to": 0,
    "palette_base_color": "#000000",
    "vis_0": True, "ord_0": 2, "_val_lbl_0": "as-sintered", "sp_color_0": "#0072B2", "extra_offset_0": 0.2, "yoff_0": 0.0,
    "scale_0": 1.0,
    **STYLE_DEFAULTS,
}


class ProjectTests(unittest.TestCase):
    def test_round_trip(self):
        files = {"a.csv": b"100,1\n101,2\n", "b.txt": b"100\t5\n101\t6\n"}
        raw = encode_project(files, SETTINGS)
        document = json.loads(raw)
        self.assertEqual(document["format"], "raman-maker-project")
        self.assertEqual(document["version"], 1)
        project = decode_project(raw)
        self.assertEqual(project["spectrum_files"], files)
        restored = project["settings"]
        self.assertEqual(restored["xrange"], (200.0, 1000.0))
        self.assertEqual(restored["marks_text"], SETTINGS["marks_text"])
        self.assertEqual(restored["ord_0"], 2)
        self.assertEqual(restored["spec_mode"], "ftir")
        self.assertEqual(restored["yunit_0"], "transmittance_percent")
        for key in SETTINGS:
            self.assertIn(key, restored, key)

    def test_every_global_key_is_validated(self):
        cleaned = clean_settings(SETTINGS)
        unknown = [k for k in GLOBAL_KEYS if k not in cleaned and k not in STYLE_DEFAULTS]
        self.assertEqual(unknown, [], "add every setting to SETTINGS so its validation is exercised")
        self.assertTrue(PATTERN_KEY.fullmatch("sp_color_12"))
        self.assertTrue(PATTERN_KEY.fullmatch("scale_3"))
        self.assertFalse(PATTERN_KEY.fullmatch("xrd_color_0"))

    def test_invalid_settings_are_rejected(self):
        for key, value in (("normalize_mode", "squared"), ("xrange", (900.0, 200.0)), ("xrange", (-5000.0, 100.0)),
                           ("baseline_lam", 99.0), ("marks_color", "red"), ("marks_text", "x" * 30000),
                           ("peak_decimals", 1.5), ("scale_0", 0.0), ("extra_offset_0", 99.0), ("legend_loc", "center"),
                           ("multi_mode", "last"), ("dpi_export", 150), ("vis_0", 1), ("ord_0", 0),
                           ("spec_mode", "uvvis"), ("ir_display", "reflectance"), ("x_reverse", "yes"), ("yunit_0", "percent")):
            with self.subTest(key=key):
                with self.assertRaises(ProjectError):
                    clean_settings({**SETTINGS, key: value})
        self.assertNotIn("unknown_key", clean_settings({**SETTINGS, "unknown_key": 1}))

    def test_damaged_projects(self):
        raw = encode_project({"a.csv": b"100,1\n101,2\n"}, SETTINGS)
        with self.assertRaises(ProjectError):
            decode_project(b"{not json")
        document = json.loads(raw)
        document["format"] = "xrd-maker-project"
        with self.assertRaises(ProjectError):
            decode_project(json.dumps(document).encode())
        document = json.loads(raw)
        document["checksums"]["a.csv"] = "0" * 64
        with self.assertRaises(ProjectError):
            decode_project(json.dumps(document).encode())
        with self.assertRaises(ProjectError):
            encode_project({"": b"x"}, SETTINGS)

    def test_fingerprint_ignores_the_timestamp(self):
        a = encode_project({"a.csv": b"1 2\n3 4\n"}, SETTINGS)
        b = encode_project({"a.csv": b"1 2\n3 4\n"}, SETTINGS)
        self.assertEqual(project_fingerprint(a), project_fingerprint(b))
        c = encode_project({"a.csv": b"1 2\n3 4\n"}, {**SETTINGS, "global_offset": 1.5})
        self.assertNotEqual(project_fingerprint(a), project_fingerprint(c))


class StyleTemplateTests(unittest.TestCase):
    def test_round_trip_keeps_only_appearance(self):
        raw = encode_style_template(SETTINGS, "Paper")
        document = json.loads(raw)
        self.assertEqual(document["format"], "raman-maker-style")
        self.assertNotIn("marks_text", document["settings"])
        self.assertNotIn("normalize_mode", document["settings"])
        self.assertNotIn("spec_mode", document["settings"], "the measurement is not part of a figure style")
        self.assertNotIn("x_reverse", document["settings"])
        restored = decode_style_template(raw)
        self.assertEqual(restored["legend_loc"], "upper left")
        self.assertEqual(restored["marks_label_mode"], "rotated")
        self.assertTrue(set(restored) <= TEMPLATE_KEYS)

    def test_xrd_maker_style_opens_without_its_degree_settings(self):
        xrd = {"format": "xrd-maker-style", "version": 1, "name": "x",
               "settings": {"font_size": 9.0, "x_label": "2θ", "major_tick": 10.0, "line_width": 1.0, "axes_box": False}}
        restored = decode_style_template(json.dumps(xrd).encode())
        self.assertEqual(restored["font_size"], 9.0)
        self.assertEqual(restored["line_width"], 1.0)
        self.assertNotIn("x_label", restored)
        self.assertNotIn("major_tick", restored)

    def test_rejects_other_documents(self):
        with self.assertRaises(ProjectError):
            decode_style_template(b"[]")
        with self.assertRaises(ProjectError):
            decode_style_template(json.dumps({"format": "raman-maker-style", "version": 1,
                                              "settings": {"marks_text": "x"}}).encode())


class MarkTests(unittest.TestCase):
    def test_parse(self):
        marks, bands = parse_marks("796 SiC (TO)\n924, Yb_{2}Si_{2}O_{7}\n954\n770-822 window\n  # a note\nabc\n1000–1100\n")
        self.assertEqual([(m.x, m.label) for m in marks], [(796.0, "SiC (TO)"), (924.0, "Yb_{2}Si_{2}O_{7}"), (954.0, "")])
        self.assertEqual([(b.low, b.high, b.label) for b in bands], [(770.0, 822.0, "window"), (1000.0, 1100.0, "")])
        self.assertEqual(mark_text(PeakMark(796.4, "SiC"), True, 1), "796.4 SiC")
        self.assertEqual(mark_text(PeakMark(796.4, "SiC"), False, 1), "SiC")
        self.assertEqual(mark_text(PeakMark(796.4), False, 0), "796")

    def test_rows_separate_neighbours(self):
        items = [(796, "796 SiC (TO)", False), (924, "924 Yb2Si2O7", False), (954, "954", False), (796, "window", False)]
        rows = top_label_rows(items, (200, 1000), 180.0, 6.0)
        self.assertEqual(rows[0], 0)
        self.assertNotEqual(rows[0], rows[3], "two labels at the same place take different rows")
        self.assertNotEqual(rows[1], rows[2], "924 and 954 are too close for one row at this width")
        wide = top_label_rows(items[:3], (200, 1000), 5000.0, 6.0)
        self.assertEqual(wide, [0, 0, 0])
        rotated = top_label_rows([(x, t, True) for x, t, _ in items[:2]], (200, 1000), 180.0, 6.0)
        self.assertEqual(rotated, [0, 0], "rotated labels are narrow enough for one row")
        self.assertEqual(top_label_rows([], (200, 1000), 180.0, 6.0), [])
        self.assertGreater(top_labels_height(*parse_marks("796 SiC\n800 TO"), x_range=(200, 1000), axes_width_points=180), 6)
        self.assertEqual(top_labels_height([], [], x_range=(200, 1000)), 0.0)

    def test_drawn_on_both_renderers(self):
        marks, bands = parse_marks("796 SiC (TO)\n924\n770-822 window\n5000 outside")
        fig, ax = plt.subplots()
        ax.plot([200, 1000], [0, 1])
        ax.set_xlim(200, 1000)
        add_mpl_bands(ax, bands, x_range=(200, 1000))
        add_mpl_marks(ax, marks, bands, x_range=(200, 1000), axes_width_points=180, font_size=6)
        self.assertEqual(len([line for line in ax.lines if line.get_linestyle() == "--"]), 2)
        self.assertEqual(sorted(t.get_text() for t in ax.texts), ["796 SiC (TO)", "924", "window"])
        plt.close(fig)
        pfig = go.Figure(go.Scatter(x=[200, 1000], y=[0, 1]))
        add_plotly_bands(pfig, bands, x_range=(200, 1000))
        add_plotly_marks(pfig, marks, bands, x_range=(200, 1000), axes_width_points=180, font_size=6)
        self.assertEqual(len(pfig.layout.shapes), 3)
        self.assertEqual(sorted(a.text for a in pfig.layout.annotations), ["796 SiC (TO)", "924", "window"])


class InsetTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(200, 1000, 801)
        self.curves = [PreparedCurve("a", x, np.sin(x / 50), "#0072B2", 0.9), PreparedCurve("b", x, np.cos(x / 50) + 2, "#D55E00", 0.9)]

    def test_matplotlib_inset(self):
        fig, ax = plt.subplots()
        for c in self.curves:
            ax.plot(c.x, c.y)
        ax.set_xlim(200, 1000)
        inset = add_mpl_zoom_inset(ax, self.curves, (900, 980), location="upper left", font_size=6)
        self.assertIsNotNone(inset)
        self.assertEqual(len(inset.lines), 2)
        self.assertEqual(inset.get_xlim(), (900.0, 980.0))
        self.assertIn("cm$^{\\mathrm{{-}1}}$", inset.texts[0].get_text())
        self.assertIsNone(add_mpl_zoom_inset(ax, self.curves, (5000, 6000)))
        with self.assertRaises(ValueError):
            add_mpl_zoom_inset(ax, self.curves, (980, 900))
        plt.close(fig)

    def test_plotly_inset(self):
        pfig = go.Figure()
        for c in self.curves:
            pfig.add_trace(go.Scatter(x=c.x, y=c.y, mode="lines"))
        pfig.update_xaxes(range=[200, 1000])
        refs = add_plotly_zoom_inset(pfig, self.curves, (900, 980), location="lower right", font_size=6)
        self.assertEqual(refs, ("x2", "y2"))
        self.assertEqual(len([t for t in pfig.data if t.xaxis == "x2"]), 3)
        self.assertIn("cm<sup>−1</sup>", pfig.layout.annotations[0].text)


if __name__ == "__main__":
    unittest.main()
