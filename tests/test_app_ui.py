"""The real UI, headless: uploads, stacking, processing, marks, projects, languages."""
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import harness
from project_io import decode_project
from raman_io import wdf_available
from style_templates import decode_style_template

THREE = ("jasco_5SiC_asSintered.csv", "jasco_5SiC_950C.txt", "renishaw_single.txt")


def downloads(at):
    """Download buttons in the sidebar (not exposed by Streamlit 1.32's AppTest)."""
    try:
        return list(at.sidebar.download_button)
    except AttributeError:
        return None


class UploadTests(unittest.TestCase):
    def test_three_files_stack_in_the_graph_app_look(self):
        at = harness.run_app(harness.data_files(*THREE))
        r = harness.report(at)
        self.assertEqual(r["names"], list(THREE), "files keep the order they were given in")
        self.assertEqual(r["labels"], ["jasco_5SiC_asSintered", "jasco_5SiC_950C", "renishaw_single"])
        levels = [c[2] for c in r["curves"]]
        self.assertEqual(levels, [0.0, 1.0, 2.0], "normalised spectra stack one height apart")
        self.assertEqual([c[1] for c in r["curves"]], ["#0072B2", "#D55E00", "#009E73"], "the paper palette in display order")
        self.assertEqual(r["mpl_legend"], ["renishaw_single", "jasco_5SiC_950C", "jasco_5SiC_asSintered"], "legend from top to bottom")
        self.assertEqual([t[3] for t in r["preview"]], [3, 2, 1], "the preview legend uses the same order")
        self.assertEqual(r["xlabel"], "Raman shift (cm$^{\\mathrm{{-}1}}$)")
        self.assertEqual(r["ylabel"], "Intensity (a.u.)")
        self.assertEqual(r["yticks"], [])
        self.assertEqual(at.session_state["xrange"], (200.0, 1000.0))
        self.assertEqual(at.session_state["major_tick"], 100.0)
        self.assertEqual(at.session_state["figure_preset"], "figapp")
        self.assertEqual(at.session_state["tick_direction"], "out")
        self.assertAlmostEqual(at.session_state["fig_width"], 88 / 25.4)
        self.assertEqual(r["png"][:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(r["pdf_header"], b"%PDF-")
        self.assertIn(b"<svg", r["svg"][:200])
        self.assertIn(r["tiff_header"][:2], (b"II", b"MM"))
        buttons = downloads(at)
        if buttons is not None:
            labels = " ".join(b.label for b in buttons)
            for word in ("PDF", "SVG", "TIFF", "PNG", "作業を保存", "スタイルを保存", "PC用"):
                self.assertIn(word, labels)

    def test_multi_spectrum_files(self):
        at = harness.run_app(harness.data_files("multi_columns.csv", "renishaw_map.txt", "book.xlsx"))
        r = harness.report(at)
        self.assertEqual(len(r["names"]), 10)
        self.assertEqual(r["labels"][:3], ["as-sintered", "750 C", "950 C"])
        self.assertIn("shared_x · A", r["labels"])
        self.assertIn("renishaw_map (X=1.3, Y=0)", r["labels"])
        self.assertEqual(len(r["curves"]), 10)

    def test_no_data_shows_the_prompt(self):
        at = harness.run_app({})
        self.assertFalse(at.exception)
        self.assertTrue(any("FT-IR スペクトル" in i.value for i in at.info))

    def test_unreadable_file_is_reported_and_the_rest_drawn(self):
        files = harness.data_files("jasco_5SiC_950C.txt")
        files["bad.txt"] = b"no numbers here\n"
        at = harness.run_app(files)
        self.assertTrue(any("bad.txt" in e.value for e in at.error))
        self.assertEqual(len(harness.report(at)["curves"]), 1)

    @unittest.skipUnless(wdf_available(), "renishawWiRE is not installed")
    def test_wdf_series_mean_and_all(self):
        at = harness.run_app(harness.data_files("depth.wdf"))
        r = harness.report(at)
        self.assertEqual(r["labels"], ["depth (mean of 40)"])
        self.assertEqual(r["mpl_legend"], ["depth (mean of 40)"])
        at = harness.rerun(at, {"multi_mode": "all"})
        r = harness.report(at)
        self.assertEqual(len(r["curves"]), 40)


class ArrangementTests(unittest.TestCase):
    def test_order_visibility_offsets_and_absolute_mode(self):
        at = harness.run_app(harness.data_files(*THREE), {"ord_0": 3, "ord_1": 1, "ord_2": 2, "vis_1": False,
                                                           "global_offset": 1.5, "extra_offset_0": 0.5})
        r = harness.report(at)
        # display order 1 → file 2 (hidden), 2 → file 3, 3 → file 1
        self.assertEqual([c[0] for c in r["curves"]], ["renishaw_single", "jasco_5SiC_asSintered"])
        self.assertEqual([c[2] for c in r["curves"]], [0.0, 2.0], "the second visible spectrum sits at 1.5 + its own 0.5")
        self.assertEqual(r["mpl_legend"], ["jasco_5SiC_asSintered", "renishaw_single"])
        at = harness.rerun(at, {"stack": False, "normalize_mode": "none", "yoff_0": 1000.0, "scale_0": 2.0, "vis_1": True})
        r = harness.report(at)
        by_name = {c[0]: c for c in r["curves"]}
        self.assertEqual(by_name["jasco_5SiC_asSintered"][2], 1000.0)
        self.assertGreater(by_name["jasco_5SiC_asSintered"][4], 7000, "raw counts scaled by 2 and shifted by 1000")
        self.assertEqual(by_name["jasco_5SiC_950C"][2], 0.0)

    def test_window_normalisation_and_hidden_panel(self):
        at = harness.run_app(harness.data_files(*THREE), {"normalize_mode": "window", "norm_window": (900.0, 980.0),
                                                           "show_panel": False, "global_offset": 0.0})
        r = harness.report(at)
        self.assertEqual(len(r["curves"]), 3)
        self.assertTrue(all(c[2] == 0.0 for c in r["curves"]), "offset 0 overlays the spectra")
        self.assertTrue(all(c[4] > 1.2 for c in r["curves"]), "the 796 peak is higher than the 924 peak that defines 1")


class ProcessingTests(unittest.TestCase):
    def test_peaks_marks_bands_inset_and_processing(self):
        at = harness.run_app(harness.data_files(*THREE), {
            "show_peaks": True, "peak_fit": True, "peak_decimals": 1, "spike_remove": True, "smooth_on": True,
            "baseline_method": "als", "show_zoom": True, "zoom_range": (900.0, 980.0), "show_side_labels": True,
            "marks_text": "796 SiC (TO)\n924 Yb_{2}Si_{2}O_{7}\n770-822 window", "show_yticks": True,
            "palette": "blues", "palette_from": 2})
        r = harness.report(at)
        self.assertEqual(r["mpl_axes"], 2, "main axes plus the inset")
        self.assertEqual(len(r["peaks"]), 9)
        self.assertEqual(r["peaks"][0]["位置 (cm⁻¹)"], 796.0)
        self.assertAlmostEqual(r["peaks"][0]["フィット中心 (cm⁻¹)"], 796.0, delta=0.2)
        self.assertGreater(len(r["yticks"]), 2)
        self.assertIn("796 SiC (TO)", r["preview_annotations"])
        self.assertIn("924 Yb<sub>2</sub>Si<sub>2</sub>O<sub>7</sub>", r["preview_annotations"])
        self.assertIn("window", r["preview_annotations"])
        self.assertEqual(r["preview_shapes"], 4, "two guide lines, one band and the inset indicator")
        self.assertEqual(r["curves"][0][1], "#000000", "the first spectrum is outside the palette range")
        self.assertEqual([c[1] for c in r["curves"][1:]], ["#519CCC", "#083C7D"], "two blue shades over the rest")
        # labels of the spectra and of the marks are drawn in the figure
        self.assertTrue(any("SiC (TO)" in t for t in r["mpl_texts"]))
        self.assertTrue(any("renishaw_single" in t for t in r["mpl_texts"]))

    def test_processing_failure_keeps_the_raw_spectrum(self):
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"), {"baseline_method": "poly", "baseline_order": 12,
                                                                          "xrange": (990.0, 1000.0)})
        self.assertFalse(at.exception)
        self.assertEqual(len(harness.report(at)["curves"]), 1)


class ProjectTests(unittest.TestCase):
    def test_project_json_restores_the_same_figure(self):
        at = harness.run_app(harness.data_files(*THREE), {"palette": "greens", "ord_0": 3, "_val_lbl_2": "Ren<i>i</i>shaw",
                                                           "marks_text": "796 SiC", "baseline_method": "linear", "show_peaks": True})
        first = harness.report(at)
        project = decode_project(first["project"])
        self.assertEqual(set(project["spectrum_files"]), set(THREE))
        self.assertEqual(project["settings"]["palette"], "greens")
        self.assertEqual(project["settings"]["_val_lbl_2"], "Ren<i>i</i>shaw")
        again = harness.run_app(project["spectrum_files"], project["settings"])
        second = harness.report(again)
        self.assertEqual(first["curves"], second["curves"])
        self.assertEqual(first["labels"], second["labels"])
        self.assertEqual(first["mpl_legend"], second["mpl_legend"])
        self.assertEqual(first["peaks"], second["peaks"])

    def test_style_template_download(self):
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"))
        at = harness.rerun(at, {"legend_loc": "lower left", "marks_fontsize": 8})
        settings = decode_style_template(harness.report(at)["style"])
        self.assertEqual(settings["legend_loc"], "lower left")
        self.assertEqual(settings["marks_fontsize"], 8)
        self.assertEqual(settings["font_size"], 7.0)

    def test_new_project_clears_the_spectra(self):
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"))
        new = next(b for b in at.sidebar.button if "新しい作業" in b.label)
        new.click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["_spectrum_bytes"], {})
        self.assertTrue(any("FT-IR スペクトル" in i.value for i in at.info))


class FtirTests(unittest.TestCase):
    def test_ftir_file_switches_the_mode_and_reverses_the_axis(self):
        at = harness.run_app(harness.data_files("ftir_abs.txt"))
        r = harness.report(at)
        self.assertEqual(at.session_state["spec_mode"], "ftir")
        self.assertTrue(at.session_state["x_reverse"])
        self.assertEqual(r["xlabel"], "Wavenumber (cm$^{\\mathrm{{-}1}}$)")
        self.assertEqual(r["ylabel"], "Absorbance (a.u.)")
        self.assertEqual(r["xlim"], (4000.0, 400.0), "high wavenumber on the left")
        self.assertEqual(r["preview_xrange"], [4000.0, 400.0])
        self.assertEqual(at.session_state["major_tick"], 500.0)
        self.assertAlmostEqual(r["curves"][0][4], 1.0, msg="absorbance normalised to its maximum")
        self.assertEqual(r["mpl_legend"], ["ftir_abs"])

    def test_raman_files_keep_the_raman_mode(self):
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"))
        self.assertEqual(at.session_state["spec_mode"], "raman")
        self.assertFalse(at.session_state["x_reverse"])
        self.assertEqual(harness.report(at)["xlim"], (200.0, 1000.0))

    def test_transmittance_display_shows_valleys_with_labels_below(self):
        at = harness.run_app(harness.data_files("ftir_abs.txt", "ftir_pct.txt"), {"ir_display": "transmittance", "show_peaks": True,
                                                                                  "normalize_mode": "none", "global_offset": 0.0})
        r = harness.report(at)
        self.assertEqual(r["ylabel"], "Transmittance (%)")
        for curve in r["curves"]:
            self.assertAlmostEqual(curve[4], 95.0, delta=1.5, msg="both files plot the same transmittance (%)")
            self.assertAlmostEqual(curve[3], 35.0, delta=1.5)
        positions = sorted({round(p["位置 (cm⁻¹)"]) for p in r["peaks"]})
        self.assertTrue(any(abs(p - 1080) <= 4 for p in positions) and any(abs(p - 3400) <= 20 for p in positions), positions)
        self.assertEqual(set(r["peaks"][0]) >= {"透過率 (%)"}, True)
        self.assertTrue(all(a[3] == "top" for a in r["peak_annotations"]), "peak values hang below the valleys")
        self.assertLess(r["ylim"][0], 30.0, "room for the labels under the lowest valley")
        # absorbance display of the same files: peaks point up, labels above
        at = harness.rerun(at, {"ir_display": "absorbance"})
        r = harness.report(at)
        self.assertTrue(all(a[3] == "bottom" for a in r["peak_annotations"]))
        self.assertAlmostEqual(r["curves"][0][4], -np.log10(0.35), delta=0.02)

    def test_y_units_override_for_a_plain_file(self):
        # a two-column file without a header is read as absorbance; the panel can declare it transmittance (0–1)
        files = harness.data_files("ftir_pct.txt")
        text = "\n".join(f"{x:.1f}\t{t / 100:.5f}" for x, t in zip(*_columns(files["ftir_pct.txt"])))
        at = harness.run_app({"plain.txt": text.encode()}, {"spec_mode": "ftir", "_mode_auto": False, "normalize_mode": "none",
                                                            "ir_display": "transmittance"})
        r = harness.report(at)
        self.assertAlmostEqual(r["curves"][0][3], 100 * 10 ** -0.95, delta=1.0, msg="a fraction read as absorbance (0.35–0.95)")
        self.assertAlmostEqual(r["curves"][0][4], 100 * 10 ** -0.35, delta=1.0)
        at = harness.rerun(at, {"yunit_0": "transmittance_fraction"})
        r = harness.report(at)
        self.assertAlmostEqual(r["curves"][0][3], 35.0, delta=1.5, msg="declared as transmittance (0–1), the bands reappear")
        self.assertAlmostEqual(r["curves"][0][4], 95.0, delta=1.5)

    def test_manual_mode_switch_and_project_round_trip(self):
        at = harness.run_app(harness.data_files("ftir_abs.txt"))
        at.sidebar.selectbox(key="spec_mode").select("raman").run()
        self.assertEqual(at.session_state["spec_mode"], "raman")
        self.assertFalse(at.session_state["_mode_auto"])
        self.assertFalse(at.session_state["x_reverse"])
        self.assertEqual(at.session_state["x_label"], "Raman shift (cm^{-1})")
        at.sidebar.selectbox(key="spec_mode").select("ftir").run()
        self.assertTrue(at.session_state["x_reverse"])
        self.assertEqual(at.session_state["y_label"], "Absorbance (a.u.)")
        at = harness.rerun(at, {"ir_display": "transmittance", "marks_text": "1080 Si–O–Si", "show_zoom": True, "zoom_range": (800.0, 1300.0)})
        first = harness.report(at)
        self.assertEqual(first["ylabel"], "Transmittance (%)")
        self.assertEqual(first["mpl_axes"], 2)
        project = decode_project(first["project"])
        self.assertEqual(project["settings"]["spec_mode"], "ftir")
        self.assertEqual(project["settings"]["ir_display"], "transmittance")
        again = harness.run_app(project["spectrum_files"], project["settings"])
        second = harness.report(again)
        self.assertEqual(first["curves"], second["curves"])
        self.assertEqual(first["xlim"], second["xlim"])
        self.assertFalse(again.session_state["_mode_auto"])


def _columns(data):
    rows = [line.split() for line in data.decode().splitlines() if line and not line.startswith("##")]
    x = [float(r[0]) for r in rows]
    y = [float(r[1]) for r in rows]
    return x, y


class LanguageTests(unittest.TestCase):
    def test_english_labels(self):
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"), {"lang_radio": "English", "show_peaks": True})
        r = harness.report(at)
        self.assertEqual(r["peaks"][0].keys() & {"Spectrum", "Position (cm⁻¹)", "Intensity"}, {"Spectrum", "Position (cm⁻¹)", "Intensity"})
        self.assertTrue(any("Spectra" in h.value for h in at.sidebar.header))


if __name__ == "__main__":
    unittest.main()
