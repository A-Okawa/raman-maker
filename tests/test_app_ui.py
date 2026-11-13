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


def _remove_buttons(at):
    return [b for b in at.button if (b.key or "").startswith("_rm_")]


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


class PasteTests(unittest.TestCase):
    """Raw text pasted into the sidebar's paste tab (Excel columns, instrument text) becomes a spectrum."""
    TWO_COLUMNS = "\n".join(f"{x}\t{1000 + 500 * np.exp(-((x - 796) / 6) ** 2):.1f}" for x in range(200, 1001, 2))

    def test_pasted_text_becomes_a_spectrum(self):
        at = harness.run_app({})
        at.text_area(key="_paste_text").input("Raman shift\tIntensity\n" + self.TWO_COLUMNS).run()
        at.button(key="_paste_add").click().run()
        r = harness.report(at)
        self.assertEqual(r["names"], ["貼り付け 1.txt"])
        self.assertEqual(r["labels"], ["貼り付け 1"])
        self.assertEqual(r["curves"][0][5], 401)
        self.assertEqual(at.session_state["_paste_text"], "", "the box is cleared once the data is in")
        self.assertTrue(any("追加しました" in s.value for s in at.success))
        # a second paste with a name of its own (a dot in it stays); the project keeps both
        at.text_area(key="_paste_text").input(self.TWO_COLUMNS).run()
        at.text_input(key="_paste_name").input("Yb2Si2O7 1.5 h").run()
        at.button(key="_paste_add").click().run()
        r = harness.report(at)
        self.assertEqual(r["names"], ["貼り付け 1.txt", "Yb2Si2O7 1.5 h.txt"])
        self.assertEqual(r["labels"], ["貼り付け 1", "Yb2Si2O7 1.5 h"])
        self.assertEqual([c[2] for c in r["curves"]], [0.0, 1.0])
        project = decode_project(r["project"])
        self.assertEqual(set(project["spectrum_files"]), {"貼り付け 1.txt", "Yb2Si2O7 1.5 h.txt"})
        again = harness.run_app(project["spectrum_files"], project["settings"])
        self.assertEqual(harness.report(again)["curves"], r["curves"])

    def test_unreadable_paste_is_reported_and_nothing_added(self):
        at = harness.run_app({})
        at.button(key="_paste_add").click().run()
        self.assertTrue(any("空です" in e.value for e in at.error))
        at.text_area(key="_paste_text").input("no numbers\nhere either").run()
        at.button(key="_paste_add").click().run()
        self.assertTrue(any("読めませんでした" in e.value for e in at.error))
        self.assertEqual(at.session_state["_spectrum_bytes"], {})
        self.assertEqual(at.session_state["_paste_text"], "no numbers\nhere either", "the text stays to be corrected")
        self.assertTrue(any("FT-IR スペクトル" in i.value for i in at.info))

    def test_pasted_ftir_text_switches_the_mode(self):
        at = harness.run_app({})
        at.text_area(key="_paste_text").input(harness.data_files("ftir_abs.txt")["ftir_abs.txt"].decode()).run()
        at.text_input(key="_paste_name").input("No_CO2.jdx").run()
        at.button(key="_paste_add").click().run()
        r = harness.report(at)
        self.assertEqual(r["names"], ["No_CO2.jdx"], "a typed text extension is kept")
        self.assertEqual(at.session_state["spec_mode"], "ftir")
        self.assertEqual(r["xlim"], (4000.0, 400.0))

    def test_restored_spectra_are_listed_and_removable(self):
        at = harness.run_app(harness.data_files(*THREE), {"_val_lbl_2": "Ren<i>i</i>shaw"})
        self.assertEqual(len(_remove_buttons(at)), 3, "one ✕ per spectrum outside the uploader")
        at.button(key="_rm_jasco_5SiC_950C.txt").click().run()
        r = harness.report(at)
        self.assertEqual(r["names"], ["jasco_5SiC_asSintered.csv", "renishaw_single.txt"])
        self.assertEqual(r["labels"], ["jasco_5SiC_asSintered", "Ren<i>i</i>shaw"], "settings follow the spectra by name")
        self.assertEqual(len(_remove_buttons(at)), 2)

    def test_labels_follow_their_spectra_through_additions_and_removals(self):
        # a first spectrum, a pasted second one (the first's label box is re-keyed), then the first is removed:
        # the survivor moves to index 0 and must not inherit the label box the first spectrum had there
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"))
        at.text_input(key="_inp_lbl_0_v0").input("SiC 950 °C").run()
        at.text_area(key="_paste_text").input(self.TWO_COLUMNS).run()
        at.button(key="_paste_add").click().run()
        self.assertEqual(harness.report(at)["labels"], ["SiC 950 °C", "貼り付け 1"])
        at.button(key="_rm_jasco_5SiC_950C.txt").click().run()
        r = harness.report(at)
        self.assertEqual(r["names"], ["貼り付け 1.txt"])
        self.assertEqual(r["labels"], ["貼り付け 1"], "the pasted spectrum keeps its own label at its new index")
        self.assertEqual(r["mpl_legend"], ["貼り付け 1"])

    def test_uploaded_files_join_restored_and_pasted_spectra(self):
        merge = harness.app_function("_merge_uploads")

        class File:
            def __init__(self, name, data):
                self.name, self._data = name, data

            def getvalue(self):
                return self._data

        current = {"restored.txt": b"r", "貼り付け 1.txt": b"p"}
        merged = merge(current, set(), [File("a.txt", b"a"), File("b.txt", b"b")])
        self.assertEqual(list(merged), ["restored.txt", "貼り付け 1.txt", "a.txt", "b.txt"], "uploads append, nothing is lost")
        merged = merge(merged, {"a.txt", "b.txt"}, [File("b.txt", b"b2")])
        self.assertEqual(merged, {"restored.txt": b"r", "貼り付け 1.txt": b"p", "b.txt": b"b2"},
                         "a file taken out of the uploader leaves; a re-read file keeps its place with the new contents")
        self.assertEqual(merge(merged, {"b.txt"}, []), {"restored.txt": b"r", "貼り付け 1.txt": b"p"}, "an emptied uploader")


class FitTests(unittest.TestCase):
    """The fit of overlapping peaks and the overlays that keep the processing checkable."""
    FIT = {"xrange": (700.0, 1000.0), "mfit_on": True, "mfit_range": (880.0, 1000.0), "mfit_centers": "925, 955",
           "mfit_max_shift": 15.0}

    def test_two_components_are_fitted_drawn_and_kept_in_the_project(self):
        plain = harness.report(harness.run_app(harness.data_files("jasco_5SiC_950C.txt"), {"xrange": (700.0, 1000.0)}))
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"), dict(self.FIT))
        r = harness.report(at)
        fit = r["fits"][0]
        self.assertTrue(fit["ok"], fit["message"])
        self.assertEqual(len(fit["components"]), 2)
        self.assertGreater(fit["r2"], 0.9)
        for (center, fwhm, height, area, share, center_err, at_limit), start in zip(fit["components"], (925, 955)):
            self.assertLess(abs(center - start), 15)
            self.assertTrue(fwhm > 0 and height > 0 and area > 0 and 0 < share < 1 and center_err > 0)
        # the figure: the spectrum, two dashed components in its colour, the dark sum; the legend is unchanged
        self.assertEqual(len(r["mpl_lines"]), len(plain["mpl_lines"]) + 3)
        dashed = [s for s in r["mpl_styles"] if s[0] == "--"]
        self.assertEqual([s[1] for s in dashed], [r["colors"][0]] * 2)
        self.assertIn("#222222", [s[1] for s in r["mpl_styles"]])
        self.assertEqual(r["mpl_legend"], plain["mpl_legend"])
        self.assertEqual(len(r["preview_all"]), len(plain["preview_all"]) + 3)
        self.assertEqual(sum(1 for t in r["preview_all"] if t[0] == "dash"), 2)
        table = at.dataframe[0].value
        self.assertEqual(len(table), 2)
        self.assertIn("面積比（対 成分1）", table.columns)
        self.assertAlmostEqual(table["面積の割合 (%)"].sum(), 100.0, delta=0.2)
        # switches: no components, shaded components
        r = harness.report(harness.rerun(at, {"mfit_show_components": False}))
        self.assertEqual(len(r["mpl_lines"]), len(plain["mpl_lines"]) + 1)
        r = harness.report(harness.rerun(at, {"mfit_show_components": True, "mfit_fill": True}))
        self.assertEqual(r["mpl_fills"], plain["mpl_fills"] + 2)
        self.assertEqual(sum(1 for t in r["preview_all"] if t[3] == "tonexty"), 2)
        # the project restores the fit
        project = decode_project(r["project"])
        self.assertEqual(project["settings"]["mfit_centers"], "925, 955")
        again = harness.report(harness.run_app(project["spectrum_files"], project["settings"]))
        self.assertEqual(again["fits"], r["fits"])

    def test_missing_or_outside_centres_and_failures_are_explained(self):
        at = harness.run_app(harness.data_files("jasco_5SiC_950C.txt"), {**self.FIT, "mfit_centers": ""})
        r = harness.report(at)
        self.assertEqual(r["fits"], [None])
        self.assertTrue(any("初期位置" in i.value for i in at.info))
        r = harness.report(harness.rerun(at, {"mfit_centers": "300, 925"}))       # 300 lies outside the fit range
        self.assertEqual(len(r["fits"][0]["components"]), 1)
        r = harness.report(harness.rerun(at, {"mfit_centers": "925", "mfit_range": (924.0, 926.0)}))
        self.assertFalse(r["fits"][0]["ok"])
        self.assertTrue(any("フィットできませんでした" in w.value for w in at.warning))

    def test_measured_data_and_baseline_overlays(self):
        files = harness.data_files("jasco_5SiC_950C.txt")
        plain = harness.report(harness.run_app(files, {"baseline_method": "als"}))
        r = harness.report(harness.run_app(files, {"baseline_method": "als", "show_raw": True, "show_baseline": True}))
        self.assertEqual(len(r["mpl_lines"]), len(plain["mpl_lines"]) + 2)
        faint, dotted = r["mpl_styles"][-2:]
        self.assertEqual((faint[0], faint[2]), ("-", 0.4))
        self.assertEqual(dotted[0], ":")
        self.assertEqual(r["curves"], plain["curves"], "the spectrum itself is not touched")
        self.assertEqual(r["mpl_legend"], plain["mpl_legend"])
        self.assertEqual(sum(1 for t in r["preview_all"] if t[0] == "dot"), 1)
        # without a baseline only the measured data are overlaid
        r = harness.report(harness.run_app(files, {"show_raw": True, "show_baseline": True}))
        self.assertEqual(len(r["mpl_lines"]), len(plain["mpl_lines"]) + 1)

    def test_ftir_fits_in_absorbance_and_draws_nothing_on_transmittance(self):
        files = harness.data_files("ftir_abs.txt")
        state = {"mfit_on": True, "mfit_range": (800.0, 1300.0), "mfit_centers": "1080", "mfit_model": "gaussian",
                 "mfit_max_shift": 40.0, "show_raw": True}
        absorbance = harness.report(harness.run_app(files, dict(state)))
        self.assertTrue(absorbance["fits"][0]["ok"])
        self.assertGreater(len(absorbance["mpl_lines"]), 1)
        transmittance = harness.report(harness.run_app(files, {**state, "ir_display": "transmittance"}))
        self.assertEqual(len(transmittance["mpl_lines"]), 1)
        self.assertEqual(transmittance["fits"][0]["components"], absorbance["fits"][0]["components"],
                         "the fit is of the absorbance whichever display is chosen")


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
