"""Quality checks (qc.py) on synthetic data with known answers.

All data here are generated in the tests (SYNTHETIC); no measured research data are used.
Each case states what is wrong with the data and the status the checks must give.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qc  # noqa: E402
import qc_text  # noqa: E402
import raman_processing  # noqa: E402
from raman_io import read_spectra  # noqa: E402
from raman_processing import (detect_peaks, fit_multipeak, fit_peak, peak_prominences_of, peak_shape,  # noqa: E402
                              process, to_absorbance)

RNG = np.random.default_rng(20261127)
X = np.arange(400.0, 650.0, 1.0)


def si_like(noise=3.0, step=1.0, fwhm=4.0, height=1000.0):
    """SYNTHETIC: one narrow Lorentzian (like Si at 520 cm⁻¹) on a flat background with Gaussian noise."""
    x = np.arange(400.0, 650.0, step)
    return x, peak_shape(x, "lorentzian", height, 520.3, fwhm) + 50 + RNG.normal(0, noise, len(x))


def text(x, y):
    return "\n".join(f"{a:.3f}\t{b:.4f}" for a, b in zip(x, y))


def statuses(items, code, subject=None):
    return [i["status"] for i in items if i["code"] == code and (subject is None or i["subject"] == subject)]


class OverallTests(unittest.TestCase):
    def test_the_worst_status_wins_and_is_never_offset(self):
        items = [qc.item("input", "a", "points", qc.PASS, 10, required=True) for _ in range(20)]
        self.assertEqual(qc.overall(items), qc.PASS)
        self.assertEqual(qc.overall(items + [qc.item("fit", "a", "chi2", qc.CAUTION, 4)]), qc.CAUTION)
        self.assertEqual(qc.overall(items + [qc.item("fit", "a", "chi2", qc.WARNING, 40),
                                             qc.item("fit", "a", "autocorr", qc.CAUTION, 0.6)]), qc.WARNING)
        self.assertEqual(qc.overall(items + [qc.item("fit", "a", "fit_status", qc.FAIL, None)]), qc.FAIL)

    def test_a_missing_required_check_is_not_a_pass(self):
        items = [qc.item("input", "a", "points", qc.PASS, 10, required=True),
                 qc.item("fit", "a", "chi2", qc.NOT_EVALUATED, None, required=True, reason="smoothed")]
        self.assertEqual(qc.overall(items), qc.NOT_EVALUATED)
        # reference values (optional) do not stop a PASS
        self.assertEqual(qc.overall(items[:1] + [qc.item("fit", "a", "r2", qc.NOT_EVALUATED, 0.99, reason="reference")]), qc.PASS)
        # a real problem is still shown when something else is unevaluated
        self.assertEqual(qc.overall(items + [qc.item("input", "a", "gaps", qc.WARNING, 1)]), qc.WARNING)
        self.assertEqual(qc.overall([]), qc.NOT_EVALUATED)


class InputTests(unittest.TestCase):
    def read(self, body, name="synthetic.txt"):
        spectrum = read_spectra(body.encode(), name)[0]
        return spectrum, qc.input_checks("s", spectrum.x, spectrum.y, spectrum.metadata)

    def test_clean_data_pass(self):
        spectrum, items = self.read(text(*si_like()))
        self.assertEqual(qc.overall(items), qc.PASS, [(i["code"], i["status"]) for i in items if i["status"] != qc.PASS])
        self.assertEqual(len(spectrum.x), 250)

    def test_missing_values_are_reported_with_their_line(self):
        rows = text(*si_like()).splitlines()
        rows[30] = "430.000\tnan"          # line 31
        rows[60] = "460.000"               # line 61: no intensity
        spectrum, items = self.read("\n".join(rows))
        self.assertEqual(len(spectrum.x), 248, "the reader goes on after the bad lines (it used to stop at line 31)")
        skipped = next(i for i in items if i["code"] == "skipped_lines")
        self.assertEqual((skipped["status"], skipped["value"], skipped["params"]["lines"]), (qc.WARNING, 2, "31, 61"))
        self.assertIn("31", qc_text.render(skipped, "ja")["reason"])

    def test_non_finite_values_are_reported(self):
        rows = text(*si_like()).splitlines()
        rows[10] = "410.000\t1e999"
        spectrum, items = self.read("\n".join(rows))
        self.assertEqual(statuses(items, "nonfinite"), [qc.WARNING])
        self.assertIn("x=410.0", next(i for i in items if i["code"] == "nonfinite")["params"]["where"])

    def test_a_numeric_table_after_the_data_is_not_mixed_in(self):
        body = text(*si_like()) + "\nSecond table\n1\t5\n2\t6\n3\t7"
        spectrum, items = self.read(body)
        self.assertEqual(len(spectrum.x), 250)
        self.assertEqual(statuses(items, "footer"), [qc.CAUTION])

    def test_declared_point_count(self):
        x, y = si_like()
        jasco = "TITLE\tsynthetic\nXUNITS\t[cm-1]\nNPOINTS\t300\nXYDATA\n" + text(x, y) + "\n##### Extended Information\n"
        _, items = self.read(jasco)
        self.assertEqual(statuses(items, "declared_points"), [qc.WARNING])
        _, items = self.read(jasco.replace("NPOINTS\t300", "NPOINTS\t250"))
        self.assertEqual(statuses(items, "declared_points"), [qc.PASS])

    def test_duplicates_gaps_and_uneven_spacing(self):
        x, y = si_like()
        x2, y2 = np.insert(x, 50, x[50]), np.insert(y, 50, y[50])
        items = qc.input_checks("s", x2, y2, {})
        self.assertEqual(statuses(items, "duplicates"), [qc.CAUTION])
        keep = np.r_[0:100, 120:len(x)]                       # 20 points missing: a gap
        items = qc.input_checks("s", x[keep], y[keep], {})
        self.assertEqual(statuses(items, "gaps"), [qc.WARNING])
        self.assertEqual(statuses(items, "spacing"), [qc.PASS], "the gap is not reported twice")
        keep = np.r_[0:100, 101:len(x)]                       # one point missing: the spacing doubles once
        items = qc.input_checks("s", x[keep], y[keep], {})
        self.assertEqual(statuses(items, "spacing"), [qc.CAUTION])
        self.assertEqual(next(i for i in items if i["code"] == "spacing")["params"]["where"], "501.0")
        # a smoothly changing spacing (spectrometer dispersion) is not flagged
        xs = 400 + np.cumsum(np.linspace(0.8, 1.6, 300))
        items = qc.input_checks("s", xs, RNG.normal(100, 1, 300) + peak_shape(xs, "gaussian", 50, 520, 10), {})
        self.assertEqual(statuses(items, "spacing") + statuses(items, "gaps"), [qc.PASS, qc.PASS])

    def test_noise_only_and_too_few_points(self):
        items = qc.input_checks("s", X, RNG.normal(100, 5, len(X)), {})
        self.assertEqual(statuses(items, "snr"), [qc.PASS], "noise has outliers above 3σ: S/N of noise alone is about 3–4")
        flat = qc.input_checks("s", X, np.full(len(X), 100.0), {})
        self.assertEqual(statuses(flat, "snr"), [qc.WARNING], "a flat line has no signal")
        self.assertEqual(statuses(flat, "saturation"), [qc.WARNING])
        tiny = qc.input_checks("s", X[:3], np.array([1.0, 2.0, 1.0]), {})
        self.assertEqual(statuses(tiny, "points"), [qc.FAIL])
        self.assertEqual(statuses(tiny, "snr"), [qc.NOT_EVALUATED])
        self.assertEqual(qc.overall(tiny), qc.FAIL)

    def test_physically_doubtful_values(self):
        x, y = si_like()
        top = y.copy()
        top[118:123] = 4000.0                                  # a flat top: detector saturation
        self.assertEqual(statuses(qc.input_checks("s", x, top, {}), "saturation"), [qc.WARNING])
        negative = y - 200                                     # a background subtracted by the instrument
        self.assertEqual(statuses(qc.input_checks("s", x, negative, {}), "negative"), [qc.CAUTION])
        # FT-IR transmittance: above 100 %, at or below 0, very high absorbance
        t = 60 - 59.995 * np.exp(-((X - 520) / 8) ** 2) + RNG.normal(0, 0.02, len(X))
        t[:5] = 101.0
        items = qc.input_checks("s", X, t, {}, kind="ftir", units="transmittance_percent",
                                absorbance=to_absorbance(t, "transmittance_percent"))
        self.assertEqual(statuses(items, "transmittance_over"), [qc.CAUTION])
        self.assertEqual(statuses(items, "absorbance_high"), [qc.CAUTION])

    def test_extreme_values_do_not_break_the_checks(self):
        for y in (np.full(len(X), 1e300) * np.linspace(1, 1.7, len(X)), np.zeros(len(X)), -np.ones(len(X))):
            with np.errstate(all="ignore"):
                items = qc.input_checks("s", X, y, {})
            self.assertTrue(all(i["status"] in qc.LEVELS for i in items))


class ProcessingTests(unittest.TestCase):
    def test_spike_removal_on_a_real_narrow_peak_is_flagged(self):
        for step, expected in ((1.0, qc.WARNING), (2.0, qc.WARNING)):
            x, y = si_like(step=step)
            _, _, info = process(x, y, spikes=6.0)
            info["spikes_on"] = True
            self.assertGreater(info["spikes"], 0, "the narrow peak is taken for a spike")
            items = qc.processing_checks("s", info, points_used=len(x))
            self.assertEqual(statuses(items, "spikes"), [expected])
            self.assertTrue(any(abs(p["x"] - 520) < 2 for p in info["spike_points"]))

    def test_a_cosmic_ray_spike_is_told_from_a_peak_top(self):
        # Even a 12 cm⁻¹ wide band (height 1000, σ = 3) loses its top point at the default threshold;
        # the one-point spike at 460 and the band top at 520 must be told apart.
        x, y = si_like(fwhm=12.0)
        y[60] += 3000.0
        _, out, info = process(x, y, spikes=6.0)
        info["spikes_on"] = True
        ratios = {round(p["x"]): p["neighbour_ratio"] for p in info["spike_points"]}
        self.assertLess(ratios[460], 0.05)
        self.assertGreater(ratios[520], 0.5)
        item = next(i for i in qc.processing_checks("s", info, points_used=len(x)) if i["code"] == "spikes")
        self.assertEqual((item["status"], item["params"]["suspect"]), (qc.WARNING, "520.0"))
        self.assertLess(out[60], 200)
        # with the spike alone (no narrow band) the check passes
        flat = 50 + RNG.normal(0, 3, len(X))
        flat[60] += 3000.0
        _, _, info = process(X, flat, spikes=6.0)
        info["spikes_on"] = True
        self.assertEqual(statuses(qc.processing_checks("s", info, points_used=len(X)), "spikes"), [qc.PASS])

    def test_an_empty_normalisation_window_is_reported_and_not_replaced(self):
        x, y = si_like()
        _, out, info = process(x, y, normalize_mode="window", normalize_window=(5000, 6000))
        np.testing.assert_array_equal(out, y)
        items = qc.processing_checks("s", info, points_used=len(x), normalize_mode="window", normalize_window=(5000, 6000))
        self.assertEqual(statuses(items, "normalization"), [qc.WARNING])
        _, _, info = process(x, np.ones(len(x)), normalize_mode="max")
        self.assertEqual(info["normalize_problem"], "flat")

    def test_transmittance_at_zero_is_reported(self):
        t = np.full(len(X), 50.0)
        t[100:104] = [0.0, -0.3, 0.0, 1e-9]
        mask = raman_processing.clipped_points(t, "transmittance_percent")
        self.assertEqual(int(mask.sum()), 4)
        items = qc.processing_checks("s", {}, points_used=len(X), x=X, clipped=mask)
        self.assertEqual(statuses(items, "clipped"), [qc.WARNING])
        self.assertEqual(next(i for i in items if i["code"] == "clipped")["params"]["where"], "500.0–503.0")

    def test_the_noise_is_measured_before_smoothing(self):
        x, y = si_like(noise=3.0)
        _, _, raw = process(x, y)
        _, _, smoothed = process(x, y, smooth_window=11)
        self.assertAlmostEqual(raw["noise"], 3.0, delta=0.4)
        self.assertAlmostEqual(smoothed["noise"], raw["noise"], delta=1e-9)
        _, _, scaled = process(x, y, normalize_mode="max")
        self.assertAlmostEqual(scaled["noise"], raw["noise"] * scaled["scale"], delta=1e-9)


class PeakTests(unittest.TestCase):
    def test_peaks_in_noise_are_flagged(self):
        y = RNG.normal(100, 5, len(X)) + peak_shape(X, "lorentzian", 200, 520, 6)
        x, out, info = process(X, y)
        indices = detect_peaks(x, out, 0.05, 5)
        snr = qc.peak_snr_values(peak_prominences_of(out, indices), info["noise"])
        self.assertGreater(len(indices), 3, "a low sensitivity picks noise maxima")
        self.assertGreater(snr[np.argmin(abs(x[indices] - 520))], 20)
        self.assertTrue((snr < 3).any())
        items = qc.peak_checks("s", x, out, indices, peak_prominences_of(out, indices), info["noise"])
        self.assertEqual(statuses(items, "peak_snr"), [qc.WARNING])

    def test_single_peak_fit_on_noise_is_flagged(self):
        noise = RNG.normal(100, 5, 1000)
        x = np.arange(1000.0)
        fitted = fit_peak(x, noise, 500.0, 15)
        if fitted["ok"]:                                       # 'ok' only means the optimiser returned
            self.assertTrue(fitted["limits"], "a fit to noise ends on a width or centre bound")
            items = qc.peak_checks("s", x, noise, [500], [1.0], 5.0, fits=[fitted])
            self.assertEqual(statuses(items, "peakfit_limit"), [qc.WARNING])
        good = fit_peak(*si_like()[0:2], 520.0, 15)
        self.assertTrue(good["ok"])
        self.assertEqual(good["limits"], [])


class FitTests(unittest.TestCase):
    XC = np.arange(1000.0, 1900.0, 1.0)
    TRUE = ((100, 1350, 120), (80, 1585, 70))

    def carbon(self, rng, noise=1.0):
        """SYNTHETIC: two Lorentzians (like carbon D and G bands) on a sloping background."""
        return (sum(peak_shape(self.XC, "lorentzian", *p) for p in self.TRUE) + 5 + 0.01 * (self.XC - 1450)
                + rng.normal(0, noise, len(self.XC)))

    def test_a_sound_fit_passes(self):
        result = fit_multipeak(self.XC, self.carbon(np.random.default_rng(1)), [1340, 1600], max_shift=30)
        items = qc.fit_checks("c", result)
        self.assertEqual(qc.overall(items), qc.PASS, [(i["code"], i["status"]) for i in items if i["status"] != qc.PASS])

    def test_the_standard_errors_cover_the_true_values(self):
        # Known answer: across noise realisations, |fitted − true| < 1.96 σ about 95 % of the time.
        hits, total = 0, 0
        for seed in range(60):
            result = fit_multipeak(self.XC, self.carbon(np.random.default_rng(seed)), [1340, 1600], max_shift=30)
            for c, (h, x0, w) in zip(result["components"], self.TRUE):
                for value, error, true in ((c["center"], c["center_err"], x0), (c["fwhm"], c["fwhm_err"], w)):
                    hits += abs(value - true) < 1.96 * error
                    total += 1
        self.assertGreater(hits / total, 0.88, hits / total)
        self.assertLess(hits / total, 0.995, hits / total)

    def test_wrong_models_are_flagged(self):
        y = self.carbon(np.random.default_rng(2))
        missing = qc.fit_checks("c", fit_multipeak(self.XC, y, [1350], max_shift=30))
        self.assertEqual(statuses(missing, "chi2") + statuses(missing, "autocorr"), [qc.WARNING, qc.WARNING])
        shape = qc.fit_checks("c", fit_multipeak(self.XC, y, [1340, 1600], model="gaussian", max_shift=30))
        self.assertEqual(statuses(shape, "autocorr"), [qc.WARNING])
        extra = qc.fit_checks("c", fit_multipeak(self.XC, y, [1340, 1480, 1600], max_shift=30))
        self.assertIn(qc.overall(extra), (qc.CAUTION, qc.WARNING), [(i["code"], i["status"]) for i in extra])

    def test_a_bad_start_ends_on_a_bound_without_errors(self):
        result = fit_multipeak(self.XC, self.carbon(np.random.default_rng(3)), [1300, 1585], max_shift=5)
        first = result["components"][0]
        self.assertEqual(first["limits"], ["center"])
        self.assertTrue(math.isnan(first["center_err"]), "no standard error for a parameter on a bound")
        self.assertFalse(math.isnan(first["fwhm_err"]))
        items = qc.fit_checks("c", result)
        self.assertEqual(statuses(items, "comp_limit"), [qc.WARNING, qc.PASS])
        self.assertEqual(qc.overall(items), qc.WARNING)

    def test_failures_give_no_numbers(self):
        y = self.carbon(np.random.default_rng(4))
        for result in (fit_multipeak(self.XC, y, [1350], window=(1349, 1351)),          # too few points
                       fit_multipeak(self.XC, np.ones(len(self.XC)), [1350])):          # flat data (zero division)
            self.assertFalse(result["ok"])
            items = qc.fit_checks("c", result)
            self.assertEqual([(i["code"], i["status"]) for i in items], [("fit_status", qc.FAIL)])
        with mock.patch.object(raman_processing, "curve_fit", side_effect=RuntimeError("Optimal parameters not found")):
            result = fit_multipeak(self.XC, y, [1340, 1600])
        self.assertFalse(result["ok"])
        self.assertEqual(result["components"], [])
        rendered = qc_text.render(qc.fit_checks("c", result)[0], "en")
        self.assertIn("no convergence", rendered["reason"])

    def test_smoothed_data_are_not_judged_by_chi_square(self):
        x = self.XC
        _, smoothed, info = process(x, self.carbon(np.random.default_rng(5)), smooth_window=11)
        result = fit_multipeak(x, smoothed, [1340, 1600], max_shift=30)
        items = qc.fit_checks("c", result, smoothed=True, noise_before_smoothing=info["noise"])
        self.assertEqual(statuses(items, "chi2"), [qc.NOT_EVALUATED])
        self.assertEqual(statuses(items, "comp_snr"), [qc.PASS, qc.PASS], "heights are compared with the unsmoothed noise")
        self.assertEqual(qc.overall(items), qc.NOT_EVALUATED, "required checks are missing, so it is not a PASS")

    def test_thresholds_can_be_changed(self):
        result = fit_multipeak(self.XC, self.carbon(np.random.default_rng(6)), [1340, 1600], max_shift=30)
        chi2 = result["chi2_red"]
        self.assertEqual(statuses(qc.fit_checks("c", result, th={"chi2_caution": chi2 / 2, "chi2_warning": chi2 * 2}), "chi2"),
                         [qc.CAUTION])


class TextTests(unittest.TestCase):
    def test_every_check_renders_in_every_language(self):
        x, y = si_like()
        _, _, info = process(x, y, spikes=6.0, normalize_mode="window", normalize_window=(5000, 6000))
        info["spikes_on"] = True
        items = (qc.input_checks("s", x, y, {"x_units_warning": "nm", "x_converted_from": "um",
                                             "_qc_read": {"skipped": [(3, "a", "not_numeric")], "skipped_count": 1,
                                                          "footer": [(9, "1 2", "after_data")], "footer_count": 1,
                                                          "cut_rows": [4], "cut_count": 1, "declared_points": 3, "points_in_file": 2}})
                 + qc.processing_checks("s", info, points_used=3, x=x, clipped=np.ones(len(x), bool),
                                        normalize_mode="window", normalize_window=(5000, 6000))
                 + qc.fit_checks("c", fit_multipeak(FitTests.XC, np.ones(len(FitTests.XC)), [1350]))
                 + qc.fit_checks("c", fit_multipeak(FitTests.XC, FitTests().carbon(np.random.default_rng(7)), [1300, 1480, 1585], max_shift=5)))
        codes = {i["code"] for i in items}
        self.assertGreater(len(codes), 20)
        for language in ("ja", "en", "th", "vi"):
            for it in items:
                text = qc_text.render(it, language, qc.default_thresholds())
                for field in ("name", "criterion", "reason"):
                    self.assertTrue(text[field], (language, it["code"], field))
                    self.assertNotIn("{", text[field], (language, it["code"], field))
        for key in qc_text.PANEL:
            for language in ("ja", "en", "th", "vi"):
                self.assertTrue(qc_text.PANEL[key][language], key)
        self.assertEqual(set(qc_text.THRESHOLD_LABELS), set(qc.THRESHOLDS))
        for code in qc_text.CHECKS:
            for field, value in qc_text.CHECKS[code].items():
                if isinstance(value, dict):
                    self.assertEqual(set(value), {"ja", "en", "th", "vi"}, (code, field))


if __name__ == "__main__":
    unittest.main()
