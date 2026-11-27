"""Processing: each step on synthetic spectra with known answers."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from raman_processing import (fit_multipeak, parse_centers, peak_shape, process as run_process,
                              baseline, baseline_als, baseline_linear, baseline_modpoly, crop, detect_peaks, fit_peak,
                              from_absorbance, normalize, peak_table, process, remove_spikes, smooth, to_absorbance)

X = np.arange(200.0, 1000.001, 0.5)


def lorentz(centre, height, hwhm):
    return height * hwhm ** 2 / ((X - centre) ** 2 + hwhm ** 2)


class CropTests(unittest.TestCase):
    def test_crop_keeps_the_interval(self):
        x, y = crop(X, X * 2, 300, 400)
        self.assertEqual(float(x[0]), 300.0)
        self.assertEqual(float(x[-1]), 400.0)
        self.assertEqual(len(x), 201)
        x2, _ = crop(X, X * 2, 400, 300)
        np.testing.assert_array_equal(x, x2)


class SpikeTests(unittest.TestCase):
    def test_single_point_spikes_are_replaced(self):
        rng = np.random.default_rng(1)
        y = 100 + lorentz(500, 1000, 5) + 3 * rng.standard_normal(len(X))
        spiked = y.copy()
        spiked[700] += 5000
        spiked[1200] += 2000
        cleaned, count = remove_spikes(spiked, 6.0)
        self.assertEqual(count, 2)
        self.assertLess(abs(cleaned[700] - y[700]), 15)
        self.assertLess(abs(cleaned[1200] - y[1200]), 15)
        # the real peak (width 5 cm-1 = 10 points) is not a spike
        self.assertAlmostEqual(float(cleaned.max()), float(y.max()), delta=15)

    def test_constant_spectrum(self):
        cleaned, count = remove_spikes(np.full(50, 3.0))
        self.assertEqual(count, 0)


class SmoothTests(unittest.TestCase):
    def test_noise_is_reduced_and_peak_kept(self):
        rng = np.random.default_rng(2)
        clean = 100 + lorentz(500, 1000, 6)
        noisy = clean + 10 * rng.standard_normal(len(X))
        smoothed = smooth(noisy, 9, 2)
        self.assertLess(np.std(smoothed - clean), np.std(noisy - clean))
        self.assertAlmostEqual(float(X[np.argmax(smoothed)]), 500.0, delta=1.0)

    def test_window_is_made_odd_and_valid(self):
        y = np.arange(20.0)
        np.testing.assert_allclose(smooth(y, 4, 2), y, atol=1e-9)
        np.testing.assert_allclose(smooth(y, 2, 3), y, atol=1e-9)
        np.testing.assert_allclose(smooth(y[:3], 11, 2), y[:3], atol=1e-9)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.background = 50 + 0.2 * (X - 200)
        self.peaks = lorentz(500, 1000, 5) + lorentz(800, 600, 4)
        self.y = self.background + self.peaks

    def test_linear_recovers_a_straight_background(self):
        base = baseline_linear(X, self.y)
        np.testing.assert_allclose(base, self.background, atol=0.5)

    def test_modpoly_stays_under_the_peaks(self):
        base = baseline_modpoly(X, self.y, 2)
        # the modified polynomial converges to a curve under the data: never above the background
        # by more than a few counts, and a little below it (the method's known bias)
        self.assertLess(float(np.max(base - self.background)), 3)
        self.assertLess(float(np.max(self.background - base)), 20)
        self.assertLess(float(base[np.argmin(np.abs(X - 500))]), self.y[np.argmin(np.abs(X - 500))] - 900)

    def test_als_stays_under_the_peaks(self):
        base = baseline_als(self.y, 1e5, 0.01)
        away = (X < 450) | (X > 850)
        self.assertLess(float(np.max(np.abs(base[away] - self.background[away]))), 10)
        self.assertLess(float(base[np.argmin(np.abs(X - 500))]), self.y[np.argmin(np.abs(X - 500))] - 900)

    def test_dispatch_and_errors(self):
        np.testing.assert_array_equal(baseline(X, self.y, "none"), 0)
        np.testing.assert_allclose(baseline(X, self.y, "offset"), float(self.y.min()))
        with self.assertRaises(ValueError):
            baseline(X, self.y, "spline")
        with self.assertRaises(ValueError):
            baseline_als(self.y, lam=-1)
        with self.assertRaises(ValueError):
            baseline_modpoly(X, self.y, order=20)


class NormalizeTests(unittest.TestCase):
    def test_modes(self):
        y = 10 + lorentz(500, 90, 5)
        self.assertAlmostEqual(float(normalize(y, "max").max()), 1.0)
        mm = normalize(y, "minmax")
        self.assertAlmostEqual(float(mm.min()), 0.0)
        self.assertAlmostEqual(float(mm.max()), 1.0)
        window = normalize(y, "window", X, (700, 900))
        self.assertAlmostEqual(float(window[(X >= 700) & (X <= 900)].max()), 1.0)
        self.assertGreater(float(window.max()), 5)
        area = normalize(y, "area", X)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        self.assertAlmostEqual(float(integrate(area, X)), 1.0, places=6)
        np.testing.assert_array_equal(normalize(y, "none"), y)
        with self.assertRaises(ValueError):
            normalize(y, "window")
        with self.assertRaises(ValueError):
            normalize(y, "nope")

    def test_window_outside_the_data_is_reported_not_replaced(self):
        # Before 1.6 an empty window silently switched to the global maximum.
        y = lorentz(500, 90, 5)
        with self.assertRaises(ValueError):
            normalize(y, "window", X, (5000, 6000))
        x, out, info = process(X, y, normalize_mode="window", normalize_window=(5000, 6000))
        np.testing.assert_array_equal(out, y)
        self.assertEqual(info["normalize_problem"], "window_empty")


class PeakTests(unittest.TestCase):
    def test_detection_and_fit(self):
        y = 20 + lorentz(500, 1000, 5) + lorentz(800, 300, 4) + lorentz(812, 100, 3)
        indices = detect_peaks(X, y, 0.01, 5.0)
        self.assertEqual([float(X[k]) for k in indices], [500.0, 800.0, 812.0])
        self.assertEqual(len(detect_peaks(X, y, 0.05, 5.0)), 2, "the shoulder at 812 has a small prominence")
        self.assertEqual(len(detect_peaks(X, y, 0.01, 20.0)), 2, "a 20 cm-1 spacing merges 800 and 812")
        fitted = fit_peak(X, y, 500.0, 15.0, "lorentzian")
        self.assertTrue(fitted["ok"])
        self.assertAlmostEqual(fitted["center"], 500.0, places=2)
        self.assertAlmostEqual(fitted["fwhm"], 10.0, delta=0.2)
        self.assertAlmostEqual(fitted["height"], 1000.0, delta=20)
        gauss = fit_peak(X, y, 500.0, 15.0, "gaussian")
        self.assertTrue(gauss["ok"])
        self.assertAlmostEqual(gauss["center"], 500.0, delta=0.1)

    def test_fit_failure_keeps_the_maximum(self):
        y = np.ones(len(X))
        result = fit_peak(X, y, 500.0, 15.0)
        self.assertFalse(result["ok"])
        self.assertTrue(np.isnan(result["fwhm"]))
        self.assertEqual(len(detect_peaks(X, y)), 0)
        with self.assertRaises(ValueError):
            fit_peak(X, y, 500.0, 15.0, "voigt")

    def test_table(self):
        y = 20 + lorentz(500, 1000, 5)
        rows = peak_table(X, y, detect_peaks(X, y), fit=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["position"], 500.0)
        self.assertTrue(rows[0]["fit_ok"])


class UnitTests(unittest.TestCase):
    def test_absorbance_and_transmittance(self):
        t = np.array([100.0, 50.0, 10.0, 1.0])
        a = to_absorbance(t, "transmittance_percent")
        np.testing.assert_allclose(a, [0.0, np.log10(2), 1.0, 2.0])
        np.testing.assert_allclose(from_absorbance(a, "transmittance_percent"), t)
        np.testing.assert_allclose(to_absorbance(t / 100, "transmittance_fraction"), a)
        np.testing.assert_allclose(from_absorbance(a, "transmittance_fraction"), t / 100)
        np.testing.assert_array_equal(to_absorbance(a, "absorbance"), a)
        np.testing.assert_array_equal(from_absorbance(a, "absorbance"), a)

    def test_zero_and_negative_transmittance_stay_finite(self):
        a = to_absorbance([0.0, -1.0, 1e-9], "transmittance_percent")
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a >= 7.9).all())
        with self.assertRaises(ValueError):
            to_absorbance([1.0], "counts")
        with self.assertRaises(ValueError):
            from_absorbance([1.0], "counts")

    def test_transmittance_bands_become_peaks_in_absorbance(self):
        t = 95 - 60 * np.exp(-0.5 * ((X - 500) / 10) ** 2)
        a = to_absorbance(t, "transmittance_percent")
        self.assertEqual(float(X[np.argmax(a)]), 500.0)
        self.assertEqual([float(X[k]) for k in detect_peaks(X, a, 0.1, 5.0)], [500.0])


class ChainTests(unittest.TestCase):
    def test_order_crop_spikes_smooth_baseline_normalize(self):
        rng = np.random.default_rng(3)
        y = 50 + 0.2 * (X - 200) + lorentz(500, 1000, 5) + 2 * rng.standard_normal(len(X))
        y[600] += 4000
        x, out, info = process(X, y, x_range=(300, 700), spikes=6.0, smooth_window=7, baseline_method="linear",
                               normalize_mode="max")
        self.assertEqual(float(x[0]), 300.0)
        self.assertEqual(info["spikes"], 1)
        self.assertEqual(len(info["baseline"]), len(x))
        self.assertAlmostEqual(float(out.max()), 1.0)
        self.assertAlmostEqual(float(x[np.argmax(out)]), 500.0, delta=0.5)
        self.assertLess(abs(float(out[0])), 0.05)

    def test_empty_range(self):
        x, out, info = process(X, X, x_range=(5000, 6000), baseline_method="als", normalize_mode="max")
        self.assertEqual(len(out), 0)


class MultiPeakFitTests(unittest.TestCase):
    """Overlapping peaks: the D and G bands of carbon as two Lorentzians on a sloping background."""
    XC = np.arange(1000.0, 1900.0, 1.0)

    def carbon(self, noise=1.0):
        rng = np.random.default_rng(3)
        return (peak_shape(self.XC, "lorentzian", 100, 1350, 120) + peak_shape(self.XC, "lorentzian", 80, 1585, 70)
                + 5 + 0.01 * (self.XC - 1450) + rng.normal(0, noise, len(self.XC)))

    def test_two_overlapping_lorentzians_are_recovered_with_errors(self):
        result = fit_multipeak(self.XC, self.carbon(), [1340, 1600], model="lorentzian", max_shift=30)
        self.assertTrue(result["ok"])
        d, g = result["components"]
        self.assertAlmostEqual(d["center"], 1350, delta=0.5)
        self.assertAlmostEqual(g["center"], 1585, delta=0.5)
        self.assertAlmostEqual(d["fwhm"], 120, delta=2)
        self.assertAlmostEqual(g["fwhm"], 70, delta=2)
        self.assertAlmostEqual(d["area"] / g["area"], 100 * 120 / (80 * 70), delta=0.03)
        self.assertAlmostEqual(d["area"], math.pi / 2 * 100 * 120, delta=200)
        self.assertAlmostEqual(d["share"] + g["share"], 1.0)
        self.assertGreater(result["r2"], 0.995)
        self.assertAlmostEqual(result["rmse"], 1.0, delta=0.15)
        for c in (d, g):                                   # the errors are of the size of the actual deviations
            self.assertTrue(0 < c["center_err"] < 0.5 and 0 < c["fwhm_err"] < 2 and 0 < c["area_err"] < 300)
            self.assertFalse(c["at_limit"])
        np.testing.assert_allclose(result["fit"], result["background"] + sum(result["curves"]))
        np.testing.assert_allclose(result["residual"], result["y"] - result["fit"])

    def test_shapes_areas_and_the_pseudo_voigt_mixing(self):
        x = np.arange(0.0, 400.0, 0.5)
        for model, factor in (("lorentzian", math.pi / 2), ("gaussian", math.sqrt(math.pi / (4 * math.log(2))))):
            y = peak_shape(x, model, 10, 200, 20)
            self.assertAlmostEqual(float(np.max(y)), 10)
            half = x[y >= 5]
            self.assertAlmostEqual(half[-1] - half[0], 20, delta=0.6)
            result = fit_multipeak(x, y, [195], model=model, background="none")
            self.assertAlmostEqual(result["components"][0]["area"], factor * 10 * 20, delta=0.01 * factor * 200)
        mixed = peak_shape(x, "pvoigt", 10, 200, 20, 0.3) + 1.0
        result = fit_multipeak(x, mixed, [203], model="pvoigt", background="constant")
        self.assertAlmostEqual(result["components"][0]["eta"], 0.3, delta=0.01)
        self.assertAlmostEqual(result["components"][0]["center"], 200, delta=0.01)
        self.assertAlmostEqual(float(result["background"][0]), 1.0, delta=0.01)

    def test_the_window_limits_and_refusals(self):
        y = self.carbon()
        inside = fit_multipeak(self.XC, y, [1585], window=(1500, 1700), max_shift=20)
        self.assertTrue(inside["ok"])
        self.assertEqual((inside["x"][0], inside["x"][-1]), (1500.0, 1700.0))
        stuck = fit_multipeak(self.XC, y, [1300, 1585], max_shift=5)          # the D band lies 50 from its start
        self.assertTrue(stuck["components"][0]["at_limit"])
        self.assertAlmostEqual(stuck["components"][0]["center"], 1305, delta=0.2)
        self.assertEqual(fit_multipeak(self.XC, y, [])["message"], "no_centers")
        self.assertEqual(fit_multipeak(self.XC, y, [2500])["message"], "center_outside")
        self.assertEqual(fit_multipeak(self.XC[:6], y[:6], [1002, 1004])["message"], "too_few_points")
        self.assertEqual(fit_multipeak(self.XC, np.ones(len(self.XC)), [1350])["message"], "flat")
        self.assertEqual(fit_multipeak(self.XC, y, list(range(1100, 1800, 50)))["message"], "too_many")
        with self.assertRaises(ValueError):
            fit_multipeak(self.XC, y, [1350], model="voigt")
        descending = fit_multipeak(self.XC[::-1], y[::-1], [1340, 1600], max_shift=30)   # FT-IR files run high → low
        self.assertAlmostEqual(descending["components"][0]["center"], 1350, delta=0.5)

    def test_diagnostics_separate_a_sound_fit_from_wrong_ones(self):
        y = self.carbon()
        good = fit_multipeak(self.XC, y, [1340, 1600], max_shift=30)
        self.assertEqual(good["warnings"], [])
        self.assertAlmostEqual(good["noise"], 1.0, delta=0.15)
        self.assertAlmostEqual(good["chi2_red"], 1.0, delta=0.3)
        self.assertLess(abs(good["autocorr"]), 0.2)
        self.assertLess(good["max_correlation"], 0.9)
        self.assertGreater(min(c["snr"] for c in good["components"]), 50)
        # one component for two bands: the residual is large and wavy
        missing = fit_multipeak(self.XC, y, [1350], max_shift=30)
        self.assertIn("misfit", missing["warnings"])
        self.assertIn("structured_residual", missing["warnings"])
        self.assertGreater(missing["chi2_red"], 10)
        # the wrong shape for the same bands
        self.assertIn("structured_residual", fit_multipeak(self.XC, y, [1340, 1600], model="gaussian", max_shift=30)["warnings"])
        # a third component where there is none: not above the noise, or traded off against its neighbour
        extra = fit_multipeak(self.XC, y, [1340, 1480, 1600], max_shift=30)
        self.assertTrue({"weak_component", "uncertain_area", "correlated", "at_limit"} & set(extra["warnings"]), extra["warnings"])
        # a start too far from the band
        self.assertIn("at_limit", fit_multipeak(self.XC, y, [1300, 1585], max_shift=5)["warnings"])
        # two components on one band cannot be told apart
        twins = fit_multipeak(self.XC, y, [1580, 1590], window=(1500, 1700), max_shift=20)
        self.assertTrue({"correlated", "uncertain_area", "at_limit", "weak_component"} & set(twins["warnings"]), twins["warnings"])

    def test_centres_are_read_from_free_text(self):
        self.assertEqual(parse_centers("1350, 1580"), [1350.0, 1580.0])
        self.assertEqual(parse_centers("1350、1580；1620\n 2700 cm"), [1350.0, 1580.0, 1620.0, 2700.0])
        self.assertEqual(parse_centers(""), [])
        self.assertEqual(parse_centers(None), [])

    def test_process_reports_the_measured_data_and_the_normalisation(self):
        y = self.carbon()
        for mode in ("none", "max", "minmax", "area"):
            x, out, info = run_process(self.XC, y, x_range=(1100, 1800), baseline_method="linear", normalize_mode=mode)
            np.testing.assert_allclose(info["raw"], y[(self.XC >= 1100) & (self.XC <= 1800)])
            np.testing.assert_allclose(out, info["scale"] * (info["raw"] - info["baseline"]) + info["shift"], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
