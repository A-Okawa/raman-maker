"""Processing: each step on synthetic spectra with known answers."""
from pathlib import Path
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from raman_processing import (baseline, baseline_als, baseline_linear, baseline_modpoly, crop, detect_peaks, fit_peak,
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

    def test_window_outside_the_data_falls_back_to_max(self):
        y = lorentz(500, 90, 5)
        self.assertAlmostEqual(float(normalize(y, "window", X, (5000, 6000)).max()), 1.0)


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


if __name__ == "__main__":
    unittest.main()
