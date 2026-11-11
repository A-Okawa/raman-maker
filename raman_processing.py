"""Spectrum processing for the figure: cropping, spike removal, smoothing, baseline
subtraction, normalisation, peak detection and single-peak fits.

Everything here works on plain NumPy arrays and never touches Streamlit, so the same
code runs in the app and in the tests. Nothing is applied unless the app asks for it;
the readers in ``raman_io`` return intensities exactly as stored.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve


BASELINE_METHODS = ("none", "offset", "linear", "poly", "als")
Y_UNITS = ("absorbance", "transmittance_percent", "transmittance_fraction")
_T_FLOOR = 1e-6              # transmittance below this (in %) is clipped before the logarithm
NORMALIZE_MODES = ("none", "max", "window", "minmax", "area")
PEAK_MODELS = ("lorentzian", "gaussian")


def _array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(array).all():
        raise ValueError("Values must be finite.")
    return array


def crop(x, y, low: float, high: float):
    """The part of a spectrum with low ≤ x ≤ high (x ascending in, ascending out)."""
    x, y = _array(x), _array(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if low > high:
        low, high = high, low
    mask = (x >= low) & (x <= high)
    return x[mask], y[mask]


# ------------------------------------------------------------------------------- FT-IR units
def to_absorbance(y, units: str = "absorbance"):
    """Absorbance from values in ``units`` (see Y_UNITS): A = -log10(T), T as a fraction.

    Transmittance at or below zero (noise in saturated bands) is clipped to 1e-8 so the
    logarithm stays finite."""
    y = _array(y)
    if units == "absorbance":
        return y.copy()
    if units == "transmittance_percent":
        fraction = y / 100.0
    elif units == "transmittance_fraction":
        fraction = y
    else:
        raise ValueError(f"Unknown y units: {units}")
    return -np.log10(np.maximum(fraction, _T_FLOOR / 100.0))


def from_absorbance(a, units: str = "absorbance"):
    """Values in ``units`` from absorbance: T = 10^(-A) (as a fraction or in %)."""
    a = _array(a)
    if units == "absorbance":
        return a.copy()
    if units == "transmittance_percent":
        return 100.0 * np.power(10.0, -a)
    if units == "transmittance_fraction":
        return np.power(10.0, -a)
    raise ValueError(f"Unknown y units: {units}")


# ------------------------------------------------------------------------------------ spikes
def remove_spikes(y, threshold: float = 6.0, window: int = 3):
    """Replace cosmic-ray spikes (Whitaker & Hayes, Chemom. Intell. Lab. Syst. 179, 82 (2018)).

    A point is a spike when the modified z-score of the intensity difference to the
    previous point exceeds ``threshold``; it is replaced by the mean of the non-spike
    points within ``window`` points on each side. Returns (cleaned y, number replaced).
    """
    y = _array(y)
    if len(y) < 5 or threshold <= 0:
        return y.copy(), 0
    diff = np.diff(y)
    median = np.median(diff)
    mad = np.median(np.abs(diff - median))
    if mad == 0:
        return y.copy(), 0
    z = 0.6745 * (diff - median) / mad
    spike = np.zeros(len(y), dtype=bool)
    spike[1:] = np.abs(z) > threshold
    # The point after a spike falls back as sharply: mark only points that stand above neighbours.
    for k in np.flatnonzero(spike):
        left = y[k - 1] if k > 0 else y[k + 1]
        right = y[k + 1] if k + 1 < len(y) else y[k - 1]
        if not (y[k] > left and y[k] > right):
            spike[k] = False
    cleaned = y.copy()
    window = max(1, int(window))
    for k in np.flatnonzero(spike):
        lo, hi = max(0, k - window), min(len(y), k + window + 1)
        neighbours = [y[j] for j in range(lo, hi) if not spike[j]]
        if neighbours:
            cleaned[k] = float(np.mean(neighbours))
    return cleaned, int(spike.sum())


# ---------------------------------------------------------------------------------- smoothing
def smooth(y, window: int = 7, polyorder: int = 2):
    """Savitzky–Golay smoothing; the window is made odd and larger than the polynomial order."""
    y = _array(y)
    window = int(window)
    polyorder = max(0, int(polyorder))
    if window <= polyorder:
        window = polyorder + 1
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) if len(y) % 2 else len(y) - 1
    if window <= polyorder or window < 3:
        return y.copy()
    return savgol_filter(y, window, polyorder)


# ----------------------------------------------------------------------------------- baseline
def baseline_linear(x, y, edge_fraction: float = 0.02):
    """A straight line through the mean of the first and last few points (at least 3)."""
    x, y = _array(x), _array(y)
    n = len(y)
    k = max(3, int(round(n * edge_fraction)))
    k = min(k, max(1, n // 2))
    x0, y0 = float(np.mean(x[:k])), float(np.mean(y[:k]))
    x1, y1 = float(np.mean(x[-k:])), float(np.mean(y[-k:]))
    if x1 == x0:
        return np.full(n, min(y0, y1))
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def baseline_modpoly(x, y, order: int = 3, iterations: int = 100, tolerance: float = 1e-3):
    """Modified polynomial baseline (Lieber & Mahadevan-Jansen, Appl. Spectrosc. 57, 1363 (2003)).

    The polynomial is fitted repeatedly to min(spectrum, previous fit), so it slides
    under the peaks.
    """
    x, y = _array(x), _array(y)
    order = int(order)
    if order < 0 or order > 12:
        raise ValueError("Polynomial order must be between 0 and 12.")
    if len(y) <= order + 1:
        return np.full(len(y), float(np.min(y)))
    span = float(np.ptp(x)) or 1.0
    xs = (x - x.mean()) / span
    work = y.copy()
    fit = work
    for _ in range(max(1, int(iterations))):
        coefficients = np.polyfit(xs, work, order)
        fit = np.polyval(coefficients, xs)
        new = np.minimum(work, fit)
        change = np.linalg.norm(new - work) / (np.linalg.norm(work) or 1.0)
        work = new
        if change < tolerance:
            break
    return np.polyval(np.polyfit(xs, work, order), xs)


def baseline_als(y, lam: float = 1e5, p: float = 0.01, iterations: int = 10):
    """Asymmetric least squares baseline (Eilers & Boelens 2005).

    ``lam`` sets the smoothness (10³ follows the spectrum closely, 10⁷ is nearly a line),
    ``p`` the asymmetry (0.001–0.1; small values keep the baseline under the peaks).
    """
    y = _array(y)
    n = len(y)
    if n < 3:
        return np.full(n, float(np.min(y)))
    if lam <= 0 or not 0 < p < 1:
        raise ValueError("lam must be positive and p between 0 and 1.")
    difference = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    penalty = lam * difference.dot(difference.transpose())
    weights = np.ones(n)
    z = y.copy()
    for _ in range(max(1, int(iterations))):
        system = sparse.diags(weights, 0) + penalty
        z = spsolve(system.tocsc(), weights * y)
        weights = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z, dtype=float)


def baseline(x, y, method: str = "none", **params):
    """The baseline for ``method`` (see BASELINE_METHODS); subtract it from y."""
    x, y = _array(x), _array(y)
    if method == "none":
        return np.zeros(len(y))
    if method == "offset":
        return np.full(len(y), float(np.min(y)))
    if method == "linear":
        return baseline_linear(x, y, params.get("edge_fraction", 0.02))
    if method == "poly":
        return baseline_modpoly(x, y, params.get("order", 3), params.get("iterations", 100))
    if method == "als":
        return baseline_als(y, params.get("lam", 1e5), params.get("p", 0.01), params.get("iterations", 10))
    raise ValueError(f"Unknown baseline method: {method}")


# -------------------------------------------------------------------------------- normalisation
def normalize(y, mode: str = "max", x=None, window=None):
    """Scale a spectrum: 'max' → max = 1, 'window' → max within the window = 1 (window on x),
    'minmax' → min 0 / max 1, 'area' → unit area (trapezoid on x), 'none' → unchanged."""
    y = _array(y)
    if mode == "none":
        return y.copy()
    if mode == "max":
        top = float(np.max(np.abs(y))) if len(y) else 0.0
        return y / top if top > 0 else y.copy()
    if mode == "minmax":
        low, high = float(np.min(y)), float(np.max(y))
        return (y - low) / (high - low) if high > low else np.zeros(len(y))
    if mode == "window":
        if x is None or window is None:
            raise ValueError("Window normalisation needs x and a (low, high) window.")
        x = _array(x)
        low, high = sorted((float(window[0]), float(window[1])))
        inside = (x >= low) & (x <= high)
        if not inside.any():
            return normalize(y, "max")
        top = float(np.max(np.abs(y[inside])))
        return y / top if top > 0 else y.copy()
    if mode == "area":
        if x is None:
            raise ValueError("Area normalisation needs x.")
        x = _array(x)
        area = float(np.trapezoid(np.abs(y), x)) if hasattr(np, "trapezoid") else float(np.trapz(np.abs(y), x))
        return y / area if area > 0 else y.copy()
    raise ValueError(f"Unknown normalisation mode: {mode}")


# -------------------------------------------------------------------------------------- peaks
def detect_peaks(x, y, prominence_fraction: float = 0.1, min_distance: float = 5.0):
    """Indices of local maxima whose prominence is at least ``prominence_fraction`` of the
    spectrum's intensity range, at least ``min_distance`` (x units) apart."""
    x, y = _array(x), _array(y)
    if len(y) < 3:
        return np.array([], dtype=int)
    span = float(np.ptp(y))
    if span <= 0:
        return np.array([], dtype=int)
    spacing = np.diff(x)
    spacing = spacing[spacing > 0]
    step = float(np.median(spacing)) if len(spacing) else 1.0
    distance = max(1, int(round(min_distance / step)))
    indices, _ = find_peaks(y, prominence=prominence_fraction * span, distance=distance)
    return indices


def _lorentzian(x, height, center, hwhm, offset, slope):
    return height * hwhm ** 2 / ((x - center) ** 2 + hwhm ** 2) + offset + slope * (x - center)


def _gaussian(x, height, center, sigma, offset, slope):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset + slope * (x - center)


def fit_peak(x, y, center_guess: float, half_window: float = 15.0, model: str = "lorentzian") -> dict:
    """Fit one peak (Lorentzian or Gaussian on a linear background) within ±half_window of
    the guess. Returns center, fwhm, height, area (model only) and ``ok``; a failed fit keeps
    the local maximum as ``center`` and sets ``ok`` False.
    """
    x, y = _array(x), _array(y)
    if model not in PEAK_MODELS:
        raise ValueError(f"Unknown peak model: {model}")
    inside = (x >= center_guess - half_window) & (x <= center_guess + half_window)
    xs, ys = x[inside], y[inside]
    result = {"center": float(center_guess), "fwhm": math.nan, "height": math.nan, "area": math.nan,
              "model": model, "ok": False, "points": int(len(xs))}
    if len(xs) < 6:
        return result
    top = int(np.argmax(ys))
    result["center"] = float(xs[top])
    edge = float(min(ys[0], ys[-1]))
    height0 = float(ys[top] - edge)
    width0 = max(float(half_window) / 4.0, float(np.median(np.diff(xs))) * 2 if len(xs) > 1 else 1.0)
    function = _lorentzian if model == "lorentzian" else _gaussian
    p0 = [max(height0, 1e-12), float(xs[top]), width0, edge, 0.0]
    span_y = float(np.ptp(ys)) or 1.0
    bounds = ([0.0, float(xs[0]), float(np.median(np.diff(xs))) / 2 if len(xs) > 1 else 1e-6, -np.inf, -np.inf],
              [np.inf, float(xs[-1]), float(half_window) * 2, np.inf, np.inf])
    try:
        params, _ = curve_fit(function, xs, ys, p0=p0, bounds=bounds, maxfev=4000)
    except (RuntimeError, ValueError):
        return result
    height, center, width, offset, slope = (float(v) for v in params)
    fit = function(xs, *params)
    residual = float(np.sqrt(np.mean((ys - fit) ** 2)))
    if span_y <= 0 or not (xs[0] < center < xs[-1]) or height < 0.05 * span_y or residual > span_y:
        return result
    if model == "lorentzian":
        fwhm, area = 2 * width, math.pi * height * width
    else:
        fwhm, area = 2 * math.sqrt(2 * math.log(2)) * width, height * width * math.sqrt(2 * math.pi)
    result.update(center=center, fwhm=float(fwhm), height=height, area=float(area), ok=True, rmse=residual)
    return result


def peak_table(x, y, indices, fit: bool = False, half_window: float = 15.0, model: str = "lorentzian") -> list[dict]:
    """One row per detected peak: position (local maximum, and the fitted centre when asked)."""
    x, y = _array(x), _array(y)
    rows = []
    for k in np.asarray(indices, dtype=int):
        row = {"position": float(x[k]), "intensity": float(y[k])}
        if fit:
            fitted = fit_peak(x, y, float(x[k]), half_window, model)
            row.update(center=fitted["center"], fwhm=fitted["fwhm"], height=fitted["height"], fit_ok=fitted["ok"])
        rows.append(row)
    return rows


def process(x, y, *, x_range=None, spikes: float | None = None, smooth_window: int = 0,
            smooth_order: int = 2, baseline_method: str = "none", baseline_params: dict | None = None,
            normalize_mode: str = "none", normalize_window=None):
    """The whole chain in the app's order: crop → spikes → smoothing → baseline → normalisation.

    Returns (x, y, info) where info reports the number of replaced spikes and the subtracted
    baseline (on the cropped x), so the app can show them.
    """
    x, y = _array(x), _array(y)
    if x_range is not None:
        x, y = crop(x, y, *x_range)
    info = {"spikes": 0, "baseline": None}
    if len(y) == 0:
        return x, y, info
    if spikes:
        y, info["spikes"] = remove_spikes(y, spikes)
    if smooth_window and smooth_window >= 3:
        y = smooth(y, smooth_window, smooth_order)
    if baseline_method != "none":
        base = baseline(x, y, baseline_method, **(baseline_params or {}))
        info["baseline"] = base
        y = y - base
    if normalize_mode != "none":
        y = normalize(y, normalize_mode, x, normalize_window)
    return x, y, info
