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


def clipped_points(y, units: str = "absorbance"):
    """Boolean mask of the points ``to_absorbance`` clips (transmittance at or below the floor)."""
    y = _array(y)
    if units == "transmittance_percent":
        return y <= _T_FLOOR
    if units == "transmittance_fraction":
        return y <= _T_FLOOR / 100.0
    return np.zeros(len(y), dtype=bool)


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


def spike_report(x, before, after, half_window: int = 10) -> list[dict]:
    """The points spike removal changed: index, x, value before and after, and ``neighbour_ratio``.

    ``neighbour_ratio`` is the height of the higher neighbour above the local median divided by the
    height of the replaced point above it. A cosmic-ray spike occupies one point, so its neighbours
    sit at the local level (ratio near 0); a real peak a few points wide lifts its neighbours too
    (a Lorentzian of FWHM 4 sampled every 1 cm⁻¹ gives about 0.8). The ratio is a warning sign only.
    """
    x, before, after = _array(x), _array(before), _array(after)
    changed = np.flatnonzero(before != after)
    out = []
    for k in changed:
        lo, hi = max(0, k - half_window), min(len(before), k + half_window + 1)
        local = np.delete(before[lo:hi], [j - lo for j in (k - 1, k, k + 1) if lo <= j < hi])
        level = float(np.median(local)) if len(local) else float(np.median(before))
        excess = float(before[k]) - level
        neighbours = [before[j] for j in (k - 1, k + 1) if 0 <= j < len(before)]
        ratio = (max(neighbours) - level) / excess if excess > 0 and neighbours else math.nan
        out.append({"index": int(k), "x": float(x[k]), "before": float(before[k]), "after": float(after[k]),
                    "neighbour_ratio": float(ratio)})
    return out


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
    'minmax' → min 0 / max 1, 'area' → unit area (trapezoid on x), 'none' → unchanged.

    A window without data points raises ValueError (``process`` then leaves the spectrum
    unnormalised and reports it); an all-zero spectrum is returned unchanged."""
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
            raise ValueError("The normalisation window contains no data points.")
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


def peak_prominences_of(y, indices):
    """Prominence of each detected peak (its height above the higher of the two bases)."""
    from scipy.signal import peak_prominences
    y = _array(y)
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return np.array([])
    return peak_prominences(y, indices)[0]


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
    step = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    limits = []
    if width <= bounds[0][2] * 1.02 or width >= bounds[1][2] * 0.98:
        limits.append("fwhm")
    if center - xs[0] <= step or xs[-1] - center <= step:
        limits.append("center")
    result["limits"] = limits
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


# ------------------------------------------------------------------ fits of overlapping peaks
PEAK_SHAPES = ("lorentzian", "gaussian", "pvoigt")
FIT_BACKGROUNDS = ("none", "constant", "linear")
_GAUSS_AREA = math.sqrt(math.pi / (4 * math.log(2)))      # area of a unit-height Gaussian per unit FWHM
_LORENTZ_AREA = math.pi / 2                                # the same for a Lorentzian
MAX_FIT_PEAKS = 12


# Thresholds of the fit diagnostics (see fit_multipeak's ``warnings``)
CHI2_HIGH = 3.0               # reduced χ² above this: the model misses structure larger than the noise
AUTOCORR_HIGH = 0.5           # lag-1 autocorrelation of the residual above this: waves, not noise
CORRELATION_HIGH = 0.98       # |correlation| of two parameters above this: they trade off against each other
SNR_LOW = 3.0                 # component height below this many noise σ
RELATIVE_ERROR_HIGH = 0.2     # standard error of an area above this fraction of the area


def estimate_noise(y) -> float:
    """Point-to-point noise σ from second differences (robust: median absolute deviation), which
    peaks and slopes that are wide compared with the point spacing do not inflate."""
    y = _array(y)
    if len(y) < 5:
        return math.nan
    second = y[:-2] - 2 * y[1:-1] + y[2:]
    mad = float(np.median(np.abs(second - np.median(second))))
    return 1.4826 * mad / math.sqrt(6.0)


def peak_shape(x, model: str, height: float, center: float, fwhm: float, eta: float = 0.5):
    """One peak of ``height`` and full width at half maximum ``fwhm``; 'pvoigt' is the pseudo-Voigt
    eta·Lorentzian + (1 − eta)·Gaussian with a common FWHM."""
    x = _array(x)
    u = (x - center) / max(float(fwhm), 1e-12)
    lorentz = 1.0 / (1.0 + 4.0 * u ** 2)
    gauss = np.exp(-4.0 * math.log(2) * u ** 2)
    if model == "lorentzian":
        return height * lorentz
    if model == "gaussian":
        return height * gauss
    if model == "pvoigt":
        return height * (eta * lorentz + (1.0 - eta) * gauss)
    raise ValueError(f"Unknown peak model: {model}")


def _area_factor(model: str, eta: float) -> float:
    if model == "lorentzian":
        return _LORENTZ_AREA
    if model == "gaussian":
        return _GAUSS_AREA
    return eta * _LORENTZ_AREA + (1.0 - eta) * _GAUSS_AREA


def fit_multipeak(x, y, centers, window=None, model: str = "lorentzian", background: str = "linear",
                  max_shift: float = 15.0) -> dict:
    """Fit the sum of len(centers) peaks (one shape for all) plus a background within ``window``.

    The number of components and their starting positions are the caller's: nothing is added or
    dropped automatically. Each centre may move by at most ±max_shift from its start; heights are
    non-negative; widths lie between two data steps and the window width.

    Returns a dict with ``ok``, ``message`` (why not, when not ok), ``x`` (the fitted points),
    ``y``, ``fit`` (total), ``background`` (curve), ``curves`` (one per component, without the
    background), ``residual``, ``r2``, ``rmse`` and ``components``: dicts with center, fwhm, height,
    area, eta, their standard errors (``*_err``, from the covariance of the fit), ``share``
    (fraction of the summed area) and ``snr`` (height over the noise).

    Diagnostics: ``noise`` (σ from second differences of the data), ``chi2_red`` (≈ 1 when the
    residual is as large as the noise), ``autocorr`` (lag-1 autocorrelation of the residual; near 0
    for noise, towards 1 for waves), ``max_correlation`` (largest |correlation| between two
    parameters) and ``warnings``, the codes of the checks that failed.
    """
    x, y = _array(x), _array(y)
    if model not in PEAK_SHAPES:
        raise ValueError(f"Unknown peak model: {model}")
    if background not in FIT_BACKGROUNDS:
        raise ValueError(f"Unknown background: {background}")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if window is not None:
        low, high = sorted((float(window[0]), float(window[1])))
        inside = (x >= low) & (x <= high)
        x, y = x[inside], y[inside]
    centers = [float(c) for c in centers]
    result = {"ok": False, "message": "", "model": model, "background_kind": background, "x": x, "y": y,
              "fit": None, "background": None, "curves": [], "residual": None, "r2": math.nan, "rmse": math.nan,
              "components": [], "noise": math.nan, "chi2_red": math.nan, "autocorr": math.nan,
              "max_correlation": math.nan, "warnings": []}
    n = len(centers)
    per_peak = 4 if model == "pvoigt" else 3
    n_background = FIT_BACKGROUNDS.index(background)
    if n == 0:
        result["message"] = "no_centers"
        return result
    if n > MAX_FIT_PEAKS:
        result["message"] = "too_many"
        return result
    if len(x) < per_peak * n + n_background + 2:
        result["message"] = "too_few_points"
        return result
    if any(not (x[0] <= c <= x[-1]) for c in centers):
        result["message"] = "center_outside"
        return result
    step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    span_x = float(x[-1] - x[0])
    span_y = float(np.ptp(y))
    if span_y <= 0 or span_x <= 0:
        result["message"] = "flat"
        return result
    x_mid = float(np.mean(x))
    floor = float(np.min(y))
    shift = max(float(max_shift), step)
    width0 = min(max(span_x / (4.0 * n), 4 * step), span_x / 2)
    p0, lower, upper = [], [], []
    for c in centers:
        height0 = max(float(np.interp(c, x, y)) - (floor if background != "none" else 0.0), 0.05 * span_y)
        p0 += [height0, c, width0] + ([0.5] if model == "pvoigt" else [])
        lower += [0.0, max(c - shift, float(x[0])), 2 * step] + ([0.0] if model == "pvoigt" else [])
        upper += [np.inf, min(c + shift, float(x[-1])), span_x] + ([1.0] if model == "pvoigt" else [])
    if n_background >= 1:
        p0.append(floor); lower.append(-np.inf); upper.append(np.inf)
    if n_background == 2:
        p0.append(0.0); lower.append(-np.inf); upper.append(np.inf)

    def split(params):
        peaks = [params[k * per_peak:(k + 1) * per_peak] for k in range(n)]
        return peaks, params[n * per_peak:]

    def background_curve(xs, tail):
        curve = np.zeros(len(xs))
        if len(tail) >= 1:
            curve = curve + tail[0]
        if len(tail) == 2:
            curve = curve + tail[1] * (xs - x_mid)
        return curve

    def total(xs, *params):
        peaks, tail = split(params)
        out = background_curve(xs, tail)
        for peak in peaks:
            out = out + peak_shape(xs, model, *peak)
        return out

    try:
        params, covariance = curve_fit(total, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        result["message"] = f"no_convergence: {exc}"
        return result
    peaks, tail = split(params)
    fit = total(x, *params)
    residual = y - fit
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    errors_ok = np.all(np.isfinite(covariance))
    components = []
    for k, peak in enumerate(peaks):
        height, center, fwhm = (float(v) for v in peak[:3])
        eta = float(peak[3]) if model == "pvoigt" else (1.0 if model == "lorentzian" else 0.0)
        factor = _area_factor(model, eta)
        area = factor * height * fwhm
        i0 = k * per_peak
        limits = []                    # parameters that ended on (or next to) a bound
        if abs(center - centers[k]) >= 0.98 * shift or center <= lower[i0 + 1] + 1e-9 * span_x or center >= upper[i0 + 1] - 1e-9 * span_x:
            limits.append("center")
        if fwhm >= 0.98 * span_x or fwhm <= 2.05 * step:
            limits.append("fwhm")
        if height <= 1e-9 * span_y:
            limits.append("height")
        if model == "pvoigt" and (eta <= 1e-3 or eta >= 1 - 1e-3):
            limits.append("eta")
        entry = {"center": center, "fwhm": fwhm, "height": height, "area": area, "eta": eta,
                 "center_err": math.nan, "fwhm_err": math.nan, "height_err": math.nan, "area_err": math.nan,
                 "eta_err": math.nan, "start": centers[k], "limits": limits,
                 "at_limit": bool(limits)}
        if errors_ok:
            block = covariance[i0:i0 + per_peak, i0:i0 + per_peak]
            sigma = np.sqrt(np.clip(np.diag(block), 0, None))
            entry.update(height_err=float(sigma[0]), center_err=float(sigma[1]), fwhm_err=float(sigma[2]))
            gradient = [factor * fwhm, 0.0, factor * height]
            if model == "pvoigt":
                entry["eta_err"] = float(sigma[3])
                gradient.append((_LORENTZ_AREA - _GAUSS_AREA) * height * fwhm)
            gradient = np.asarray(gradient)
            entry["area_err"] = float(math.sqrt(max(float(gradient @ block @ gradient), 0.0)))
            # A covariance-based error describes a parameter free to move both ways; on a bound it does not.
            for name in limits:
                entry[f"{name}_err"] = math.nan
            if {"height", "fwhm", "eta"} & set(limits):
                entry["area_err"] = math.nan
        components.append(entry)
    total_area = sum(c["area"] for c in components)
    noise = estimate_noise(y)
    for entry in components:
        entry["share"] = entry["area"] / total_area if total_area > 0 else math.nan
        entry["snr"] = entry["height"] / noise if noise > 0 else math.inf
    # ---- diagnostics: is the fit as good as the noise allows, and are the parameters determined?
    freedom = len(x) - len(params)
    chi2_red = ss_res / freedom / noise ** 2 if freedom > 0 and noise > 0 else math.nan
    centred = residual - np.mean(residual)
    denominator = float(np.sum(centred ** 2))
    autocorr = float(np.sum(centred[1:] * centred[:-1]) / denominator) if denominator > 0 else math.nan
    max_correlation = math.nan
    if errors_ok and len(params) > 1:
        sigma_all = np.sqrt(np.clip(np.diag(covariance), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = covariance / np.outer(sigma_all, sigma_all)
        off_diagonal = np.abs(correlation[~np.eye(len(params), dtype=bool)])
        off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
        max_correlation = float(np.max(off_diagonal)) if len(off_diagonal) else math.nan
    warnings = []
    if not errors_ok:
        warnings.append("no_errors")
    if math.isfinite(chi2_red) and chi2_red > CHI2_HIGH:
        warnings.append("misfit")
    if math.isfinite(autocorr) and autocorr > AUTOCORR_HIGH:
        warnings.append("structured_residual")
    if math.isfinite(max_correlation) and max_correlation > CORRELATION_HIGH:
        warnings.append("correlated")
    if any(c["at_limit"] for c in components):
        warnings.append("at_limit")
    if any(c["snr"] < SNR_LOW for c in components):
        warnings.append("weak_component")
    if any(math.isfinite(c["area_err"]) and c["area"] > 0 and c["area_err"] / c["area"] > RELATIVE_ERROR_HIGH for c in components):
        warnings.append("uncertain_area")
    if any(c["fwhm"] < 4 * step for c in components):
        warnings.append("narrow_component")
    result.update(ok=True, fit=fit, background=background_curve(x, tail),
                  curves=[peak_shape(x, model, *peak) for peak in peaks], residual=residual,
                  r2=1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
                  rmse=math.sqrt(ss_res / len(x)), components=components, noise=noise, chi2_red=chi2_red,
                  autocorr=autocorr, max_correlation=max_correlation, warnings=warnings)
    return result


def parse_centers(text: str) -> list[float]:
    """Numbers separated by commas, spaces, semicolons or line breaks; anything else is skipped."""
    values = []
    for token in str(text or "").replace("、", ",").replace("，", ",").replace("；", ",").replace(";", ",").replace(",", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def process(x, y, *, x_range=None, spikes: float | None = None, smooth_window: int = 0,
            smooth_order: int = 2, baseline_method: str = "none", baseline_params: dict | None = None,
            normalize_mode: str = "none", normalize_window=None):
    """The whole chain in the app's order: crop → spikes → smoothing → baseline → normalisation.

    Returns (x, y, info) where info reports the number of replaced spikes and their details
    (``spike_points``, see spike_report), the subtracted baseline and the cropped data as measured
    (``raw``), both on the cropped x, the normalisation as ``scale`` and ``shift`` (and
    ``normalize_problem`` when it could not be applied), and ``noise``: the point-to-point noise of
    the data before smoothing, on the scale of the returned spectrum.
    """
    x, y = _array(x), _array(y)
    if x_range is not None:
        x, y = crop(x, y, *x_range)
    info = {"spikes": 0, "baseline": None, "raw": y.copy(), "scale": 1.0, "shift": 0.0, "spike_points": [],
            "noise": math.nan, "normalize_problem": None, "smoothed": False}
    if len(y) == 0:
        return x, y, info
    if spikes:
        cleaned, info["spikes"] = remove_spikes(y, spikes)
        info["spike_points"] = spike_report(x, y, cleaned)
        y = cleaned
    # Noise of the data before smoothing (smoothing hides it); scaled with the normalisation below.
    noise = estimate_noise(y)
    if smooth_window and smooth_window >= 3:
        info["smoothed"] = True
        y = smooth(y, smooth_window, smooth_order)
    if baseline_method != "none":
        base = baseline(x, y, baseline_method, **(baseline_params or {}))
        info["baseline"] = base
        y = y - base
    if normalize_mode != "none":
        before = y
        if normalize_mode == "window" and normalize_window is not None and \
                not ((x >= min(normalize_window)) & (x <= max(normalize_window))).any():
            info["normalize_problem"] = "window_empty"         # left unnormalised, reported by the QC
        else:
            y = normalize(y, normalize_mode, x, normalize_window)
            if np.ptp(before) == 0:
                info["normalize_problem"] = "flat"               # nothing to scale by
        # Normalisation is y → scale·y + shift; the app uses the pair to draw the measured data and
        # the subtracted baseline on the scale of the processed spectrum.
        if np.ptp(before) > 0:
            info["scale"] = float(np.ptp(y) / np.ptp(before))
            info["shift"] = float(np.mean(y) - info["scale"] * np.mean(before))
    info["noise"] = float(noise * abs(info["scale"])) if math.isfinite(noise) else math.nan
    return x, y, info
