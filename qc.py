"""Quality checks for Raman Maker: input data, processing, peaks and peak fits.

Each check returns an *item*: a dict with the section, the spectrum (and component) it concerns, a
check code, a status, the value it measured, whether it is required, and parameters for the text
(``qc_text.render``). Statuses:

- PASS            the value meets the criterion
- CAUTION         a minor point to be aware of
- WARNING         check or re-analyse before using the result
- FAIL            the analysis failed or does not meet the criterion; no result is reported
- NOT EVALUATED   the check could not be made (reference values are also shown this way)

The overall status is the worst evaluated status, so a problem is never offset by other items that
pass. If every evaluated item passes but a required item could not be evaluated, the overall
status is NOT EVALUATED rather than PASS.

A program that ran without error is not the same as a result that is scientifically sound; these
checks look for the second only as far as the listed criteria go. They do not judge peak
assignments or whether a baseline is physically right.

The thresholds are provisional (see THRESHOLDS for the reasoning) and can be changed by the user.
Nothing here modifies data.
"""

from __future__ import annotations

import math

import numpy as np

from raman_processing import estimate_noise

PASS, CAUTION, WARNING, FAIL, NOT_EVALUATED = "PASS", "CAUTION", "WARNING", "FAIL", "NOT EVALUATED"
LEVELS = (PASS, CAUTION, WARNING, FAIL, NOT_EVALUATED)
_RANK = {PASS: 0, CAUTION: 1, WARNING: 2, FAIL: 3}

# key: (default, minimum, maximum, basis). Every value is provisional and can be changed in the app.
THRESHOLDS = {
    "snr": (3.0, 1.0, 20.0,
            "Height (or prominence) over the noise σ. 3 is the usual detection limit (3σ)."),
    "chi2_caution": (3.0, 1.0, 100.0,
                     "Reduced χ² is about 1 when the residual is as large as the noise. The noise is estimated "
                     "from the data (second differences), so some margin is allowed before CAUTION."),
    "chi2_warning": (10.0, 1.0, 1000.0,
                     "Reduced χ² above this: the residual is several times the noise; the model misses structure."),
    "autocorr": (0.5, 0.05, 0.99,
                 "Lag-1 autocorrelation of the residual. White noise gives about 0 (within ±2/√n); waves "
                 "left by a missing component or a wrong shape push it towards 1."),
    "correlation": (0.98, 0.5, 0.9999,
                    "Largest |correlation| between two fitted parameters. Near 1 the two trade off and their "
                    "individual values are not determined by the data."),
    "area_rel_err": (0.2, 0.01, 1.0,
                     "Standard error of a component area relative to the area. Above 20 % area ratios are "
                     "hard to compare between spectra."),
    "spike_neighbour": (0.2, 0.01, 1.0,
                        "A cosmic-ray spike occupies one point and leaves its neighbours at the local level; a "
                        "real peak a few points wide lifts them (a Lorentzian of FWHM 4 cm⁻¹ sampled every "
                        "1 cm⁻¹ gives about 0.8)."),
    "step_jump": (1.5, 1.05, 10.0,
                  "Ratio of two adjacent point spacings. Spectrometer dispersion changes the spacing slowly; a "
                  "jump means joined or edited data. Smoothing, ALS, spike removal and the noise estimate "
                  "work on point indices and assume even spacing."),
    "gap": (5.0, 1.5, 100.0,
            "A spacing this many times the local median spacing is treated as a missing range."),
    "absorbance_high": (3.0, 1.0, 8.0,
                        "Absorbance above 3 means less than 0.1 % of the light is transmitted; detector noise "
                        "and stray light dominate and band shapes are unreliable."),
}


def default_thresholds() -> dict:
    return {key: value[0] for key, value in THRESHOLDS.items()}


def _thresholds(th: dict | None) -> dict:
    out = default_thresholds()
    for key, value in (th or {}).items():
        if key in out and isinstance(value, (int, float)) and math.isfinite(value):
            out[key] = float(value)
    return out


def item(section: str, subject: str, code: str, status: str, value=None, *, required: bool = False,
         component: int | None = None, **params) -> dict:
    assert status in LEVELS, status
    return {"section": section, "subject": subject, "component": component, "code": code, "status": status,
            "value": value, "required": required, "params": params}


def overall(items: list[dict]) -> str:
    """The worst evaluated status; NOT EVALUATED when a required check is missing and nothing is worse
    than PASS, or when nothing was evaluated."""
    evaluated = [i["status"] for i in items if i["status"] != NOT_EVALUATED]
    worst = max(evaluated, key=_RANK.get) if evaluated else None
    if worst in (CAUTION, WARNING, FAIL):
        return worst
    if worst is None or any(i["required"] and i["status"] == NOT_EVALUATED for i in items):
        return NOT_EVALUATED
    return PASS


def counts(items: list[dict]) -> dict:
    return {level: sum(1 for i in items if i["status"] == level) for level in LEVELS}


# ------------------------------------------------------------------------------------- formatting
def fmt(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def where(values, digits: int = 1, limit: int = 8) -> str:
    """'520.3, 1001.2, … (+4)' for a list of x values."""
    values = [v for v in values if v is not None]
    text = ", ".join(fmt(v, digits) for v in values[:limit])
    return text + (f", … (+{len(values) - limit})" if len(values) > limit else "")


def ranges(x, mask, digits: int = 1, limit: int = 6) -> str:
    """Contiguous runs of ``mask`` as 'a–b' on x."""
    x, mask = np.asarray(x, dtype=float), np.asarray(mask, dtype=bool)
    runs, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            runs.append(fmt(x[k], digits) if j == k else f"{fmt(x[k], digits)}–{fmt(x[j], digits)}")
            k = j + 1
        else:
            k += 1
    text = ", ".join(runs[:limit])
    return text + (f", … (+{len(runs) - limit})" if len(runs) > limit else "")


def lines(entries, limit: int = 10) -> str:
    """'9, 14, 15' for recorded line numbers."""
    numbers = [str(e[0] if isinstance(e, (tuple, list)) else e) for e in entries]
    text = ", ".join(numbers[:limit])
    return text + (f", … (+{len(numbers) - limit})" if len(numbers) > limit else "")


# ------------------------------------------------------------------------------------ input data
def input_checks(subject: str, x, y, metadata: dict, *, kind: str = "raman", units: str = "absorbance",
                 absorbance=None, th: dict | None = None) -> list[dict]:
    """Checks of one spectrum as read (before any processing).

    ``y`` is the data as stored; for FT-IR ``units`` says what they are and ``absorbance`` is the
    converted signal (used for the noise and the high-absorbance check)."""
    th = _thresholds(th)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    read = dict(metadata.get("_qc_read") or {})
    out = []
    add = out.append
    n = len(x)
    add(item("input", subject, "points", FAIL if n < 5 else PASS, n, required=True))
    add(item("input", subject, "x_range", PASS, f"{fmt(x[0])}–{fmt(x[-1])} cm⁻¹" if n else "—"))

    skipped = read.get("skipped") or []
    if "skipped_count" in read:
        count = int(read.get("skipped_count") or 0)
        add(item("input", subject, "skipped_lines", WARNING if count else PASS, count,
                 lines=lines(skipped), text=" | ".join(t for _, t, _ in skipped[:3])))
    footer = read.get("footer") or []
    if footer:
        numeric_after = [f for f in footer if f[2] == "after_data"]
        add(item("input", subject, "footer", CAUTION if numeric_after else PASS, int(read.get("footer_count") or len(footer)),
                 lines=lines(footer), numeric=len(numeric_after)))
    if read.get("cut_count"):
        add(item("input", subject, "cut_rows", CAUTION, int(read["cut_count"]), lines=lines(read.get("cut_rows") or [])))
    nonfinite = int(read.get("nonfinite_count") or 0)
    add(item("input", subject, "nonfinite", WARNING if nonfinite else PASS, nonfinite,
             where=", ".join(f"#{k}" + (f" (x={fmt(v)})" if v is not None else "") for k, v in (read.get("nonfinite") or [])[:8])))
    declared = read.get("declared_points")
    in_file = read.get("points_in_file")
    if declared is not None and in_file is not None:
        add(item("input", subject, "declared_points", PASS if int(declared) == int(in_file) else WARNING,
                 f"{in_file} / {declared}", declared=declared, read=in_file))
    if metadata.get("x_units_warning"):
        add(item("input", subject, "x_units", CAUTION, str(metadata["x_units_warning"]), units=str(metadata["x_units_warning"])))
    if metadata.get("x_converted_from"):
        add(item("input", subject, "x_converted", PASS, str(metadata["x_converted_from"]), units=str(metadata["x_converted_from"])))

    if n >= 2:
        steps = np.diff(x)
        duplicate = steps == 0
        add(item("input", subject, "duplicates", CAUTION if duplicate.any() else PASS, int(duplicate.sum()),
                 where=where(x[1:][duplicate])))
        positive = steps[steps > 0]
        if len(positive) >= 3:
            # local median spacing (11 steps), so a slowly changing dispersion is not flagged
            padded = np.pad(positive, 5, mode="edge")
            local = np.array([np.median(padded[k:k + 11]) for k in range(len(positive))])
            xs = x[1:][steps > 0]
            gap = positive > th["gap"] * local
            ratio = positive[1:] / positive[:-1]
            jump = np.zeros(len(positive), dtype=bool)
            for k in np.flatnonzero(np.maximum(ratio, 1 / ratio) > th["step_jump"]):
                jump[k + 1 if ratio[k] > 1 else k] = True     # mark the larger of the two steps once
            jump &= ~gap
            add(item("input", subject, "gaps", WARNING if gap.any() else PASS, int(gap.sum()),
                     where=where(xs[gap]), median_step=fmt(float(np.median(positive)), 2)))
            add(item("input", subject, "spacing", CAUTION if jump.any() else PASS,
                     fmt(float(np.max(np.maximum(ratio, 1 / ratio))), 2) if len(ratio) else "—",
                     where=where(xs[jump]), median_step=fmt(float(np.median(positive)), 2)))

    signal = np.asarray(absorbance, dtype=float) if absorbance is not None else y
    noise = estimate_noise(signal) if n >= 5 else math.nan
    if n >= 5 and math.isfinite(noise):
        top = float(np.max(signal)) - float(np.median(signal))
        snr = 0.0 if top <= 0 else top / noise if noise > 0 else math.inf      # a flat line has no signal
        add(item("input", subject, "snr", WARNING if snr < th["snr"] else PASS, fmt(snr, 1) if math.isfinite(snr) else "∞",
                 required=True, noise=fmt(noise, 4 if noise < 1 else 2)))
    else:
        add(item("input", subject, "snr", NOT_EVALUATED, None, required=True, reason="too_few"))

    if kind == "raman" and n >= 5 and math.isfinite(noise):
        negative = y < -3 * noise
        add(item("input", subject, "negative", CAUTION if negative.any() else PASS, int(negative.sum()),
                 minimum=fmt(float(np.min(y)), 2), where=ranges(x, negative)))
    if n >= 3:
        top = np.max(signal)
        at_top = signal == top
        run, best = 0, 0
        for flag in at_top:
            run = run + 1 if flag else 0
            best = max(best, run)
        add(item("input", subject, "saturation", WARNING if best >= 3 else PASS, best, where=ranges(x, at_top)))
    if kind == "ftir":
        if units in ("transmittance_percent", "transmittance_fraction"):
            limit = 100.0 if units == "transmittance_percent" else 1.0
            over = y > limit * 1.005
            add(item("input", subject, "transmittance_over", CAUTION if over.any() else PASS, int(over.sum()),
                     where=ranges(x, over)))
        if absorbance is not None:
            high = signal > th["absorbance_high"]
            add(item("input", subject, "absorbance_high", CAUTION if high.any() else PASS, int(high.sum()),
                     where=ranges(x, high), limit=fmt(th["absorbance_high"], 1)))
    return out


# ------------------------------------------------------------------------------------ processing
def processing_checks(subject: str, info: dict, *, points_used: int, x=None, clipped=None,
                      normalize_mode: str = "none", normalize_window=None, th: dict | None = None) -> list[dict]:
    """Checks of what processing did to one spectrum (info from raman_processing.process)."""
    th = _thresholds(th)
    out = [item("processing", subject, "points_used", FAIL if points_used < 5 else PASS, points_used, required=True)]
    spikes = info.get("spike_points") or []
    if info.get("spikes_on"):
        suspect = [p for p in spikes if math.isfinite(p["neighbour_ratio"]) and p["neighbour_ratio"] >= th["spike_neighbour"]]
        out.append(item("processing", subject, "spikes", WARNING if suspect else PASS, len(spikes),
                        where=where([p["x"] for p in spikes]), suspect=where([p["x"] for p in suspect]),
                        suspect_count=len(suspect)))
    problem = info.get("normalize_problem")
    if normalize_mode != "none":
        out.append(item("processing", subject, "normalization", WARNING if problem else PASS, normalize_mode,
                        problem=problem or "", window=f"{fmt(min(normalize_window))}–{fmt(max(normalize_window))}"
                        if normalize_window is not None else ""))
    if clipped is not None and x is not None:
        clipped = np.asarray(clipped, dtype=bool)
        out.append(item("processing", subject, "clipped", WARNING if clipped.any() else PASS, int(clipped.sum()),
                        where=ranges(x, clipped)))
    return out


# ------------------------------------------------------------------------------------------ peaks
def peak_checks(subject: str, x, signal, indices, prominences, noise: float, *, fits=None,
                th: dict | None = None) -> list[dict]:
    """Peak S/N (prominence over the noise of the data before smoothing) and single-peak fits."""
    th = _thresholds(th)
    x = np.asarray(x, dtype=float)
    out = []
    if not len(indices):
        return [item("peaks", subject, "peak_snr", PASS, 0, where="")]
    if not (math.isfinite(noise) and noise > 0):
        out.append(item("peaks", subject, "peak_snr", NOT_EVALUATED, None, required=True, reason="no_noise"))
    else:
        snr = np.asarray(prominences, dtype=float) / noise
        weak = snr < th["snr"]
        out.append(item("peaks", subject, "peak_snr", WARNING if weak.any() else PASS, int(weak.sum()), required=True,
                        where=where(x[np.asarray(indices)[weak]]), total=len(indices)))
    if fits is not None:
        failed = [x[k] for k, f in zip(indices, fits) if not f["ok"]]
        limited = [x[k] for k, f in zip(indices, fits) if f["ok"] and f.get("limits")]
        out.append(item("peaks", subject, "peakfit_limit", WARNING if limited else PASS, len(limited), where=where(limited)))
        out.append(item("peaks", subject, "peakfit_failed", CAUTION if failed else PASS, len(failed), where=where(failed)))
    return out


def peak_snr_values(prominences, noise: float):
    """S/N of each peak (NaN when the noise is unknown)."""
    prominences = np.asarray(prominences, dtype=float)
    if not (math.isfinite(noise) and noise > 0):
        return np.full(len(prominences), math.nan)
    return prominences / noise


# ------------------------------------------------------------------------------------------- fits
def fit_checks(subject: str, result: dict, *, smoothed: bool = False, noise_before_smoothing: float = math.nan,
               th: dict | None = None) -> list[dict]:
    """Checks of one multi-peak fit (raman_processing.fit_multipeak).

    On smoothed data the fit's own noise estimate is too small, so χ², the residual autocorrelation
    and the errors are not evaluated; component heights are compared with the noise of the data
    before smoothing (``noise_before_smoothing``) when it is known."""
    th = _thresholds(th)
    out = []
    add = out.append
    if not result.get("ok"):
        add(item("fit", subject, "fit_status", FAIL, None, required=True, message=str(result.get("message", ""))))
        return out
    add(item("fit", subject, "fit_status", PASS, None, required=True, message=""))
    n_params = len(result["components"]) * (4 if result.get("model") == "pvoigt" else 3) + \
        {"none": 0, "constant": 1, "linear": 2}.get(result.get("background_kind"), 0)
    add(item("fit", subject, "dof", PASS, f"{len(result['x'])} / {n_params}", points=len(result["x"]), parameters=n_params))
    chi2 = result.get("chi2_red", math.nan)
    if smoothed:
        add(item("fit", subject, "chi2", NOT_EVALUATED, fmt(chi2, 2), required=True, reason="smoothed"))
        add(item("fit", subject, "autocorr", NOT_EVALUATED, fmt(result.get("autocorr"), 2), required=True, reason="smoothed"))
    else:
        if math.isfinite(chi2):
            status = PASS if chi2 <= th["chi2_caution"] else CAUTION if chi2 <= th["chi2_warning"] else WARNING
            add(item("fit", subject, "chi2", status, fmt(chi2, 2), required=True))
        else:
            add(item("fit", subject, "chi2", NOT_EVALUATED, None, required=True, reason="no_noise"))
        auto = result.get("autocorr", math.nan)
        if math.isfinite(auto):
            add(item("fit", subject, "autocorr", WARNING if auto > th["autocorr"] else PASS, fmt(auto, 2), required=True))
        else:
            add(item("fit", subject, "autocorr", NOT_EVALUATED, None, required=True, reason="no_noise"))
    corr = result.get("max_correlation", math.nan)
    if math.isfinite(corr):
        add(item("fit", subject, "correlation", WARNING if corr >= th["correlation"] else PASS, fmt(corr, 3), required=True))
    else:
        single = n_params <= 1
        add(item("fit", subject, "correlation", PASS if single else NOT_EVALUATED, None, required=True,
                 reason="" if single else "no_errors"))
        if not single:
            add(item("fit", subject, "no_errors", WARNING, None))
    noise = result.get("noise", math.nan)
    for number, c in enumerate(result["components"], start=1):
        snr = c.get("snr", math.nan)
        if smoothed:
            snr = c["height"] / noise_before_smoothing if math.isfinite(noise_before_smoothing) and noise_before_smoothing > 0 else math.nan
        if math.isfinite(snr):
            add(item("fit", subject, "comp_snr", WARNING if snr < th["snr"] else PASS, fmt(snr, 1), component=number))
        else:
            add(item("fit", subject, "comp_snr", NOT_EVALUATED, None, component=number, reason="no_noise"))
        if c.get("limits"):
            add(item("fit", subject, "comp_limit", WARNING, ", ".join(c["limits"]), component=number,
                     parameters=", ".join(c["limits"])))
        else:
            add(item("fit", subject, "comp_limit", PASS, "", component=number))
        if smoothed:
            add(item("fit", subject, "area_err", NOT_EVALUATED, None, component=number, reason="smoothed"))
        elif math.isfinite(c.get("area_err", math.nan)) and c["area"] > 0:
            rel = c["area_err"] / c["area"]
            add(item("fit", subject, "area_err", CAUTION if rel > th["area_rel_err"] else PASS, f"{100 * rel:.1f} %",
                     component=number))
        else:
            add(item("fit", subject, "area_err", NOT_EVALUATED, None, component=number,
                     reason="at_limit" if c.get("limits") else "no_errors"))
        step = float(np.median(np.diff(np.sort(result["x"])))) if len(result["x"]) > 1 else math.nan
        if math.isfinite(step) and step > 0:
            add(item("fit", subject, "narrow", WARNING if c["fwhm"] < 4 * step else PASS, fmt(c["fwhm"] / step, 1),
                     component=number))
    add(item("fit", subject, "r2", NOT_EVALUATED, fmt(result.get("r2"), 5), reason="reference"))
    add(item("fit", subject, "rmse", NOT_EVALUATED, fmt(result.get("rmse"), 4), reason="reference", noise=fmt(noise, 4)))
    return out


def worst_of(items: list[dict], subject: str | None = None, section: str | None = None,
             component: int | None = None) -> str:
    """Overall status of the items for one spectrum / section / component (fit-wide items count for
    every component)."""
    chosen = [i for i in items if (subject is None or i["subject"] == subject)
              and (section is None or i["section"] == section)
              and (component is None or i["component"] in (None, component))]
    return overall(chosen)
