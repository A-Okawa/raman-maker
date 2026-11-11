"""Write the synthetic spectra in tests/data (three Lorentzian peaks on a sloping background)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
X = np.arange(200.0, 1000.001, 0.5)


def spectrum(peaks, seed=0, noise=20.0, base=300.0):
    rng = np.random.default_rng(seed)
    y = base + 0.05 * (X - 200) + noise * rng.standard_normal(len(X))
    for centre, height, hwhm in peaks:
        y += height * hwhm ** 2 / ((X - centre) ** 2 + hwhm ** 2)
    return y


def jasco_text(name, y, delimiter):
    lines = ["TITLE" + delimiter + name, "DATA TYPE" + delimiter + "RAMAN SPECTRUM", "ORIGIN" + delimiter + "JASCO",
             "OWNER" + delimiter, "DATE" + delimiter + "26/05/29", "TIME" + delimiter + "14:03:21",
             "SPECTROMETER/DATA SYSTEM" + delimiter + "NRS-7200", "LOCALE" + delimiter + "1041", "RESOLUTION" + delimiter,
             "DELTAX" + delimiter + "-0.5", "XUNITS" + delimiter + "cm-1", "YUNITS" + delimiter + "Int.",
             "FIRSTX" + delimiter + "1000", "LASTX" + delimiter + "200", "NPOINTS" + delimiter + str(len(X)),
             "FIRSTY" + delimiter + f"{y[-1]:.4f}", "MAXY" + delimiter + f"{y.max():.4f}", "MINY" + delimiter + f"{y.min():.4f}",
             "XYDATA"]
    lines += [f"{xi:g}{delimiter}{yi:.4f}" for xi, yi in zip(X[::-1], y[::-1])]          # descending, as exported
    lines += ["", "##### Extended Information", "[Comments]", "Sample name" + delimiter + name, "[Detailed Information]",
              "Laser wavelength" + delimiter + "532.11 nm", "Exposure time" + delimiter + "20 sec",
              "Accumulation" + delimiter + "10", "Objective" + delimiter + "MPLFLN 20x", "Grating" + delimiter + "2400 l/mm",
              "Laser power" + delimiter + "6.6 mW", "Wavenumber shift correction" + delimiter + "0.90 cm-1"]
    return "\r\n".join(lines) + "\r\n"


# ---- FT-IR: the JCAMP-like export (## header, then x y rows) and JCAMP-DX (X++(Y..Y)) in AFFN and DIFDUP form
XW = np.linspace(400.0, 4000.0, 1801)        # wavenumber, 2 cm-1 steps


def ir_transmittance(seed=5):
    """A transmittance spectrum (%): Si–O–Si at 1080, O–H around 3400, a C–H doublet; 95 % baseline."""
    rng = np.random.default_rng(seed)
    t = 95 - 60 * np.exp(-0.5 * ((XW - 1080) / 25) ** 2) - 30 * np.exp(-0.5 * ((XW - 3400) / 120) ** 2) \
        - 8 * np.exp(-0.5 * ((XW - 2920) / 10) ** 2) - 5 * np.exp(-0.5 * ((XW - 2850) / 10) ** 2)
    return t + 0.3 * rng.standard_normal(len(XW))


def ir_export_text(title, xunits, yunits, x, y):
    """The ``##KEY=VALUE`` + ``x \t y`` layout of the FT-IR text export (x descending, as exported)."""
    head = f"##TITLE={title}\r\n##DATA TYPE=INFRARED SPECTRUM\r\n##XUNITS={xunits}\r\n##YUNITS={yunits}\r\n"
    return head + "".join(f"{xi:.6f}  \t{yi:.6f} \r\n" for xi, yi in zip(x[::-1], y[::-1]))


_SQZ, _SQZN, _DIF, _DIFN, _DUP = "@ABCDEFGHI", "@abcdefghi", "%JKLMNOPQR", "%jklmnopqr", "STUVWXYZs"


def _sqz(v):
    d = str(abs(int(round(v))))
    return (_SQZ if v >= 0 else _SQZN)[int(d[0])] + d[1:]


def _dif(v):
    d = str(abs(int(round(v))))
    return (_DIF if v >= 0 else _DIFN)[int(d[0])] + d[1:]


def _dup(n):
    d = str(n)
    return _DUP[int(d[0]) - 1] + d[1:]


def jcamp_difdup(x, yi, per_line=10):
    """(X++(Y..Y)) lines in DIFDUP form: the first y of a line in SQZ form, differences in DIF form,
    repeats in DUP form, and each line after the first starting with the previous line's last y."""
    out, k = [], 0
    while k < len(yi):
        chunk, start = yi[k:k + per_line], k
        if k > 0:
            chunk, start = np.concatenate(([yi[k - 1]], chunk)), k - 1
        parts = [_sqz(chunk[0])]
        diffs, j = np.diff(chunk), 0
        while j < len(diffs):
            run = 1
            while j + run < len(diffs) and diffs[j + run] == diffs[j]:
                run += 1
            parts.append(_dif(diffs[j]) + (_dup(run) if run > 1 else ""))
            j += run
        out.append(f"{x[start]:.0f}" + "".join(parts))
        k += per_line
    return "\n".join(out)


def jcamp_text(title, x, y, yunits, compressed, yfactor=0.001):
    head = (f"##TITLE={title}\n##JCAMP-DX=4.24\n##DATA TYPE=INFRARED SPECTRUM\n##XUNITS=1/CM\n##YUNITS={yunits}\n"
            f"##XFACTOR=1.0\n##YFACTOR={yfactor if compressed else 1.0}\n##FIRSTX={x[0]:.0f}\n##LASTX={x[-1]:.0f}\n"
            f"##NPOINTS={len(x)}\n##DELTAX={x[1] - x[0]:.1f}\n##XYDATA=(X++(Y..Y))\n")
    if compressed:
        body = jcamp_difdup(x, np.round(y / yfactor).astype(int))
    else:
        lines, per_line = [], 6
        for k in range(0, len(x), per_line):
            lines.append(" ".join([f"{x[k]:.1f}"] + [f"{v:.4f}" for v in y[k:k + per_line]]))
        body = "\n".join(lines)
    return head + body + "\n##END=\n"


def main():
    DATA.mkdir(exist_ok=True)
    t_percent = ir_transmittance()
    absorbance = -np.log10(t_percent / 100.0)
    (DATA / "ftir_abs.txt").write_bytes(ir_export_text("sample A", "1/CM", "Abs", XW, absorbance).encode("utf-8"))
    (DATA / "ftir_pct.txt").write_bytes(ir_export_text("sample T", "1/CM", "%T", XW, t_percent).encode("utf-8"))
    xd = XW[::-1]                                              # JCAMP-DX files usually run 4000 → 400
    (DATA / "ftir_affn.jdx").write_text(jcamp_text("affn", xd, t_percent[::-1], "TRANSMITTANCE", compressed=False))
    (DATA / "ftir_difdup.jdx").write_text(jcamp_text("difdup", xd, t_percent[::-1], "TRANSMITTANCE", compressed=True))
    y1 = spectrum([(796, 3000, 4), (924, 1500, 3), (954, 800, 3)], seed=0)
    y2 = spectrum([(797.5, 2800, 4), (924, 1600, 3), (954, 900, 3)], seed=1)
    y3 = spectrum([(799, 2500, 4), (925, 1700, 3), (955, 1000, 3)], seed=2)
    (DATA / "jasco_5SiC_asSintered.csv").write_text(jasco_text("5SiC as-sintered", y1, ","), encoding="cp932")
    (DATA / "jasco_5SiC_950C.txt").write_text(jasco_text("5SiC 950C", y2, "\t"), encoding="utf-8")
    (DATA / "renishaw_single.txt").write_text(
        "#Wave\t#Intensity\n" + "\n".join(f"{xi:.3f}\t{yi:.2f}" for xi, yi in zip(X[::-1], y3[::-1])) + "\n")
    rows = []
    for px, py in [(0.0, 0.0), (1.3, 0.0), (0.0, 1.3)]:
        yy = spectrum([(796 + px, 3000, 4), (924, 1500, 3)], seed=3)
        rows += [f"{px:.2f}\t{py:.2f}\t{xi:.3f}\t{yi:.2f}" for xi, yi in zip(X[::-1], yy[::-1])]
    (DATA / "renishaw_map.txt").write_text("#X\t#Y\t#Wave\t#Intensity\n" + "\n".join(rows) + "\n")
    (DATA / "rruff_quartz.txt").write_text(
        "##NAMES=Quartz\n##RRUFFID=R040031\n##IDEAL CHEMISTRY=SiO_2_\n##LASER_WAVELENGTH=532\n##END=\n"
        + "\n".join(f"{xi:.4f}, {yi:.4f}" for xi, yi in zip(X, spectrum([(464, 5000, 3), (206, 1500, 5)], seed=4)))
        + "\n##END=\n")
    (DATA / "multi_columns.csv").write_text(
        "Raman shift (cm-1),as-sintered,750 C,950 C\n"
        + "\n".join(f"{xi:g},{a:.2f},{b:.2f},{c:.2f}" for xi, a, b, c in zip(X, y1, y2, y3)) + "\n")
    (DATA / "two_col.xy").write_text("\n".join(f"{xi:.2f} {yi:.3f}" for xi, yi in zip(X, y1)) + "\n")
    (DATA / "decimal_comma.csv").write_text(
        "\n".join(f"{xi:.2f};{yi:.3f}".replace(".", ",") for xi, yi in zip(X, y1)) + "\n")
    with pd.ExcelWriter(DATA / "book.xlsx") as writer:
        pd.DataFrame({"shift": X, "A": y1, "B": y2}).to_excel(writer, sheet_name="shared_x", index=False)
        pd.DataFrame({"x1": X, "y1": y1, "x2": X + 1, "y2": y3}).to_excel(writer, sheet_name="pairs", index=False)


if __name__ == "__main__":
    main()
