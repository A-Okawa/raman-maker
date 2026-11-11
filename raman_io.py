"""Offline readers for Raman spectra (one or more spectra per file).

Supported layouts
-----------------
- JASCO Spectra Manager text / CSV exports (``TITLE … XYDATA … ##### Extended Information``),
  as written by the NRS series with either tab- or comma-separated fields; the extended
  information (laser, exposure, accumulation, …) is kept as metadata.
- Renishaw WiRE text exports (``#Wave  #Intensity``; map exports with ``#X #Y #Wave #Intensity``
  become one spectrum per position) and WiRE ``.wdf`` files when the optional
  ``renishawWiRE`` package is installed (a single spectrum, or the spectra of a series / map,
  either averaged or read one by one).
- Files with a ``##KEY=VALUE`` header: FT-IR text exports (``##DATA TYPE=INFRARED SPECTRUM``,
  ``##YUNITS=Abs`` or ``%T``, then ``x y`` rows), RRUFF database text (``##NAMES=…``), and
  JCAMP-DX (``##XYDATA=(X++(Y..Y))`` with AFFN or SQZ/DIF/DUP rows, ``(XY..XY)`` pairs, one
  spectrum per ``##TITLE=`` block). Wavelength axes (µm, nm) are converted to cm⁻¹.
- Generic text: two columns (x, y), one x column with several y columns, or repeated
  (x, y) column pairs; tab, comma, semicolon or space separated; an optional header row
  names the spectra. ``.xy``, ``.txt``, ``.csv``, ``.dat``, ``.asc``, ``.prn``, ``.tsv``.
- Excel workbooks (``.xlsx``): every sheet, with the same column rules as generic text.

Intensities are returned as stored (absorbance or transmittance stay what the file says;
``native_y_units`` reports which). Raman shifts / wavenumbers are returned in ascending order.
Unsupported or damaged files raise ``RamanParseError`` instead of guessing a column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np


SUPPORTED_EXTENSIONS = ("csv", "txt", "xy", "dat", "asc", "prn", "tsv", "jdx", "dx", "jcamp", "wdf", "xlsx")
_MAX_POINTS = 2_000_000
_MAX_SPECTRA = 500          # spectra read from one file (maps are averaged beyond this, see multi=)
_DEFAULT_MULTI_LIMIT = 200  # spectra kept with multi="all"
_EXPORT_HELP = ("Export the spectrum as a JASCO text/CSV file, a Renishaw text export, "
                "or a two-column text/CSV file.")


class RamanParseError(ValueError):
    """The file is damaged, ambiguous, or uses an unsupported format."""


@dataclass
class RamanSpectrum:
    name: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> np.ndarray:
        return np.column_stack((self.x, self.y))


# --------------------------------------------------------------------------------------- helpers
def _text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data:
        raise RamanParseError("Binary data was found where a text file was expected. " + _EXPORT_HELP)
    for encoding in ("utf-8-sig", "cp932", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise RamanParseError("The text encoding could not be read. Save a UTF-8 text/CSV export.")


_NUMBER = re.compile(r"^[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+\-]?\d+)?$")


def _is_number(token: str) -> bool:
    return bool(_NUMBER.match(token.strip()))


def _number(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def _split(line: str, delimiter: str | None) -> list[str]:
    line = line.strip()
    if not line:
        return []
    if delimiter is None:
        return [t for t in re.split(r"[\s,;]+", line) if t]
    if delimiter == " ":
        return [t for t in re.split(r"\s+", line) if t]
    return [t.strip() for t in line.split(delimiter)]


def _detect_delimiter(lines: list[str]) -> str | None:
    """The field separator of the data rows: tab, comma, semicolon or whitespace."""
    sample = [line for line in lines[:50] if line.strip()]
    if not sample:
        return None
    if all("\t" in line for line in sample):
        return "\t"
    if all(";" in line for line in sample):
        return ";"
    if all("," in line for line in sample):
        # "1,5;2,3" would have been caught by the semicolon branch; a comma here separates fields.
        return ","
    return " "


def _decimal_comma(tokens: list[str]) -> list[str]:
    """Fields such as '1,5' (German exports) → '1.5' when no field contains a period."""
    if any("." in t for t in tokens):
        return tokens
    return [t.replace(",", ".") for t in tokens]


def _stem(filename: str) -> str:
    return Path(filename).stem or filename


def _spectrum(name: str, x, y, metadata: dict) -> RamanSpectrum:
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if len(x) != len(y) or not len(x):
        raise RamanParseError("A spectrum has missing or mismatched shift/intensity values.")
    if len(x) > _MAX_POINTS:
        raise RamanParseError("A spectrum has more than 2,000,000 points.")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        raise RamanParseError("A spectrum needs at least two finite points.")
    order = np.argsort(x, kind="stable")
    return RamanSpectrum(name, x[order], y[order], dict(metadata))


def _monotonic(values: np.ndarray) -> bool:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return False
    steps = np.diff(values)
    return bool(np.all(steps >= 0) or np.all(steps <= 0)) and len(np.unique(values)) >= 2


def _columns_to_spectra(filename: str, names: list[str], columns: np.ndarray, metadata: dict,
                        prefix: str = "") -> list[RamanSpectrum]:
    """Numeric columns → spectra.

    One monotonic x column followed by y columns shares the x; repeated (x, y) pairs each
    keep their own x. Column names (from a header row) name the spectra.
    """
    n_cols = columns.shape[1]
    if n_cols < 2:
        raise RamanParseError("At least two numeric columns (Raman shift and intensity) are needed.")
    mono = [_monotonic(columns[:, j]) for j in range(n_cols)]
    if not mono[0]:
        raise RamanParseError("The first numeric column must be the Raman shift (monotonic). " + _EXPORT_HELP)
    spectra, x_index, j = [], 0, 1
    while j < n_cols:
        if mono[j] and j + 1 < n_cols and not mono[j + 1]:
            x_index = j          # a new (x, y) pair
            j += 1
            continue
        x, y = columns[:, x_index], columns[:, j]
        if n_cols == 2 or (names and names[j]):
            label = names[j] if names and names[j] else _stem(filename)
            name = _stem(filename) if n_cols == 2 else f"{prefix}{label}"
        else:
            name = f"{prefix}{_stem(filename)} [{len(spectra) + 1}]"
        meta = dict(metadata)
        if names and names[j]:
            meta["y_name"] = names[j]
        if names and names[x_index]:
            meta["x_name"] = names[x_index]
        spectra.append(_spectrum(name, x, y, meta))
        j += 1
    if not spectra:
        raise RamanParseError("No intensity column was found next to the Raman shift column.")
    return spectra


# ----------------------------------------------------------------------------------------- JASCO
_JASCO_MARK = re.compile(r"^\s*XYDATA\s*$", re.I)


def _is_jasco(text: str) -> bool:
    head = text[:8000]
    return bool(re.search(r"^\s*XYDATA\s*$", head, re.I | re.M)) and \
        bool(re.search(r"^\s*(TITLE|ORIGIN|XUNITS|NPOINTS|DELTAX)\b", head, re.I | re.M))


def _key_value(line: str) -> tuple[str, str]:
    for delimiter in ("\t", ","):
        if delimiter in line:
            key, value = line.split(delimiter, 1)
            return key.strip(), value.strip()
    parts = line.split(None, 1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (line.strip(), "")


def _read_jasco(text: str, filename: str) -> list[RamanSpectrum]:
    lines = text.splitlines()
    metadata: dict[str, Any] = {"format": "JASCO"}
    rows = []
    section = None
    state = "header"
    for line in lines:
        stripped = line.strip()
        if state == "header":
            if _JASCO_MARK.match(stripped):
                state = "data"
                continue
            if stripped:
                key, value = _key_value(stripped)
                if key:
                    metadata[key] = value
        elif state == "data":
            if not stripped:
                continue
            tokens = _split(stripped, None)
            if len(tokens) >= 2 and all(_is_number(t) for t in tokens[:2]):
                rows.append((_number(tokens[0]), _number(tokens[1])))
                continue
            state = "extended"
        if state == "extended":
            if not stripped or stripped.startswith("#####"):
                continue
            m = re.match(r"^\[(.+)\]$", stripped)
            if m:
                section = m.group(1).strip()
                continue
            key, value = _key_value(stripped)
            if key:
                metadata[f"{section}: {key}" if section else key] = value
    if not rows:
        raise RamanParseError("No XYDATA rows were found in the JASCO file.")
    data = np.asarray(rows, dtype=float)
    name = _stem(filename)
    x_units = str(metadata.get("XUNITS", ""))
    if x_units and not re.search(r"cm|raman|shift|wave", x_units, re.I):
        metadata["x_units_warning"] = x_units
    return [_spectrum(name, data[:, 0], data[:, 1], metadata)]


# ------------------------------------------------------------------------------- JCAMP-like (##KEY=)
# Files whose header is made of ``##KEY=VALUE`` lines: FT-IR text exports (``##TITLE= ##DATA TYPE=INFRARED
# SPECTRUM ##XUNITS=1/CM ##YUNITS=Abs`` followed by x y rows), RRUFF database text (``##NAMES= ##RRUFFID=``)
# and full JCAMP-DX files (``##XYDATA=(X++(Y..Y))`` with AFFN or SQZ/DIF/DUP-compressed rows, or
# ``(XY..XY)`` pairs). A file with several ``##TITLE=`` blocks gives one spectrum per block.
_SQZ = {c: v for v, c in enumerate("@ABCDEFGHI")}
_SQZ.update({c: -v for v, c in enumerate("@abcdefghi") if v})
_DIF = {c: v for v, c in enumerate("%JKLMNOPQR")}
_DIF.update({c: -v for v, c in enumerate("%jklmnopqr") if v})
_DUP = {c: v for v, c in enumerate("STUVWXYZs", start=1)}
_JCAMP_TOKEN = re.compile(r"[@A-Ia-i%J-Rj-rS-Zs][0-9.]*|[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?|\?")
# In a compressed line E/e are the SQZ digits 5/-5, never an exponent (compressed lines carry no exponents).
_JCAMP_TOKEN_COMPRESSED = re.compile(r"[@A-Ia-i%J-Rj-rS-Zs][0-9.]*|[+\-]?(?:\d+\.?\d*|\.\d+)|\?")
_JCAMP_COMPRESSED = re.compile(r"[@A-DF-Ia-df-i%J-Rj-rS-Zs]")


def _is_jcamp_like(text: str) -> bool:
    return text.lstrip().startswith("##")


def _jcamp_tokens(line: str) -> list[tuple[str, float]]:
    """(kind, value) for every number of a JCAMP data line: kind is 'num' (absolute, AFFN or SQZ),
    'dif' (difference to the previous y) or 'dup' (repeat count); '?' (missing) becomes NaN."""
    tokens = []
    line = line.replace(",", " ").replace(";", " ")
    pattern = _JCAMP_TOKEN_COMPRESSED if _JCAMP_COMPRESSED.search(line) else _JCAMP_TOKEN
    for match in pattern.finditer(line):
        token = match.group(0)
        head = token[0]
        if token == "?":
            tokens.append(("num", math.nan))
        elif head in _SQZ:
            digits = token[1:]
            value = float(str(abs(_SQZ[head])) + digits) if digits or _SQZ[head] else 0.0
            tokens.append(("num", -value if _SQZ[head] < 0 else value))
        elif head in _DIF:
            digits = token[1:]
            value = float(str(abs(_DIF[head])) + digits) if digits or _DIF[head] else 0.0
            tokens.append(("dif", -value if _DIF[head] < 0 else value))
        elif head in _DUP:
            tokens.append(("dup", float(str(_DUP[head]) + token[1:])))
        else:
            tokens.append(("num", float(token)))
    return tokens


def _decode_xpp(lines: list[str], x_factor: float, y_factor: float, first_x, last_x, n_points) -> tuple[np.ndarray, np.ndarray]:
    """``(X++(Y..Y))``: each line starts with the x of its first y; the y values are equally spaced."""
    ys, x_starts, line_counts, skipped = [], [], [], []
    previous_dif = False
    for line in lines:
        tokens = _jcamp_tokens(line)
        if not tokens:
            continue
        kind, x_value = tokens[0]
        if kind != "num":
            raise RamanParseError("A JCAMP-DX data line does not start with an x value.")
        values = []
        last_kind, last = "num", math.nan
        for kind, value in tokens[1:]:
            if kind == "num":
                values.append(value)
                last, last_kind = value, "num"
            elif kind == "dif":
                last = last + value
                values.append(last)
                last_kind = "dif"
                last_dif = value
            else:                                   # dup: repeat the previous value or difference
                count = int(value) - 1
                for _ in range(count):
                    if last_kind == "dif":
                        last = last + last_dif
                    values.append(last)
        dropped = 0
        if previous_dif and values:
            values = values[1:]                     # the first y of a line after DIF data is a check value
            dropped = 1
        previous_dif = any(kind == "dif" for kind, _ in tokens[1:])
        if not values:
            continue
        x_starts.append(x_value * x_factor)
        line_counts.append(len(values))
        skipped.append(dropped)
        ys.extend(values)
    if not ys:
        raise RamanParseError("No data values were found in the JCAMP-DX XYDATA block.")
    total = len(ys)
    if first_x is not None and last_x is not None and total > 1:
        step = (last_x - first_x) / (total - 1) if n_points in (None, total) or n_points < 2 else (last_x - first_x) / (n_points - 1)
    else:
        if len(x_starts) > 1:
            step = (x_starts[1] - x_starts[0]) / (line_counts[0] - skipped[1])   # a check value sits on the previous line's last x
        else:
            raise RamanParseError("The JCAMP-DX file lacks FIRSTX/LASTX, so the x spacing is unknown.")
    xs = []
    for start, count, dropped in zip(x_starts, line_counts, skipped):
        xs.extend(start + (k + dropped) * step for k in range(count))   # a dropped check value sat at x_value
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float) * y_factor


def _decode_xy_pairs(lines: list[str], x_factor: float, y_factor: float) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for line in lines:
        values.extend(value for kind, value in _jcamp_tokens(line) if kind == "num")
    if len(values) < 4:
        raise RamanParseError("No (x, y) pairs were found in the JCAMP-DX data.")
    if len(values) % 2:
        values = values[:-1]
    data = np.asarray(values, dtype=float).reshape(-1, 2)
    return data[:, 0] * x_factor, data[:, 1] * y_factor


def _finish_jcamp_block(filename: str, metadata: dict, plain_rows: list, data_lines: list, data_form: str | None,
                        block_index: int, block_count: int) -> RamanSpectrum | None:
    metadata = dict(metadata)
    x_factor = float(metadata.get("XFACTOR", 1) or 1) if _is_number(str(metadata.get("XFACTOR", "1"))) else 1.0
    y_factor = float(metadata.get("YFACTOR", 1) or 1) if _is_number(str(metadata.get("YFACTOR", "1"))) else 1.0

    def number_or_none(key):
        value = str(metadata.get(key, "")).strip()
        return _number(value) if _is_number(value) else None

    if data_form and data_lines:
        form = data_form.upper().replace(" ", "")
        if "X++" in form:
            x, y = _decode_xpp(data_lines, x_factor, y_factor, number_or_none("FIRSTX"), number_or_none("LASTX"),
                               number_or_none("NPOINTS"))
        elif "XY" in form:
            x, y = _decode_xy_pairs(data_lines, x_factor, y_factor)
        else:
            raise RamanParseError(f"Unsupported JCAMP-DX data form: {data_form}")
    elif plain_rows:
        data = np.asarray(plain_rows, dtype=float)
        x, y = data[:, 0], data[:, 1]
    else:
        return None
    units = str(metadata.get("XUNITS", "")).strip().upper()
    if re.search(r"MICROMETER|MICRON|^UM$|µM", units):
        with np.errstate(divide="ignore"):
            x = np.where(x > 0, 1e4 / x, np.nan)
        metadata["x_converted_from"] = metadata["XUNITS"]
    elif re.search(r"NANOMETER|^NM$", units):
        with np.errstate(divide="ignore"):
            x = np.where(x > 0, 1e7 / x, np.nan)
        metadata["x_converted_from"] = metadata["XUNITS"]
    elif units and not re.search(r"CM|RAMAN|SHIFT|WAVE", units):
        metadata["x_units_warning"] = metadata["XUNITS"]
    rruff_name = str(metadata.get("NAMES") or "").strip()
    rruff_id = str(metadata.get("RRUFFID") or "").strip()
    if rruff_name or rruff_id:
        metadata["format"] = "RRUFF"
        label = f"{rruff_name} ({rruff_id})" if rruff_name and rruff_id else rruff_name or rruff_id
    elif block_count > 1:
        label = str(metadata.get("TITLE") or "").strip() or f"{_stem(filename)} [{block_index + 1}]"
    else:
        label = _stem(filename)
    return _spectrum(label, x, y, metadata)


def _read_jcamp_like(text: str, filename: str) -> list[RamanSpectrum]:
    blocks: list[dict] = []
    current: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("$$", 1)[0].rstrip() if not raw_line.lstrip().startswith("$$") else ""
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("##"):
            key, _, value = stripped[2:].partition("=")
            key, value = key.strip(), value.strip()
            if not key:
                continue
            if key.upper() == "TITLE" and (current is None or current["seen_title"]):
                current = {"metadata": {"format": "JCAMP"}, "rows": [], "data_lines": [], "form": None,
                           "seen_title": True, "in_data": False}
                blocks.append(current)
            if current is None:
                current = {"metadata": {"format": "JCAMP"}, "rows": [], "data_lines": [], "form": None,
                           "seen_title": False, "in_data": False}
                blocks.append(current)
            upper = key.upper()
            if upper in ("XYDATA", "XYPOINTS", "PEAK TABLE", "DATA TABLE"):
                current["form"] = value
                current["in_data"] = True
                continue
            if upper == "END":
                current["in_data"] = False
                continue
            current["in_data"] = False
            if upper == "TITLE":
                current["seen_title"] = True
            current["metadata"][key] = value
            continue
        if current is None:
            continue
        if current["form"] is not None:
            if current["in_data"]:
                current["data_lines"].append(stripped)
            continue
        tokens = _split(stripped, None)
        if len(tokens) >= 2 and all(_is_number(t) for t in tokens[:2]):
            current["rows"].append((_number(tokens[0]), _number(tokens[1])))
    spectra = []
    for index, block in enumerate(blocks):
        spectrum = _finish_jcamp_block(filename, block["metadata"], block["rows"], block["data_lines"],
                                       block["form"], index, len(blocks))
        if spectrum is not None:
            spectra.append(spectrum)
    if not spectra:
        raise RamanParseError("No data rows were found after the ## header lines.")
    return spectra


# ----------------------------------------------------------------------------------- generic text
_COMMENT = re.compile(r"^\s*(#|//|%|!)")


def _read_generic_text(text: str, filename: str) -> list[RamanSpectrum]:
    lines = text.splitlines()
    comments, header, data_lines = [], None, []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _COMMENT.match(stripped):
            comments.append(stripped)
            continue
        if not data_lines and header is None:
            tokens = _split(stripped, None)
            if tokens and not _is_number(tokens[0]):
                header = stripped
                continue
        data_lines.append(stripped)
    if not data_lines:
        raise RamanParseError("No numeric rows were found. " + _EXPORT_HELP)
    delimiter = _detect_delimiter(data_lines)
    rows, width = [], None
    for line in data_lines:
        tokens = _decimal_comma(_split(line, delimiter))
        if not tokens or not all(_is_number(t) for t in tokens):
            if rows:
                break          # trailing text (a footer): stop at the first bad row after data
            continue           # leading junk before the data
        if width is None:
            width = len(tokens)
        if len(tokens) != width:
            if len(tokens) > width:
                tokens = tokens[:width]
            else:
                continue
        rows.append([_number(t) for t in tokens])
    if not rows or width is None or width < 2:
        raise RamanParseError("At least two numeric columns (Raman shift and intensity) are needed. " + _EXPORT_HELP)
    columns = np.asarray(rows, dtype=float)
    names = _column_names(header, comments, delimiter, width)
    metadata: dict[str, Any] = {"format": "text"}
    if names and any(re.search(r"wave|shift|raman", n, re.I) for n in names) and len(names) > 2:
        return _read_named_columns(filename, names, columns, metadata)
    return _columns_to_spectra(filename, names, columns, metadata)


def _column_names(header: str | None, comments: list[str], delimiter: str | None, width: int) -> list[str]:
    candidates = ([header] if header else []) + comments[::-1]
    for line in candidates:
        cleaned = re.sub(r"^\s*(#|//|%|!)+\s*", "", line)
        tokens = [t.lstrip("#").strip() for t in _split(cleaned, delimiter)]
        if delimiter == " " or delimiter is None:
            tokens = [t.lstrip("#").strip() for t in re.split(r"\s{2,}|\t|,|;", cleaned) if t.strip()]
            if len(tokens) != width:
                tokens = [t.lstrip("#").strip() for t in cleaned.split() if t.strip()]
        if len(tokens) == width and not all(_is_number(t) for t in tokens if t):
            return tokens
    return []


def _read_named_columns(filename: str, names: list[str], columns: np.ndarray, metadata: dict) -> list[RamanSpectrum]:
    """Renishaw-style exports: coordinate columns, then the shift and intensity columns."""
    lower = [n.lower() for n in names]
    x_col = next((j for j, n in enumerate(lower) if re.search(r"wave|shift|raman", n)), None)
    y_col = next((j for j, n in enumerate(lower) if re.search(r"intens|count|signal|absorb|transmit|reflect|%\s*t\b", n)), None)
    if x_col is None or y_col is None:
        return _columns_to_spectra(filename, names, columns, metadata)
    coordinate_cols = [j for j in range(columns.shape[1]) if j not in (x_col, y_col)]
    metadata = dict(metadata, x_name=names[x_col], y_name=names[y_col])
    if not coordinate_cols:
        return [_spectrum(_stem(filename), columns[:, x_col], columns[:, y_col], metadata)]
    keys = columns[:, coordinate_cols]
    _, first_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_index)
    spectra = []
    for rank, group in enumerate(order):
        rows = inverse.ravel() == group
        label = ", ".join(f"{names[c]}={keys[first_index[group], k]:g}" for k, c in enumerate(coordinate_cols))
        meta = dict(metadata, position=label)
        spectra.append(_spectrum(f"{_stem(filename)} ({label})", columns[rows, x_col], columns[rows, y_col], meta))
        if len(spectra) >= _MAX_SPECTRA:
            break
    return spectra


# ------------------------------------------------------------------------------------------ Excel
def _is_xlsx(data: bytes, filename: str) -> bool:
    return data[:2] == b"PK" and (filename.lower().endswith(".xlsx") or b"xl/" in data[:4096])


def _read_excel(data: bytes, filename: str) -> list[RamanSpectrum]:
    import pandas as pd
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
    except Exception as exc:           # openpyxl errors, password-protected books, …
        raise RamanParseError(f"The workbook could not be read: {exc}") from exc
    spectra = []
    for sheet_name, frame in sheets.items():
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        if frame.empty or frame.shape[1] < 2:
            continue
        first = frame.iloc[0]
        numeric_first = pd.to_numeric(first, errors="coerce")
        names = []
        if numeric_first.isna().sum() >= max(1, len(first) // 2):
            names = ["" if (isinstance(v, float) and math.isnan(v)) else str(v).strip() for v in first]
            frame = frame.iloc[1:]
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        keep = [j for j in range(numeric.shape[1]) if numeric.iloc[:, j].notna().sum() >= 2]
        if len(keep) < 2:
            continue
        numeric = numeric.iloc[:, keep]
        names = [names[j] for j in keep] if names else []
        columns = numeric.to_numpy(dtype=float)
        prefix = f"{sheet_name} · " if len(sheets) > 1 else ""
        try:
            spectra.extend(_columns_to_spectra(filename, names, columns, {"format": "xlsx", "sheet": str(sheet_name)}, prefix))
        except RamanParseError:
            continue
    if not spectra:
        raise RamanParseError("No sheet with a Raman shift column and intensity columns was found.")
    return spectra


# -------------------------------------------------------------------------------------------- WDF
def wdf_available() -> bool:
    try:
        import renishawWiRE  # noqa: F401
    except ImportError:
        return False
    return True


def _check_wdf_blocks(data: bytes) -> None:
    """Walk the block table (4-byte id, 4-byte uid, 8-byte size) before handing the file to the
    reader, which loops forever on a block of size 0 or a truncated file."""
    if data[:4] != b"WDF1":
        raise RamanParseError("Not a Renishaw WiRE .wdf file (missing WDF1 header).")
    n, pos, blocks = len(data), 0, 0
    while pos + 16 <= n:
        size = int.from_bytes(data[pos + 8:pos + 16], "little", signed=True)
        if size < 16 or pos + size > n:
            raise RamanParseError("The .wdf file is truncated or damaged.")
        pos += size
        blocks += 1
        if blocks > 100_000:
            raise RamanParseError("The .wdf file has an unreasonable number of blocks.")
    if blocks == 0:
        raise RamanParseError("The .wdf file is truncated or damaged.")


def _read_wdf(data: bytes, filename: str, multi: str, limit: int) -> list[RamanSpectrum]:
    try:
        from renishawWiRE import WDFReader
    except ImportError as exc:
        raise RamanParseError("Reading .wdf files needs the optional renishawWiRE package "
                              "(pip install renishawWiRE), or export the spectrum as text from WiRE.") from exc
    _check_wdf_blocks(data)
    with tempfile.NamedTemporaryFile(suffix=".wdf", delete=False) as handle:
        handle.write(data)
        path = handle.name
    reader = None
    try:
        try:
            reader = WDFReader(path)
        except Exception as exc:
            raise RamanParseError(f"The .wdf file could not be read: {exc}") from exc
        x = np.asarray(reader.xdata, dtype=float)
        spectra = np.asarray(reader.spectra, dtype=float)

        def _plain(value):
            value = getattr(value, "name", value)         # enum members by name
            if isinstance(value, float):
                return None if not math.isfinite(value) else round(value, 2)
            return value
        metadata: dict[str, Any] = {
            "format": "WDF", "title": getattr(reader, "title", ""),
            "laser_wavelength_nm": _plain(getattr(reader, "laser_length", None)),
            "count": getattr(reader, "count", None), "measurement_type": _plain(getattr(reader, "measurement_type", None)),
            "scan_type": _plain(getattr(reader, "scan_type", None)),
            "accumulation_count": getattr(reader, "accumulation_count", None),
        }
        xpos = getattr(reader, "xpos", None)
        ypos = getattr(reader, "ypos", None)
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        os.unlink(path)
    if spectra.ndim == 1:
        return [_spectrum(_stem(filename), x, spectra, metadata)]
    table = spectra.reshape(-1, spectra.shape[-1])
    if table.shape[1] != len(x):
        raise RamanParseError("The .wdf spectra do not match the Raman-shift axis length.")
    count = table.shape[0]
    if multi == "mean" or count == 1:
        meta = dict(metadata, averaged_spectra=count)
        label = _stem(filename) if count == 1 else f"{_stem(filename)} (mean of {count})"
        return [_spectrum(label, x, table.mean(axis=0), meta)]
    if multi not in ("all", "first"):
        raise RamanParseError("multi must be 'mean', 'all' or 'first'.")
    n = 1 if multi == "first" else min(count, max(1, int(limit)))
    result = []
    for k in range(n):
        label = f"{_stem(filename)} [{k + 1}]"
        meta = dict(metadata, index=k + 1)
        try:
            if xpos is not None and ypos is not None and len(xpos) == count and len(ypos) == count:
                label = f"{_stem(filename)} ({float(xpos[k]):g}, {float(ypos[k]):g})"
                meta["position"] = (float(xpos[k]), float(ypos[k]))
        except (TypeError, ValueError):
            pass
        result.append(_spectrum(label, x, table[k], meta))
    if multi == "all" and n < count:
        result[0].metadata["spectra_not_read"] = count - n
    return result


# ------------------------------------------------------------------------------------------ entry
def read_spectra(data: bytes, filename: str = "spectrum.txt", *, multi: str = "mean",
                 limit: int = _DEFAULT_MULTI_LIMIT) -> list[RamanSpectrum]:
    """All spectra in one file.

    ``multi`` chooses what a file with many spectra (a .wdf map or series) becomes:
    ``"mean"`` one averaged spectrum, ``"all"`` one spectrum each (up to ``limit``),
    ``"first"`` the first spectrum only. Text exports with several columns always give
    one spectrum per column.
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise RamanParseError("The file is empty.")
    data = bytes(data)
    lower = filename.lower()
    if lower.endswith(".jws"):
        raise RamanParseError("JASCO .jws files are binary; export them as text or CSV from Spectra Manager.")
    if data[:4] == b"WDF1" or lower.endswith(".wdf"):
        return _read_wdf(data, filename, multi, limit)
    if _is_xlsx(data, filename):
        return _read_excel(data, filename)
    text = _text(data)
    if _is_jasco(text):
        return _read_jasco(text, filename)
    if _is_jcamp_like(text):
        return _read_jcamp_like(text, filename)
    return _read_generic_text(text, filename)


def condition_summary(metadata: dict) -> list[tuple[str, str]]:
    """Measurement conditions worth showing (laser, exposure, accumulation, objective, grating…)."""
    wanted = re.compile(r"laser|wavelength|exposure|accumulat|objective|grating|slit|power|resolution|date|time|"
                        r"title|position|averaged|count|scan|measurement|names|rruffid|locality|spectrometer|"
                        r"units|data type|instrument|sample|origin|owner|x_converted_from|x_name|y_name", re.I)
    rows = []
    for key, value in metadata.items():
        if key in ("format", "x_units_warning", "spectra_not_read") or value in (None, ""):
            continue
        if str(key).upper() == "TITLE" and "\ufffd" in str(value):
            continue                          # a title the instrument wrote in an encoding the export lost
        if wanted.search(str(key)):
            rows.append((str(key), str(value)))
    return rows[:40]


def spectrum_kind(metadata: dict) -> str | None:
    """'ftir' or 'raman' when the file says what it is (DATA TYPE, TITLE), else None."""
    text = " ".join(str(metadata.get(key, "")) for key in ("DATA TYPE", "DATATYPE", "data_type", "Data type"))
    if re.search(r"INFRARED|\bIR\b|FT-?IR", text, re.I):
        return "ftir"
    if re.search(r"RAMAN", text, re.I):
        return "raman"
    if metadata.get("format") == "WDF" or any("laser" in str(k).lower() for k in metadata):
        return "raman"
    return None


def native_y_units(metadata: dict, y=None) -> str | None:
    """'absorbance', 'transmittance_percent', 'transmittance_fraction' or None (unknown).

    Read from YUNITS (or a column name kept as ``y_name``); a transmittance without a '%' is
    taken as a fraction when every value is at most 1.5."""
    text = " ".join(str(metadata.get(key, "")) for key in ("YUNITS", "y_name"))
    if re.search(r"ABSORB|KUBELKA|\bABS\b|^A$|LOG\s*\(1/", text, re.I):
        return "absorbance"
    if re.search(r"TRANSMIT|REFLECT|%\s*T|\bT\b|%\s*R", text, re.I):
        if "%" in text:
            return "transmittance_percent"
        if y is not None and len(y) and float(np.nanmax(y)) <= 1.5:
            return "transmittance_fraction"
        return "transmittance_percent"
    return None
