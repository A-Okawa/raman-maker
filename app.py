import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import inspect
import os
import re
from datetime import datetime
from pathlib import Path

from project_io import (
    GLOBAL_KEYS, PATTERN_KEY, X_LIMITS, ProjectError, encode_project, decode_project,
    atomic_save, project_fingerprint, clean_settings,
)
from raman_io import (read_spectra, RamanParseError, SUPPORTED_EXTENSIONS, condition_summary, wdf_available,
                      spectrum_kind, native_y_units)
from raman_processing import (process, detect_peaks, fit_peak, fit_multipeak, parse_centers, to_absorbance,
                              from_absorbance, MAX_FIT_PEAKS)
from figure_style import (
    STYLE_DEFAULTS, STYLE_KEYS, FIGURE_PRESETS_MM, FONT_FAMILIES, METRIC_COMPATIBLE, mm_to_inches, inches_to_mm,
    font_rcparams, apply_mpl_style, export_figure, resolve_font, css_font_stack, validate_style_settings,
)
from palettes import (
    PALETTES, PALETTE_NAMES, DEFAULT_PALETTE, REFERENCE_COLORS, MAX_SHADE_STEPS,
    palette_sequence, swatch_colors, load_colors, color_emoji,
)
from rich_text import rich, to_html, plain
from style_templates import encode_style_template, decode_style_template
from plot_extras import (
    PreparedCurve, parse_marks, add_mpl_marks, add_plotly_marks, add_mpl_bands, add_plotly_bands,
    add_mpl_zoom_inset, add_plotly_zoom_inset, top_labels_height, UNIT_HTML,
)

APP_VERSION = "1.3"
LAST_UPDATED = "2026-11-13"
LOCAL_MODE = os.environ.get("RAMAN_MAKER_LOCAL") == "1"
# Named colours in the graph app's format; copy that app's colors.csv here to share its names.
COLORS_FILE = Path(__file__).resolve().parent / "colors.csv"
# Axis titles in the graph app's markup (<i>…</i>, _{…}, ^{…}), per measurement and y display.
DEFAULT_X_LABEL = "Raman shift (cm^{-1})"
DEFAULT_Y_LABEL = "Intensity (a.u.)"
MODE_LABELS = {("raman", "intensity"): (DEFAULT_X_LABEL, DEFAULT_Y_LABEL),
               ("ftir", "absorbance"): ("Wavenumber (cm^{-1})", "Absorbance (a.u.)"),
               ("ftir", "transmittance"): ("Wavenumber (cm^{-1})", "Transmittance (%)")}
KNOWN_LABELS = {label for pair in MODE_LABELS.values() for label in pair}   # defaults the app may swap
Y_UNIT_CHOICES = ("auto", "absorbance", "transmittance_percent", "transmittance_fraction")
LOCAL_DATA_DIR = (Path(os.environ["RAMAN_MAKER_DATA_DIR"]).expanduser()
                  if LOCAL_MODE and os.environ.get("RAMAN_MAKER_DATA_DIR") else None)

# ===== 翻訳辞書 / Translation dictionary =====
TRANSLATIONS = {
    "ja": {
        "password_prompt": "パスワードを入力してください",
        "login": "ログイン",
        "wrong_password": "パスワードが違います",
        "app_caption": "ラマン・FT-IR スペクトルの重ね合わせ・積み重ね・ベースライン補正・ピーク注記・論文用 PDF / SVG / TIFF 出力",
        "label": "ラベル",
        "add_italic": "＋斜体",
        "italic_help": "末尾に `<i>TEXT</i>` を追加。TEXT を書き換えて使用。文字を選んで Ctrl+I（Mac は ⌘+I）でも斜体",
        "add_subscript": "＋x₂",
        "subscript_help": "下付き：末尾に `_{N}` を追加。N を書き換えて使用。文字を選んで Ctrl+,（カンマ）でも下付き",
        "add_superscript": "＋x²",
        "superscript_help": "上付き：末尾に `^{N}` を追加。N を書き換えて使用。文字を選んで Ctrl+.（ピリオド）でも上付き",
        "change_color": "🎨 色を変更",
        "session_header": "💾 セッション",
        "load_session": "保存した作業を開く (.json)",
        "project_help": "スペクトル、表示順、色、ラベル、処理、図の設定をまとめて保存・再開できます。",
        "project_loaded": "保存した作業を読み込みました。",
        "project_error": "作業ファイルを開けませんでした: ",
        "new_project": "新しい作業を始める",
        "resume_local": "前回の自動保存から再開",
        "autosaved": "PCに自動保存済み · {time}",
        "autosave_error": "自動保存できませんでした。作業ファイルをダウンロードして保存してください: ",
        "local_mode": "PC版 · データと自動保存はこのPC内に保存されます。",
        "data_header": "📂 スペクトル",
        "upload": "スペクトル（JASCO・Renishaw TXT/WDF・JCAMP-DX / FT-IR テキスト・RRUFF・XY/CSV・Excel）",
        "native_help": "JASCO Spectra Manager の CSV/TXT、Renishaw WiRE の TXT/WDF、`##TITLE=` で始まる FT-IR テキスト・JCAMP-DX、RRUFF、2列以上の XY/CSV/TXT、Excel を読み込めます。列が複数あるファイルは列ごとに別スペクトルになります。",
        "input_file_tab": "📄 ファイル",
        "input_paste_tab": "📋 貼り付け",
        "paste_label": "生データを貼り付け（波数・強度の 2 列以上）",
        "paste_help": "Excel や測定ソフトからコピーした数値をそのまま貼り付けます。見出し行があれば列名がスペクトル名になります。JASCO・Renishaw のテキストや、`##TITLE=` で始まる JCAMP-DX / FT-IR テキストの中身も貼れます。",
        "paste_placeholder": "400    0.0123\n402    0.0130\n404    0.0141\n…",
        "paste_name": "スペクトル名（省略可）",
        "paste_add": "➕ スペクトルに追加",
        "paste_added": "追加しました: ",
        "paste_empty": "貼り付け欄が空です。",
        "paste_error": "貼り付けたデータを読めませんでした: ",
        "paste_default": "貼り付け {n}",
        "other_sources": "作業ファイル・貼り付けのスペクトル（{n}）",
        "remove_source": "このスペクトルを外す",
        "remove_source_help": "ファイル欄にないスペクトルです。押すと外れます（元に戻すには開き直すか貼り直してください）。",
        "mode": "測定の種類",
        "mode_raman": "ラマン",
        "mode_ftir": "FT-IR",
        "mode_help": "ファイルに DATA TYPE があれば自動で切り替わります。FT-IR では横軸を波数（高波数が左）、縦軸を吸光度または透過率で描きます。",
        "ir_display": "縦軸の表示（FT-IR）",
        "ir_absorbance": "吸光度",
        "ir_transmittance": "透過率 (%)",
        "ir_display_help": "前処理（ベースライン・正規化）は吸光度で行い、表示のときに透過率に変換します（T = 10^(−A)）。透過率のファイルも吸光度で表示できます。",
        "x_reverse": "横軸を反転（左が高波数）",
        "y_unit": "データの縦軸",
        "y_unit_auto": "自動（ファイルの記載）",
        "y_unit_abs": "吸光度",
        "y_unit_pct": "透過率 (%)",
        "y_unit_frac": "透過率 (0–1)",
        "y_unit_help": "ファイルの YUNITS（Abs・%T など）から判定します。記載がないファイルは吸光度として読むので、透過率のデータはここで選んでください。",
        "read_error": "スペクトルを読み込めませんでした: ",
        "duplicate_names": "同名のファイルがあります。ファイル名を変更して選び直してください（現在の作業は保持しています）: ",
        "project_save_error": "作業ファイルを保存できません。データ量やラベルを確認してください: ",
        "scan_count": "{files} ファイル · {scans} スペクトル",
        "data_range": "測定データの全範囲を表示",
        "download_desktop": "PC用をダウンロード (.zip)",
        "desktop_help": "Mac・Windows対応。初回セットアップ後はオフラインで使用できます。",
        "graph_settings": "⚙️ グラフ設定",
        "xrange": "ラマンシフトの範囲 (cm⁻¹)",
        "xrange_ir": "波数の範囲 (cm⁻¹)",
        "show_legend_cb": "凡例を表示",
        "tick_settings": "目盛り設定",
        "major_tick": "主目盛り間隔 (cm⁻¹)",
        "show_minor_cb": "副目盛りを表示",
        "minor_tick": "副目盛り間隔 (cm⁻¹)",
        "sample_labels": "サンプルラベル（グラフ内）",
        "show_side_labels_cb": "グラフ内にラベルを表示",
        "label_pos": "ラベル位置",
        "left": "左",
        "right": "右",
        "label_fontsize": "ラベル文字サイズ",
        "label_offset_x": "横オフセット (cm⁻¹)",
        "label_offset_x_help": "正=グラフ内側、負=グラフ外側",
        "label_offset_y": "縦オフセット（スペクトル高さ比）",
        "label_offset_y_help": "0=ベースライン、1.0=ピーク付近",
        "fig_size": "📐 図サイズ・出力",
        "save_session": "💾 作業を保存・ダウンロード (.json)",
        "show_panel": "⚙️ スペクトル設定パネルを表示",
        "spectra_header": "#### スペクトル",
        "display_order": "表示順",
        "show_cb": "表示する",
        "offset_adj": "オフセット調整（最大値に対する比）",
        "y_position": "Y位置（絶対値）",
        "scale": "強度の倍率",
        "drag_zoom": "ドラッグで範囲ズーム ／ ダブルクリックでリセット",
        "interactive_preview": "インタラクティブ",
        "export_preview": "保存画像のプレビュー",
        "export_preview_caption": "図サイズ: {width:.1f} × {height:.1f} mm · PNG / TIFF: {dpi} DPI（プレビューは150 DPI）",
        "save_tiff": "📥 TIFF として保存 ({dpi} DPI)",
        "save_png": "📥 PNG として保存 ({dpi} DPI)",
        "upload_prompt": "ラマン・FT-IR スペクトルのファイルを選ぶか、生データを貼り付けるか、保存した作業ファイル（.json）を開いてください。",
        "conditions": "ℹ️ 測定条件",
    },
    "en": {
        "password_prompt": "Enter password",
        "login": "Login",
        "wrong_password": "Incorrect password",
        "app_caption": "Raman and FT-IR spectra: overlay or stack · baseline correction · peak marks · publication-quality PDF / SVG / TIFF export",
        "label": "Label",
        "add_italic": "+Italic",
        "italic_help": "Appends `<i>TEXT</i>`; replace TEXT. Or select text and press Ctrl+I (⌘+I on a Mac).",
        "add_subscript": "+x₂",
        "subscript_help": "Subscript: appends `_{N}`; replace N. Or select text and press Ctrl+, (comma).",
        "add_superscript": "+x²",
        "superscript_help": "Superscript: appends `^{N}`; replace N. Or select text and press Ctrl+. (period).",
        "change_color": "🎨 Change color",
        "session_header": "💾 Session",
        "load_session": "Open saved project (.json)",
        "project_help": "Save and reopen spectra, order, colors, labels, processing and figure settings together.",
        "project_loaded": "Project restored.",
        "project_error": "Could not open the project: ",
        "new_project": "Start a new project",
        "resume_local": "Resume the previous autosave",
        "autosaved": "Autosaved on this PC · {time}",
        "autosave_error": "Could not autosave. Download the project to keep your work: ",
        "local_mode": "PC edition · Data and autosaves stay on this computer.",
        "data_header": "📂 Spectra",
        "upload": "Spectra (JASCO, Renishaw TXT/WDF, JCAMP-DX / FT-IR text, RRUFF, XY/CSV, Excel)",
        "native_help": "Reads JASCO Spectra Manager CSV/TXT, Renishaw WiRE TXT/WDF, FT-IR text and JCAMP-DX files starting with `##TITLE=`, RRUFF, XY/CSV/TXT with two or more columns, and Excel. Extra columns become separate spectra.",
        "input_file_tab": "📄 Files",
        "input_paste_tab": "📋 Paste",
        "paste_label": "Paste raw data (two or more columns: wavenumber, intensity)",
        "paste_help": "Paste numbers copied from Excel or the instrument software as they are. A header row names the spectra. The contents of JASCO and Renishaw text files and of JCAMP-DX / FT-IR text starting with `##TITLE=` work too.",
        "paste_placeholder": "400    0.0123\n402    0.0130\n404    0.0141\n…",
        "paste_name": "Spectrum name (optional)",
        "paste_add": "➕ Add to the spectra",
        "paste_added": "Added: ",
        "paste_empty": "The paste box is empty.",
        "paste_error": "The pasted data could not be read: ",
        "paste_default": "Pasted {n}",
        "other_sources": "Spectra from projects and pasting ({n})",
        "remove_source": "Remove this spectrum",
        "remove_source_help": "Spectra that are not in the file box. A click removes one (reopen or paste again to bring it back).",
        "mode": "Measurement",
        "mode_raman": "Raman",
        "mode_ftir": "FT-IR",
        "mode_help": "Switches automatically when the file states its DATA TYPE. FT-IR draws wavenumber (high values on the left) against absorbance or transmittance.",
        "ir_display": "Y axis (FT-IR)",
        "ir_absorbance": "Absorbance",
        "ir_transmittance": "Transmittance (%)",
        "ir_display_help": "Baselines and normalisation are applied in absorbance; the display converts to transmittance (T = 10^(−A)). Transmittance files can be shown as absorbance too.",
        "x_reverse": "Reverse the x axis (high wavenumber on the left)",
        "y_unit": "Y units of the data",
        "y_unit_auto": "Automatic (from the file)",
        "y_unit_abs": "Absorbance",
        "y_unit_pct": "Transmittance (%)",
        "y_unit_frac": "Transmittance (0–1)",
        "y_unit_help": "Taken from the file's YUNITS (Abs, %T, …). Files without it are read as absorbance, so choose transmittance here for such data.",
        "read_error": "Could not read the spectrum: ",
        "duplicate_names": "Files have duplicate names. Rename them and select again; your current work is kept: ",
        "project_save_error": "Could not save the project. Check the data size and labels: ",
        "scan_count": "{files} files · {scans} spectra",
        "data_range": "Show the full measured range",
        "download_desktop": "Download PC edition (.zip)",
        "desktop_help": "For Mac and Windows. Works offline after the first setup.",
        "graph_settings": "⚙️ Graph Settings",
        "xrange": "Raman shift range (cm⁻¹)",
        "xrange_ir": "Wavenumber range (cm⁻¹)",
        "show_legend_cb": "Show legend",
        "tick_settings": "Tick settings",
        "major_tick": "Major tick interval (cm⁻¹)",
        "show_minor_cb": "Show minor ticks",
        "minor_tick": "Minor tick interval (cm⁻¹)",
        "sample_labels": "Sample labels (in plot)",
        "show_side_labels_cb": "Show labels in plot",
        "label_pos": "Label position",
        "left": "Left",
        "right": "Right",
        "label_fontsize": "Label font size",
        "label_offset_x": "Horizontal offset (cm⁻¹)",
        "label_offset_x_help": "Positive=inside, Negative=outside",
        "label_offset_y": "Vertical offset (spectrum height ratio)",
        "label_offset_y_help": "0=baseline, 1.0=near peak",
        "fig_size": "📐 Figure Size & Export",
        "save_session": "💾 Save and download project (.json)",
        "show_panel": "⚙️ Show spectrum settings panel",
        "spectra_header": "#### Spectra",
        "display_order": "Display order",
        "show_cb": "Show",
        "offset_adj": "Offset adjustment (fraction of the maximum)",
        "y_position": "Y position (absolute)",
        "scale": "Intensity scale factor",
        "drag_zoom": "Drag to zoom / Double-click to reset",
        "interactive_preview": "Interactive",
        "export_preview": "Export preview",
        "export_preview_caption": "Figure size: {width:.1f} × {height:.1f} mm · PNG / TIFF: {dpi} DPI (preview: 150 DPI)",
        "save_tiff": "📥 Save as TIFF ({dpi} DPI)",
        "save_png": "📥 Save as PNG ({dpi} DPI)",
        "upload_prompt": "Choose Raman or FT-IR spectrum files, paste raw data, or open a saved project (.json) to continue.",
        "conditions": "ℹ️ Measurement conditions",
    },
}

st.set_page_config(page_title="Raman Maker", page_icon="🔬", layout="wide")


def _wide(function):
    """Full width on old and new Streamlit: width="stretch" where it exists, else use_container_width."""
    parameters = inspect.signature(function).parameters
    if "width" in parameters and "int" not in str(parameters["width"].annotation):
        return {"width": "stretch"}
    if "use_container_width" in parameters:
        return {"use_container_width": True}
    if "use_column_width" in parameters:
        return {"use_column_width": True}
    return {}


WIDE_BUTTON, WIDE_DOWNLOAD, WIDE_POPOVER = _wide(st.button), _wide(st.download_button), _wide(st.popover)
WIDE_CHART, WIDE_IMAGE = _wide(st.plotly_chart), _wide(st.image)

# ===== 言語選択 / Language selection =====
_lang_option = st.sidebar.radio("🌐 Language / 言語", ["日本語", "English"], horizontal=True, key="lang_radio")
_lang_code = "en" if _lang_option == "English" else "ja"
T = TRANSLATIONS[_lang_code]


def ui(ja, en):
    return en if _lang_code == "en" else ja


def _widget_default(key, default, checkbox=False, number=False):
    # A restored value has one owner: session state. Passing a second widget
    # default causes Streamlit's duplicate-default warning after project loads.
    if st.session_state.get(key) is None:
        st.session_state[key] = default
    return False if checkbox else "min" if number else None


# ===== パスワード認証 =====
def check_password():
    if LOCAL_MODE:
        return True
    if st.session_state.get("authenticated"):
        return True
    try:
        # Without any secrets.toml (a PC, a fresh deployment) no password is required; st.secrets itself
        # would report the missing file on the page.
        pwd = st.secrets.get("password", "") if st.secrets.load_if_toml_exists() else ""
    except Exception:
        pwd = ""
    if not pwd:
        return True
    st.title("🔬 Raman Maker")
    entered = st.text_input(T["password_prompt"], type="password")
    if st.button(T["login"]):
        if entered == pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error(T["wrong_password"])
    return False


if not check_password():
    st.stop()

st.title("Raman Maker")
st.caption(T["app_caption"])
if LOCAL_MODE:
    st.caption(T["local_mode"])


# ===== RestoredFile =====
class RestoredFile:
    """A spectrum of a loaded file (uploaded now or restored from a project)."""
    def __init__(self, name: str, data: bytes, spectrum=None, source: str = "", siblings: int = 1):
        self.name = name
        self._data = data
        self.spectrum = spectrum
        self.source = source
        self.siblings = siblings      # spectra in the same file

    def read(self) -> bytes:
        return self._data

    @property
    def default_label(self) -> str:
        """The reader's name: the file stem, a column or sheet name, a position, an RRUFF mineral name."""
        return self.spectrum.name


# ===== ラベル入力（書式ボタン付き） =====
def label_input(key: str, default: str = "") -> str:
    val_key = f"_val_{key}"
    ver_key = f"_ver_{key}"
    if val_key not in st.session_state:
        st.session_state[val_key] = default
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0
    inp_key = f"_inp_{key}_v{st.session_state[ver_key]}"
    if inp_key not in st.session_state:
        st.session_state[inp_key] = st.session_state[val_key]
    inp = st.text_input(T["label"], key=inp_key)
    st.session_state[val_key] = inp
    c1, c2, c3 = st.columns(3)
    if c1.button(T["add_italic"], key=f"_bi_{key}", help=T["italic_help"], **WIDE_BUTTON):
        st.session_state[val_key] = inp + "<i>TEXT</i>"
        st.session_state[ver_key] += 1
        st.rerun()
    if c2.button(T["add_subscript"], key=f"_bs_{key}", help=T["subscript_help"], **WIDE_BUTTON):
        st.session_state[val_key] = inp + "_{N}"
        st.session_state[ver_key] += 1
        st.rerun()
    if c3.button(T["add_superscript"], key=f"_bp_{key}", help=T["superscript_help"], **WIDE_BUTTON):
        st.session_state[val_key] = inp + "^{N}"
        st.session_state[ver_key] += 1
        st.rerun()
    return st.session_state[val_key]


# ===== データ読み込み =====
@st.cache_data(show_spinner=False, max_entries=64)
def _read_cached(data: bytes, filename: str, multi: str):
    return read_spectra(data, filename, multi=multi)


# ===== 色（グラフ描画appと同じ色の組み合わせ） =====
OTHER_COLOR = "—"


def named_colors():
    return load_colors(COLORS_FILE)


def _int_or(value, default):
    return value if type(value) is int else default


def spectrum_display_order():
    """Spectrum indices in display order (the 表示順 numbers; equal numbers keep the file order)."""
    state = st.session_state
    n = len(state.get("_spectrum_names") or [])
    return sorted(range(n), key=lambda i: (_int_or(state.get(f"ord_{i}"), i + 1), i))


def palette_range(n):
    """First and last display positions (1-based, inclusive) that take the palette; the rest are the base colour."""
    state = st.session_state
    lo = min(max(_int_or(state.get("palette_from"), 1), 1), max(n, 1))
    to = _int_or(state.get("palette_to"), 0)
    hi = n if to <= 0 else min(max(to, lo), n)
    return lo, hi


def palette_base_color():
    value = st.session_state.get("palette_base_color")
    return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else "#000000"


def palette_items():
    state = st.session_state
    return [(f"sp_color_{i}", state.get(f"vis_{i}", True) is not False) for i in spectrum_display_order()]


def palette_plan():
    """The palette colour for every spectrum key, and the number of coloured items.

    Visible spectra within the chosen range take the palette in display order (shades spread
    over them, as the graph app spreads them over its series); hidden ones continue after them.
    Spectra outside the range (a base measurement, for example) take the base colour.
    """
    state = st.session_state
    name = state.get("palette", DEFAULT_PALETTE)
    name = name if name in PALETTES else DEFAULT_PALETTE
    reverse, steps = bool(state.get("palette_reverse", False)), _int_or(state.get("palette_steps"), 0)
    items = palette_items()
    lo, hi = palette_range(len(items))
    shaded = items[lo - 1:hi]
    visible = [key for key, shown in shaded if shown]
    hidden = [key for key, shown in shaded if not shown]
    plan = {key: palette_base_color() for key, _ in items[:lo - 1] + items[hi:]}
    plan.update(zip(visible, palette_sequence(name, len(visible), reverse, steps)))
    plan.update(zip(hidden, palette_sequence(name, len(shaded), reverse, steps)[len(visible):]))
    return plan, len(visible)


def _apply_palette():
    plan, _ = palette_plan()
    for key, shown in palette_items():
        if shown:
            st.session_state[key] = plan[key]
    for key in list(st.session_state):
        if str(key).startswith(("_palview_", "_cname_")):
            del st.session_state[key]


def _set_palette_range(n):
    lo, hi = st.session_state.get("_palette_range", (1, n))
    st.session_state["palette_from"] = int(lo)
    st.session_state["palette_to"] = 0 if int(hi) >= n else int(hi)
    _apply_palette()


def top_to_bottom(levels):
    """Positions ordered as the curves appear from top to bottom, for the legend."""
    return sorted(range(len(levels)), key=lambda k: -levels[k])


def _set_color(key, value):
    st.session_state[key] = value


def _named_color_chosen(key, name_key):
    value = named_colors().get(st.session_state.get(name_key))
    if value:
        st.session_state[key] = value


def swatch_set_colors(set_key, n_items):
    return swatch_colors(set_key, n_items, bool(st.session_state.get("palette_reverse", False)),
                         _int_or(st.session_state.get("palette_steps"), 0))


def color_picker_popover(key: str, default_hex: str, n_items: int = 1):
    """Colour of one spectrum: swatches of the graph app's palettes, colours by name, or any colour."""
    _widget_default(key, default_hex)
    current = str(st.session_state[key])
    st.markdown(
        f'<div style="background:{current};height:20px;border-radius:5px;'
        f'border:1px solid #ccc;margin-bottom:4px"></div>',
        unsafe_allow_html=True,
    )
    with st.popover(T["change_color"], **WIDE_POPOVER):
        view_key = f"_palview_{key}"
        if st.session_state.get(view_key) not in PALETTES:
            st.session_state[view_key] = st.session_state.get("palette", DEFAULT_PALETTE)
        st.selectbox(ui("色の組み合わせ", "Palette"), list(PALETTES), key=view_key,
                     format_func=lambda k: PALETTE_NAMES[_lang_code][k],
                     help=ui("グラフ描画appと同じ組み合わせです。濃淡は、色を付けるスペクトルの数（または「濃淡の段数」）で分けた色で、"
                             "一番薄い色でも白地とのコントラストが 3:1 以上です。",
                             "The graph app's palettes. Shades are split by the number of coloured spectra "
                             "(or the shade steps); even the lightest has at least 3:1 contrast against white."))
        colors = swatch_set_colors(st.session_state[view_key], n_items)
        cols = st.columns(len(colors))
        for j, hex_c in enumerate(colors):
            with cols[j]:
                selected = hex_c.lower() == current.lower()
                st.markdown(
                    f'<div style="background:{hex_c};height:24px;border-radius:3px;'
                    f'border:{"3px solid #333" if selected else "1px solid #ccc"};'
                    f'margin-bottom:2px"></div>',
                    unsafe_allow_html=True,
                )
                st.button("✓" if selected else " ", key=f"{key}_sw_{j}", help=hex_c.upper(),
                          on_click=_set_color, args=(key, hex_c), **WIDE_BUTTON)
        names = named_colors()
        name_key = f"_cname_{key}"
        st.session_state[name_key] = next((n for n, h in names.items() if h.lower() == current.lower()), OTHER_COLOR)
        st.selectbox(ui("名前で選ぶ", "By name"), [OTHER_COLOR] + list(names), key=name_key,
                     on_change=_named_color_chosen, args=(key, name_key),
                     format_func=lambda n: n if n == OTHER_COLOR else f"{color_emoji(names[n])} {n}  {names[n]}",
                     help=ui("colors.csv の色（グラフ描画appと同じ書式「名前,#RRGGBB」）。グラフ描画appの colors.csv をこのアプリのフォルダに置くと、同じ名前の色を選べます。",
                             "Colours from colors.csv (the graph app's format, name,#RRGGBB). Put the graph app's colors.csv in this app's folder to share its names."))
        st.color_picker(ui("自由に選ぶ", "Any colour"), key=key)
    return st.session_state[key]


# ===== セッション保存/復元 =====
for _key in list(st.session_state):
    if _key != "lang_radio" and (_key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(_key)):
        st.session_state[_key] = st.session_state[_key]

_TRANSIENT = re.compile(r"_(?:inp|ver)_lbl_.*|_(?:palview|cname)_.*")


def build_session_json() -> bytes:
    settings = {key: value for key, value in st.session_state.items()
                if key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(key)}
    return encode_project(st.session_state.get("_spectrum_bytes", {}), settings)


def restore_session(json_bytes: bytes):
    project = decode_project(json_bytes)
    for key in list(st.session_state):
        if key in GLOBAL_KEYS or PATTERN_KEY.fullmatch(key) or _TRANSIENT.fullmatch(key):
            del st.session_state[key]
    st.session_state["_spectrum_bytes"] = project["spectrum_files"]
    st.session_state.pop("_upload_error", None)
    st.session_state.update(project["settings"])
    st.session_state["_file_epoch"] = st.session_state.get("_file_epoch", 0) + 1
    st.session_state["_spectrum_names"] = None
    st.session_state["_ticks_auto"] = "major_tick" not in project["settings"]
    st.session_state["_mode_auto"] = "spec_mode" not in project["settings"]
    st.session_state["_xrange_auto"] = "xrange" not in project["settings"]
    st.session_state["_margin_top_auto"] = "margin_top" not in project["settings"]
    st.session_state["_project_loaded"] = True
    st.session_state.pop("_project_error", None)
    st.session_state.pop("_autosave_fingerprint", None)


def _open_uploaded_project():
    upload = st.session_state.get("_session_upload")
    if upload is not None:
        try:
            restore_session(upload.getvalue())
        except (ProjectError, ValueError) as exc:
            st.session_state["_project_error"] = str(exc)


def _open_local_autosave():
    try:
        restore_session((LOCAL_DATA_DIR / "autosave.json").read_bytes())
    except (OSError, ProjectError) as exc:
        st.session_state["_project_error"] = str(exc)


def _open_style_template():
    upload = st.session_state.get("_style_upload")
    if upload is None:
        return
    try:
        settings = decode_style_template(upload.getvalue())
        clean_settings({**{key: st.session_state[key] for key in STYLE_KEYS if key in st.session_state}, **settings})
        st.session_state.update(settings)
        st.session_state.pop("_size_signature", None)
        st.session_state["_style_loaded"] = True
        st.session_state.pop("_style_error", None)
    except (ProjectError, ValueError) as exc:
        st.session_state["_style_error"] = str(exc)


def _apply_figure_preset():
    name = st.session_state["figure_preset"]
    preset = FIGURE_PRESETS_MM.get(name)
    if preset:
        st.session_state["fig_width"], st.session_state["fig_height"] = map(mm_to_inches, preset)
        if name == "figapp":
            # The graph app's defaults: Arial 7 pt labels, 6.5 pt ticks and legend, 0.5 pt axes, 0.9 pt lines.
            st.session_state.update(font_family="Arial", font_size=7.0, tick_font_size=6.5, legend_font_size=6.5,
                label_fontsize=7, annotation_font_size=6, marks_fontsize=6,
                line_width=0.9, reference_line_width=0.6, axes_line_width=0.5, tick_length=2.5, tick_direction="out",
                axes_box=False,
                margin_left=.06, margin_right=.98, margin_bottom=.14, margin_top=.96)
        else:
            compact = name == "single"
            size = 9.0 if compact else 11.0
            st.session_state.update(font_size=size, tick_font_size=size, legend_font_size=size,
                label_fontsize=8 if compact else 10, annotation_font_size=6 if compact else 7,
                marks_fontsize=6 if compact else 7,
                margin_left=.16 if compact else .12, margin_bottom=.22 if compact else .15)
    st.session_state.pop("_size_signature", None)
    st.session_state["_margin_top_auto"] = True


def _update_figure_dimension(axis):
    value = st.session_state[f"_figure_{axis}"]
    st.session_state[f"fig_{axis}"] = mm_to_inches(value) if st.session_state["size_unit"] == "mm" else value
    st.session_state["figure_preset"] = "custom"


def _update_margin(key):
    value = st.session_state[f"_percent_{key}"] / 100
    st.session_state[key] = 1 - value if key in ("margin_right", "margin_top") else value
    st.session_state["_margin_top_auto"] = False


def _new_project():
    for key in list(st.session_state):
        if (key in GLOBAL_KEYS and key != "lang_radio") or PATTERN_KEY.fullmatch(key) or _TRANSIENT.fullmatch(key):
            del st.session_state[key]
    st.session_state["_spectrum_bytes"] = {}
    st.session_state.pop("_upload_error", None)
    for key in ("_paste_count", "_paste_error", "_paste_added"):
        st.session_state.pop(key, None)
    st.session_state["_file_epoch"] = st.session_state.get("_file_epoch", 0) + 1
    st.session_state["_spectrum_names"] = None
    st.session_state.pop("_project_loaded", None)
    st.session_state.pop("_project_error", None)
    st.session_state.pop("_preset_initialised", None)      # the new project starts in the graph app's look again
    st.session_state.pop("_ticks_auto", None)
    st.session_state.pop("_xrange_auto", None)
    st.session_state.pop("_mode_auto", None)
    st.session_state.pop("_margin_top_auto", None)


def _merge_uploads(current: dict, previous: set, files) -> dict:
    """The uploader's files join the spectra already loaded (from a project or pasted): files taken out of
    the uploader leave, new or re-chosen files enter (the latest contents win), everything else stays in order."""
    names = {file.name for file in files}
    merged = {name: data for name, data in current.items() if name not in previous or name in names}
    for file in files:
        merged[file.name] = file.getvalue()
    return merged


def _uploaded_files_changed(widget_key):
    files = st.session_state.get(widget_key) or []
    names = [file.name for file in files]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        st.session_state["_upload_error"] = ", ".join(duplicates)
        return
    st.session_state.pop("_upload_error", None)
    seen_key = f"_seen_{widget_key}"          # the uploader's files as of its previous change
    st.session_state["_spectrum_bytes"] = _merge_uploads(st.session_state.get("_spectrum_bytes", {}),
                                                         st.session_state.get(seen_key, set()), files)
    st.session_state[seen_key] = set(names)


_TEXT_SUFFIX = re.compile(r"\.(?:txt|csv|xy|dat|asc|prn|tsv|jdx|dx|jcamp)$", re.I)


def _pasted_source_name(name: str, existing) -> str:
    """A file-like name for pasted data ('.txt' unless a text extension was typed), unique among the spectra."""
    if not _TEXT_SUFFIX.search(name):
        name += ".txt"
    stem, suffix = os.path.splitext(name)
    candidate, k = name, 2
    while candidate in existing:
        candidate = f"{stem} ({k}){suffix}"
        k += 1
    return candidate


def _add_pasted_spectrum():
    text = (st.session_state.get("_paste_text") or "").strip()
    if not text:
        st.session_state["_paste_error"] = T["paste_empty"]
        return
    count = st.session_state.get("_paste_count", 0) + 1
    typed = (st.session_state.get("_paste_name") or "").strip()
    files = dict(st.session_state.get("_spectrum_bytes", {}))
    name = _pasted_source_name(typed or T["paste_default"].format(n=count), files)
    data = (text + "\n").encode("utf-8")
    try:
        _read_cached(data, name, st.session_state.get("multi_mode", "mean"))     # the same reader as for files
    except RamanParseError as exc:
        st.session_state["_paste_error"] = T["paste_error"] + str(exc)
        return
    files[name] = data
    st.session_state["_spectrum_bytes"] = files
    st.session_state["_paste_count"] = count
    st.session_state["_paste_text"] = ""
    st.session_state["_paste_name"] = ""
    st.session_state.pop("_paste_error", None)
    st.session_state["_paste_added"] = name


def _remove_source(name):
    files = dict(st.session_state.get("_spectrum_bytes", {}))
    files.pop(name, None)
    st.session_state["_spectrum_bytes"] = files


def _remap_file_session_state(old_names: list, new_names: list, key_tmpls: list):
    """When the set of spectra changes, move the index-based settings by spectrum name.

    The label boxes are widgets keyed by index and a version. Every remap hands out a version that was
    never used, so a spectrum landing on an index another spectrum held before does not inherit its text.
    """
    old_vals: dict[str, dict] = {}
    for old_i, name in enumerate(old_names):
        old_vals[name] = {tmpl: st.session_state.get(tmpl.format(old_i)) for tmpl in key_tmpls}
    for old_i in range(len(old_names)):
        for tmpl in key_tmpls:
            st.session_state.pop(tmpl.format(old_i), None)
    version = st.session_state.get("_label_version", 0) + 1
    st.session_state["_label_version"] = version
    for new_i, name in enumerate(new_names):
        for tmpl in key_tmpls:
            if tmpl.startswith("_ver_"):
                st.session_state[tmpl.format(new_i)] = version
                continue
            val = old_vals.get(name, {}).get(tmpl)
            if val is not None:
                st.session_state[tmpl.format(new_i)] = val


# ===== 図の初期値（ウィジェットを作る前に決める） =====
for _key, _value in STYLE_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
st.session_state.setdefault("fig_width", mm_to_inches(88.0))
st.session_state.setdefault("fig_height", mm_to_inches(70.0))
if st.session_state.get("_preset_initialised") is None and st.session_state.get("font_size") is None:
    # A new session starts in the graph app's look (88 × 70 mm, Arial 7 pt, thin lines, outward ticks).
    st.session_state["_preset_initialised"] = True
    st.session_state["figure_preset"] = "figapp"
    _apply_figure_preset()

# ===== サイドバー =====
# Data first (files or pasted text), then the project section, then the figure settings.
st.sidebar.caption(f"Raman Maker {APP_VERSION} · Last updated: {LAST_UPDATED}")
_file_epoch = st.session_state.get("_file_epoch", 0)
st.sidebar.header(T["data_header"])
st.sidebar.caption(T["native_help"])

# ----- the files are parsed before the widgets below are drawn, so the measurement type (Raman / FT-IR)
# can follow what the files say; the uploader's callback has already stored the bytes.
_widget_default("multi_mode", "mean")
multi_mode = st.session_state["multi_mode"]
active = []
for _source_name, _source_bytes in st.session_state.get("_spectrum_bytes", {}).items():
    try:
        _spectra = _read_cached(_source_bytes, _source_name, multi_mode)
        for _k, _spectrum in enumerate(_spectra):
            _name = _source_name if len(_spectra) == 1 else f"{_source_name} [{_k + 1}] {_spectrum.name}"
            active.append(RestoredFile(_name, _source_bytes, _spectrum, _source_name, len(_spectra)))
    except RamanParseError as _error:
        st.error(T["read_error"] + f"{_source_name}: {_error}")

_new_names = [file.name for file in active]
_old_names = st.session_state.get("_spectrum_names")
if _old_names is not None and _old_names != _new_names:
    _remap_file_session_state(_old_names, _new_names,
        ["vis_{}", "ord_{}", "_val_lbl_{}", "_ver_lbl_{}", "sp_color_{}", "extra_offset_{}", "yoff_{}", "scale_{}", "yunit_{}"])
st.session_state["_spectrum_names"] = _new_names

# ----- measurement: Raman or FT-IR (auto-detected from the files until chosen by hand)
if st.session_state.get("_mode_auto") is None:
    st.session_state["_mode_auto"] = st.session_state.get("spec_mode") is None   # a restored project keeps its choice
_widget_default("spec_mode", "raman")
_widget_default("ir_display", "absorbance")


def _apply_mode_labels():
    """Axis titles follow the measurement and y display unless the person wrote their own."""
    mode, display = st.session_state["spec_mode"], st.session_state["ir_display"]
    x_default, y_default = MODE_LABELS[(mode, display if mode == "ftir" else "intensity")]
    if st.session_state.get("x_label") in (None, *KNOWN_LABELS):
        st.session_state["x_label"] = x_default
    if st.session_state.get("y_label") in (None, *KNOWN_LABELS):
        st.session_state["y_label"] = y_default


def _mode_chosen():
    st.session_state["_mode_auto"] = False
    st.session_state["x_reverse"] = st.session_state["spec_mode"] == "ftir"
    _apply_mode_labels()


if active and st.session_state["_mode_auto"]:
    _kinds = {spectrum_kind(f.spectrum.metadata) for f in active} - {None}
    _detected = "ftir" if "ftir" in _kinds else "raman" if "raman" in _kinds else st.session_state["spec_mode"]
    if _detected != st.session_state["spec_mode"]:
        st.session_state["spec_mode"] = _detected
        st.session_state["x_reverse"] = _detected == "ftir"
        _apply_mode_labels()
# ----- data in: files, or raw text pasted from Excel / the instrument software
_upload_key = f"spectrum_upload_{_file_epoch}"
_file_tab, _paste_tab = st.sidebar.tabs([T["input_file_tab"], T["input_paste_tab"]])
with _file_tab:
    uploaded_files = st.file_uploader(
        T["upload"], type=list(SUPPORTED_EXTENSIONS), accept_multiple_files=True,
        key=_upload_key, on_change=_uploaded_files_changed, args=(_upload_key,),
    )
    if st.session_state.get("_upload_error"):
        st.error(T["duplicate_names"] + st.session_state["_upload_error"])
with _paste_tab:
    st.text_area(T["paste_label"], key="_paste_text", height=150, placeholder=T["paste_placeholder"], help=T["paste_help"])
    st.text_input(T["paste_name"], key="_paste_name",
                  placeholder=T["paste_default"].format(n=st.session_state.get("_paste_count", 0) + 1))
    st.button(T["paste_add"], key="_paste_add", on_click=_add_pasted_spectrum, **WIDE_BUTTON)
    if st.session_state.get("_paste_error"):
        st.error(st.session_state["_paste_error"])
    if st.session_state.get("_paste_added"):
        st.success(T["paste_added"] + st.session_state.pop("_paste_added"))
spec_mode = st.sidebar.selectbox(T["mode"], ["raman", "ftir"], key="spec_mode", on_change=_mode_chosen,
                                 format_func=lambda v: T["mode_raman"] if v == "raman" else T["mode_ftir"], help=T["mode_help"])
IS_FTIR = spec_mode == "ftir"
if IS_FTIR:
    st.sidebar.selectbox(T["ir_display"], ["absorbance", "transmittance"], key="ir_display", on_change=_apply_mode_labels,
                         format_func=lambda v: T["ir_absorbance"] if v == "absorbance" else T["ir_transmittance"],
                         help=T["ir_display_help"])
ir_display = st.session_state["ir_display"] if IS_FTIR else "intensity"
PEAKS_DOWN = IS_FTIR and ir_display == "transmittance"
st.sidebar.selectbox(
    ui("多点測定ファイル（WDF のマップ・連続測定）の扱い", "Files with many spectra (WDF maps and series)"),
    ["mean", "all", "first"], key="multi_mode",
    format_func=lambda v: {"mean": ui("平均して 1 本にする", "Average into one spectrum"),
                           "all": ui("1 本ずつ読み込む（最大 200 本）", "Read one by one (up to 200)"),
                           "first": ui("最初の 1 本だけ", "First spectrum only")}[v],
    help=ui("テキスト書き出しの列や Excel の列は、この設定に関わらず列ごとに別スペクトルになります。"
            + ("" if wdf_available() else "　.wdf の読み込みには renishawWiRE パッケージが必要です（未インストール）。"),
            "Columns of text exports and Excel sheets always become separate spectra."
            + ("" if wdf_available() else " Reading .wdf needs the renishawWiRE package (not installed).")))
if active:
    st.sidebar.caption(T["scan_count"].format(files=len(st.session_state.get("_spectrum_bytes", {})), scans=len(active)))
# Spectra that are not in the uploader (restored from a project, or pasted) are listed here, each removable.
_uploader_names = {file.name for file in (uploaded_files or [])}
_other_sources = [name for name in st.session_state.get("_spectrum_bytes", {}) if name not in _uploader_names]
if _other_sources:
    with st.sidebar.expander(T["other_sources"].format(n=len(_other_sources))):
        st.caption(T["remove_source_help"])
        for _name in _other_sources:
            st.button(f"✕ {_name}", key=f"_rm_{_name}", help=T["remove_source"], on_click=_remove_source, args=(_name,),
                      **WIDE_BUTTON)
for _file in active:
    _warn = _file.spectrum.metadata.get("x_units_warning")
    if _warn:
        st.sidebar.warning(ui(f"{_file.source}: 横軸の単位が「{_warn}」です。cm⁻¹ として描きます。",
                              f"{_file.source}: the x units are '{_warn}'. Drawn as cm⁻¹."))
    _converted = _file.spectrum.metadata.get("x_converted_from")
    if _converted:
        st.sidebar.info(ui(f"{_file.source}: 横軸を {_converted} から波数 (cm⁻¹) に換算しました。",
                           f"{_file.source}: the x axis was converted from {_converted} to wavenumber (cm⁻¹)."))
    _skipped = _file.spectrum.metadata.get("spectra_not_read")
    if _skipped:
        st.sidebar.info(ui(f"{_file.source}: 上限を超えた {_skipped} 本は読み込んでいません。",
                           f"{_file.source}: {_skipped} spectra beyond the limit were not read."))

# ----- projects: open, resume the autosave, start afresh (the download button is drawn with the figure)
st.sidebar.header(T["session_header"])
st.sidebar.caption(T["project_help"])
session_upload = st.sidebar.file_uploader(T["load_session"], type=["json"], key="_session_upload",
                                          on_change=_open_uploaded_project)
if st.session_state.pop("_project_loaded", False):
    st.sidebar.success(T["project_loaded"])
if st.session_state.get("_project_error"):
    st.sidebar.error(T["project_error"] + st.session_state["_project_error"])
if LOCAL_DATA_DIR and (LOCAL_DATA_DIR / "autosave.json").is_file() and not st.session_state.get("_spectrum_bytes"):
    st.sidebar.button(T["resume_local"], on_click=_open_local_autosave, **WIDE_BUTTON)
st.sidebar.button(T["new_project"], on_click=_new_project, **WIDE_BUTTON)
autosave_status = st.sidebar.empty()

# ----- graph settings
st.sidebar.header(T["graph_settings"])
_data_min = min(float(f.spectrum.x.min()) for f in active) if active else 0.0
_data_max = max(float(f.spectrum.x.max()) for f in active) if active else 4000.0
_slider_lo = max(X_LIMITS[0], float(np.floor(_data_min / 100) * 100 - 100)) if active else 0.0
_slider_hi = min(X_LIMITS[1], float(np.ceil(_data_max / 100) * 100 + 100)) if active else 4000.0
if _slider_hi <= _slider_lo:
    _slider_hi = _slider_lo + 100.0
# The range follows the data until the person moves the slider (a restored project keeps its own).
if st.session_state.get("_xrange_auto") is None:
    st.session_state["_xrange_auto"] = st.session_state.get("xrange") is None
_data_range = (float(np.floor(_data_min / 10) * 10), float(np.ceil(_data_max / 10) * 10)) if active else (100.0, 2000.0)
if active and st.session_state["_xrange_auto"]:
    st.session_state["xrange"] = _data_range


def _xrange_edited():
    st.session_state["_xrange_auto"] = False


def _show_data_range():
    st.session_state["xrange"] = _data_range
    st.session_state["_xrange_auto"] = True


if active:
    st.sidebar.button(T["data_range"], on_click=_show_data_range, **WIDE_BUTTON)
_current = st.session_state.get("xrange") or (100.0, 2000.0)
_current = (float(min(max(_current[0], _slider_lo), _slider_hi - 5)), float(min(max(_current[1], _slider_lo + 5), _slider_hi)))
if _current[0] >= _current[1]:
    _current = (_slider_lo, _slider_hi)
st.session_state["xrange"] = _current
xrange = st.sidebar.slider(T["xrange_ir"] if IS_FTIR else T["xrange"], _slider_lo, _slider_hi, step=5.0, key="xrange", on_change=_xrange_edited)
x_min, x_max = float(xrange[0]), float(xrange[1])

_widget_default("normalize_mode", "max")
normalize_mode = st.sidebar.selectbox(ui("正規化", "Normalisation"), ["none", "max", "window", "minmax", "area"], key="normalize_mode",
    format_func=lambda v: {"none": ui("しない（元の強度）", "None (as measured)"), "max": ui("最大値 = 1", "Maximum = 1"),
                           "window": ui("指定範囲の最大値 = 1", "Maximum within a window = 1"),
                           "minmax": ui("最小 0・最大 1", "Minimum 0, maximum 1"), "area": ui("面積 = 1", "Unit area")}[v],
    help=ui("表示範囲・前処理のあとに掛けます。「指定範囲の最大値」は、基準にするピーク（例：SiC 796 cm⁻¹）の範囲で揃えるときに使います。",
            "Applied after the range and the processing. 'Within a window' scales to a reference peak (e.g. SiC at 796 cm⁻¹)."))
if normalize_mode == "window":
    _nw = st.session_state.get("norm_window")
    if not (isinstance(_nw, (list, tuple)) and len(_nw) == 2 and x_min <= _nw[0] < _nw[1] <= x_max):
        st.session_state["norm_window"] = (x_min, x_max)
    norm_window = st.sidebar.slider(ui("正規化の基準範囲 (cm⁻¹)", "Normalisation window (cm⁻¹)"), x_min, x_max, step=1.0, key="norm_window")
else:
    norm_window = None
stack = st.sidebar.checkbox(ui("積み重ねる（自動オフセット）", "Stack spectra (automatic offsets)"),
                            value=_widget_default("stack", True, checkbox=True), key="stack",
                            help=ui("表示順に下から積みます。外すと、各スペクトルの Y位置と倍率を手で決めます。",
                                    "Stacks in display order from the bottom. Unchecked: set each spectrum's Y position and scale by hand."))
global_offset = (st.sidebar.slider(ui("オフセット（倍率）", "Offset (multiplier)"), 0.0, 3.0,
                                   _widget_default("global_offset", 1.0), step=0.05, key="global_offset",
                                   help=ui("前のスペクトルの最大値に対する比。0 で重ね描きになります。",
                                           "Fraction of the previous spectrum's maximum; 0 overlays the spectra."))
                 if stack else 0.0)
if "show_legend" not in st.session_state:
    st.session_state["show_legend"] = True
show_legend = st.sidebar.checkbox(T["show_legend_cb"], key="show_legend")
_widget_default("legend_loc", "upper right")
legend_loc = st.sidebar.selectbox(ui("凡例の位置", "Legend position"),
    ["upper right", "upper left", "lower right", "lower left", "best"], key="legend_loc",
    format_func=lambda v: {"upper right": ui("右上", "Upper right"), "upper left": ui("左上", "Upper left"),
                           "lower right": ui("右下", "Lower right"), "lower left": ui("左下", "Lower left"),
                           "best": ui("自動", "Automatic")}[v]) if show_legend else "upper right"
show_yticks = st.sidebar.checkbox(ui("縦軸の目盛りを表示", "Show y-axis ticks"),
                                  value=_widget_default("show_yticks", False, checkbox=True), key="show_yticks")

with st.sidebar.expander(ui("前処理（スパイク除去・平滑化・ベースライン）", "Processing (spikes, smoothing, baseline)")):
    st.caption(ui("順番：表示範囲で切り出し → スパイク除去 → 平滑化 → ベースライン差し引き → 正規化。元のデータは変えません。"
                  + ("　FT-IR では吸光度に直してから処理し、表示のときに透過率へ戻します。" if IS_FTIR else ""),
                  "Order: crop to the range → spikes → smoothing → baseline subtraction → normalisation. The data files stay as they are."
                  + (" FT-IR data are processed in absorbance and converted back for a transmittance display." if IS_FTIR else "")))
    spike_remove = st.checkbox(ui("宇宙線スパイクを除去", "Remove cosmic-ray spikes"),
                               value=_widget_default("spike_remove", False, checkbox=True), key="spike_remove")
    spike_threshold = st.number_input(ui("スパイクのしきい値（修正 z スコア）", "Spike threshold (modified z-score)"), 1.0, 50.0,
                                      value=_widget_default("spike_threshold", 6.0, number=True), step=0.5, key="spike_threshold",
                                      help=ui("隣の点との差の修正 z スコアがこの値を超える点を、周りの平均で置き換えます（Whitaker–Hayes 法）。小さいほど多く除きます。",
                                              "Points whose intensity jump exceeds this modified z-score are replaced by the mean of their neighbours (Whitaker–Hayes). Smaller removes more.")) if spike_remove else 6.0
    smooth_on = st.checkbox(ui("平滑化（Savitzky–Golay）", "Smooth (Savitzky–Golay)"),
                            value=_widget_default("smooth_on", False, checkbox=True), key="smooth_on")
    if smooth_on:
        _sc1, _sc2 = st.columns(2)
        smooth_window = _sc1.number_input(ui("窓の点数（奇数）", "Window (points, odd)"), 3, 101,
                                          value=_widget_default("smooth_window", 7, number=True), step=2, key="smooth_window")
        smooth_order = _sc2.number_input(ui("多項式の次数", "Polynomial order"), 1, 5,
                                         value=_widget_default("smooth_order", 2, number=True), step=1, key="smooth_order")
    else:
        smooth_window, smooth_order = 0, 2
    _widget_default("baseline_method", "none")
    baseline_method = st.selectbox(ui("ベースライン", "Baseline"), ["none", "offset", "linear", "poly", "als"], key="baseline_method",
        format_func=lambda v: {"none": ui("差し引かない", "None"), "offset": ui("最小値を 0 にする", "Shift the minimum to 0"),
                               "linear": ui("直線（両端を結ぶ）", "Straight line through the ends"),
                               "poly": ui("多項式（ピークの下をなぞる）", "Polynomial under the peaks"),
                               "als": ui("非対称最小二乗 (ALS)", "Asymmetric least squares (ALS)")}[v],
        help=ui("多項式は Lieber & Mahadevan-Jansen の反復フィット、ALS は Eilers & Boelens の方法です。",
                "Polynomial: the iterative fit of Lieber & Mahadevan-Jansen; ALS: Eilers & Boelens."))
    baseline_params = {}
    if baseline_method == "poly":
        baseline_params["order"] = st.number_input(ui("多項式の次数", "Polynomial order"), 0, 12,
                                                   value=_widget_default("baseline_order", 3, number=True), step=1, key="baseline_order")
    elif baseline_method == "als":
        _bc1, _bc2 = st.columns(2)
        _lam = _bc1.number_input(ui("滑らかさ log₁₀λ", "Smoothness log₁₀λ"), 1.0, 10.0,
                                 value=_widget_default("baseline_lam", 5.0, number=True), step=0.5, key="baseline_lam",
                                 help=ui("3 でスペクトルに沿い、7 でほぼ直線になります。", "3 follows the spectrum closely, 7 is nearly a line."))
        _p = _bc2.number_input(ui("非対称度 p", "Asymmetry p"), 0.0001, 0.5,
                               value=_widget_default("baseline_p", 0.01, number=True), step=0.005, format="%.4f", key="baseline_p",
                               help=ui("小さいほどベースラインがピークの下に留まります。", "Smaller keeps the baseline under the peaks."))
        baseline_params.update(lam=float(10 ** _lam), p=float(_p))
    st.markdown(ui("**処理の確認**", "**Check the processing**"))
    show_raw = st.checkbox(ui("処理前のデータを薄く重ねる", "Overlay the data as measured (faint)"),
                           value=_widget_default("show_raw", False, checkbox=True), key="show_raw",
                           help=ui("スパイク除去・平滑化・ベースライン差し引きの前のデータを、処理後と同じ縮尺で薄い線で描きます。処理でピークの形や位置が変わっていないかを確かめられます。保存する図にも出ます。",
                                   "Draws the data before spike removal, smoothing and baseline subtraction as a faint line on the scale of the processed spectrum, to check that the processing has not changed peak shapes or positions. It appears in the saved figure too."))
    show_baseline = st.checkbox(ui("差し引いたベースラインを重ねる", "Overlay the subtracted baseline"),
                                value=_widget_default("show_baseline", False, checkbox=True), key="show_baseline",
                                help=ui("処理前のデータの下に、差し引いたベースラインを点線で描きます。", "Draws the subtracted baseline as a dotted line under the data as measured."))

st.sidebar.subheader(T["tick_settings"])
# Tick spacing follows the displayed range until the person sets it (a restored project keeps its own).
_span = x_max - x_min
_default_major, _default_minor = (100.0, 50.0) if _span <= 1500 else (500.0, 100.0) if _span <= 4500 else (1000.0, 200.0)
if st.session_state.get("_ticks_auto") is None:
    st.session_state["_ticks_auto"] = st.session_state.get("major_tick") is None
if st.session_state["_ticks_auto"]:
    st.session_state["major_tick"], st.session_state["minor_tick"] = _default_major, _default_minor


def _ticks_edited():
    st.session_state["_ticks_auto"] = False


major_tick = st.sidebar.number_input(T["major_tick"], min_value=1.0, max_value=2000.0,
                                     value=_widget_default("major_tick", _default_major, number=True), step=10.0, key="major_tick",
                                     on_change=_ticks_edited)
show_minor = st.sidebar.checkbox(T["show_minor_cb"], value=_widget_default("show_minor", True, checkbox=True), key="show_minor")
minor_tick = st.sidebar.number_input(T["minor_tick"], min_value=0.5, max_value=1000.0,
                                     value=_widget_default("minor_tick", _default_minor, number=True), step=5.0, key="minor_tick",
                                     on_change=_ticks_edited) if show_minor else None

st.sidebar.subheader(ui("軸のラベル", "Axis labels"))
_axis_label_help = ui(
    "グラフ描画appと同じ書き方です。文字を選んで Ctrl+I で斜体、Ctrl+B で太字、Ctrl+,（カンマ）で下付き、"
    "Ctrl+.（ピリオド）で上付き（Mac は ⌘ でも可）。入力欄では `<i>θ</i>`、`_{2}`、`^{-1}` のような記号で出て、"
    "Enter で図に反映されます。直接書いても同じです。空欄にするとラベルを出しません。",
    "Same markup as the graph app. Select text and press Ctrl+I for italic, Ctrl+B bold, Ctrl+, (comma) subscript, "
    "Ctrl+. (period) superscript (⌘ on a Mac). The box shows codes such as `<i>θ</i>`, `_{2}`, `^{-1}`; press Enter "
    "to update the figure. Typing them works too. Leave empty for no label.")
_apply_mode_labels()      # a default title follows the measurement / y display; a title of one's own stays
x_title = st.sidebar.text_input(ui("横軸", "x axis"), key="x_label", help=_axis_label_help)
y_title = st.sidebar.text_input(ui("縦軸", "y axis"), key="y_label", help=_axis_label_help)
st.sidebar.caption(ui("例：`Raman shift (cm^{-1})`、`Wavenumber (cm^{-1})`、`Yb_{2}Si_{2}O_{7}`、`<i>E</i>_{2g}`。スペクトルのラベルも同じ書き方です。",
                      "e.g. `Raman shift (cm^{-1})`, `Wavenumber (cm^{-1})`, `Yb_{2}Si_{2}O_{7}`, `<i>E</i>_{2g}`. Spectrum labels use the same markup."))
x_reverse = st.sidebar.checkbox(T["x_reverse"], value=_widget_default("x_reverse", IS_FTIR, checkbox=True), key="x_reverse",
                                help=ui("FT-IR の慣例（4000 → 400 cm⁻¹）。ラマンでも使えます。", "The FT-IR convention (4000 → 400 cm⁻¹); available for Raman too."))

_LR_OPTIONS = ["左", "右"]
_lr_fmt = lambda x: T["left"] if x == "左" else T["right"]
st.sidebar.subheader(T["sample_labels"])
show_side_labels = st.sidebar.checkbox(T["show_side_labels_cb"], value=_widget_default("show_side_labels", False, checkbox=True), key="show_side_labels")
if show_side_labels:
    _widget_default("label_side", "右")
    label_side = st.sidebar.radio(T["label_pos"], _LR_OPTIONS, format_func=_lr_fmt, horizontal=True, key="label_side")
    label_fontsize = st.sidebar.slider(T["label_fontsize"], 4, 28, _widget_default("label_fontsize", 9), key="label_fontsize")
    label_offset_x = st.sidebar.slider(T["label_offset_x"], -200.0, 200.0, _widget_default("label_offset_x", 20.0), step=5.0,
                                       help=T["label_offset_x_help"], key="label_offset_x")
    label_offset_y = st.sidebar.slider(T["label_offset_y"], -0.5, 1.5, _widget_default("label_offset_y", 0.1), step=0.01,
                                       help=T["label_offset_y_help"], key="label_offset_y")
else:
    label_side, label_fontsize, label_offset_x, label_offset_y = "右", 9, 20.0, 0.1

st.sidebar.header(T["fig_size"])
st.sidebar.selectbox(ui("論文図のひな形", "Publication figure preset"), list(FIGURE_PRESETS_MM),
    format_func=lambda key: {"custom": ui("自由指定", "Custom"),
        "single": ui("1段幅の例 · 85 × 70 mm", "Single column example · 85 × 70 mm"),
        "double": ui("2段幅の例 · 180 × 120 mm", "Double column example · 180 × 120 mm"),
        "figapp": ui("グラフ描画appと同じ体裁 · 88 × 70 mm", "Graph app look · 88 × 70 mm")}[key],
    key="figure_preset", on_change=_apply_figure_preset)
st.sidebar.caption(ui("ひな形は寸法・文字サイズ・余白を設定します。「グラフ描画appと同じ体裁」は Arial 7 pt・目盛り 6.5 pt・細い線で、グラフ描画app・XRD描画appの図と並べられます。",
                      "Presets set dimensions, font sizes and margins. 'Graph app look' uses Arial 7 pt, 6.5 pt ticks and thin lines to sit next to the graph app's and XRD Maker's figures."))
size_unit = st.sidebar.radio(ui("寸法の単位", "Size unit"), ["mm", "inch"], horizontal=True, key="size_unit")
_signature = (size_unit, st.session_state["fig_width"], st.session_state["fig_height"])
if st.session_state.get("_size_signature") != _signature:
    for _axis in ("width", "height"):
        _value = st.session_state[f"fig_{_axis}"]
        st.session_state[f"_figure_{_axis}"] = inches_to_mm(_value) if size_unit == "mm" else _value
    st.session_state["_size_signature"] = _signature
for _axis, _label in (("width", ui("図の幅", "Figure width")), ("height", ui("図の高さ", "Figure height"))):
    st.sidebar.number_input(f"{_label} ({size_unit})", min_value=25.0 if size_unit == "mm" else 25 / 25.4,
        max_value=508.0 if size_unit == "mm" else 20.0, step=1.0 if size_unit == "mm" else 0.1,
        format="%.2f", key=f"_figure_{_axis}", on_change=_update_figure_dimension, args=(_axis,))
fig_width, fig_height = st.session_state["fig_width"], st.session_state["fig_height"]
dpi_export = st.sidebar.selectbox("Output DPI", [300, 600], index=0, key="dpi_export")
st.session_state["font_size"] = float(st.session_state.get("font_size") or 7.0)
for _key in ("tick_font_size", "legend_font_size"):
    st.session_state[_key] = float(st.session_state.get(_key) or st.session_state["font_size"])
st.sidebar.caption(ui("文字の大きさ (pt)", "Text sizes (pt)"))
_size_columns = st.sidebar.columns(3)
for _column, _key, _label in ((_size_columns[0], "font_size", ui("軸ラベル", "Axis labels")),
                              (_size_columns[1], "tick_font_size", ui("目盛り", "Ticks")),
                              (_size_columns[2], "legend_font_size", ui("凡例", "Legend"))):
    _column.number_input(_label, min_value=4.0, max_value=28.0, step=0.5, format="%.1f", key=_key)
font_size = st.session_state["font_size"]
tick_font_size = st.session_state["tick_font_size"]
legend_font_size = st.session_state["legend_font_size"]
# Tick labels on the y axis need room between the frame and the axis title: widen the left margin
# while they are shown (and put it back when they are hidden) until the margins are edited by hand.
if st.session_state.get("_margin_top_auto", True):
    if show_yticks and st.session_state["margin_left"] < 0.13:
        st.session_state["_margin_left_before"] = st.session_state["margin_left"]
        st.session_state["margin_left"] = 0.15
    elif not show_yticks and st.session_state.get("_margin_left_before") is not None and st.session_state["margin_left"] == 0.15:
        st.session_state["margin_left"] = st.session_state.pop("_margin_left_before")
with st.sidebar.expander(ui("線・フォント・余白", "Lines, fonts and margins")):
    st.selectbox(ui("フォント", "Font family"), list(FONT_FAMILIES), key="font_family",
                 help=ui("グラフ描画appと同じ選択肢です。斜体・下付き・上付きも同じ書体で描きます。",
                         "The graph app's choices. Italics, subscripts and superscripts use the same font."))
    _actual_font = resolve_font(st.session_state["font_family"])
    if _actual_font != st.session_state["font_family"]:
        if _actual_font in METRIC_COMPATIBLE.get(st.session_state["font_family"], ()):
            st.caption(ui(f"この環境には {st.session_state['font_family']} がないため、字幅が同じ {_actual_font} で描きます。"
                          f"SVG には {st.session_state['font_family']} と書き込むので、手元で開けば {st.session_state['font_family']} になります。",
                          f"{st.session_state['font_family']} is not installed here, so {_actual_font} (same character widths) is used. "
                          f"SVG files name {st.session_state['font_family']}, so they open in it on your computer."))
        else:
            st.caption(ui(f"この環境では {_actual_font} を使用します。", f"This environment uses {_actual_font}."))
    for _key, _label in (("line_width", ui("スペクトルの線幅 (pt)", "Spectrum line width (pt)")),
                        ("reference_line_width", ui("注記の線幅 (pt)", "Mark line width (pt)")),
                        ("axes_line_width", ui("軸・目盛りの線幅 (pt)", "Axes and tick width (pt)"))):
        st.number_input(_label, .1, 5., step=.1, key=_key)
    st.number_input(ui("目盛りの長さ (pt)", "Tick length (pt)"), .5, 10., step=.5, key="tick_length",
                    help=ui("主目盛りの長さ。副目盛りはその半分です。", "Major tick length; minor ticks are half as long."))
    st.selectbox(ui("目盛りの向き", "Tick direction"), ["auto", "in", "out"], key="tick_direction",
                 format_func=lambda value: {"auto": ui("標準（内向き）", "Default (inward)"), "in": ui("内向き", "Inward"), "out": ui("外向き", "Outward")}[value])
    st.checkbox(ui("上と右の枠線を描く", "Draw the top and right frame"), key="axes_box")
    for _key, _label in (("margin_left", ui("左余白 (%)", "Left margin (%)")),
                        ("margin_right", ui("右余白 (%)", "Right margin (%)")),
                        ("margin_bottom", ui("下余白 (%)", "Bottom margin (%)")),
                        ("margin_top", ui("上余白 (%)", "Top margin (%)"))):
        _fraction = st.session_state[_key]
        st.session_state[f"_percent_{_key}"] = round(100 * (1 - _fraction if _key in ("margin_right", "margin_top") else _fraction), 8)
        st.number_input(_label, 0., 100., step=1., key=f"_percent_{_key}", on_change=_update_margin, args=(_key,))
    st.checkbox(ui("背景を透明にする", "Transparent background"), key="transparent_background")
    st.checkbox(ui("SVGの文字を編集可能にする", "Keep SVG text editable"), key="svg_text_editable")
    st.caption(ui("編集可能なSVGの表示には、開くPCにも使用フォントが必要です。", "Editable SVG text uses fonts installed on the computer opening it."))
plot_style = {key: st.session_state[key] for key in STYLE_KEYS}
try:
    plot_style = validate_style_settings(plot_style, fill_defaults=True)
except ValueError:
    st.sidebar.error(ui("余白の合計が100%未満になるよう調整してください。", "Adjust opposing margins so their sum is below 100%."))
    st.stop()

with st.sidebar.expander(ui("作図スタイルの保存・読み込み", "Save and load figure styles")):
    st.caption(ui("寸法・線・フォント・ラベル位置・注記の体裁を再利用できます。測定データや試料名は含みません。XRD描画appのスタイルも開けます（共通の項目だけ）。",
                  "Reuse dimensions, lines, fonts, label positions and mark styles. Styles do not include measurements or sample names. XRD Maker styles open too (shared settings only)."))
    st.file_uploader(ui("スタイルを開く (.json)", "Open figure style (.json)"), type=["json"], key="_style_upload", on_change=_open_style_template)
    if st.session_state.pop("_style_loaded", False):
        st.success(ui("作図スタイルを読み込みました。", "Figure style loaded."))
    if st.session_state.get("_style_error"):
        st.error(st.session_state["_style_error"])
    st.download_button(ui("スタイルを保存 (.json)", "Save figure style (.json)"),
        encode_style_template(dict(st.session_state)), file_name="raman_figure_style.json", mime="application/json", **WIDE_DOWNLOAD)

with st.sidebar.expander(ui("ピーク・注記・拡大図", "Peaks, marks and zoom inset")):
    st.markdown(ui("**ピークの自動検出**", "**Automatic peak detection**"))
    show_peaks = st.checkbox(ui("ピーク位置を表示", "Show peak positions"), value=_widget_default("show_peaks", False, checkbox=True), key="show_peaks")
    peak_prom = st.slider(ui("ピーク感度（プロミネンス比）", "Peak sensitivity (prominence fraction)"), 0.005, 0.9,
                          _widget_default("peak_prom", 0.1), step=0.005, key="peak_prom",
                          help=ui("スペクトルの強度幅に対するプロミネンスの比。小さいほど弱いピークも拾います。",
                                  "Prominence as a fraction of the spectrum's intensity range. Smaller finds weaker peaks."))
    _pc1, _pc2 = st.columns(2)
    peak_distance = _pc1.number_input(ui("最小間隔 (cm⁻¹)", "Minimum spacing (cm⁻¹)"), 1.0, 500.0,
                                      value=_widget_default("peak_distance", 5.0, number=True), step=1.0, key="peak_distance")
    peak_decimals = _pc2.number_input(ui("小数点以下の桁", "Decimals"), 0, 3, value=_widget_default("peak_decimals", 0, number=True), step=1, key="peak_decimals")
    peak_label_rotate = st.checkbox(ui("ピーク値を縦書き（90°）にする", "Rotate peak values (90°)"),
                                    value=_widget_default("peak_label_rotate", True, checkbox=True), key="peak_label_rotate")
    peak_fit = st.checkbox(ui("ピーク位置をフィットで求める（ピーク一覧）", "Fit peak positions (peak table)"),
                           value=_widget_default("peak_fit", False, checkbox=True), key="peak_fit",
                           help=ui("検出した各ピークの ± 半幅の範囲を、直線ベースライン付きのローレンツ／ガウス関数でフィットします。図の値は極大点、ピーク一覧にフィット中心と半値幅が出ます。",
                                   "Fits each detected peak within ± half-window with a Lorentzian or Gaussian on a linear background. The figure shows the maxima; the peak table lists fitted centres and FWHM."))
    if peak_fit:
        _pf1, _pf2 = st.columns(2)
        _widget_default("peak_model", "lorentzian")
        peak_model = _pf1.selectbox(ui("関数", "Model"), ["lorentzian", "gaussian"], key="peak_model",
                                    format_func=lambda v: {"lorentzian": ui("ローレンツ", "Lorentzian"), "gaussian": ui("ガウス", "Gaussian")}[v])
        peak_half_window = _pf2.number_input(ui("フィットの半幅 (cm⁻¹)", "Fit half-window (cm⁻¹)"), 2.0, 200.0,
                                             value=_widget_default("peak_half_window", 15.0, number=True), step=1.0, key="peak_half_window")
    else:
        peak_model, peak_half_window = "lorentzian", 15.0
    st.markdown(ui("**手入力の注記（位置の線と帯）**", "**Hand-written marks (guide lines and bands)**"))
    _widget_default("marks_text", "")
    marks_text = st.text_area(ui("1 行に 1 つ：`1080 Si–O–Si`、`3400`、帯は `3200-3600 O–H`", "One per line: `1080 Si–O–Si`, `3400`, a band: `3200-3600 O–H`")
                              if IS_FTIR else ui("1 行に 1 つ：`796 SiC (TO)`、`924`、帯は `770-822 window`", "One per line: `796 SiC (TO)`, `924`, a band: `770-822 window`"),
                              key="marks_text", height=110,
                              help=ui("数値のあとに空白かカンマを置いてラベルを書きます。ラベルには `_{2}`、`<i>E</i>` などの書式が使えます。`#` 以降はメモです。",
                                      "A value, then a space or comma, then the label; labels take the same markup (`_{2}`, `<i>E</i>`). Text after `#` is a note."))
    marks, bands = parse_marks(marks_text)
    _widget_default("marks_style", "dashed")
    _widget_default("marks_label_mode", "top")
    _mk1, _mk2 = st.columns(2)
    marks_style = _mk1.selectbox(ui("線の種類", "Line style"), ["dashed", "dotted", "solid", "none"], key="marks_style",
                                 format_func=lambda v: {"dashed": ui("破線", "Dashed"), "dotted": ui("点線", "Dotted"), "solid": ui("実線", "Solid"), "none": ui("線なし", "No line")}[v])
    marks_label_mode = _mk2.selectbox(ui("ラベルの置き方", "Label placement"), ["top", "rotated", "none"], key="marks_label_mode",
                                      format_func=lambda v: {"top": ui("上に横書き", "Above, upright"), "rotated": ui("上に縦書き", "Above, rotated"), "none": ui("ラベルなし", "No label")}[v])
    _mk3, _mk4, _mk5 = st.columns(3)
    marks_show_value = _mk3.checkbox(ui("数値も表示", "Show the value"), value=_widget_default("marks_show_value", True, checkbox=True), key="marks_show_value")
    marks_decimals = _mk4.number_input(ui("桁", "Decimals"), 0, 3, value=_widget_default("marks_decimals", 0, number=True), step=1, key="marks_decimals")
    marks_fontsize = _mk5.number_input(ui("文字 (pt)", "Size (pt)"), 4, 20, value=_widget_default("marks_fontsize", 6, number=True), step=1, key="marks_fontsize")
    _widget_default("marks_color", "#555555")
    _widget_default("bands_color", "#BBBBBB")
    _mk6, _mk7 = st.columns(2)
    marks_color = _mk6.color_picker(ui("線とラベルの色", "Line and label colour"), key="marks_color")
    bands_color = _mk7.color_picker(ui("帯の色", "Band colour"), key="bands_color")
    _mk8, _mk9 = st.columns(2)
    bands_alpha = _mk8.slider(ui("帯の濃さ", "Band opacity"), 0.0, 1.0, _widget_default("bands_alpha", 0.3), step=0.05, key="bands_alpha")
    bands_label = _mk9.checkbox(ui("帯のラベルを表示", "Show band labels"), value=_widget_default("bands_label", True, checkbox=True), key="bands_label")
    _label_height = top_labels_height(marks, bands, x_range=(x_min, x_max), label_mode=marks_label_mode, show_value=marks_show_value,
                                      decimals=marks_decimals, show_band_labels=bands_label, plain=plain,
                                      axes_width_points=fig_width * 72 * (plot_style["margin_right"] - plot_style["margin_left"]),
                                      font_size=marks_fontsize)
    if _label_height > 0:
        _needed_top = max(round(1 - (_label_height + 3) / (fig_height * 72), 3), plot_style["margin_bottom"] + 0.05)
        if plot_style["margin_top"] > _needed_top + 0.002:
            if st.session_state.get("_margin_top_auto", True):
                # The labels sit above the axes: widen the top margin for them until the person sets margins by hand.
                st.session_state["margin_top"] = plot_style["margin_top"] = _needed_top
                st.caption(ui(f"ラベルが収まるように上余白を {100 * (1 - _needed_top):.0f}% にしました（余白を手で変えると自動調整は止まります）。",
                              f"The top margin was set to {100 * (1 - _needed_top):.0f}% for the labels (editing a margin by hand stops this)."))
            else:
                st.caption(ui(f"ラベルは図の上に出ます。上余白を {100 * (1 - _needed_top):.0f}% 以上にすると切れません。",
                              f"The labels sit above the axes; a top margin of {100 * (1 - _needed_top):.0f}% or more keeps them inside."))
                st.button(ui("上余白をラベルに合わせる", "Fit the top margin to the labels"), key="_fit_top_margin",
                          on_click=lambda: st.session_state.update(margin_top=_needed_top), **WIDE_BUTTON)
    st.markdown(ui("**拡大図**", "**Zoom inset**"))
    show_zoom = st.checkbox(ui("拡大図を表示", "Show zoom inset"), value=_widget_default("show_zoom", False, checkbox=True), key="show_zoom")
    _zr = st.session_state.get("zoom_range")
    if not (isinstance(_zr, (list, tuple)) and len(_zr) == 2 and x_min <= _zr[0] < _zr[1] <= x_max):
        st.session_state["zoom_range"] = (x_min + 0.25 * (x_max - x_min), x_min + 0.45 * (x_max - x_min))
    zoom_range = st.slider(ui("拡大する範囲 (cm⁻¹)", "Inset range (cm⁻¹)"), x_min, x_max, step=1.0, key="zoom_range")
    _widget_default("zoom_location", "upper left")
    zoom_location = st.selectbox(ui("拡大図の位置", "Inset position"), ["upper left", "upper right", "lower left", "lower right"],
        format_func=lambda value: {"upper left": ui("左上", "Upper left"), "upper right": ui("右上", "Upper right"), "lower left": ui("左下", "Lower left"), "lower right": ui("右下", "Lower right")}[value], key="zoom_location")
    annotation_font_size = st.slider(ui("ピーク値・拡大図の文字サイズ", "Peak value and inset font size"), 4, 20, _widget_default("annotation_font_size", 6), key="annotation_font_size")

with st.sidebar.expander(ui("ピーク分離フィット", "Fit overlapping peaks")):
    st.caption(ui("指定した範囲を、指定した数のピークの和でフィットします。成分の数と初期位置は自分で決めます（自動では増やしません）。前処理後のデータに対して、表示中のすべてのスペクトルを同じ条件でフィットします。",
                  "Fits the chosen range with the sum of the peaks you list. The number of components and their starting positions are yours (none is added automatically). All visible spectra are fitted on the processed data with the same settings."))
    mfit_on = st.checkbox(ui("フィットする", "Fit"), value=_widget_default("mfit_on", False, checkbox=True), key="mfit_on")
    _mr = st.session_state.get("mfit_range")
    if not (isinstance(_mr, (list, tuple)) and len(_mr) == 2 and x_min <= _mr[0] < _mr[1] <= x_max):
        st.session_state["mfit_range"] = (x_min, x_max)
    mfit_range = st.slider(ui("フィットする範囲 (cm⁻¹)", "Fit range (cm⁻¹)"), x_min, x_max, step=1.0, key="mfit_range")
    _widget_default("mfit_centers", "")
    mfit_centers_text = st.text_input(ui("ピークの初期位置 (cm⁻¹)", "Starting positions (cm⁻¹)"), key="mfit_centers",
                                      placeholder=ui("例: 1350, 1580", "e.g. 1350, 1580"),
                                      help=ui(f"カンマか空白で区切ります（最大 {MAX_FIT_PEAKS} 本）。フィット範囲の外の値は使いません。",
                                              f"Separate with commas or spaces (up to {MAX_FIT_PEAKS}). Values outside the fit range are not used."))
    _mf1, _mf2 = st.columns(2)
    _widget_default("mfit_model", "lorentzian")
    mfit_model = _mf1.selectbox(ui("関数", "Model"), ["lorentzian", "gaussian", "pvoigt"], key="mfit_model",
                                format_func=lambda v: {"lorentzian": ui("ローレンツ", "Lorentzian"), "gaussian": ui("ガウス", "Gaussian"),
                                                       "pvoigt": ui("擬フォークト", "Pseudo-Voigt")}[v],
                                help=ui("結晶のラマンバンドはローレンツ、ガラス・非晶質や装置幅が効く場合はガウス寄りです。擬フォークトは両者の混合比 η も合わせるので自由度が増えます。",
                                        "Raman bands of crystals are close to Lorentzian; glasses, amorphous phases and instrument broadening are closer to Gaussian. The pseudo-Voigt also fits the mixing ratio η, which adds freedom."))
    _widget_default("mfit_background", "linear")
    mfit_background = _mf2.selectbox(ui("バックグラウンド", "Background"), ["none", "constant", "linear"], key="mfit_background",
                                     format_func=lambda v: {"none": ui("なし", "None"), "constant": ui("定数", "Constant"),
                                                            "linear": ui("直線", "Linear")}[v])
    mfit_max_shift = st.number_input(ui("中心の動ける幅 ± (cm⁻¹)", "Allowed centre shift ± (cm⁻¹)"), 1.0, 500.0,
                                     value=_widget_default("mfit_max_shift", 15.0, number=True), step=1.0, key="mfit_max_shift",
                                     help=ui("各ピークの中心は初期位置からこの幅までしか動きません。上限に張り付いた成分は結果の表に印が付きます。",
                                             "Each centre may move this far from its starting position. Components that end at a limit are flagged in the results."))
    mfit_show_total = st.checkbox(ui("合計の曲線を図に描く", "Draw the summed curve"), value=_widget_default("mfit_show_total", True, checkbox=True), key="mfit_show_total")
    mfit_show_components = st.checkbox(ui("各成分を図に描く", "Draw the components"), value=_widget_default("mfit_show_components", True, checkbox=True), key="mfit_show_components")
    mfit_fill = st.checkbox(ui("成分を薄く塗る", "Shade the components"), value=_widget_default("mfit_fill", False, checkbox=True), key="mfit_fill")
mfit_centers = [c for c in parse_centers(mfit_centers_text) if mfit_range[0] <= c <= mfit_range[1]][:MAX_FIT_PEAKS]

st.sidebar.divider()


# ===== メインエリア =====
show_panel = st.toggle(T["show_panel"], value=_widget_default("show_panel", True, checkbox=True), key="show_panel")

if active:
    if show_panel:
        col_graph, col_settings = st.columns([7, 3])
    else:
        col_graph = st.container()
        col_settings = None

    for _key, _value in (("palette", DEFAULT_PALETTE), ("palette_reverse", False), ("palette_steps", 0),
                         ("palette_from", 1), ("palette_to", 0), ("palette_base_color", "#000000")):
        st.session_state.setdefault(_key, _value)
    if st.session_state["palette"] not in PALETTES:
        st.session_state["palette"] = DEFAULT_PALETTE
    palette_defaults, palette_count = palette_plan()

    orders, visibles, labels, colors_sel, abs_offsets, scales = [], [], [], [], [], []
    if show_panel and col_settings is not None:
        with col_settings:
            with st.container(height=700):
                st.markdown(ui("#### 色", "#### Colours"))
                st.selectbox(ui("色の組み合わせ", "Palette"), list(PALETTES), key="palette",
                    format_func=lambda key: PALETTE_NAMES[_lang_code][key], on_change=_apply_palette,
                    help=ui("グラフ描画app・XRD描画appと同じ色の組み合わせです。選ぶと、表示中のスペクトルに表示順で色が付きます。"
                            "濃淡と viridis は、一番薄い色でも白地とのコントラストが 3:1 以上（細い線の目安）になる濃さから始まります。",
                            "The same palettes as the graph app and XRD Maker. Choosing one colours the visible spectra in display order. "
                            "Shades and viridis start at a colour with at least 3:1 contrast against white (the guideline for thin lines)."))
                if len(active) >= 2:
                    _n = len(active)
                    _lo, _hi = palette_range(_n)
                    st.session_state["_palette_range"] = (_lo, _hi)
                    st.slider(ui("色を付ける範囲（表示順）", "Coloured spectra (display order)"), min_value=1, max_value=_n,
                        step=1, key="_palette_range", on_change=_set_palette_range, args=(_n,),
                        help=ui("この範囲のスペクトルだけに色の組み合わせ（濃淡）を付けます。範囲外（基準の測定など）は「範囲外の色」（初期値は黒）になります。",
                                "Only these spectra take the palette (shades). Spectra outside the range, such as a base measurement, take the outside colour (black by default)."))
                    _names = [plain(st.session_state.get(f"_val_lbl_{i}") or os.path.splitext(active[i].name)[0])
                              for i in spectrum_display_order()]
                    _outside = _names[:_lo - 1] + _names[_hi:]
                    _shaded = _names[_lo - 1:_hi]
                    st.caption(ui("色の組み合わせ：", "Palette: ") + (f"{_shaded[0]} 〜 {_shaded[-1]}" if len(_shaded) > 1 else _shaded[0])
                               + (ui("　／　範囲外：", "  /  outside: ") + "、".join(_outside) if _outside else ""))
                    if _outside:
                        st.color_picker(ui("範囲外の色", "Colour outside the range"), key="palette_base_color", on_change=_apply_palette)
                st.checkbox(ui("濃淡（色の順）を逆に", "Reverse shades"), key="palette_reverse", on_change=_apply_palette,
                    help=ui("「…の濃淡」では 1 つ目のスペクトルが濃い色になります。ほかの組み合わせでは色の順が逆になります。",
                            "With shades, the first spectrum gets the dark one; other palettes reverse the order."))
                _pc1, _pc2 = st.columns([0.45, 0.55])
                _pc1.number_input(ui("濃淡の段数", "Shade steps"), min_value=0, max_value=MAX_SHADE_STEPS, step=1,
                    key="palette_steps", on_change=_apply_palette,
                    help=ui("0 はスペクトルの数で分けます。ほかの図と系列の数が違うときは、その図の系列の数を入れると同じ濃さになります（濃淡と viridis）。",
                            "0 splits the shades by the number of spectra. Enter the number of series of another figure to get its shades (shades and viridis)."))
                _pc2.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
                _pc2.button(ui("色を当てはめ直す", "Re-apply"), on_click=_apply_palette, **WIDE_BUTTON,
                    help=ui("スペクトルごとに変えた色を戻すときや、表示順・表示を変えたあとに押します。",
                            "Restores the palette after changing single colours, the display order or visibility."))

                st.markdown(T["spectra_header"])
                for i, f in enumerate(active):
                    default_name = f.default_label
                    default_hex = palette_defaults.get(f"sp_color_{i}", REFERENCE_COLORS[0])
                    with st.expander(f"**{i + 1}. {default_name}**", expanded=True):
                        order = st.number_input(T["display_order"], value=_widget_default(f"ord_{i}", i + 1, number=True),
                                                min_value=1, max_value=10000, key=f"ord_{i}")
                        visible = st.checkbox(T["show_cb"], value=_widget_default(f"vis_{i}", True, checkbox=True), key=f"vis_{i}")
                        label = label_input(key=f"lbl_{i}", default=default_name)
                        chosen_color = color_picker_popover(f"sp_color_{i}", default_hex, palette_count)
                        if IS_FTIR:
                            _detected_units = native_y_units(f.spectrum.metadata, f.spectrum.y)
                            _unit_names = {"auto": T["y_unit_auto"] + (f"：{T['y_unit_' + {'absorbance': 'abs', 'transmittance_percent': 'pct', 'transmittance_fraction': 'frac'}[_detected_units]]}" if _detected_units else ui("：記載なし → 吸光度", ": not stated → absorbance")),
                                           "absorbance": T["y_unit_abs"], "transmittance_percent": T["y_unit_pct"], "transmittance_fraction": T["y_unit_frac"]}
                            _widget_default(f"yunit_{i}", "auto")
                            st.selectbox(T["y_unit"], list(Y_UNIT_CHOICES), key=f"yunit_{i}", format_func=lambda v, names=_unit_names: names[v],
                                         help=T["y_unit_help"])
                        if stack:
                            eoff_key = f"extra_offset_{i}"
                            if eoff_key not in st.session_state:
                                st.session_state[eoff_key] = 0.0
                            st.slider(T["offset_adj"], min_value=-5.0, max_value=15.0, step=0.05, key=eoff_key)
                            abs_offsets.append(0.0)
                            scales.append(1.0)
                        else:
                            _yc1, _yc2 = st.columns(2)
                            _top = float(np.max(np.abs(f.spectrum.y))) if (normalize_mode == "none" and len(f.spectrum.y)) else 1.0
                            _step = float(10 ** (np.floor(np.log10(max(_top, 1e-9))) - 1))
                            yoff = _yc1.number_input(T["y_position"], value=_widget_default(f"yoff_{i}", 0.0, number=True),
                                                     step=_step, format="%.3f", key=f"yoff_{i}")
                            scale = _yc2.number_input(T["scale"], value=_widget_default(f"scale_{i}", 1.0, number=True),
                                                      min_value=0.0001, max_value=1e6, step=0.1, format="%.4f", key=f"scale_{i}")
                            abs_offsets.append(float(yoff))
                            scales.append(float(scale))
                        _conditions = condition_summary(f.spectrum.metadata)
                        if _conditions:
                            with st.popover(T["conditions"], **WIDE_POPOVER):
                                st.table(pd.DataFrame(_conditions, columns=[ui("項目", "Item"), ui("値", "Value")]))
                    orders.append(order)
                    visibles.append(visible)
                    labels.append(label)
                    colors_sel.append(chosen_color)
    else:
        for i, f in enumerate(active):
            key = f"sp_color_{i}"
            orders.append(st.session_state.get(f"ord_{i}", i + 1))
            visibles.append(st.session_state.get(f"vis_{i}", True))
            labels.append(st.session_state.get(f"_val_lbl_{i}", f.default_label))
            colors_sel.append(st.session_state.get(key, palette_defaults.get(key, REFERENCE_COLORS[0])))
            if stack:
                st.session_state.setdefault(f"extra_offset_{i}", 0.0)
                abs_offsets.append(0.0)
                scales.append(1.0)
            else:
                abs_offsets.append(float(st.session_state.get(f"yoff_{i}", 0.0)))
                scales.append(float(st.session_state.get(f"scale_{i}", 1.0)))
    sort_idx = sorted(range(len(active)), key=lambda i: orders[i])

    # ===== 前処理（表示範囲・スパイク・平滑化・ベースライン・正規化） =====
    _prepared: dict = {}
    _spikes_removed = 0

    def data_units(i):
        """The y units of a spectrum's file: the panel's choice, else what the file says, else absorbance."""
        choice = st.session_state.get(f"yunit_{i}", "auto")
        if choice in Y_UNIT_CHOICES[1:]:
            return choice
        return native_y_units(active[i].spectrum.metadata, active[i].spectrum.y) or "absorbance"

    def prepared(i):
        """(x, y as displayed, info); FT-IR data are processed in absorbance and converted for display.
        info['signal'] is the processed signal with peaks pointing up (used for peak detection)."""
        if i not in _prepared:
            spectrum = active[i].spectrum
            y_in = to_absorbance(spectrum.y, data_units(i)) if IS_FTIR else spectrum.y
            try:
                x, y, info = process(
                    spectrum.x, y_in, x_range=(x_min, x_max), spikes=spike_threshold if spike_remove else None,
                    smooth_window=smooth_window if smooth_on else 0, smooth_order=smooth_order,
                    baseline_method=baseline_method, baseline_params=baseline_params,
                    normalize_mode=normalize_mode, normalize_window=norm_window,
                )
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                st.warning(ui(f"{active[i].name}: 前処理できませんでした（{exc}）。元のデータを表示します。",
                              f"{active[i].name}: processing failed ({exc}); showing the data as measured."))
                mask = (spectrum.x >= x_min) & (spectrum.x <= x_max)
                x, y, info = spectrum.x[mask], y_in[mask], {"spikes": 0, "baseline": None, "raw": y_in[mask]}
            info["signal"] = y
            if PEAKS_DOWN:
                y = from_absorbance(y, "transmittance_percent")
            _prepared[i] = (x, y, info)
        return _prepared[i]

    def arranged():
        """Visible spectra in display order with their plotted intensities and stacking levels."""
        cumulative = 0.0
        rows = []
        for i in sort_idx:
            if not visibles[i]:
                continue
            x, y, info = prepared(i)
            if len(y) == 0:
                continue
            y_min, y_max = float(np.min(y)), float(np.max(y))
            height = max(y_max, 1e-9) if y_max > 0 else max(y_max - y_min, 1e-9)
            if stack:
                extra = float(st.session_state.get(f"extra_offset_{i}", 0.0))
                level = cumulative + extra * height
                y_plot = y + level
                cumulative += global_offset * height
                label_y = level + y_min + label_offset_y * (y_max - y_min)
            else:
                y_plot = y * scales[i] + abs_offsets[i]
                level = abs_offsets[i]
                label_y = level + y_min * scales[i] + label_offset_y * (y_max - y_min) * scales[i]
            # lift: processed value → plotted value (the stacking offset, or the manual scale and position)
            lift = (1.0, level) if stack else (scales[i], abs_offsets[i])
            rows.append(dict(index=i, x=x, y=y, y_plot=y_plot, level=level, label_y=label_y, signal=info["signal"],
                             color=colors_sel[i], label=labels[i], spikes=info["spikes"], lift=lift,
                             raw=info.get("raw"), baseline=info.get("baseline"),
                             norm=(info.get("scale", 1.0), info.get("shift", 0.0))))
        return rows

    curves = arranged()
    _spikes_removed = sum(r["spikes"] for r in curves)
    axes_width_points = fig_width * 72 * (plot_style["margin_right"] - plot_style["margin_left"])

    # Overlays live on the processed (absorbance) scale, so a transmittance display draws none.
    OVERLAYS = not PEAKS_DOWN

    def overlay_curves(row):
        """(kind, x, plotted y) of the data as measured and of the subtracted baseline, on the scale of
        the processed spectrum: both go through the normalisation and the row's offset."""
        out = []
        if not OVERLAYS or row["raw"] is None or len(row["raw"]) != len(row["x"]):
            return out
        scale, shift = row["norm"]
        a, b = row["lift"]
        if show_raw:
            out.append(("raw", row["x"], a * (scale * row["raw"] + shift) + b))
        if show_baseline and row["baseline"] is not None:
            out.append(("baseline", row["x"], a * (scale * row["baseline"] + shift) + b))
        return out

    _fits: dict = {}

    def fit_of(row):
        """The multi-peak fit of a spectrum's processed signal (None when fitting is off or has no centres)."""
        if not (mfit_on and mfit_centers):
            return None
        if row["index"] not in _fits:
            _fits[row["index"]] = fit_multipeak(row["x"], row["signal"], mfit_centers, window=mfit_range, model=mfit_model,
                                                background=mfit_background, max_shift=mfit_max_shift)
        return _fits[row["index"]]

    def fit_curves(row):
        """(kind, x, plotted y[, plotted background]) for the summed curve and the components; each
        component is drawn on the fitted background so that it sits on the data."""
        result = fit_of(row)
        if not OVERLAYS or result is None or not result["ok"]:
            return []
        a, b = row["lift"]
        out = []
        base = a * result["background"] + b
        if mfit_show_components:
            out += [("component", result["x"], a * (curve + result["background"]) + b, base) for curve in result["curves"]]
        if mfit_show_total:
            out.append(("total", result["x"], a * result["fit"] + b, base))
        return out

    FIT_TOTAL_COLOR = "#222222"

    def _rgba(color, alpha):
        from matplotlib.colors import to_rgb
        r, g, b = (int(round(255 * v)) for v in to_rgb(color))
        return f"rgba({r},{g},{b},{alpha})"

    def peak_indices(row):
        """Peaks of the processed signal (absorption bands point up in absorbance, so transmittance
        valleys are found too)."""
        return detect_peaks(row["x"], row["signal"], peak_prom, peak_distance) if show_peaks else np.array([], dtype=int)

    Y_VALUE_NAME = ({"absorbance": ui("吸光度", "Absorbance"), "transmittance": ui("透過率 (%)", "Transmittance (%)")}[ir_display]
                    if IS_FTIR else ui("強度", "Intensity"))
    X_VALUE_NAME = "Wavenumber (cm-1)" if IS_FTIR else "Raman shift (cm-1)"
    FILE_STEM = "ftir" if IS_FTIR else "raman"

    def peak_rows():
        rows = []
        for row in curves:
            for k in peak_indices(row):
                entry = {ui("スペクトル", "Spectrum"): plain(row["label"]), ui("位置 (cm⁻¹)", "Position (cm⁻¹)"): round(float(row["x"][k]), 2),
                         Y_VALUE_NAME: float(row["y"][k])}
                if peak_fit:
                    fitted = fit_peak(row["x"], row["signal"], float(row["x"][k]), peak_half_window, peak_model)
                    entry.update({ui("フィット中心 (cm⁻¹)", "Fitted centre (cm⁻¹)"): round(fitted["center"], 2) if fitted["ok"] else None,
                                  ui("半値幅 (cm⁻¹)", "FWHM (cm⁻¹)"): round(fitted["fwhm"], 2) if fitted["ok"] else None,
                                  ui("高さ", "Height"): fitted["height"] if fitted["ok"] else None})
                rows.append(entry)
        return rows

    def plotted_y_limits():
        """Explicit y limits only when peak values hang below transmittance valleys: the lowest
        label needs room under the curves (matplotlib and Plotly would clip it at the frame)."""
        if not (PEAKS_DOWN and show_peaks and curves):
            return None
        low = min(float(np.min(r["y_plot"])) for r in curves)
        high = max(float(np.max(r["y_plot"])) for r in curves)
        span = max(high - low, 1e-9)
        label_points = (0.55 * annotation_font_size * (int(peak_decimals) + 5) if peak_label_rotate
                        else annotation_font_size * 1.2) + 4
        axes_height_points = max(fig_height * 72 * (plot_style["margin_top"] - plot_style["margin_bottom"]), 1.0)
        return low - span * (label_points / axes_height_points) / max(1 - label_points / axes_height_points - 0.05, 0.3), high + 0.05 * span

    y_limits = plotted_y_limits()

    def side_label_position():
        """x of the in-plot sample labels and their alignment; "left" means the left edge of the axes
        even when the axis runs from high to low wavenumber."""
        left_edge, right_edge = (x_max, x_min) if x_reverse else (x_min, x_max)
        inward = -label_offset_x if x_reverse else label_offset_x
        if label_side == "左":
            return left_edge + inward, "left"
        return right_edge - inward, "right"

    # ===== 図の生成 =====
    @plt.rc_context()
    def build_figure():
        from matplotlib.ticker import MultipleLocator, MaxNLocator
        plt.rcParams.update({**font_rcparams(plot_style), "font.size": font_size, "axes.labelsize": font_size,
                             "xtick.labelsize": tick_font_size, "ytick.labelsize": tick_font_size,
                             "legend.fontsize": legend_font_size, "legend.title_fontsize": legend_font_size})
        major_len, minor_len = plot_style["tick_length"], plot_style["tick_length"] / 2
        direction = "in" if plot_style["tick_direction"] == "auto" else plot_style["tick_direction"]
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.tick_params(which="major", axis="x", length=major_len, direction=direction)
        if show_minor and minor_tick:
            ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
            ax.tick_params(which="minor", axis="x", length=minor_len, direction=direction)
        if not plot_style["axes_box"]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        prepared_curves, legend_lines, legend_levels, side_labels = [], [], [], []
        for row in curves:
            line, = ax.plot(row["x"], row["y_plot"], color=row["color"], linewidth=plot_style["line_width"], label=rich(row["label"]))
            legend_lines.append(line)
            legend_levels.append(row["level"])
            prepared_curves.append(PreparedCurve(rich(row["label"]), row["x"], row["y_plot"], row["color"], plot_style["line_width"]))
            side_labels.append((row["label_y"], row["color"], row["label"]))
            for kind, ox, oy in overlay_curves(row):
                ax.plot(ox, oy, color=row["color"], linewidth=plot_style["line_width"] * 0.7, alpha=0.4 if kind == "raw" else 0.8,
                        linestyle="-" if kind == "raw" else ":", zorder=line.get_zorder() - 0.5)
            for kind, fx, fy, fbase in fit_curves(row):
                if kind == "component":
                    if mfit_fill:
                        ax.fill_between(fx, fbase, fy, color=row["color"], alpha=0.15, linewidth=0)
                    ax.plot(fx, fy, color=row["color"], linewidth=plot_style["line_width"] * 0.7, linestyle="--")
                else:
                    ax.plot(fx, fy, color=FIT_TOTAL_COLOR, linewidth=plot_style["line_width"] * 0.7, linestyle="-")
            for k in peak_indices(row):
                ax.annotate(f"{row['x'][k]:.{int(peak_decimals)}f}", xy=(row["x"][k], row["y_plot"][k]),
                            xytext=(0, -3 if PEAKS_DOWN else 3), textcoords="offset points", ha="center",
                            va="top" if PEAKS_DOWN else "bottom",
                            rotation=90 if peak_label_rotate else 0, fontsize=annotation_font_size, color=row["color"])
        if show_side_labels:
            for y_c, col, txt in side_labels:
                x_pos, ha = side_label_position()
                ax.text(x_pos, y_c, rich(txt), color=col, fontsize=label_fontsize, ha=ha, va="center",
                        bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
        ax.set_ylabel(rich(y_title))
        if show_yticks:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.tick_params(which="major", axis="y", length=major_len, direction=direction)
        else:
            ax.set_yticks([])
        ax.set_xlabel(rich(x_title))
        ax.set_xlim((x_max, x_min) if x_reverse else (x_min, x_max))
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        add_mpl_bands(ax, bands, color=bands_color, alpha=bands_alpha, x_range=(x_min, x_max))
        add_mpl_marks(ax, marks, bands, color=marks_color,
                      linestyle={"dashed": "--", "dotted": ":", "solid": "-", "none": "none"}[marks_style],
                      linewidth=plot_style["reference_line_width"], label_mode=marks_label_mode, show_value=marks_show_value,
                      decimals=marks_decimals, font_size=marks_fontsize, rich=rich, plain=plain, x_range=(x_min, x_max),
                      show_band_labels=bands_label, axes_width_points=axes_width_points)
        if show_legend and legend_lines:
            ordered = [legend_lines[k] for k in top_to_bottom(legend_levels)]
            ax.legend(handles=ordered, loc=legend_loc, frameon=False)
        apply_mpl_style(fig, plot_style)
        if show_zoom and x_min <= zoom_range[0] < zoom_range[1] <= x_max:
            add_mpl_zoom_inset(ax, prepared_curves, zoom_range, location=zoom_location, font_size=annotation_font_size,
                               reverse_x=x_reverse)
        return fig

    # ===== インタラクティブ Plotly プレビュー =====
    def build_plotly_figure():
        pfig = go.Figure()
        prepared_curves, legend_levels, legend_traces = [], [], []
        family = css_font_stack(plot_style["font_family"])
        # Text of 7 pt is unreadable on screen: the preview enlarges text, ticks and lines together
        # (the saved figure keeps the real sizes; the "保存画像のプレビュー" tab shows those).
        pv = max(1.0, min(2.4, 12.0 / max(font_size, 1.0)))
        pt = 4 / 3 * pv                        # points → preview pixels
        for row in curves:
            name = to_html(row["label"])
            pfig.add_trace(go.Scatter(
                x=row["x"], y=row["y_plot"], name=name, mode="lines", showlegend=show_legend,
                line=dict(color=row["color"], width=plot_style["line_width"] * pt),
                hovertemplate=plain(row["label"]) + "<br>%{x:.1f} " + UNIT_HTML + "<br>" + ("T=" if PEAKS_DOWN else "A=" if IS_FTIR else "I=") + "%{y:.4g}<extra></extra>",
            ))
            legend_traces.append(len(pfig.data) - 1)
            legend_levels.append(row["level"])
            prepared_curves.append(PreparedCurve(name, row["x"], row["y_plot"], row["color"], plot_style["line_width"] * pv))
            for kind, ox, oy in overlay_curves(row):
                pfig.add_trace(go.Scatter(
                    x=ox, y=oy, mode="lines", showlegend=False, opacity=0.4 if kind == "raw" else 0.8,
                    line=dict(color=row["color"], width=plot_style["line_width"] * pt * 0.7, dash="solid" if kind == "raw" else "dot"),
                    hovertemplate=(ui("処理前", "As measured") if kind == "raw" else ui("ベースライン", "Baseline")) + "<br>%{x:.1f} " + UNIT_HTML + "<extra></extra>"))
            _component_number = 0
            for kind, fx, fy, fbase in fit_curves(row):
                if kind == "component":
                    _component_number += 1
                    if mfit_fill:
                        pfig.add_trace(go.Scatter(x=fx, y=fbase, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    pfig.add_trace(go.Scatter(
                        x=fx, y=fy, mode="lines", showlegend=False, fill="tonexty" if mfit_fill else None,
                        fillcolor=_rgba(row["color"], 0.15) if mfit_fill else None,
                        line=dict(color=row["color"], width=plot_style["line_width"] * pt * 0.7, dash="dash"),
                        hovertemplate=ui(f"成分 {_component_number}", f"Component {_component_number}") + "<br>%{x:.1f} " + UNIT_HTML + "<extra></extra>"))
                else:
                    pfig.add_trace(go.Scatter(
                        x=fx, y=fy, mode="lines", showlegend=False,
                        line=dict(color=FIT_TOTAL_COLOR, width=plot_style["line_width"] * pt * 0.7),
                        hovertemplate=ui("フィットの合計", "Fitted sum") + "<br>%{x:.1f} " + UNIT_HTML + "<extra></extra>"))
            for k in peak_indices(row):
                pfig.add_annotation(x=float(row["x"][k]), y=float(row["y_plot"][k]), text=f"{row['x'][k]:.{int(peak_decimals)}f}",
                                    showarrow=False, textangle=-90 if peak_label_rotate else 0, xanchor="center",
                                    yanchor="top" if PEAKS_DOWN else "bottom", yshift=-3 if PEAKS_DOWN else 3,
                                    font=dict(size=annotation_font_size * pv, color=row["color"], family=family))
            if show_side_labels:
                x_pos, _ha = side_label_position()
                pfig.add_trace(go.Scatter(
                    x=[x_pos], y=[row["label_y"]], mode="text", text=[name],
                    textposition="middle right" if _ha == "left" else "middle left",
                    textfont=dict(color=row["color"], size=label_fontsize * pv, family=family),
                    showlegend=False, hoverinfo="skip",
                ))
        for rank, k in enumerate(top_to_bottom(legend_levels), start=1):
            pfig.data[legend_traces[k]].legendrank = rank
        add_plotly_bands(pfig, bands, color=bands_color, alpha=bands_alpha, x_range=(x_min, x_max))
        add_plotly_marks(pfig, marks, bands, color=marks_color,
                         linestyle={"dashed": "dash", "dotted": "dot", "solid": "solid", "none": "none"}[marks_style],
                         linewidth=plot_style["reference_line_width"] * pv, label_mode=marks_label_mode, show_value=marks_show_value,
                         decimals=marks_decimals, font_size=marks_fontsize * pv, to_html=to_html, plain=plain, x_range=(x_min, x_max),
                         show_band_labels=bands_label, axes_width_points=axes_width_points * pv, font_family=family)
        tick_px = plot_style["tick_length"] * pt
        ticks = {"in": "inside", "out": "outside"}.get(plot_style["tick_direction"], "inside")
        minor_cfg = dict(ticks=ticks, dtick=minor_tick, ticklen=tick_px / 2, showgrid=False) if (show_minor and minor_tick) else {}
        pfig.update_xaxes(range=[x_max, x_min] if x_reverse else [x_min, x_max], dtick=major_tick, ticks=ticks, ticklen=tick_px,
                          showline=True, linecolor="black", linewidth=plot_style["axes_line_width"] * pt,
                          mirror=plot_style["axes_box"], showgrid=False, zeroline=False,
                          tickfont=dict(color="black", size=tick_font_size * pv), title_text=to_html(x_title),
                          title_font=dict(color="black", size=font_size * pv), minor=minor_cfg)
        pfig.update_yaxes(range=list(y_limits) if y_limits is not None else None,
                          showticklabels=show_yticks, ticks=ticks if show_yticks else "", ticklen=tick_px,
                          showline=True, linecolor="black", linewidth=plot_style["axes_line_width"] * pt,
                          mirror=plot_style["axes_box"], showgrid=False, zeroline=False,
                          tickfont=dict(color="black", size=tick_font_size * pv), title_text=to_html(y_title),
                          title_font=dict(color="black", size=font_size * pv))
        legend_position = {"upper right": dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
                           "upper left": dict(x=0.0, y=1.0, xanchor="left", yanchor="top"),
                           "lower right": dict(x=1.0, y=0.0, xanchor="right", yanchor="bottom"),
                           "lower left": dict(x=0.0, y=0.0, xanchor="left", yanchor="bottom"),
                           "best": dict(x=1.0, y=1.0, xanchor="right", yanchor="top")}[legend_loc]
        pfig.update_layout(
            dragmode="zoom",
            font=dict(family=family, size=font_size * pv, color="black"),
            height=max(360, min(1800, int(fig_height * 96 * pv))),
            plot_bgcolor="rgba(255,255,255,0)" if plot_style["transparent_background"] else "white",
            paper_bgcolor="rgba(255,255,255,0)" if plot_style["transparent_background"] else "white",
            showlegend=show_legend,
            legend=dict(font=dict(color="black", size=legend_font_size * pv), **legend_position),
            margin=dict(l=int(fig_width * 96 * pv * plot_style["margin_left"]),
                        r=int(fig_width * 96 * pv * (1 - plot_style["margin_right"])),
                        t=int(fig_height * 96 * pv * (1 - plot_style["margin_top"])),
                        b=int(fig_height * 96 * pv * plot_style["margin_bottom"])),
        )
        if show_zoom and x_min <= zoom_range[0] < zoom_range[1] <= x_max:
            add_plotly_zoom_inset(pfig, prepared_curves, zoom_range, location=zoom_location, font_size=annotation_font_size * pv,
                                  reverse_x=x_reverse)
        return pfig

    # ===== 描画 & ダウンロード =====
    with col_graph:
        interactive_tab, export_tab = st.tabs([T["interactive_preview"], T["export_preview"]])
        pfig = build_plotly_figure()
        with interactive_tab:
            st.caption(T["drag_zoom"] + (ui(f"　／　スパイク除去: {_spikes_removed} 点を置き換え", f"  /  spikes replaced: {_spikes_removed}")
                                         if spike_remove else ""))
            st.plotly_chart(pfig, config={"scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                          "displaylogo": False}, **WIDE_CHART)
        fig = build_figure()
        png_bytes = export_figure(fig, "png", 150, style=plot_style)
        pdf_bytes = export_figure(fig, "pdf", dpi_export, style=plot_style)
        svg_bytes = export_figure(fig, "svg", dpi_export, style=plot_style)
        _raster_ok = fig_width * fig_height * dpi_export ** 2 <= 36_000_000
        png_export_bytes = export_figure(fig, "png", dpi_export, style=plot_style) if _raster_ok else None
        tiff_bytes = export_figure(fig, "tiff", dpi_export, style=plot_style) if _raster_ok else None
        plt.close(fig)
        with export_tab:
            st.image(png_bytes, **WIDE_IMAGE)
            st.caption(T["export_preview_caption"].format(width=inches_to_mm(fig_width), height=inches_to_mm(fig_height), dpi=dpi_export))
            st.caption(ui("軸や文字が端で切れる場合は、余白または文字サイズを調整してください。注記のラベルは図の上に出るので、上余白を広げると収まります。",
                          "If text or axes are clipped, adjust the margins or font sizes. Mark labels sit above the axes; widen the top margin to fit them."))

        if show_peaks:
            with st.expander(ui("ピーク一覧", "Peak table"), expanded=False):
                rows = peak_rows()
                if rows:
                    table = pd.DataFrame(rows)
                    st.dataframe(table, hide_index=True)
                    st.download_button(ui("📥 ピーク一覧を CSV で保存", "📥 Save the peak table as CSV"),
                        data=table.to_csv(index=False).encode("utf-8-sig"), file_name=f"{FILE_STEM}_peaks.csv", mime="text/csv", **WIDE_DOWNLOAD)
                    st.caption(ui("位置は表示範囲・前処理後の極大点です。フィットを有効にすると、ローレンツ／ガウス関数の中心と半値幅を併記します。",
                                  "Positions are local maxima after the range and processing. With fitting on, the fitted centre and FWHM are listed too."))
                else:
                    st.caption(ui("検出されたピークはありません。感度を下げてみてください。", "No peaks were detected; try a lower sensitivity."))
        if mfit_on:
            with st.expander(ui("ピーク分離フィットの結果", "Results of the peak fit"), expanded=True):
                if not mfit_centers:
                    st.info(ui("サイドバーの「ピーク分離フィット」に、ピークの初期位置（フィット範囲の中）を入力してください。",
                               "Enter the starting positions (inside the fit range) under “Fit overlapping peaks” in the sidebar."))
                else:
                    _reasons = {"too_few_points": ui("範囲内の点が足りません", "too few points in the range"),
                                "center_outside": ui("初期位置がデータの範囲外です", "a starting position lies outside the data"),
                                "flat": ui("範囲内のデータが平坦です", "the data are flat in the range")}
                    fit_rows, curve_frames, residual_fig = [], [], go.Figure()
                    for row in curves:
                        result = fit_of(row)
                        name = plain(row["label"])
                        if not result["ok"]:
                            _why = result["message"]
                            st.warning(f"{name}: " + ui("フィットできませんでした（", "the fit failed (")
                                       + _reasons.get(_why, ui("収束しませんでした", "no convergence")) + ui("）。", ")."))
                            continue
                        first = result["components"][0]
                        for number, c in enumerate(result["components"], start=1):
                            fit_rows.append({
                                ui("スペクトル", "Spectrum"): name, ui("成分", "Component"): number,
                                ui("中心 (cm⁻¹)", "Centre (cm⁻¹)"): round(c["center"], 2), ui("± 中心", "± centre"): round(c["center_err"], 2),
                                ui("半値幅 (cm⁻¹)", "FWHM (cm⁻¹)"): round(c["fwhm"], 2), ui("± 半値幅", "± FWHM"): round(c["fwhm_err"], 2),
                                ui("高さ", "Height"): c["height"], ui("± 高さ", "± height"): c["height_err"],
                                ui("面積", "Area"): c["area"], ui("± 面積", "± area"): c["area_err"],
                                ui("面積の割合 (%)", "Area share (%)"): round(100 * c["share"], 1),
                                ui("面積比（対 成分1）", "Area ratio (to no. 1)"): c["area"] / first["area"] if first["area"] > 0 else None,
                                ui("高さ比（対 成分1）", "Height ratio (to no. 1)"): c["height"] / first["height"] if first["height"] > 0 else None,
                                **({"η": round(c["eta"], 3)} if mfit_model == "pvoigt" else {}),
                                "R²": round(result["r2"], 5), "RMSE": result["rmse"],
                                ui("上限・下限に到達", "At a limit"): "⚠" if c["at_limit"] else "",
                            })
                        frame = {ui("スペクトル", "Spectrum"): name, X_VALUE_NAME: result["x"], ui("データ", "Data"): result["y"],
                                 ui("フィットの合計", "Fitted sum"): result["fit"], ui("バックグラウンド", "Background"): result["background"],
                                 ui("残差", "Residual"): result["residual"]}
                        for number, curve in enumerate(result["curves"], start=1):
                            frame[ui(f"成分 {number}", f"Component {number}")] = curve
                        curve_frames.append(pd.DataFrame(frame))
                        residual_fig.add_trace(go.Scatter(x=result["x"], y=result["residual"], mode="lines", name=name,
                                                          line=dict(color=row["color"], width=1.2)))
                    if fit_rows:
                        fit_table = pd.DataFrame(fit_rows)
                        st.dataframe(fit_table, hide_index=True)
                        if any(r[ui("上限・下限に到達", "At a limit")] for r in fit_rows):
                            st.warning(ui("⚠ の成分は、中心が動ける幅の端か、幅の上限・下限に張り付いています。初期位置・成分の数・範囲を見直してください。",
                                          "Components marked ⚠ ended at the edge of the allowed centre shift or at a width limit. Review the starting positions, the number of components or the range."))
                        residual_fig.add_hline(y=0, line=dict(color="black", width=0.8))
                        residual_fig.update_layout(height=220, margin=dict(l=60, r=20, t=30, b=40), plot_bgcolor="white", paper_bgcolor="white",
                                                   title=dict(text=ui("残差（データ − フィット）", "Residual (data − fit)"), font=dict(size=13, color="black")),
                                                   showlegend=len(residual_fig.data) > 1, font=dict(color="black"))
                        residual_fig.update_xaxes(range=[mfit_range[1], mfit_range[0]] if x_reverse else list(mfit_range), showline=True,
                                                  linecolor="black", showgrid=False, zeroline=False, ticks="outside")
                        residual_fig.update_yaxes(showline=True, linecolor="black", showgrid=False, zeroline=False, ticks="outside")
                        st.plotly_chart(residual_fig, config={"displaylogo": False}, **WIDE_CHART)
                        _d1, _d2 = st.columns(2)
                        _d1.download_button(ui("📥 フィット結果を CSV で保存", "📥 Save the fit results as CSV"),
                            data=fit_table.to_csv(index=False).encode("utf-8-sig"), file_name=f"{FILE_STEM}_fit.csv", mime="text/csv", **WIDE_DOWNLOAD)
                        _d2.download_button(ui("📥 フィット曲線を CSV で保存", "📥 Save the fitted curves as CSV"),
                            data=pd.concat(curve_frames, ignore_index=True).to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"{FILE_STEM}_fit_curves.csv", mime="text/csv", **WIDE_DOWNLOAD)
                        st.caption(ui("前処理後のデータ" + ("（吸光度）" if IS_FTIR else "") + "に対する最小二乗フィットです。± は共分散行列から求めた標準誤差で、点ごとの雑音が独立であることを仮定しています。"
                                      "残差に波打ちが残る場合は、成分の数か関数が合っていません。面積・高さは正規化とベースラインの設定に依存するので、比で比べてください。"
                                      + ("　透過率の表示では曲線は図に重ねません（吸光度の表示で確認できます）。" if PEAKS_DOWN else ""),
                                      "A least-squares fit of the processed data" + (" (absorbance)" if IS_FTIR else "") + ". ± is the standard error from the covariance matrix and assumes independent noise at each point. "
                                      "Waves left in the residual mean that the number of components or the model does not match. Areas and heights depend on the normalisation and the baseline, so compare ratios."
                                      + (" The curves are not drawn on a transmittance display (see them in absorbance)." if PEAKS_DOWN else "")))
        with st.expander(ui("処理後のデータ", "Processed data"), expanded=False):
            frames = [pd.DataFrame({ui("スペクトル", "Spectrum"): plain(row["label"]), X_VALUE_NAME: row["x"],
                                    Y_VALUE_NAME + ui("（処理後）", " (processed)"): row["y"],
                                    Y_VALUE_NAME + ui("（図の位置）", " (as plotted)"): row["y_plot"]}) for row in curves]
            if frames:
                processed = pd.concat(frames, ignore_index=True)
                st.download_button(ui("📥 処理後のデータを CSV で保存", "📥 Save the processed data as CSV"),
                    data=processed.to_csv(index=False).encode("utf-8-sig"), file_name=f"{FILE_STEM}_processed.csv", mime="text/csv", **WIDE_DOWNLOAD)
                st.caption(ui("表示範囲で切り出し、スパイク除去・平滑化・ベースライン・正規化を掛けた値と、オフセットを足した図の位置の値です（縦持ち）。",
                              "Values after the range, spikes, smoothing, baseline and normalisation, and the plotted values with offsets (long format)."))

        st.sidebar.download_button(ui("📥 PDF として保存", "📥 Save as PDF"), data=pdf_bytes, file_name=f"{FILE_STEM}_result.pdf",
                                   mime="application/pdf", **WIDE_DOWNLOAD)
        st.sidebar.download_button(ui("📥 SVG として保存", "📥 Save as SVG"), data=svg_bytes, file_name=f"{FILE_STEM}_result.svg",
                                   mime="image/svg+xml", **WIDE_DOWNLOAD)
        if not _raster_ok:
            st.sidebar.warning(ui("PNG / TIFFは3,600万画素までです。図の寸法またはDPIを下げてください。PDF / SVGは保存できます。",
                                  "For PNG / TIFF, reduce the dimensions or DPI to stay within 36 megapixels. PDF / SVG are available."))
        else:
            st.sidebar.download_button(T["save_tiff"].format(dpi=dpi_export), data=tiff_bytes, file_name=f"{FILE_STEM}_result.tiff",
                                       mime="image/tiff", key=f"dl_tiff_{dpi_export}", **WIDE_DOWNLOAD)
            st.sidebar.download_button(T["save_png"].format(dpi=dpi_export), data=png_export_bytes, file_name=f"{FILE_STEM}_result.png",
                                       mime="image/png", key=f"dl_png_{dpi_export}", **WIDE_DOWNLOAD)
        st.sidebar.divider()
else:
    st.info(T["upload_prompt"])
    st.markdown(ui(
        "- **読める形式**：JASCO Spectra Manager の CSV/TXT（`XYDATA` 付き）、Renishaw WiRE の TXT 書き出しと .wdf、"
        "`##TITLE=` で始まる FT-IR テキスト・JCAMP-DX（`(X++(Y..Y))` の圧縮形式も可）、RRUFF の TXT、"
        "2 列以上の XY/CSV/TXT/DAT、Excel（.xlsx）。\n"
        "- **FT-IR**：ファイルの `DATA TYPE=INFRARED SPECTRUM` で自動的に FT-IR モード（横軸は高波数が左、縦軸は吸光度／透過率）になります。\n"
        "- **複数の列**：1 列目をラマンシフト、2 列目以降を別々のスペクトルとして読みます。`x, y, x, y` の並びも読めます。\n"
        "- **色・文字・ひな形**：グラフ描画app・XRD描画appと同じ色の組み合わせ、`<i>…</i>`・`_{…}`・`^{…}` の書式、Arial 7 pt の「グラフ描画appと同じ体裁」。",
        "- **Formats**: JASCO Spectra Manager CSV/TXT (with `XYDATA`), Renishaw WiRE TXT exports and .wdf, "
        "FT-IR text and JCAMP-DX files starting with `##TITLE=` (compressed `(X++(Y..Y))` too), RRUFF TXT, "
        "XY/CSV/TXT/DAT with two or more columns, Excel (.xlsx).\n"
        "- **FT-IR**: a file with `DATA TYPE=INFRARED SPECTRUM` switches to the FT-IR mode (high wavenumber on the left, absorbance or transmittance).\n"
        "- **Several columns**: the first column is the Raman shift and each further column a spectrum; `x, y, x, y` layouts work too.\n"
        "- **Colours, text, presets**: the graph app's and XRD Maker's palettes, the `<i>…</i>`, `_{…}`, `^{…}` markup and the Arial 7 pt 'graph app look'."))

if st.session_state.get("_spectrum_bytes"):
    try:
        _project_bytes = build_session_json()
        st.sidebar.download_button(T["save_session"], data=_project_bytes, file_name="raman_project.json",
                                   mime="application/json", **WIDE_DOWNLOAD)
        if LOCAL_DATA_DIR:
            try:
                _fingerprint = project_fingerprint(_project_bytes)
                if _fingerprint != st.session_state.get("_autosave_fingerprint"):
                    atomic_save(LOCAL_DATA_DIR / "autosave.json", _project_bytes)
                    st.session_state["_autosave_fingerprint"] = _fingerprint
                    st.session_state["_autosave_time"] = datetime.now().astimezone().strftime("%H:%M:%S")
                autosave_status.caption(T["autosaved"].format(time=st.session_state.get("_autosave_time", "")))
            except (OSError, ProjectError) as _error:
                autosave_status.warning(T["autosave_error"] + str(_error))
    except ProjectError as _error:
        autosave_status.warning(T["project_save_error"] + str(_error))

st.sidebar.divider()
try:
    from build_bundle import create_bundle_bytes
    st.sidebar.download_button(T["download_desktop"], data=create_bundle_bytes(), file_name="Raman-Maker-PC.zip",
                               mime="application/zip", **WIDE_DOWNLOAD)
    st.sidebar.caption(T["desktop_help"])
except (ImportError, FileNotFoundError):
    pass

# ===== 書式のショートカット（グラフ描画app・XRD描画appと同じ） =====
_SHORTCUTS_HTML = """<script>
(function () {
  const P = window.parent;
  if (!P || !P.document) return;
  const handler = function (e) {
    const el = e.target;
    if (!el || !(el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && (el.type === 'text' || !el.type)))) return;
    if (el.getAttribute('role') === 'combobox' || (el.closest && el.closest('[data-baseweb="select"]'))) return;
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    const k = e.key;
    let open = null, close = null;
    if (k === 'i' || k === 'I') { open = '<i>'; close = '</i>'; }
    else if (k === 'b' || k === 'B') { open = '<b>'; close = '</b>'; }
    else if (k === ',' || k === '=') { open = '_{'; close = '}'; }
    else if (k === '.' || k === '+') { open = '^{'; close = '}'; }
    if (!open) return;
    e.preventDefault();
    e.stopPropagation();
    const v = el.value || '';
    const a = (el.selectionStart == null) ? v.length : el.selectionStart;
    const b = (el.selectionEnd == null) ? a : el.selectionEnd;
    const nv = v.slice(0, a) + open + v.slice(a, b) + close + v.slice(b);
    const proto = el.tagName === 'TEXTAREA' ? P.HTMLTextAreaElement.prototype : P.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, nv);
    el.dispatchEvent(new P.Event('input', { bubbles: true }));
    el.setSelectionRange(a + open.length, b + open.length);
  };
  if (P.__ramanMakerKeys) P.document.removeEventListener('keydown', P.__ramanMakerKeys, true);
  P.__ramanMakerKeys = handler;
  P.document.addEventListener('keydown', handler, true);
})();
</script>"""
if hasattr(st, "iframe"):
    st.iframe(_SHORTCUTS_HTML, height="content")
else:
    import streamlit.components.v1 as components
    components.html(_SHORTCUTS_HTML, height=0)
