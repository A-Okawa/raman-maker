"""Readers: every supported layout, the metadata they keep, and the errors they raise."""
from pathlib import Path
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from raman_io import (RamanParseError, SUPPORTED_EXTENSIONS, condition_summary, native_y_units, read_spectra,
                      spectrum_kind, wdf_available)
sys.path.insert(0, str(APP_DIR / "tests"))
import make_test_data

DATA = APP_DIR / "tests" / "data"


def load(name, **kwargs):
    return read_spectra((DATA / name).read_bytes(), name, **kwargs)


class JascoTests(unittest.TestCase):
    def test_comma_csv_in_cp932(self):
        spectra = load("jasco_5SiC_asSintered.csv")
        self.assertEqual(len(spectra), 1)
        s = spectra[0]
        self.assertEqual(s.name, "jasco_5SiC_asSintered")
        self.assertEqual(len(s.x), 1601)
        self.assertTrue(np.all(np.diff(s.x) > 0), "Raman shifts are returned ascending even when exported descending")
        self.assertAlmostEqual(float(s.x[0]), 200.0)
        self.assertAlmostEqual(float(s.x[-1]), 1000.0)
        self.assertEqual(s.metadata["TITLE"], "5SiC as-sintered")
        self.assertEqual(s.metadata["XUNITS"], "cm-1")
        self.assertEqual(s.metadata["Detailed Information: Laser wavelength"], "532.11 nm")
        self.assertEqual(s.metadata["Detailed Information: Wavenumber shift correction"], "0.90 cm-1")
        self.assertNotIn("x_units_warning", s.metadata)

    def test_tab_text(self):
        s = load("jasco_5SiC_950C.txt")[0]
        self.assertEqual(s.metadata["format"], "JASCO")
        self.assertEqual(s.metadata["Detailed Information: Accumulation"], "10")
        self.assertAlmostEqual(float(s.y[np.argmin(np.abs(s.x - 797.5))]), float(s.y.max()), delta=60)

    def test_peak_intensity_is_as_stored(self):
        s = load("jasco_5SiC_asSintered.csv")[0]
        # the stored value at 796 cm-1 (3000 on the background) survives unchanged
        self.assertGreater(float(s.y[np.argmin(np.abs(s.x - 796))]), 3000)

    def test_wavelength_axis_is_flagged(self):
        text = (DATA / "jasco_5SiC_950C.txt").read_text(encoding="utf-8").replace("XUNITS\tcm-1", "XUNITS\tnm")
        s = read_spectra(text.encode("utf-8"), "wl.txt")[0]
        self.assertEqual(s.metadata["x_units_warning"], "nm")

    def test_condition_summary_lists_measurement_conditions(self):
        rows = dict(condition_summary(load("jasco_5SiC_950C.txt")[0].metadata))
        self.assertEqual(rows["TITLE"], "5SiC 950C")
        self.assertIn("Detailed Information: Laser wavelength", rows)
        self.assertNotIn("format", rows)


class RenishawTextTests(unittest.TestCase):
    def test_single_spectrum_with_hash_header(self):
        s = load("renishaw_single.txt")[0]
        self.assertEqual(s.name, "renishaw_single")
        self.assertEqual(len(s.x), 1601)
        self.assertTrue(np.all(np.diff(s.x) > 0))

    def test_map_export_becomes_one_spectrum_per_position(self):
        spectra = load("renishaw_map.txt")
        self.assertEqual([s.name for s in spectra],
                         ["renishaw_map (X=0, Y=0)", "renishaw_map (X=1.3, Y=0)", "renishaw_map (X=0, Y=1.3)"])
        self.assertTrue(all(len(s.x) == 1601 for s in spectra))
        self.assertEqual(spectra[1].metadata["position"], "X=1.3, Y=0")


class RruffTests(unittest.TestCase):
    def test_name_and_id(self):
        s = load("rruff_quartz.txt")[0]
        self.assertEqual(s.name, "Quartz (R040031)")
        self.assertEqual(s.metadata["IDEAL CHEMISTRY"], "SiO_2_")
        self.assertEqual(len(s.x), 1601)


class GenericTextTests(unittest.TestCase):
    def test_header_names_the_columns(self):
        spectra = load("multi_columns.csv")
        self.assertEqual([s.name for s in spectra], ["as-sintered", "750 C", "950 C"])
        self.assertTrue(all(len(s.x) == 1601 for s in spectra))

    def test_two_columns_space_separated(self):
        s = load("two_col.xy")[0]
        self.assertEqual(s.name, "two_col")
        self.assertEqual(len(s.x), 1601)

    def test_decimal_comma_with_semicolons(self):
        s = load("decimal_comma.csv")[0]
        self.assertAlmostEqual(float(s.x[1]), 200.5)
        reference = load("two_col.xy")[0]
        np.testing.assert_allclose(s.y, reference.y, atol=5e-4)

    def test_columns_without_header_are_numbered(self):
        text = "\n".join(f"{x:g}\t{x * 2:g}\t{x * 3:g}" for x in range(100, 110))
        spectra = read_spectra(text.encode(), "plain.txt")
        self.assertEqual([s.name for s in spectra], ["plain [1]", "plain [2]"])

    def test_repeated_xy_pairs(self):
        text = "\n".join(f"{x:g},{(x % 3):g},{x + 1:g},{(x % 5):g}" for x in range(100, 130))
        spectra = read_spectra(text.encode(), "pairs.csv")
        self.assertEqual(len(spectra), 2)
        self.assertAlmostEqual(float(spectra[1].x[0]), 101.0)

    def test_footer_text_stops_the_data(self):
        text = "100 1\n101 2\n102 3\nEND OF DATA\n"
        s = read_spectra(text.encode(), "f.txt")[0]
        self.assertEqual(len(s.x), 3)

    def test_errors(self):
        with self.assertRaises(RamanParseError):
            read_spectra(b"", "empty.txt")
        with self.assertRaises(RamanParseError):
            read_spectra(b"only text\nno numbers\n", "text.txt")
        with self.assertRaises(RamanParseError):
            read_spectra(b"100\n101\n102\n", "one_column.txt")
        with self.assertRaises(RamanParseError):
            read_spectra(b"\x00\x01\x02" * 50, "binary.txt")
        with self.assertRaises(RamanParseError):
            read_spectra(b"junk", "spectrum.jws")
        with self.assertRaises(RamanParseError):
            read_spectra(b"5 1\n3 2\n9 4\n1 1\n", "notmono.txt")    # the first column must be the shift


class ExcelTests(unittest.TestCase):
    def test_shared_x_and_pairs_in_two_sheets(self):
        spectra = load("book.xlsx")
        self.assertEqual([s.name for s in spectra], ["shared_x · A", "shared_x · B", "pairs · y1", "pairs · y2"])
        self.assertAlmostEqual(float(spectra[3].x[0]), 201.0)
        self.assertEqual(spectra[0].metadata["sheet"], "shared_x")


@unittest.skipUnless(wdf_available(), "renishawWiRE is not installed")
class WdfTests(unittest.TestCase):
    def test_single_spectrum(self):
        s = load("sp.wdf")[0]
        self.assertEqual(s.name, "sp")
        self.assertEqual(len(s.x), 1015)
        self.assertTrue(np.all(np.diff(s.x) > 0))
        self.assertEqual(s.metadata["count"], 1)
        self.assertAlmostEqual(s.metadata["laser_wavelength_nm"], 532.25, places=2)

    def test_series_is_averaged_by_default(self):
        s = load("depth.wdf")[0]
        self.assertEqual(s.name, "depth (mean of 40)")
        self.assertEqual(s.metadata["averaged_spectra"], 40)
        everyone = load("depth.wdf", multi="all")
        np.testing.assert_allclose(np.mean([e.y for e in everyone], axis=0), s.y, rtol=1e-6)

    def test_series_one_by_one_with_a_limit(self):
        spectra = load("depth.wdf", multi="all", limit=5)
        self.assertEqual(len(spectra), 5)
        self.assertEqual(spectra[0].metadata["spectra_not_read"], 35)
        self.assertEqual(len(load("depth.wdf", multi="first")), 1)

    def test_damaged_file(self):
        with self.assertRaises(RamanParseError):
            read_spectra(b"WDF1" + b"\x00" * 100, "broken.wdf")


class FtirTests(unittest.TestCase):
    """The FT-IR text export (## header + x y rows) and JCAMP-DX."""

    def setUp(self):
        self.t_percent = make_test_data.ir_transmittance()

    def test_export_text_absorbance(self):
        s = load("ftir_abs.txt")[0]
        self.assertEqual(s.name, "ftir_abs")
        self.assertEqual(len(s.x), 1801)
        self.assertTrue(np.all(np.diff(s.x) > 0), "exported high → low, returned ascending")
        self.assertEqual(s.metadata["DATA TYPE"], "INFRARED SPECTRUM")
        self.assertEqual(spectrum_kind(s.metadata), "ftir")
        self.assertEqual(native_y_units(s.metadata, s.y), "absorbance")
        np.testing.assert_allclose(s.y, -np.log10(self.t_percent / 100), atol=1e-5)
        self.assertEqual(dict(condition_summary(s.metadata))["YUNITS"], "Abs")

    def test_export_text_transmittance(self):
        s = load("ftir_pct.txt")[0]
        self.assertEqual(native_y_units(s.metadata, s.y), "transmittance_percent")
        np.testing.assert_allclose(s.y, self.t_percent, atol=1e-5)

    def test_jcamp_affn_and_difdup(self):
        for name, tolerance in (("ftir_affn.jdx", 1e-4), ("ftir_difdup.jdx", 1e-3)):
            with self.subTest(name=name):
                s = load(name)[0]
                self.assertEqual(len(s.x), 1801)
                np.testing.assert_allclose(s.x, make_test_data.XW)
                np.testing.assert_allclose(s.y, self.t_percent, atol=tolerance)
                self.assertEqual(native_y_units(s.metadata, s.y), "transmittance_percent")
                self.assertEqual(spectrum_kind(s.metadata), "ftir")

    def test_difdup_without_firstx_uses_the_line_spacing(self):
        text = "\n".join(line for line in (DATA / "ftir_difdup.jdx").read_text().splitlines()
                         if not line.startswith(("##FIRSTX", "##LASTX", "##NPOINTS")))
        s = read_spectra(text.encode(), "nofirst.jdx")[0]
        np.testing.assert_allclose(s.x, make_test_data.XW)

    def test_sqz_e_is_a_digit_not_an_exponent(self):
        # E12 = 512 (SQZ), J2 = +12, T repeats the difference once more, K5 = +25
        text = "##TITLE=e\n##XUNITS=1/CM\n##YUNITS=Abs\n##FIRSTX=1000\n##LASTX=1003\n##NPOINTS=4\n##XYDATA=(X++(Y..Y))\n1000E12J2TK5\n##END=\n"
        s = read_spectra(text.encode(), "e.jdx")[0]
        self.assertEqual(list(s.y), [512.0, 524.0, 536.0, 561.0])
        self.assertEqual(list(s.x), [1000.0, 1001.0, 1002.0, 1003.0])
        affn = "##TITLE=e\n##XUNITS=1/CM\n##YUNITS=Abs\n##FIRSTX=1000\n##LASTX=1002\n##NPOINTS=3\n##XYDATA=(X++(Y..Y))\n1000 1.5E-2 2E+0 -3e1\n##END=\n"
        self.assertEqual(list(read_spectra(affn.encode(), "e2.jdx")[0].y), [0.015, 2.0, -30.0])

    def test_xy_pairs_and_blocks(self):
        pairs = "##TITLE=p\n##DATA TYPE=INFRARED SPECTRUM\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n##XYDATA=(XY..XY)\n400, 0.1; 402, 0.2\n404, 0.15\n##END=\n"
        s = read_spectra(pairs.encode(), "pairs.jdx")[0]
        self.assertEqual(list(s.x), [400.0, 402.0, 404.0])
        self.assertEqual(list(s.y), [0.1, 0.2, 0.15])
        compound = ("##TITLE=link\n##BLOCKS=2\n##TITLE=first\n##XUNITS=1/CM\n##YUNITS=Abs\n400 0.1\n500 0.2\n##END=\n"
                    "##TITLE=second\n##XUNITS=1/CM\n##YUNITS=%T\n400 90\n500 80\n##END=\n##END=\n")
        spectra = read_spectra(compound.encode(), "two.jdx")
        self.assertEqual([t.name for t in spectra], ["first", "second"])
        self.assertEqual(native_y_units(spectra[1].metadata, spectra[1].y), "transmittance_percent")

    def test_wavelength_axes_become_wavenumbers(self):
        um = "##TITLE=u\n##XUNITS=MICROMETERS\n##YUNITS=%T\n2.5\t90\n5.0\t50\n10.0\t80\n"
        s = read_spectra(um.encode(), "um.txt")[0]
        np.testing.assert_allclose(s.x, [1000.0, 2000.0, 4000.0])
        self.assertEqual(list(s.y), [80.0, 50.0, 90.0])
        self.assertEqual(s.metadata["x_converted_from"], "MICROMETERS")
        nm = "##TITLE=n\n##XUNITS=NANOMETERS\n##YUNITS=Abs\n2500\t0.1\n5000\t0.2\n"
        np.testing.assert_allclose(read_spectra(nm.encode(), "nm.txt")[0].x, [2000.0, 4000.0])

    def test_units_from_column_names(self):
        text = "Wavenumber (cm-1),Transmittance (%)\n" + "\n".join(f"{x},{90 - x / 100}" for x in range(400, 1000, 2)) + "\n"
        s = read_spectra(text.encode(), "omnic.csv")[0]
        self.assertEqual(native_y_units(s.metadata, s.y), "transmittance_percent")
        fraction = "wavenumber,transmittance\n" + "\n".join(f"{x},{0.9 - x / 10000}" for x in range(400, 1000, 2)) + "\n"
        s = read_spectra(fraction.encode(), "frac.csv")[0]
        self.assertEqual(native_y_units(s.metadata, s.y), "transmittance_fraction")
        self.assertIsNone(native_y_units(load("two_col.xy")[0].metadata))
        self.assertIsNone(spectrum_kind(load("two_col.xy")[0].metadata))

    def test_damaged_jcamp(self):
        with self.assertRaises(RamanParseError):
            read_spectra(b"##TITLE=x\n##XUNITS=1/CM\n##XYDATA=(X++(Y..Y))\n##END=\n", "empty.jdx")
        with self.assertRaises(RamanParseError):
            read_spectra(b"##TITLE=x\n##XYDATA=(X++(Y..Y))\n1000 1 2 3\n##END=\n", "nostep.jdx")
        with self.assertRaises(RamanParseError):
            read_spectra(b"##TITLE=x\n##NTUPLES=IR\n##DATA TABLE=(T++(R..R)), XYDATA\n1 2 3\n##END=\n", "ntuples.jdx")


class ExtensionTests(unittest.TestCase):
    def test_supported_extensions(self):
        for ext in ("csv", "txt", "xy", "dat", "jdx", "dx", "wdf", "xlsx"):
            self.assertIn(ext, SUPPORTED_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
