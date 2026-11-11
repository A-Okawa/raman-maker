# Test data

- `jasco_*.csv/.txt`, `renishaw_*.txt`, `rruff_quartz.txt`, `multi_columns.csv`, `two_col.xy`,
  `decimal_comma.csv`, `book.xlsx`: synthetic Raman spectra written by `tests/make_test_data.py`
  (three Lorentzian peaks on a sloping background) in the layouts the readers support.
- `ftir_abs.txt`, `ftir_pct.txt` (the `##TITLE=` FT-IR text export in absorbance / %T), `ftir_affn.jdx`,
  `ftir_difdup.jdx` (JCAMP-DX `(X++(Y..Y))`, plain and DIFDUP-compressed): one synthetic transmittance
  spectrum, also from `make_test_data.py`.
- `sp.wdf` (single spectrum) and `depth.wdf` (series of 40 spectra): example files of the
  py-wdf-reader project (https://github.com/alchem0x2A/py-wdf-reader, MIT License, © 2022 T. Tian),
  used to test the optional `.wdf` reader.
