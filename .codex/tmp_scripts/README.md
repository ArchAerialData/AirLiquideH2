Tmp Scripts Overview

Purpose
- Ad-hoc helper scripts for quick inspection and validation during development.
- Run these from the repo root; they avoid clutter by living under `.codex/tmp_scripts`.

Prerequisites
- Python deps used by these scripts: pandas, openpyxl, fastkml, shapely, beautifulsoup4, lxml, rtree
- Install: `pip install pandas openpyxl fastkml shapely beautifulsoup4 lxml rtree`
- KMZ path is controlled by `modules/config.py` via `DEFAULT_KMZ_PATH` (currently `kmz/AirLiquideH2.kmz`).

How To Run (from repo root)
- `python .codex/tmp_scripts/tmp_check_align.py`
- `python .codex/tmp_scripts/tmp_excel_probe.py`
- `python .codex/tmp_scripts/tmp_probe_kmz.py`
- `python .codex/tmp_scripts/tmp_kmz_test.py`
- `python .codex/tmp_scripts/tmp_check_temp.py`
- `python .codex/tmp_scripts/tmp_check_ts.py`
- `python .codex/tmp_scripts/tmp_peek_ts.py`
- `python .codex/tmp_scripts/tmp_read_hdr.py`
- `python .codex/tmp_scripts/tmp_probe_single.py`
- `python .codex/tmp_scripts/tmp_parse_kml.py`

Script Summaries
- `tmp_check_align.py`: Verifies header and data alignment rules (gold columns centered, `Route_Desc` left). Prints alignments for a few cells.
- `tmp_excel_probe.py`: Dumps first rows and reports header fill colors for the generated XLSX.
- `tmp_probe_kmz.py`: Prints KMZ-enriched columns (PLID, BeginMeasu, EndMeasure, Route_Desc, Class_Loca, Diameter, Product) from the latest output.
- `tmp_kmz_test.py`: Loads the KMZ index using `DEFAULT_KMZ_PATH` and performs a sample nearest lookup.
- `tmp_check_temp.py`: Parses a CSV, attaches source metadata, converts to the final output schema (checks Date + TEMP (F)).
- `tmp_check_ts.py`: Parses a CSV and reports timestamp parsing status.
- `tmp_peek_ts.py`: Shows raw `Time Stamp` strings to diagnose BOM/leading character issues.
- `tmp_read_hdr.py`: Reads the header row from an output file.
- `tmp_probe_single.py`: Prints the header and a handful of data rows from the single_csv test output.
- `tmp_parse_kml.py`: Minimal KML structure probing using fastkml (used for troubleshooting KML nesting).

Typical Pipeline Runs
- Single CSV test folder: `python main.py "test_folders\single_csv"`
- Two CSVs (after confirming single_csv): `python main.py "test_folders\two_csvs"`

Notes
- Scripts that import `modules.*` add the repo root to `sys.path` at runtime, so you can run them directly from `.codex/tmp_scripts`.
- Outputs are written next to the input folder (`Combined_Extracted.xlsx`). Close the file before re-running to avoid PermissionError on save.

