Air Liquide H2 Processing Toolkit
=================================

This repository hosts the tooling used to ingest raw Air Liquide flight CSV files, enrich them with KMZ alignment data, and export downstream workbooks and shapefiles.

Scripts at a Glance
-------------------
- `main.py` – Full combined workbook export (all readings).
- `clean.py` – Deduplicated export keeping the highest PPM per (BeginMeasu, EndMeasure) pair.
- `hits.py` – High-PPM export with Excel plus KMZ and heatmap assets.
- `template-filler.py` – Fills TrackerTemplate rows with earliest matched dates and produces an XLSX tracker.

Getting Started
---------------
- Python 3.10 or newer is recommended.
- No manual dependency install is required; the first run of any script auto-installs packages listed in `requirements.txt`.
- Each command accepts one or more directory paths. Paths can be passed on the command line or provided interactively when prompted.

Common Output Locations
-----------------------
- Excel exports: `Excel-Reports/` under the chosen input directory (or its common parent).
- KMZ exports: `KMZ/` alongside the Excel output (hits flow only).
- Temporary per-row KML files: `temp_kml/` (cleared automatically each run).
- Tracker template outputs: configurable via `--output` (default is the template path with `.xlsx`).

Configuration
-------------
Optional `config.yaml` (or `config.yml`) placed in the working directory or repo root controls shared defaults:
- `kmz_path`, `kmz_distance_threshold`, `kmz_buffer_meters`
- `hits_threshold`
- `pin_spacing_meters`
- `output_format`, `include_separator_rows`, `separator_style`
- Advanced options such as `columns_to_extract` and `header_renames`

See the comments in `config.yaml` for descriptions and defaults. Script-specific CLI flags override YAML settings when available.

Script Details & Usage
----------------------

main.py — Combined Workbook Export
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Purpose:
- Parse every CSV found under one or more input directories.
- Enrich with KMZ metadata when the configured KMZ exists.
- Produce a multi-sheet Excel workbook (`ALL` sheet plus one sheet per date).

Core controls:
- KMZ path and distance threshold via `config.yaml`.
- Output file name: `Excel-Reports/Combined_Extracted.xlsx` by default (per input root/common parent).

Copy/paste commands:
- `python main.py "C:\path\to\FlightData"`
- `python main.py "C:\data\Day1" "C:\data\Day2"`

clean.py — Deduplicated Workbook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Purpose:
- Run the same ingest and enrichment pipeline as `main.py`.
- Per source CSV, keep only the highest PPM measurement for each (BeginMeasu, EndMeasure) pair.
- Write `Excel-Reports/Combined_Extracted_Clean.xlsx`.

Core controls mirror `main.py` (config-driven KMZ settings, same output conventions).

Copy/paste commands:
- `python clean.py "C:\path\to\FlightData"`
- `python clean.py "C:\data\Day1" "C:\data\Day2"`

hits.py — High-PPM Export + KMZ Assets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Purpose:
- Filter the combined dataset to measurements at or above the configured PPM threshold (default 5.0).
- Produce `Excel-Reports/Combined_Extracted_Hits.xlsx` and `.csv`.
- Generate KMZ deliverables (per-row placemarks and a heatmap overlay).

Core controls:
- `hits_threshold` (YAML) to change the PPM cutoff.
- `pin_spacing_meters` (YAML) to adjust placemark density in the “All” KMZ.
- `kmz_*` settings as in `main.py`.

Copy/paste commands:
- `python hits.py "C:\path\to\FlightData"`
- `python hits.py "C:\data\Day1"` (with custom settings from YAML if needed)

template-filler.py — Tracker Date Populator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Purpose:
- Recursively scan input directories for CSV files and reuse the existing pipeline to derive PLID, BeginMeasu, EndMeasure, and associated dates.
- Compare the derived values to `TrackerTemplate.csv` rows.
- Fill the template `Date` column with the earliest matching date (format `MM/DD/YYYY`) or `N/A` when no match is found.
- Save an XLSX copy of the template (and optionally write back to the CSV when `--update-template-csv` is supplied).

Key options:
- `--template PATH` – Path to the Tracker template CSV (defaults to `test_folders/.example_output/TrackerTemplate.csv` when present).
- `--output PATH` – Destination XLSX. Defaults to the template path with `.xlsx`.
- `--kmz PATH` – Override the KMZ file used for enrichment.
- `--max-distance FLOAT` – Override the KMZ match distance (meters).
- `--skip-kmz` – Bypass KMZ enrichment entirely.
- `--update-template-csv` – Also overwrite the CSV with the filled dates.

Copy/paste commands:
- `python template-filler.py "test_folders/.example_output/2025_09" --output "test_outputs/xlsx/TrackerTemplate_filled.xlsx"`
- `python template-filler.py "D:\field-data" --template "C:\shared\TrackerTemplate.csv" --kmz "D:\alignments\AirLiquideH2.kmz" --max-distance 200`
- `python template-filler.py "C:\data" --skip-kmz --update-template-csv > "logs\template-filler.log" 2>&1`

Controlling Outputs
-------------------
- Destination folders: override with `--output` for `template-filler.py`; other scripts derive paths automatically based on the provided inputs.
- KMZ behavior: adjust via YAML or `--kmz` / `--max-distance` / `--skip-kmz` (template-filler).
- Thresholds and formatting: update `config.yaml` for all other scripts; `hits_threshold` and `pin_spacing_meters` directly influence workbook contents and KMZ density.
- Logging: redirect stdout/stderr to files when needed (see sample command above).

Tips
----
- Supply multiple directories to merge separate flight days in one run.
- The KMZ index is loaded once per execution; keep the path reachable for best results.
- Temporary `temp_kml/` contents are regenerated each run; do not place permanent files there.
- Re-run `template-filler.py` whenever new field CSVs are added to ensure the tracker stays current.
