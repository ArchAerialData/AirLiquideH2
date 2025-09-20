Air Liquide H2 CSV -> Excel + KMZ Exports
=========================================

**Overview**
- Entry points:
  - `main.py` — Full dataset export with KMZ enrichment
  - `clean.py` — Same as main, but per-CSV dedupe to one row per unique `(BeginMeasu, EndMeasure)` with max PPM
  - `hits.py` — Filters to rows with PPM >= threshold and writes Excel plus KMZ bundles
- Workbook format (all flows):
  - `ALL` sheet (yellow tab) plus one sheet per unique Date (e.g., `Sept 2, 2025`)
  - Columns (order):
    - Gold (KMZ): `PLID, BeginMeasu, EndMeasure, Class_Loca, Diameter, Product, Route_Desc`
    - Blue (CSV): `Date, PPM, Longitude, Latitude, View in Google Earth, TEMP (F), Serial No., Source_File`
    - Diagnostics appended at end when present: `KMZ_Match_Method, Nearest_Distance_Meters`

**Key Features**
- Robust header fix + column extraction from raw flight CSVs
- KMZ enrichment with distance checks
  - Local meters projection for distance calculation
  - Buffer-first (15.24 m) then nearest fallback
  - 150 m cap: rows beyond threshold are excluded
  - Attributes parsed from KML description tables; fallback to Placemark name for PLID
- XLSX formatting
  - Blue/gold headers; autosized columns; date/number formats
  - PPM >= 5.00 highlighted in yellow; rows with `Nearest_Distance_Meters > 500` shaded light grey
- "View in Google Earth" column
  - Per‑row KML written to `temp_kml/` and hyperlinked as "View Placemark"
  - Styled balloon (client-like): header band + zebra table; lat/long to 7 decimals
  - PPM rounded to 2 decimals for KML `<name>` and balloon
  - Unique filenames: `PPM_{value_2dp}_r{rowIndex}.kml`
  - `temp_kml` is cleared each run

**Install & Run**
- Python 3.10+
- First run auto‑installs requirements from `requirements.txt`
- Run via CLI (or drag‑and‑drop parent folder on Windows):
  - `python main.py "path\to\parent_folder"`
  - `python clean.py "path\to\parent_folder"`
  - `python hits.py "path\to\parent_folder"`

**Outputs**
- Excel-Reports: XLSX/CSV exports
  - `Excel-Reports/Combined_Extracted.xlsx` (main)
  - `Excel-Reports/Combined_Extracted_Clean.xlsx` (clean)
  - `Excel-Reports/Combined_Extracted_Hits.xlsx` and `.csv` (hits flow)
- KMZ: map exports (hits flow)
  - `KMZ/Combined_Extracted_Hits_Placemarks.kmz` — all per‑row placemark KMLs
  - `KMZ/Combined_Extracted_Hits_All.kmz` — heatmap overlay + spaced placemark pins (decluttered)

**Hits Flow**
- Threshold (default): keeps rows with `PPM >= 5.0`
- Placemarks.kmz: bundles every `temp_kml/*.kml` with a `doc.kml` of NetworkLinks
- All.kmz: embeds the heatmap (`overlay.png`) and a spaced subset of pins to reduce clutter
  - Guarantees the highest single PPM pin shows
  - If there are ties for highest, it chooses the one closest to the dataset centroid
  - Even spacing "every N meters" along the principal direction (via PCA) from the top PPM centerpoint

**Heatmap**
- Resolution: 512x512 PNG overlay
- Bounds: tight LatLonBox with ~0.0003° padding
- Interpolation: inverse‑distance weighting (power 2)
- Color ramp: green → yellow → orange → red → purple
- Single‑value mode: smooth hot‑core kernel (orange→red→purple)

**Settings**
- Pin spacing in All.kmz
  - File: `modules/config.py`
  - Keys: `DEFAULT_PIN_SPACING_METERS` (code default) and `pin_spacing_meters` (YAML)
  - Larger values = fewer pins; smaller values = denser pins
- Hits threshold
  - File: `hits.py` (code default) or YAML
  - Keys: `HITS_THRESHOLD = 5.0` (code default) and `hits_threshold` (YAML)
- KMZ enrichment
  - File: `modules/config.py`
  - Keys: `DEFAULT_KMZ_PATH`, `DEFAULT_KMZ_DISTANCE_THRESHOLD` (code defaults) and `kmz_path`, `kmz_distance_threshold` (YAML)
- Temp KML styling and naming
  - File: `modules/xlsx_multisheet_writer.py`
  - Controls balloon HTML, PPM rounding (2dp), coordinate display (7dp), filename format `PPM_{value_2dp}_r{row}.kml`
- Heatmap rendering tweaks
  - File: `hits.py` (helpers around `_heatmap_snippet_and_assets`)
  - Adjust: grid size (W/H), padding, kernel sigma, color stops
- Output folders
  - Excel -> `Excel-Reports/` and KMZ -> `KMZ/`
  - Base output location is the input folder (or common parent for multiple inputs)

**Configuration via YAML**
- Place `config.yaml` (or `config.yml`) in either the working directory or project root.
- Examples:
  - Minimal to tweak hits and spacing
    ```yaml
    hits_threshold: 5.3
    pin_spacing_meters: 35
    ```
  - With KMZ settings
    ```yaml
    kmz_path: kmz/AirLiquideH2.kmz
    kmz_distance_threshold: 150
    kmz_buffer_meters: 15.24
    ```
  - Other optional keys also supported: `columns_to_extract`, `header_renames`, `output_format`, `include_separator_rows`.
 - A commented `config.yaml` is included in the repo — edit values directly to customize behavior.

**Dedup Logic (clean.py)**
- Per CSV, compute unique `(BeginMeasu, EndMeasure)`
- Keep the row with the highest PPM for each pair (ties keep one)
- Combine deduped groups into the final workbook

**Notes**
- Diagnostics, when present, are appended strictly at the end and do not alter the core column order

**Settings Explained**
- `hits_threshold`
  - What it does: Sets the minimum gas reading (in PPM) that counts as a “hit.” Rows below this value are not included in the hits Excel or KMZ exports.
  - When to change: Raise it to focus on stronger detections; lower it to see more points.
  - Typical range: 5.0–7.0 PPM.
- `pin_spacing_meters`
  - What it does: Controls how spread out the pins are in the “All” KMZ. The app always shows the single highest PPM reading, then places additional pins every N meters along the main direction of travel.
  - When to change: Increase to declutter a tight cluster; decrease to show more detail.
  - Typical range: 20–50 meters.
- `kmz_path`
  - What it does: Points to a reference KMZ (e.g., pipeline alignment) used to enrich the CSV rows with fields like PLID, product, etc.
  - When to change: If your project uses a different network or alignment file.
- `kmz_distance_threshold`
  - What it does: The maximum allowed distance (in meters) between a reading and the nearest feature in the reference KMZ for it to be considered a match. Rows farther than this are excluded.
  - When to change: Tighten to be more conservative; loosen if your GPS or reference alignment is rough.
- `kmz_buffer_meters`
  - What it does: The first “capture radius” used when finding a nearby feature before falling back to the pure nearest feature. Think of it as the half‑width of a corridor around lines.
  - Notes: 15.24 meters ≈ 50 feet.
- `columns_to_extract` (advanced)
  - What it does: Lets you restrict which columns from the raw CSVs are kept during processing.
  - When to change: Only if your source files have many extra fields you want to ignore.
- `header_renames` (advanced)
  - What it does: Maps raw column names to the standardized names the app expects (useful when headers vary between files).
  - When to change: If incoming CSV headers differ from the expected names.
- `output_format` (advanced)
  - What it does: Output type for the workbook. The app writes Excel by default and also writes a CSV variant for the hits flow.
- `include_separator_rows` and `separator_style` (advanced)
  - What they do: Optional visual separator rows in the Excel sheets and their styling. Safe to leave as defaults.

Defaults if no YAML is present
- hits_threshold: 5.0
- pin_spacing_meters: 25.0 meters
- kmz_distance_threshold: 150.0 meters
- kmz_buffer_meters: 15.24 meters
- kmz_path: kmz/AirLiquideH2.kmz
- Output folders: Excel files -> `Excel-Reports/`, KMZ files -> `KMZ/`
