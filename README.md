Air Liquide H2 – CSV → KMZ‑enriched XLSX Pipelines
==================================================

Overview
- Two entry points:
  - `main.py`: Full dataset export with KMZ enrichment
  - `clean.py`: Same steps, but per‑CSV dedupe to one row per unique (BeginMeasu, EndMeasure) with max PPM
- Output format: Multi‑sheet XLSX
  - Sheet 1: `ALL` (yellow tab) – single header row; no group banners
  - One sheet per unique Date (ascending), named like `Sept 2, 2025`
- Columns (order):
  - Gold (KMZ): PLID, BeginMeasu, EndMeasure, Class_Loca, Diameter, Product, Route_Desc
  - Blue (CSV): Date, PPM, Longitude, Latitude, View in Google Earth, TEMP (F), Serial No., Source_File
  - Diagnostics (appended at end if present): KMZ_Match_Method, Nearest_Distance_Meters

Key Features
- Header fix + column extraction from raw flight CSVs
- KMZ enrichment with high‑accuracy matching:
  - Projected distances in meters (EPSG:3857)
  - Buffer‑first (15.24 m) then nearest‑fallback
  - 150 m cap: rows beyond threshold are excluded
  - Attributes parsed from KML description table with fallback to Placemark name for PLID
  - Diagnostics appended at end
- XLSX formatting:
  - Blue/gold headers; autosized columns; date/number formats
  - PPM ≥ 5.00 highlighted in yellow
  - Rows with Nearest_Distance_Meters > 500 shaded light grey
- “View in Google Earth” column:
  - Per‑row KML written to `temp_kml/` and hyperlinked as “View Placemark”
  - Placemark balloon mimics client style: centered “{PPM} PPM”, blue header band, zebra table; lat/long to 7 decimals
  - `temp_kml` is cleared each run

Install & Run
1) Python 3.10+
2) First run auto‑installs requirements if missing (`requirements.txt`):
   - pandas, openpyxl, fastkml, shapely, beautifulsoup4, lxml, rtree, pyproj, PyYAML
3) Drag‑and‑drop a parent folder onto the script (Windows) or run via CLI:
   - `python main.py "path\to\parent_folder"`
   - `python clean.py "path\to\parent_folder"`
4) Outputs
   - `Combined_Extracted.xlsx` (main)
   - `Combined_Extracted_Clean.xlsx` (clean)

Dedup Logic (clean.py)
- Per CSV, compute unique (BeginMeasu, EndMeasure)
- Keep the row with the highest PPM for each pair (ties keep one)
- Combine deduped groups into the final workbook

Config
- KMZ path: `kmz/AirLiquideH2.kmz` (modules/config.py)
- KMZ distance threshold: 150 m
- Buffer size: 15.24 m

Notes
- Only rows with a matched feature within 150 m are included
- Diagnostics (if present) are appended strictly at the end and do not alter core column order
