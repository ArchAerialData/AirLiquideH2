KMZ Matching Refactor — Plan (Projection + Buffer + Fallback)

Objective
- Improve KMZ enrichment hit‑rate and correctness by:
  - Computing distances in meters via projection (pyproj + shapely.transform)
  - Using a buffer‑first match strategy, then nearest‑fallback (never leave blank)
  - Keeping our existing HTML attribute extraction for gold fields
  - Appending any new diagnostic columns at the very end of the current table layout (do not reorder existing columns)
  - Visually flagging potential outliers: rows with Nearest_Distance_Meters > 500 highlighted light grey

Current Gaps (why gold columns can be blank)
- Degrees→meters approximation (deg*111,000) can misclassify proximity.
- Hard cutoff (e.g., ≤100 m) leaves non‑matches blank.
- Rtree candidate set is small and built on geographic bounds; dense systems may miss close features.

What We’ll Borrow From the Reference Script
- Project geometries and points to a metric CRS (EPSG:3857) for accurate meters.
- Precompute buffered geometries around lines (e.g., 15.24 m / 50 ft).
- Match order:
  1) If point is inside any buffer → pick closest line among those hits.
  2) Else → pick the single nearest line (fallback) and still enrich (record distance + method).

Scope of Refactor (modules to touch)
1) modules/kmz_lookup.py
   - Add pyproj dependency and store per‑feature:
     - original geometry (lon/lat)
     - projected LineString (3857)
     - projected buffer polygon (buffer_meters)
   - Build rtree index on projected bounds for better candidate retrieval.
   - Replace deg→m distance with true projected distance.
   - Implement buffer‑first, nearest‑fallback logic.
   - Keep HTML description table parsing; if a key is absent, fall back to Placemark name for PLID.
   - Return both attributes and diagnostics (distance, method).

2) modules/csv_parser.py (enrich_with_kmz)
   - Consume the new diagnostics from KMZIndex.lookup and set the appended columns.
   - Preserve existing gold field population (PLID, BeginMeasu, EndMeasure, Route_Desc, Class_Loca, Diameter, Product).

3) modules/output_schema.py
   - Keep OUTPUT_COLUMNS exactly as is (do not alter order).
   - Append new diagnostics columns at the end only if present:
     - KMZ_Match_Method (string; values: "buffer_hit" | "nearest_fallback")
     - Nearest_Distance_Meters (float; true meters from projected geometry)
   - Ensure reindex uses OUTPUT_COLUMNS + any extras in that order.

4) modules/xlsx_multisheet_writer.py and modules/xlsx_writer.py
   - No header order changes; if appended diagnostics exist, write them as trailing columns.
   - Apply number format for Nearest_Distance_Meters (0.00) when present.
   - Add conditional formatting to highlight entire data rows where Nearest_Distance_Meters > 500 using a light grey fill (e.g., FFEFEFEF). This rule applies on every sheet (ALL and per‑date) and does not alter values.

Appended Columns (added strictly at the end)
- KMZ_Match_Method: string; how the match was chosen.
- Nearest_Distance_Meters: float; true meter distance for the chosen feature.
Notes:
- These are additive only; they will appear after Source_File and will not interrupt any existing columns.
- If diagnostics are not desired later, we can hide them or gate via config without changing the core order.
 - No additional diagnostics beyond these two will be added at this time.

Matching Algorithm (per row)
1) Project point (lon, lat) to EPSG:3857 via a cached Transformer.
2) Get candidates from rtree built on projected bounds. Use either:
   - index.intersection with a small search window around the point (e.g., ±R meters), or
   - index.nearest with a larger candidate count (e.g., 50).
3) Among candidates, check buffer.contains(point). If any hits:
   - Compute true line distance (projected) for those hits; pick minimum → method = buffer_hit.
4) Else:
   - Compute true line distance to each candidate; pick minimum → method = nearest_fallback.
5) Return feature attributes + diagnostics: distance (meters) and method.

Configuration Knobs
- buffer_meters (default: 15.24) — confirmed
- candidate_count (default: 50) if using nearest; or search_radius_m (default: 200) if using intersection window
- projection (fixed to 3857 for now; can switch to UTM later if needed)

Performance Considerations
- Precomputing projected & buffered geometries increases memory but reduces per‑point CPU.
- rtree on projected bounds improves candidate quality.
- Sequential points likely match similar features; a future optimization could seed search with last matched feature (not required for first pass).

Data Integrity & Formatting
- Gold attribute values will still come from the KML description table where present.
- For PLID/BeginMeasu/EndMeasure, we keep values as parsed and write as numbers to avoid “stored as text”; Excel display formatting remains the same.
- New diagnostics are appended; existing layout unchanged.

Backwards Compatibility
- No existing column order changes.
- If the writer sees no diagnostic columns, output is identical to today.
- After refactor, blanks in gold fields should largely disappear due to nearest fallback; distance column allows downstream filtering if a row is too far.
 - Additionally, rows with distance > 500 m will be visually highlighted (light grey) so potential mismatches stand out without removing data.

Testing & Validation
- Unit‑style checks on KMZIndex:
  - Feature count before/after parse (fastkml vs lxml fallback) unchanged.
  - Distances match known values in a small synthetic KML.
  - Buffer hit vs nearest fallback behaves as expected at thresholds.
- Integration on test_folders:
  - single_csv and two_csvs: compare enrichment hit‑rates before/after.
  - Verify diagnostics columns appended at end.
  - Confirm Excel number formats and alignments unchanged; PPM >= 5 highlighting unaffected.
  - Confirm “ALL” + per‑Date sheets still render correctly.

Implementation Steps
1) kmz_lookup.py
   - Add pyproj import and a cached Transformer (epsg:4326 → epsg:3857).
   - During _extract_features, compute and store projected LineString + bounds.
   - Build rtree on projected bounds.
   - Add per‑feature projected buffer polygon (buffer_meters from config).
   - Rework lookup() to implement buffer‑first, nearest‑fallback; return attributes + {"Match_Method", "Distance_Meters"}.
2) csv_parser.py: enrich_with_kmz
   - Capture diagnostics; map to output columns KMZ_Match_Method and Nearest_Distance_Meters (only set if present).
3) output_schema.py
   - Extend to_output_df to accept trailing diagnostics if present and reindex as OUTPUT_COLUMNS + trailing extras.
4) writers (xlsx_multisheet_writer.py and xlsx_writer.py)
   - If diagnostics columns exist, write and autosize; set number format for Nearest_Distance_Meters (0.00).
   - Add conditional formatting per sheet to shade entire data rows light grey where Nearest_Distance_Meters > 500 (e.g., using a formula rule anchored to the distance column).
5) Validate end‑to‑end on test folders; review enrichment coverage.

Decisions
- Nearest fallback: Always apply; do not drop far matches. Additionally, highlight rows where distance > 500 m in light grey so they can be reviewed.
- Default buffer_meters: 15.24 (50 ft).
- Additional diagnostics: None beyond KMZ_Match_Method and Nearest_Distance_Meters.
