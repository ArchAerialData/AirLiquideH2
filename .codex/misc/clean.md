Clean Pipeline (clean.py) — Design Plan

Overview
- Goal: Provide a sister CLI to main.py named clean.py that runs the same ingestion and enrichment steps, but dedupes rows per CSV so that only a single row remains for each unique pair of KMZ measures (BeginMeasu, EndMeasure).
- Output: A single XLSX combining all deduped CSVs with the exact same workbook layout, colors, and group headers as main.py currently produces.
- Invocation: Same as main.py — supports drag-and-drop of folders on Windows or CLI paths; recursively discovers CSVs under provided folders.

Dedup Rule (per CSV)
- Grouping Keys: BeginMeasu and EndMeasure (exact string headers in the final output schema).
- Selection: Within each group, keep the row with the highest PPM.
- Tie-breaking: If multiple rows share the same max PPM within a group, keep exactly one of them (first occurrence is acceptable).
- Scope: Dedup is applied independently per CSV. After deduping each CSV, the results are combined into one XLSX with group boundaries corresponding to each input CSV.
- Zeros are valid: Values 0 or 0.00 for BeginMeasu/EndMeasure/PPM are considered valid and must NOT be treated as missing.

Data Flow (per CSV)
1) Parse raw CSV with header fix and column selection (same logic as main.py).
2) Attach source metadata columns (Source_File, Path, Project_Name, etc.).
3) KMZ enrichment via KMZIndex lookup (Route_Name → PLID; other KMZ fields) using DEFAULT_KMZ_PATH.
4) Project to final output schema (Date only; TEMP (F) from Celsius; reorder columns to match current XLSX).
5) Deduplicate within this projected DataFrame using (BeginMeasu, EndMeasure) → row with max PPM.
6) Collect per-CSV deduped frames.
7) Combine frames with updated group boundaries.
8) Write the XLSX with the same writer and formatting as main.py (grey merged project row per group, repeated blue/gold header row per group, column widths, number formats, and alignment rules).

Implementation Notes
- New entrypoint: clean.py (mirrors main.py structure).
  - Reuse modules: io_utils.resolve_input_paths, file_discovery.find_csv_files, csv_parser.parse_raw_csv, csv_parser.attach_source_metadata, csv_parser.enrich_with_kmz, output_schema.to_output_df, aggregator.combine, xlsx_writer.write_with_separators.
  - After to_output_df, apply dedup function before aggregating.
- Additional module(s): exactly one new helper module `modules/dedupe.py` to hold the dedup logic, plus the `clean.py` entrypoint. All other functionality reuses existing modules.
- Dedup function (proposed signature):
  ```python
  def dedupe_by_measure(df: pd.DataFrame) -> pd.DataFrame:
      # Precondition: df is in final output schema with columns BeginMeasu, EndMeasure, PPM
      # Ensure numerics for PPM/measure columns (coerce errors → NaN)
      work = df.copy()
      for c in ("BeginMeasu", "EndMeasure", "PPM"):
          if c in work.columns:
              work[c] = pd.to_numeric(work[c], errors="coerce")
      # Drop rows without measures/PPM (zeros are valid; only NaNs are dropped)
      work = work.dropna(subset=["BeginMeasu", "EndMeasure", "PPM"])  # zeros remain
      # idxmax returns the first occurrence of the max PPM within each (BeginMeasu, EndMeasure) group
      keep_idx = work.groupby(["BeginMeasu", "EndMeasure"]) ["PPM"].idxmax()
      return work.loc[keep_idx].sort_values(["BeginMeasu", "EndMeasure"]).reset_index(drop=True)
  ```
- Group boundaries: The aggregator already derives boundaries from the length of each per-CSV DataFrame. We will pass the deduped frame so the boundaries reflect post-dedup counts.
- Writer: Reuse current writer so formatting remains identical. It already styles gold vs blue headers, merged project rows, date format, numeric formats for PLID/BeginMeasu/EndMeasure, and alignment (gold columns centered except Route_Desc left-aligned).
- Output file name: Default to `Combined_Extracted_Clean.xlsx` to avoid overwriting main.py’s output. Configurable if needed.

CLI/UX
- Drag-and-drop or CLI accepts one or more folder paths (same as main.py).
- Recursively discovers all `.csv`/`.CSV` files.
- Logs the discovered files and the dedup summary per CSV (e.g., original rows vs kept rows, number of measure-pairs).
- Writes one workbook to the parent/common folder of the inputs.

Edge Cases & Rules
- Missing KMZ measures: If (BeginMeasu, EndMeasure) are missing for a row, that row cannot be deduped by the rule. Options:
  - Default: drop those rows from the output (recommended to maintain a clean measure-based view). Zero values are valid and must be kept; only NaN/non-numeric are treated as missing. We can optionally log how many rows were dropped for missing measures.
  - Alternative: keep the first occurrence of each missing-measure row; we can add a toggle if needed.
- Non-numeric PPM: Coerce to numeric; rows with NaN PPM are excluded from the dedup groups by default. If a group has no numeric PPM, we can keep the first row arbitrarily; callout for confirmation if that’s important.
- Ties: idxmax keeps the first occurrence — consistent and deterministic with stable source order.
- Performance: The dedup is O(n) per CSV with a single groupby and idxmax; scalable for large files.

Testing Plan
- Single-file flow: Run `python clean.py "test_folders\single_csv"`, confirm the resulting XLSX has far fewer rows (one per unique measure pair), and the formatting matches main.py.
- Multi-file flow: Run `python clean.py "test_folders\two_csvs"` to confirm two groups with their own merged headers and repeated blue/gold header rows. Verify per-group dedup stats in logs.
- Validation queries:
  - Count unique (BeginMeasu, EndMeasure) pairs equals number of data rows written for each CSV group.
  - For each pair, verify PPM is the maximum in the original group (ties allowed to pick any single row).
  - Ensure PLID/BeginMeasu/EndMeasure remain numeric in Excel (no "stored as text" warnings).
  - Date formatted mm/dd/yyyy; Route_Desc left-aligned; other gold columns centered.

Open Questions (for follow-up)
- Should rows lacking BeginMeasu/EndMeasure be dropped (default) or retained separately? If retained, where should they appear (e.g., at the end of each group) and with what rule?
- Preferred output filename for clean.py (proposed: Combined_Extracted_Clean.xlsx)?
- Any filtering on distance threshold before dedup (e.g., use only matches within a stricter distance)?

Acceptance Criteria
- clean.py processes the same inputs as main.py and writes a single workbook with the same formatting and grouping.
- Within each CSV group, the number of written rows equals the count of unique (BeginMeasu, EndMeasure).
- For every (BeginMeasu, EndMeasure), the retained row has the maximum PPM observed for that pair.
- No Excel "number stored as text" warnings for PLID, BeginMeasu, EndMeasure.
