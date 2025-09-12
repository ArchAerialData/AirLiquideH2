Multi‑Sheet Output Plan (ALL + Date Sheets)

Summary
- Replace per‑CSV merged banner rows and repeated headers with a multi‑sheet workbook structure:
  - Sheet 1: "ALL" — every row from the run (single header in row 1; no merged grey row)
  - Additional sheets: one per unique Date value (from the final output column "Date")
  - Each sheet uses the same columns, header styling, numeric formats, and alignments as the current output
- Applies to both pipelines (main.py and clean.py). For clean.py, "ALL" uses deduped data; per‑date sheets reflect the same deduped data filtered by date.

What I validated in your example
- File: test_folders\\two_csvs\\Combined_Extracted_Clean.xlsx
  - Sheets present: ["ALL", "Sept 2, 2025", "Sept 3, 2025", "Sept 4, 2025"]
  - Tab color for "ALL": FFFFFF00 (yellow)
  - Header row in row 1 with the new column order you approved:
    [PLID, BeginMeasu, EndMeasure, Class_Loca, Diameter, Product, Route_Desc, Date, PPM, Longitude, Latitude, TEMP (F), Serial No., Source_File]

Proposed Approach
- Add a new writer module (e.g., modules/xlsx_multisheet_writer.py) to implement this without disturbing the existing grouped writer.
- Keep modules/xlsx_writer.write_with_separators intact for backward compatibility and one‑sheet legacy runs.
- Provide a single entry point function, e.g.:
  - write_all_and_date_sheets(combined_df, output_path, separator_style, config)
  - Reuses current header styling, alignment rules, and numeric formats.

Behavior Details
- "ALL" sheet
  - Single header row at row 1 with blue/gold fill as today.
  - No merged/bannner rows.
  - Writes all rows from the DataFrame (deduped or not depending on the pipeline).
  - Applies number formats: PLID = 0; BeginMeasu/EndMeasure = 0.00; Date = mm/dd/yyyy.
  - Applies alignment: gold columns centered except Route_Desc left‑aligned.
  - Sets the tab color for "ALL" to FFFFFF00 (from your example). Configurable.

- Per‑Date sheets
  - Identify unique, non‑NaN values in the "Date" column (date objects); zeros are not applicable here.
  - For each unique Date, create a new sheet copy of the header and only the rows for that Date.
  - Header styling and data formatting identical to "ALL".
  - Sheet name format: "Sept D, YYYY" (explicitly using "Sept" for September to match your example; see Name Formatting below).
  - Default tab color: none (as in your example), or configurable.

Name Formatting (Date → Sheet Name)
- Desired: "Sept 2, 2025" vs typical strftime("%b %d, %Y") → "Sep 2, 2025".
- Implement a custom mapping for months to enforce "Sept" specifically:
  - {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}
- Sanitize for Excel sheet name constraints (≤31 chars, no : \\ / ? * [ ]). If collision occurs, append a numeric suffix.

Integration Points
- main.py and clean.py:
  - Both will use the new multi‑sheet writer by default (no toggle).
  - Replace the grouped writer call with modules/xlsx_multisheet_writer.write_all_and_date_sheets(...).

Styling & Formats (unchanged from current header rules)
- Header fills: BLUE for CSV fields, GOLD for KMZ fields.
- Header font: white, bold, centered (Route_Desc header left‑aligned if desired — currently centered in header, left in data).
- Data alignment: KMZ (gold) columns centered except Route_Desc left‑aligned.
- Number formats: PLID = "0"; BeginMeasu/EndMeasure = "0.00"; Date = "mm/dd/yyyy".
- Column autosize per sheet.

Edge Cases & Limits
- No Date rows (NaT):
  - Kept on the "ALL" sheet; excluded from per‑Date sheets (unless we decide to add a "No Date" sheet — optional later).
- Many distinct dates: Potentially many sheets; Excel’s sheet limit is high but practical UX may suffer. We will enforce a reasonable upper limit with a warning if requested.
- Name collisions: After sanitization, if two dates format to the same label (unlikely), append " (2)", "(3)", etc.

Config Details (finalized)
- Multi‑sheet mode: Enabled by default for both main.py and clean.py (no toggle).
- SHEET_ALL_NAME: "ALL".
- ALL_TAB_COLOR: "FFFFFF00" (yellow, per example).
- DATE_SHEET_TAB_COLOR: None (no color for date sheets, per preference).
- DATE_SHEET_NAME_FORMAT: use custom month map to force "Sept" labeling (e.g., "Sept 2, 2025").
- SHEET ORDER: After "ALL", date sheets will be ordered ascending by date.

Conditional Formatting / Highlighting
- Requirement: Highlight any cells in the PPM column with value >= 5.00 across all sheets.
- Implementation: Apply an OpenPyXL conditional formatting rule (CellIsRule) to the PPM column range on each sheet:
  - Range: from the first data row (row 2 on ALL; row 2 on each date sheet) down to the last row in that sheet
  - Condition: ">= 5"
  - Fill: solid yellow (same hex as tab color or a standard highlight, e.g., FFFFFF00)
- This preserves values and number formats; only background fill is applied when condition matches.

Implementation Steps
1) Create modules/xlsx_multisheet_writer.py with write_all_and_date_sheets(...):
   - Build workbook; add "ALL"; style header; write combined_df; apply formats; autosize; set tab color.
   - Add conditional formatting on PPM column (>= 5.00) for ALL sheet.
   - Identify unique Date values (drop NaT). Sort ascending.
   - For each date: create sheet with name via month‑map (e.g., "Sept 2, 2025"); write header and filtered rows; apply formats; autosize; add conditional formatting on PPM column.
2) Add helper for date→sheet‑name conversion (with "Sept" mapping) and sheet name sanitization.
3) Replace writer invocation in both main.py and clean.py to use the new multi‑sheet writer by default.
4) Validate with test_folders\\two_csvs and a synthetic multi‑date dataset; confirm ordering and highlighting.

Validation Plan
- Verify workbook sheets: first is "ALL" with the correct tab color; one sheet per unique Date with expected names (e.g., "Sept 2, 2025").
- Confirm identical header and column order on all sheets.
- Confirm number formats and alignments match current rules.
- Confirm "ALL" contains the union of all rows (deduped or not depending on pipeline).
- Confirm each per‑Date sheet contains only rows for that date.
- Spot‑check column widths appear reasonable on each sheet.

Open Questions
- Do you want the new multi‑sheet mode applied to both main.py and clean.py by default, or only to clean.py initially?
- Should per‑Date sheet tabs have colors? (Your example shows only "ALL" colored). If yes, please provide the hex.
- Any desired sort order for sheets (e.g., ascending by date) and rows (e.g., by PLID or BeginMeasu)?
