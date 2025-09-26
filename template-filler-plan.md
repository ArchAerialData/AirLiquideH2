# Template Filler Implementation Plan

1. **Assess existing pipeline assets**  
   - Inspect `hits.py`, CSV parsing modules, and KMZ utilities to confirm how PLID/Measure/Date fields are produced.  
   - Success: Clear mapping of required columns and dependencies documented.
2. **Define template-filler requirements**  
   - Capture input expectations (recursive CSV scan, KMZ enrichment, template path), matching rules, and output formatting (earliest date, `N/A`, `MM/DD/YYYY`, XLSX).  
   - Success: Written acceptance criteria for matching logic and outputs.
3. **Implement `template-filler.py`**  
   - Wire CLI argument parsing (`--template`, `--output`, KMZ overrides) and reuse `resolve_input_paths`.  
   - Reuse CSV parsing plus KMZ enrichment to build a PLID/measure -> earliest date lookup.  
   - Update the template `Date` column with matches or `N/A` and emit XLSX (optionally refresh CSV).  
   - Success: Script runs without syntax errors and produces the filled workbook.
4. **Validate workflow**  
   - Smoke-test parsing via `python -m compileall template-filler.py` (already executed).  
   - Dry-run against sample data once KMZ dependencies are available; confirm earliest-date selection and `N/A` behavior.  
   - Success: Validation log or manual review confirming expected matches.
5. **Operational guidance**  
   - Document invocation examples and follow-up steps (e.g., rerun when new field data arrives, keep KMZ path current).  
   - Success: Users can reproduce the process confidently following the checklist above.
