from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from modules.io_utils import resolve_input_paths, derive_output_path
from modules.file_discovery import find_csv_files
from modules.logging_utils import get_logger
from modules.output_schema import to_output_df
from modules.dedupe import dedupe_by_measure


logger = get_logger(__name__)


def _ensure_requirements() -> None:
    import importlib, subprocess, sys
    needed = [
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
        ("fastkml", "fastkml"),
        ("shapely", "shapely"),
        ("bs4", "beautifulsoup4"),
        ("lxml", "lxml"),
        ("rtree", "rtree"),
        ("pyproj", "pyproj"),
    ]
    missing = []
    for mod, _ in needed:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        req = str(Path(__file__).parent / "requirements.txt")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])
        except Exception:
            pass


def run(argv: list[str]) -> None:
    _ensure_requirements()

    # Lazy imports of heavy deps after ensuring requirements
    from modules.csv_parser import (
        parse_raw_csv,
        attach_source_metadata,
        ParsedCSV,
        enrich_with_kmz,
    )
    from modules.csv_schema import infer_source_info
    from modules.kmz_lookup import KMZIndex
    from modules.aggregator import combine
    from modules.xlsx_multisheet_writer import write_all_and_date_sheets
    from modules.config import SeparatorStyle, DEFAULT_KMZ_PATH, DEFAULT_KMZ_DISTANCE_THRESHOLD
    """Deduped processing pipeline (clean variant).

    - Same ingestion, enrichment, and projection as main.py
    - Per CSV, dedupe rows to one per (BeginMeasu, EndMeasure) with max PPM
    - Combine all deduped CSVs into a single XLSX with identical formatting
    """
    input_dirs = resolve_input_paths(argv)
    if not input_dirs:
        logger.warning("No input directories provided. Exiting.")
        return

    logger.info("Input directories:")
    for p in input_dirs:
        logger.info(f"  - {p}")

    csv_files = find_csv_files(input_dirs)
    if not csv_files:
        logger.warning("No CSV files found under the provided directories.")
        return

    logger.info(f"Discovered {len(csv_files)} CSV files (recursive).")

    # Load KMZ index (configurable path)
    kmz_path = Path(DEFAULT_KMZ_PATH)
    if kmz_path.exists():
        logger.info(f"Loading KMZ spatial index from: {kmz_path}")
        try:
            kmz_index = KMZIndex(kmz_path)
        except Exception as e:
            logger.warning(f"Failed to initialize KMZ index ({e}). Proceeding without enrichment.")
            kmz_index = None
    else:
        logger.warning("KMZ file not found. Proceeding without spatial enrichment.")
        kmz_index = None

    parsed_csvs: list[ParsedCSV] = []
    for csv_path in csv_files:
        try:
            logger.info(f"Processing: {csv_path}")

            # Parse and clean CSV
            cleaned_df, project_name = parse_raw_csv(csv_path)
            source_info = infer_source_info(csv_path)
            cleaned_df = attach_source_metadata(cleaned_df, csv_path, project_name)

            parsed_csv = ParsedCSV(
                df=cleaned_df,
                source_info=source_info,
                project_name=project_name,
            )

            # Enrich with KMZ if available
            if kmz_index is not None:
                parsed_csv = enrich_with_kmz(parsed_csv, kmz_index, max_distance=DEFAULT_KMZ_DISTANCE_THRESHOLD)

            # Project to final output schema
            projected_df = to_output_df(parsed_csv.df)
            original_count = len(projected_df)

            # Deduplicate per CSV
            deduped_df = dedupe_by_measure(projected_df)
            kept_count = len(deduped_df)
            logger.info(f"  Deduped rows: kept {kept_count} of {original_count} (unique measure pairs)")

            parsed_csvs.append(ParsedCSV(
                df=deduped_df,
                source_info=parsed_csv.source_info,
                project_name=parsed_csv.project_name,
                enriched=parsed_csv.enriched,
            ))

        except Exception as e:
            logger.error(f"Failed to process {csv_path}: {e}")

    if not parsed_csvs:
        logger.error("No CSVs were successfully processed.")
        return

    # Combine all CSVs
    logger.info("Combining deduped CSV data...")
    combined_df, boundaries = combine(parsed_csvs)

    # Write output (multi-sheet by default)
    out_path = derive_output_path(input_dirs, preferred_name="Excel-Reports/Combined_Extracted_Clean.xlsx")
    logger.info(f"Writing output to: {out_path}")

    separator_style = SeparatorStyle()
    write_all_and_date_sheets(combined_df, out_path, separator_style)

    logger.info("Clean (deduped) processing complete!")


if __name__ == "__main__":
    run(sys.argv[1:])
