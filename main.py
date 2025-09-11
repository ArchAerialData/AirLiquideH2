# Entry point for the Air Liquide H2 CSV merge skeleton.
# This file intentionally avoids doing real work until you specify the schema.
from __future__ import annotations

import sys
from pathlib import Path

from modules.io_utils import resolve_input_paths, derive_output_path
from modules.file_discovery import find_csv_files
from modules.logging_utils import get_logger
from modules.csv_parser import (
    parse_raw_csv,
    attach_source_metadata,
    ParsedCSV,
    enrich_with_kmz,
)
from modules.csv_schema import infer_source_info
from modules.kmz_lookup import KMZIndex
from modules.aggregator import combine
from modules.xlsx_writer import write_with_separators
from modules.config import SeparatorStyle, DEFAULT_KMZ_PATH, DEFAULT_KMZ_DISTANCE_THRESHOLD
from modules.output_schema import to_output_df

logger = get_logger(__name__)

def run(argv: list[str]) -> None:
    """Complete processing pipeline."""
    # Setup
    input_dirs = resolve_input_paths(argv)
    if not input_dirs:
        logger.warning("No input directories provided. Exiting.")
        return

    logger.info("Input directories:")
    for p in input_dirs:
        logger.info(f"  - {p}")

    # Discover CSV files
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

    # Process each CSV
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

            # Project to final output schema (Date formatting, TEMP F, PLID, etc.)
            projected = parsed_csv
            projected.df = to_output_df(parsed_csv.df)
            parsed_csvs.append(projected)

        except Exception as e:
            logger.error(f"Failed to process {csv_path}: {e}")

    if not parsed_csvs:
        logger.error("No CSVs were successfully processed.")
        return

    # Combine all CSVs
    logger.info("Combining CSV data...")
    combined_df, boundaries = combine(parsed_csvs)

    # Write output
    out_path = derive_output_path(input_dirs, preferred_name="Combined_Extracted.xlsx")
    logger.info(f"Writing output to: {out_path}")

    separator_style = SeparatorStyle()
    write_with_separators(combined_df, boundaries, out_path, separator_style)

    logger.info("Processing complete!")


if __name__ == "__main__":
    run(sys.argv[1:])
