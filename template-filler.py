from __future__ import annotations

import argparse
import sys
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from modules.io_utils import resolve_input_paths
from modules.file_discovery import find_csv_files
from modules.logging_utils import get_logger
from modules.csv_parser import parse_raw_csv, attach_source_metadata, ParsedCSV, enrich_with_kmz
from modules.csv_schema import infer_source_info
from modules.kmz_lookup import KMZIndex
from modules.config import DEFAULT_KMZ_DISTANCE_THRESHOLD, DEFAULT_KMZ_PATH, load_default_config
from modules.output_schema import to_output_df

try:
    from hits import _ensure_requirements
except Exception:
    def _ensure_requirements() -> None:
        """Fallback no-op when hits._ensure_requirements is unavailable."""
        return None


logger = get_logger(__name__)

_DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "test_folders" / ".example_output" / "TrackerTemplate.csv"
if not _DEFAULT_TEMPLATE.exists():
    _DEFAULT_TEMPLATE = None

NormalizedKey = Tuple[Decimal, Decimal, Decimal]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill TrackerTemplate.csv with earliest dates matching PLID, BeginMeasu, and "
            "EndMeasure extracted from recursively parsed CSV files."
        )
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        help="One or more root folders to scan for CSV files (recursive).",
    )
    default_template = str(_DEFAULT_TEMPLATE) if _DEFAULT_TEMPLATE else None
    parser.add_argument(
        "-t",
        "--template",
        default=default_template,
        required=default_template is None,
        help=(
            "Path to TrackerTemplate.csv. "
            + ("Defaults to the repo example template." if default_template else "")
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination XLSX path. Defaults to the template path with a .xlsx extension.",
    )
    parser.add_argument(
        "--kmz",
        help="Optional override for the KMZ file used to enrich measurements.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        help="Maximum KMZ match distance in meters (overrides config).",
    )
    parser.add_argument(
        "--skip-kmz",
        action="store_true",
        help="Skip KMZ enrichment even if a KMZ file is available.",
    )
    parser.add_argument(
        "--update-template-csv",
        action="store_true",
        help="Overwrite the template CSV with the filled Date values.",
    )
    return parser.parse_args(argv)


def _normalize_decimal(value: object, digits: int) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        token = value.strip()
        if not token or token.lower() == "nan":
            return None
    else:
        token = str(value)
    try:
        dec = Decimal(token)
    except InvalidOperation:
        try:
            dec = Decimal(str(float(token)))
        except (InvalidOperation, ValueError, TypeError):
            return None
    quantizer = Decimal(1).scaleb(-digits) if digits > 0 else Decimal(1)
    try:
        return dec.quantize(quantizer, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


def _build_key(plid: object, begin: object, end: object) -> Optional[NormalizedKey]:
    plid_norm = _normalize_decimal(plid, digits=0)
    begin_norm = _normalize_decimal(begin, digits=5)
    end_norm = _normalize_decimal(end, digits=5)
    if plid_norm is None or begin_norm is None or end_norm is None:
        return None
    return (plid_norm, begin_norm, end_norm)


def _update_lookup_from_output(df: pd.DataFrame, lookup: Dict[NormalizedKey, date]) -> int:
    if df.empty:
        return 0
    required = {"PLID", "BeginMeasu", "EndMeasure", "Date"}
    if not required.issubset(df.columns):
        return 0
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.date
    plids = df["PLID"]
    begins = df["BeginMeasu"]
    ends = df["EndMeasure"]
    updated = 0
    for plid, begin, end, dt in zip(plids, begins, ends, dates):
        key = _build_key(plid, begin, end)
        if key is None or dt is None or pd.isna(dt):
            continue
        existing = lookup.get(key)
        if existing is None or dt < existing:
            lookup[key] = dt
        updated += 1
    return updated


def collect_measure_dates(
    csv_files: Iterable[Path],
    kmz_index: Optional[KMZIndex],
    max_distance: float,
) -> Dict[NormalizedKey, date]:
    lookup: Dict[NormalizedKey, date] = {}
    processed = 0
    for csv_path in csv_files:
        processed += 1
        logger.info(f"Processing CSV: {csv_path}")
        try:
            cleaned_df, project_name = parse_raw_csv(csv_path)
            source_info = infer_source_info(csv_path)
            cleaned_df = attach_source_metadata(cleaned_df, csv_path, project_name)
            parsed = ParsedCSV(
                df=cleaned_df,
                source_info=source_info,
                project_name=project_name,
            )
            if kmz_index is not None:
                parsed = enrich_with_kmz(parsed, kmz_index, max_distance=max_distance)
            output_df = to_output_df(parsed.df)
            _update_lookup_from_output(output_df, lookup)
        except Exception as exc:
            logger.error(f"Failed to process {csv_path}: {exc}")
    logger.info(
        f"Collected {len(lookup)} unique PLID/measure ranges across {processed} file(s)."
    )
    return lookup


def fill_template(template_path: Path, lookup: Dict[NormalizedKey, date]) -> Tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(template_path)
    if "Date" not in df.columns:
        df["Date"] = ""
    filled = 0
    unmatched = 0
    for idx, row in df.iterrows():
        key = _build_key(row.get("PLID"), row.get("BeginMeasu"), row.get("EndMeasure"))
        if key is None:
            df.at[idx, "Date"] = "N/A"
            unmatched += 1
            continue
        match_date = lookup.get(key)
        if match_date is None:
            df.at[idx, "Date"] = "N/A"
            unmatched += 1
        else:
            df.at[idx, "Date"] = match_date.strftime("%m/%d/%Y")
            filled += 1
    df["Date"] = df["Date"].fillna("N/A").astype(str)
    return df, filled, unmatched


def write_outputs(
    df: pd.DataFrame,
    template_path: Path,
    output_path: Path,
    update_csv: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    logger.info(f"Tracker XLSX written to {output_path}")
    if update_csv:
        df.to_csv(template_path, index=False)
        logger.info(f"Template CSV updated in-place at {template_path}")


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_requirements()
    args = parse_args(argv or sys.argv[1:])

    if not args.template:
        logger.error("A template CSV path must be provided with --template.")
        return 1

    template_path = Path(args.template).expanduser().resolve()
    if not template_path.exists():
        logger.error(f"Template CSV not found: {template_path}")
        return 1

    input_dirs = resolve_input_paths(args.input_dirs)
    if not input_dirs:
        logger.error("No input directories provided. Cannot proceed.")
        return 1

    csv_files = find_csv_files(input_dirs)
    template_resolved = template_path.resolve()
    csv_files = [p for p in csv_files if p.resolve() != template_resolved]
    if not csv_files:
        logger.warning("No CSV files discovered under the provided directories.")

    cfg = load_default_config()
    kmz_path = Path(args.kmz or cfg.kmz_path or DEFAULT_KMZ_PATH).expanduser().resolve()
    kmz_index: Optional[KMZIndex] = None
    if args.skip_kmz:
        logger.info("Skipping KMZ enrichment (--skip-kmz supplied).")
    else:
        if kmz_path.exists():
            try:
                kmz_index = KMZIndex(kmz_path)
                logger.info(f"Loaded KMZ index from {kmz_path}")
            except Exception as exc:
                logger.warning(f"Unable to initialize KMZ index ({exc}). Continuing without KMZ.")
                kmz_index = None
        else:
            logger.warning(f"KMZ file not found at {kmz_path}. Proceeding without KMZ enrichment.")

    max_distance = float(
        args.max_distance
        or getattr(cfg, "kmz_distance_threshold", DEFAULT_KMZ_DISTANCE_THRESHOLD)
        or DEFAULT_KMZ_DISTANCE_THRESHOLD
    )

    if kmz_index is None:
        logger.warning("KMZ enrichment disabled; PLID/measure matches may be empty.")

    lookup = collect_measure_dates(csv_files, kmz_index, max_distance)
    if not lookup:
        logger.warning("No matching PLID/Begin/End dates were found. Template will be filled with N/A.")

    filled_df, matched, unmatched = fill_template(template_path, lookup)
    total_rows = len(filled_df)
    logger.info(
        f"Template update complete: {matched} matched, {unmatched} set to N/A, {total_rows} rows total."
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output else template_path.with_suffix(".xlsx")
    )
    write_outputs(filled_df, template_path, output_path, update_csv=args.update_template_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
