from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from modules.io_utils import resolve_input_paths
from modules.file_discovery import find_csv_files
from modules.logging_utils import get_logger
from modules.output_schema import to_output_df


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


def _month_name(dt: datetime) -> str:
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
    return names[dt.month - 1]


def _time_suffix(dt: datetime) -> str:
    hh = dt.strftime("%I").lstrip("0") or "0"
    mm = dt.strftime("%M")
    ampm = dt.strftime("%p").lower()
    return f"{hh}-{mm}{ampm}"


def _build_output_paths(input_dirs: List[Path], prefix: str) -> Tuple[Path, Path, str]:
    # Determine parent indicator
    parent = input_dirs[0].name if input_dirs else "UNKNOWN"
    if len(input_dirs) > 1:
        parent = "MULTI"

    now = datetime.now()
    stamp = f"{_month_name(now)}_{now.day}_{_time_suffix(now)}"
    base_name = f"{prefix}_{parent}_{stamp}"

    out_dir = Path("test_outputs") / "xlsx"
    log_dir = Path("test_outputs") / "xlsx_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = out_dir / f"{base_name}.xlsx"
    log_path = log_dir / f"{base_name}.txt"
    return xlsx_path, log_path, base_name


def _reasons_for_zero(cleaned_df: pd.DataFrame, projected_df: pd.DataFrame) -> List[str]:
    reasons: List[str] = []
    required = ["Time Stamp", "Longitude", "Latitude", "PPM"]
    for col in required:
        if col not in cleaned_df.columns:
            reasons.append(f"missing column '{col}' (header inference)")
        else:
            share = pd.to_numeric(cleaned_df[col], errors="coerce").notna().mean() if col != "Time Stamp" else pd.to_datetime(cleaned_df[col], errors="coerce").notna().mean()
            if share < 0.1:
                reasons.append(f"column '{col}' mostly empty ({share:.0%})")

    # Dedupe relies on Begin/End/PPM in projected
    for col in ["BeginMeasu", "EndMeasure", "PPM"]:
        if col not in projected_df.columns:
            reasons.append(f"missing projected '{col}'")
        else:
            share = pd.to_numeric(projected_df[col], errors="coerce").notna().mean()
            if share < 0.05:
                tag = "(no KMZ match)" if col in {"BeginMeasu", "EndMeasure"} else ""
                reasons.append(f"projected '{col}' mostly NaN {tag}")
    if not reasons:
        reasons.append("no dedupe groups (all PPM NaN or measures NaN)")
    return reasons


def run(argv: list[str]) -> None:
    _ensure_requirements()

    # Lazy imports after requirements check
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

    input_dirs = resolve_input_paths(argv)
    if not input_dirs:
        logger.warning("No input directories provided. Exiting.")
        return

    xlsx_path, log_path, base_name = _build_output_paths(input_dirs, prefix="CLEAN")

    csv_files = find_csv_files(input_dirs)
    errors: List[str] = []
    file_summaries: List[str] = []
    parsed_csvs: List[ParsedCSV] = []

    # KMZ index (optional)
    kmz_index = None
    kmz_path = Path(DEFAULT_KMZ_PATH)
    if kmz_path.exists():
        try:
            kmz_index = KMZIndex(kmz_path)
        except Exception as e:
            errors.append(f"KMZ init failed: {e}")

    for csv_path in csv_files:
        try:
            cleaned_df, project_name = parse_raw_csv(csv_path)
            source_info = infer_source_info(csv_path)
            cleaned_df = attach_source_metadata(cleaned_df, csv_path, project_name)

            parsed = ParsedCSV(df=cleaned_df, source_info=source_info, project_name=project_name)
            if kmz_index is not None:
                parsed = enrich_with_kmz(parsed, kmz_index, max_distance=DEFAULT_KMZ_DISTANCE_THRESHOLD)

            projected_df = to_output_df(parsed.df)
            original_count = len(projected_df)

            # Deduplicate per CSV (clean variant)
            from modules.dedupe import dedupe_by_measure
            deduped_df = dedupe_by_measure(projected_df)
            kept_count = len(deduped_df)

            if kept_count == 0:
                reasons = ", ".join(_reasons_for_zero(cleaned_df, projected_df))
                file_summaries.append(f"{csv_path.name}: rows=0; reasons: {reasons}")
            else:
                file_summaries.append(f"{csv_path.name}: rows={kept_count} (from {original_count})")

            parsed_csvs.append(ParsedCSV(
                df=deduped_df,
                source_info=parsed.source_info,
                project_name=parsed.project_name,
                enriched=parsed.enriched,
            ))
        except Exception as e:
            errors.append(f"{csv_path.name}: {e}")

    combined_df, boundaries = combine(parsed_csvs)

    # Write XLSX to test_outputs/xlsx with custom name
    sep_style = SeparatorStyle()
    try:
        from modules.xlsx_multisheet_writer import write_all_and_date_sheets
        write_all_and_date_sheets(combined_df, xlsx_path, sep_style)
    except Exception as e:
        errors.append(f"XLSX write failed: {e}")

    # Create log file alongside
    lines: List[str] = []
    lines.append(f"Script: test_clean.py")
    lines.append(f"Output: {xlsx_path}")
    lines.append(f"CSV files discovered: {len(csv_files)}")
    lines.append(f"Total combined rows: {len(combined_df)}")
    lines.append("")
    lines.append("Per-file summary:")
    lines.extend([f"  - {s}" for s in file_summaries])
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend([f"  - {e}" for e in errors])
    log_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"Wrote test XLSX: {xlsx_path}")
    logger.info(f"Wrote test log:  {log_path}")


if __name__ == "__main__":
    run(sys.argv[1:])

