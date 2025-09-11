from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .csv_schema import (
    RAW_CSV_COLUMN_MAPPING,
    CSVSchema,
    COLUMN_DTYPES,
    SourceInfo,
    infer_source_info,
    fix_csv_headers,
)
from .kmz_lookup import KMZIndex


@dataclass
class ParsedCSV:
    df: pd.DataFrame
    source_info: SourceInfo
    project_name: str
    enriched: bool = False


def parse_raw_csv(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    """Parse raw CSV with header correction and column extraction.

    Returns:
        - Cleaned DataFrame with target columns only (canonical headers)
        - Project name extracted from row 2 (A2)
    """
    df = pd.read_csv(csv_path, header=None, encoding="utf-8-sig")

    # Extract project name from row 1 (0-based index)
    project_name = str(df.iloc[1, 0]) if len(df) > 1 else "Unknown"

    # Get headers from row 2 and fix missing PPM header
    if len(df) <= 2:
        raise ValueError(f"Invalid CSV structure in {csv_path}")

    raw_headers = df.iloc[2].copy()
    headers = fix_csv_headers(raw_headers)

    # Extract data starting from row 3
    data_df = df.iloc[3:].copy()
    data_df.columns = headers

    # Select only target columns using RAW_CSV_COLUMN_MAPPING (by position)
    # and rename to canonical names.
    schema = CSVSchema()

    # Build a mapping from the actual header label at each position -> canonical name
    positional_labels = {pos: headers.iloc[pos] for pos in RAW_CSV_COLUMN_MAPPING.keys() if pos < len(headers)}
    selected_cols = [positional_labels[pos] for pos in RAW_CSV_COLUMN_MAPPING.keys() if pos in positional_labels]

    cleaned_df = data_df[selected_cols].copy()
    rename_map: Dict[str, str] = {positional_labels[pos]: RAW_CSV_COLUMN_MAPPING[pos] for pos in positional_labels}
    cleaned_df = cleaned_df.rename(columns=rename_map)

    # Apply data type conversions and validations
    cleaned_df = validate_and_convert_types(cleaned_df, schema)

    return cleaned_df, project_name


def validate_and_convert_types(df: pd.DataFrame, schema: Optional[CSVSchema] = None) -> pd.DataFrame:
    """Convert columns to proper data types and handle errors."""
    schema = schema or CSVSchema()
    out = df.copy()

    # Convert timestamp: strip leading stray characters (e.g., '?') then parse
    if schema.timestamp_col in out.columns:
        ts = out[schema.timestamp_col].astype(str).str.strip()
        # Remove BOM or stray leading markers (e.g., '\ufeff', '?')
        ts = ts.str.replace('\ufeff', '', regex=False)
        ts = ts.str.replace(r"^\?+", "", regex=True)
        # Common format observed: YYYY-MM-DD HH:MM:SS
        parsed = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        # Fallback if strict format fails
        parsed = parsed.fillna(pd.to_datetime(ts, errors="coerce"))
        out[schema.timestamp_col] = parsed

    # Convert numeric columns
    for col in [schema.longitude_col, schema.latitude_col, schema.temperature_col, schema.ppm_col]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Convert serial to string
    if schema.serial_col in out.columns:
        out[schema.serial_col] = out[schema.serial_col].astype("string")

    # Remove rows with invalid coordinates
    out = out.dropna(subset=[c for c in [schema.longitude_col, schema.latitude_col] if c in out.columns])
    return out


def attach_source_metadata(df: pd.DataFrame, csv_path: Path, project_name: str) -> pd.DataFrame:
    """Add source tracking columns."""
    out = df.copy()
    out["Source_File"] = csv_path.name
    out["Source_Path"] = str(csv_path)
    out["Project_Name"] = project_name
    out["Date_Folder"] = csv_path.parent.name
    return out


def enrich_with_kmz(parsed_csv: ParsedCSV, kmz_index: KMZIndex, max_distance: float = 100.0) -> ParsedCSV:
    """Enrich CSV data with KMZ polyline attributes."""
    df = parsed_csv.df.copy()

    kmz_columns = {
        "KMZ_Route_Name": "",
        "KMZ_Route_Desc": "",
        "KMZ_Diameter": "",
        "KMZ_Product": "",
        "KMZ_Class_Loca": "",
        "KMZ_BeginMeasu": "",
        "KMZ_EndMeasure": "",
        "KMZ_Distance_Meters": "",
    }
    for col, default in kmz_columns.items():
        if col not in df.columns:
            df[col] = default

    schema = CSVSchema()
    for idx, row in df.iterrows():
        lat = row.get(schema.latitude_col)
        lon = row.get(schema.longitude_col)
        if pd.isna(lat) or pd.isna(lon):
            continue
        res = kmz_index.lookup(float(lat), float(lon), max_distance_meters=max_distance)
        if not res:
            continue
        field_mapping = {
            "Route_Name": "KMZ_Route_Name",
            "Route_Desc": "KMZ_Route_Desc",
            "Diameter": "KMZ_Diameter",
            "Product": "KMZ_Product",
            "Class_Loca": "KMZ_Class_Loca",
            "BeginMeasu": "KMZ_BeginMeasu",
            "EndMeasure": "KMZ_EndMeasure",
            "Distance_Meters": "KMZ_Distance_Meters",
        }
        for kmz_field, df_col in field_mapping.items():
            if kmz_field in res:
                df.at[idx, df_col] = res[kmz_field]

    return ParsedCSV(
        df=df,
        source_info=parsed_csv.source_info,
        project_name=parsed_csv.project_name,
        enriched=True,
    )


# Backwards-compatible stub kept for now (not used in the main pipeline)
class Schema:
    def __init__(self, columns_to_extract: list[str]):
        self.columns_to_extract = columns_to_extract


def parse_csv(path: Path, schema: Schema, header_renames: Optional[Dict[str, str]] = None) -> ParsedCSV:
    df = pd.read_csv(path)
    if header_renames:
        df = df.rename(columns=header_renames)
    cols = [c for c in schema.columns_to_extract if c in df.columns] if schema.columns_to_extract else df.columns.tolist()
    df = df.loc[:, cols]
    src = infer_source_info(path)
    return ParsedCSV(df=df, source_info=src, project_name="", enriched=False)
