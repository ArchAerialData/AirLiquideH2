from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class SourceInfo:
    """Describes where a CSV came from (for grouping/traceability)."""
    date_folder: Optional[str]
    flight_folder: Optional[str]
    filename: str
    filepath: Path


def infer_source_info(csv_path: Path) -> SourceInfo:
    """Attempt to infer date/flight from the folder structure.
    e.g.: .../__NoHitsFlights/2025_09_02/Flight1/AL-Flight 1.CSV
    """
    parts = csv_path.parts
    date_folder = None
    flight_folder = None

    # naive pass: look for patterns like YYYY_MM_DD and 'Flight*'
    for p in parts:
        if (
            len(p) == 10
            and p[4] == "_"
            and p[7] == "_"
            and p[:4].isdigit()
            and p[5:7].isdigit()
            and p[8:10].isdigit()
        ):
            date_folder = p
        if p.lower().startswith("flight"):
            flight_folder = p

    return SourceInfo(
        date_folder=date_folder,
        flight_folder=flight_folder,
        filename=csv_path.name,
        filepath=csv_path.resolve(),
    )


"""
CSV schema and header utilities
--------------------------------
Defines canonical column names, mappings, and simple header fixes
for the Air Liquide H2 CSVs. Kept feature-modular in this module.
"""

# Map raw CSV zero-based column positions -> canonical headers
RAW_CSV_COLUMN_MAPPING: dict[int, str] = {
    0: "Time Stamp",      # Column A
    2: "Longitude",       # Column C
    3: "Latitude",        # Column D
    4: "Temperature",     # Column E (normalize odd raw header to a safe name)
    # Note: "PPM" can drift across files; index 11 is the most common,
    # but downstream code will also attempt pattern-based inference.
    11: "PPM",            # Column L (typical)
    13: "Serial No.",     # Column N
}


@dataclass(frozen=True)
class CSVSchema:
    # Core extracted columns
    timestamp_col: str = "Time Stamp"
    longitude_col: str = "Longitude"
    latitude_col: str = "Latitude"
    temperature_col: str = "Temperature"
    ppm_col: str = "PPM"
    serial_col: str = "Serial No."

    # KMZ enrichment columns
    kmz_route_name: str = "KMZ_Route_Name"
    kmz_route_desc: str = "KMZ_Route_Desc"
    kmz_diameter: str = "KMZ_Diameter"
    kmz_product: str = "KMZ_Product"
    kmz_class_loca: str = "KMZ_Class_Loca"
    kmz_begin_measu: str = "KMZ_BeginMeasu"
    kmz_end_measure: str = "KMZ_EndMeasure"
    kmz_distance_meters: str = "KMZ_Distance_Meters"


# Desired pandas dtypes for the core extracted columns
COLUMN_DTYPES: dict[str, str] = {
    "Time Stamp": "datetime64[ns]",
    "Longitude": "float64",
    "Latitude": "float64",
    "Temperature": "float64",
    "PPM": "float64",
    "Serial No.": "string",
}


def fix_csv_headers(raw_headers: pd.Series) -> pd.Series:
    """Normalize header text but do not coerce aliases to PPM.

    We now expect unedited raw CSVs where hydrogen readings live under a
    header such as "H2 %" (e.g., cell L3). We intentionally keep that label
    intact and derive a numeric "PPM" column later as `H2 % * 10000`.

    Older, pilot-edited files that already include a real "PPM" column will
    continue to work because we won't overwrite or rename anything here.
    """
    corrected = raw_headers.copy()
    return corrected.astype(str).str.replace("\ufeff", "", regex=False).str.strip()


# Backwards-compat lightweight schema container still used in early skeleton
class Schema:
    def __init__(self, columns_to_extract: List[str]):
        self.columns_to_extract = columns_to_extract


def load_schema(columns_to_extract: List[str]) -> Schema:
    # Later: accept renames, type conversions, derived fields, etc.
    return Schema(columns_to_extract=columns_to_extract)

