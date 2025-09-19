from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .csv_schema import CSVSchema


# Final output column order to match the sample workbook
OUTPUT_COLUMNS: List[str] = [
    # KMZ fields first (gold)
    "PLID",          # from KMZ Route_Name
    "BeginMeasu",    # from KMZ
    "EndMeasure",    # from KMZ
    "Class_Loca",    # from KMZ
    "Diameter",      # from KMZ
    "Product",       # from KMZ
    "Route_Desc",    # from KMZ
    # CSV fields after (blue)
    "Date",          # from Time Stamp (date only)
    "PPM",           # from CSV
    "Longitude",     # from CSV
    "Latitude",      # from CSV
    "View in Google Earth",  # hyperlink to local temp KML
    "TEMP (F)",      # derived from Temperature (C)
    "Serial No.",    # from CSV
    "Source_File",   # from metadata
]

# Optional trailing diagnostics to append (exact order)
TRAILING_EXTRAS: List[str] = [
    "KMZ_Match_Method",
    "Nearest_Distance_Meters",
]

# Grouping of headers for styling
CSV_HEADERS = {"Date", "PPM", "Longitude", "Latitude", "TEMP (F)", "Serial No.", "Source_File"}
KMZ_HEADERS = {"PLID", "BeginMeasu", "EndMeasure", "Route_Desc", "Class_Loca", "Diameter", "Product"}


def to_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Project a cleaned+enriched dataframe to the final output schema.

    - Date: from 'Time Stamp' (date only)
    - TEMP (F): from 'Temperature' in C => F
    - PLID: from 'KMZ_Route_Name'
    - Other KMZ fields copied as-is if present
    - Missing KMZ fields are left blank
    - Append trailing diagnostics at the end if present
    """
    schema = CSVSchema()
    out = pd.DataFrame()

    # Date as Python date object (writer will set mm/dd/yyyy format)
    if schema.timestamp_col in df.columns:
        out["Date"] = pd.to_datetime(df[schema.timestamp_col], errors="coerce").dt.date
    else:
        out["Date"] = pd.NaT

    # PLID from KMZ_Route_Name (numeric for Excel)
    if "KMZ_Route_Name" in df.columns:
        out["PLID"] = pd.to_numeric(df["KMZ_Route_Name"], errors="coerce")
    else:
        out["PLID"] = pd.NA

    # PPM
    if schema.ppm_col in df.columns:
        out["PPM"] = pd.to_numeric(df[schema.ppm_col], errors="coerce")
    else:
        out["PPM"] = pd.NA

    # KMZ attributes
    for src, dst in [
        ("KMZ_BeginMeasu", "BeginMeasu"),
        ("KMZ_EndMeasure", "EndMeasure"),
        ("KMZ_Route_Desc", "Route_Desc"),
        ("KMZ_Class_Loca", "Class_Loca"),
        ("KMZ_Diameter", "Diameter"),
        ("KMZ_Product", "Product"),
    ]:
        if src in df.columns and dst in {"BeginMeasu", "EndMeasure", "Class_Loca", "Diameter"}:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            out[dst] = df[src] if src in df.columns else ""

    # Coordinates
    out["Longitude"] = pd.to_numeric(df.get(schema.longitude_col, pd.Series(dtype=float)), errors="coerce")
    out["Latitude"] = pd.to_numeric(df.get(schema.latitude_col, pd.Series(dtype=float)), errors="coerce")

    # Temperature conversion C->F
    if schema.temperature_col in df.columns:
        c = pd.to_numeric(df[schema.temperature_col], errors="coerce")
        out["TEMP (F)"] = (c * 9.0 / 5.0) + 32.0
    else:
        out["TEMP (F)"] = pd.NA

    # Serial + source (create empty series if missing)
    if schema.serial_col in df.columns:
        out["Serial No."] = df[schema.serial_col].astype("string")
    else:
        out["Serial No."] = pd.Series([""] * len(out), index=out.index, dtype="string")
    if "Source_File" in df.columns:
        out["Source_File"] = df["Source_File"].astype("string")
    else:
        out["Source_File"] = pd.Series([""] * len(out), index=out.index, dtype="string")

    # Append trailing diagnostics if present in input
    appended_cols: List[str] = []
    for extra in TRAILING_EXTRAS:
        if extra in df.columns:
            # Pass through, ensuring numeric formatting for distance later
            out[extra] = df[extra]
            appended_cols.append(extra)

    # Reorder: base columns followed by appended diagnostics (if any)
    return out.reindex(columns=OUTPUT_COLUMNS + appended_cols)
