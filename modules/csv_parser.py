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
    # Preferred: unedited raw format => ignore first two rows, header is row 3
    # If that doesn't look right, fall back to the older robust loader.
    def _extract_project_name(path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
                _ = f.readline()  # created-by line
                line2 = f.readline().strip()
                return line2 if line2 else path.stem
        except Exception:
            return path.stem

    df = None  # type: ignore[assignment]
    project_name = csv_path.stem
    try:
        df_try = pd.read_csv(csv_path, header=None, encoding="utf-8-sig", skiprows=2)
        # Heuristic: confirm the first remaining row contains the usual headers
        if not df_try.empty:
            tokens = set(df_try.iloc[0].astype(str).str.strip())
            if ("Longitude" in tokens and "Latitude" in tokens) or ("Time Stamp" in tokens or "Time Stam" in tokens):
                df = df_try
                project_name = _extract_project_name(csv_path)
    except Exception:
        df = None

    if df is None:
        # Robust CSV loading: try utf-8-sig with C engine; fallback to python engine; last resort: latin-1
        try:
            df = pd.read_csv(csv_path, header=None, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(csv_path, header=None, encoding="utf-8-sig", engine="python")
            except Exception:
                df = pd.read_csv(csv_path, header=None, encoding="latin-1", engine="python", on_bad_lines="skip")
        # Try to extract project name from row 1 (0-based index), else fallback to file stem
        project_name = str(df.iloc[1, 0]) if len(df) > 1 else csv_path.stem

    # Detect a header row within the first few lines; fallback to headerless layout
    header_row_idx: Optional[int] = None
    expected_tokens = {"Time Stamp", "Longitude", "Latitude", "Serial No.", "PPM"}
    probe_rows = min(6, len(df))
    for r in range(probe_rows):
        row_vals = df.iloc[r].astype(str).str.strip()
        if any(tok in set(row_vals) for tok in expected_tokens) or (
            ("Longitude" in set(row_vals)) and ("Latitude" in set(row_vals))
        ):
            header_row_idx = r
            break

    schema = CSVSchema()
    if header_row_idx is not None:
        # Use detected header row
        raw_headers = df.iloc[header_row_idx].copy()
        headers = fix_csv_headers(raw_headers)

        data_df = df.iloc[header_row_idx + 1 :].copy()
        data_df.columns = headers

        # New behavior: if the raw file provides an "H2 %" column and no
        # explicit PPM column, derive PPM as (H2 % * 10000).
        def _norm_label(s: str) -> str:
            s = str(s).replace("\ufeff", "").strip().lower()
            return "".join(ch for ch in s if ch.isalnum() or ch.isspace())

        labels_normalized = { _norm_label(c): c for c in data_df.columns }
        has_ppm = any(_norm_label(c) == "ppm" for c in data_df.columns)
        # Look for common ways the instrument labels the percent column
        h2_percent_col = None
        for cand in ["h2 %", "h2%", "h2 percent", "h2 pct", "h2percentage"]:
            key = _norm_label(cand)
            if key in labels_normalized:
                h2_percent_col = labels_normalized[key]
                break
        if not has_ppm and h2_percent_col is not None:
            try:
                ppm_values = pd.to_numeric(data_df[h2_percent_col], errors="coerce") * 10000.0
                data_df["PPM"] = ppm_values
            except Exception:
                # If conversion fails, leave as-is and let downstream inference try
                pass

        # Name-based mapping with normalization and fuzzy aliases
        def norm(s: str) -> str:
            s = str(s).replace("\ufeff", "").strip().lower()
            out = []
            for ch in s:
                if ch.isalnum() or ch.isspace():
                    out.append(ch)
            return "".join(out)

        # Rebuild availability map after potential PPM derivation
        available = {norm(c): c for c in data_df.columns}

        def pick(*candidates: str) -> Optional[str]:
            for cand in candidates:
                n = norm(cand)
                if n in available:
                    return available[n]
            # substring fallback (e.g., "temp (c)")
            for key, orig in available.items():
                for cand in candidates:
                    n = norm(cand)
                    if n and n in key:
                        return orig
            return None

        col_time = pick("time stamp", "timestamp", "time", "date time", "date")
        col_lon = pick("longitude", "lon", "long")
        col_lat = pick("latitude", "lat")
        col_temp = pick("temperature", "temp")
        col_ppm = pick("ppm")
        col_serial = pick("serial no.", "serial", "serial number", "serialno")

        chosen: Dict[str, Optional[str]] = {
            "Time Stamp": col_time,
            "Longitude": col_lon,
            "Latitude": col_lat,
            "Temperature": col_temp,
            "PPM": col_ppm,
            "Serial No.": col_serial,
        }

        # If a column labelled as PPM exists but is effectively empty (e.g.,
        # header repair guessed the wrong place), treat it as missing so that
        # pattern-based inference can step in.
        if chosen.get("PPM") in data_df.columns:
            try:
                _ppm_vals = pd.to_numeric(data_df[chosen["PPM"]], errors="coerce")  # type: ignore[index]
                if _ppm_vals.notna().mean() < 0.2:
                    chosen["PPM"] = None
            except Exception:
                chosen["PPM"] = None

        # If any required column is missing, first try pattern-based inference
        # over the observed columns. If still missing, fall back to positional mapping.
        selection: Dict[str, str] = {}
        pattern_needed = {k for k, v in chosen.items() if v is None}

        def _to_numeric(series) -> pd.Series:
            # Handle duplicate column names where data_df[c] can return a DataFrame
            try:
                import pandas as _pd
                if isinstance(series, _pd.DataFrame):
                    series = series.iloc[:, 0]
            except Exception:
                pass
            return pd.to_numeric(series, errors="coerce")

        def _avg_dec_places(series: pd.Series) -> float:
            def decs(x: object) -> int:
                try:
                    s = str(x)
                    if "." in s:
                        return len(s.split(".")[-1].rstrip("0"))
                except Exception:
                    pass
                return 0
            vals = series.dropna().head(1000)
            if len(vals) == 0:
                return 0.0
            return sum(decs(v) for v in vals) / float(len(vals))

        def _ratio(mask: pd.Series) -> float:
            denom = mask.shape[0] if mask is not None else 0
            if denom == 0:
                return 0.0
            return float(mask.sum()) / float(denom)

        numeric_cols = {c for c in data_df.columns if _to_numeric(data_df[c]).notna().sum() > 0}

        inferred: Dict[str, str] = {}

        # Timestamp by parse rate
        if "Time Stamp" in pattern_needed:
            best_col = None
            best_score = 0.0
            for c in data_df.columns:
                s = data_df[c].astype(str).str.strip()
                parsed = pd.to_datetime(s, errors="coerce")
                score = parsed.notna().mean()
                if score > 0.8 and score >= best_score:
                    best_score, best_col = score, c
            if best_col:
                inferred["Time Stamp"] = best_col

        # Latitude / Longitude using ranges + decimal precision
        def _is_lat(series: pd.Series) -> bool:
            x = _to_numeric(series)
            return _ratio(x.between(-90, 90)) > 0.95 and _avg_dec_places(series) >= 5.0

        def _is_lon(series: pd.Series) -> bool:
            x = _to_numeric(series)
            return _ratio(x.between(-180, 180)) > 0.95 and _avg_dec_places(series) >= 5.0

        if any(k in pattern_needed for k in ("Longitude", "Latitude")):
            lat_cands, lon_cands = [], []
            for c in numeric_cols:
                s = data_df[c]
                if _is_lat(s):
                    lat_cands.append((c, _avg_dec_places(s)))
                if _is_lon(s):
                    lon_cands.append((c, _avg_dec_places(s)))
            # Prefer highest decimal precision
            lat_cands.sort(key=lambda t: t[1], reverse=True)
            lon_cands.sort(key=lambda t: t[1], reverse=True)
            if "Latitude" in pattern_needed and lat_cands:
                inferred["Latitude"] = lat_cands[0][0]
            if "Longitude" in pattern_needed and lon_cands:
                # Avoid selecting the same column for both
                for cand, _ in lon_cands:
                    if cand != inferred.get("Latitude"):
                        inferred["Longitude"] = cand
                        break

        # Temperature within a plausible C range
        if "Temperature" in pattern_needed:
            best = None
            for c in numeric_cols:
                x = _to_numeric(data_df[c])
                if _ratio(x.between(-50, 80)) > 0.9:
                    best = c if best is None else best
            if best:
                inferred["Temperature"] = best

        # PPM: usually very small numbers with a lot of decimals; choose the
        # numeric column (not already taken) with the highest combination of
        # proportion <1 and <10, with decent decimal precision.
        if "PPM" in pattern_needed:
            best_col, best_score = None, -1.0
            taken = set(inferred.values())
            for c in numeric_cols:
                if c in taken:
                    continue
                x = _to_numeric(data_df[c])
                if x.notna().sum() == 0:
                    continue
                p_lt1 = _ratio((x.abs() < 1.0))
                p_lt10 = _ratio((x.abs() < 10.0))
                decs = _avg_dec_places(data_df[c])
                score = p_lt1 * 2.0 + p_lt10 + (decs / 10.0)
                # Hard exclude very large ranges typical of pressure/humidity
                if x.quantile(0.95) > 1000:
                    continue
                if score > best_score:
                    best_score, best_col = score, c
            if best_col:
                inferred["PPM"] = best_col

        # Serial No.: mostly identical alphanumeric token across rows
        if "Serial No." in pattern_needed:
            best_col, best_score = None, 0.0
            for c in data_df.columns:
                s = data_df[c].astype(str).str.strip()
                s = s[s != ""]
                if s.empty:
                    continue
                mode = s.mode(dropna=True)
                if mode.empty:
                    continue
                top = mode.iloc[0]
                share = (s == top).mean()
                # Damp score if values look like datetimes
                dt_share = pd.to_datetime(s, errors="coerce").notna().mean()
                score = max(0.0, share - 0.2 * dt_share)
                if score > 0.6 and score >= best_score:
                    best_score, best_col = score, c
            if best_col:
                inferred["Serial No."] = best_col

        # Merge in any inferred columns where name-picking failed
        for canon in ["Time Stamp", "Longitude", "Latitude", "Temperature", "PPM", "Serial No."]:
            actual = chosen.get(canon)
            if actual is not None and actual in data_df.columns:
                selection[canon] = actual
            elif canon in inferred:
                selection[canon] = inferred[canon]
            else:
                # fallback by position mapping if within bounds
                pos = None
                for p, name in RAW_CSV_COLUMN_MAPPING.items():
                    if name == canon and p < len(headers):
                        pos = p
                        break
                if pos is not None:
                    selection[canon] = headers.iloc[pos]

        selected_actual = [selection[k] for k in [
            "Time Stamp","Longitude","Latitude","Temperature","PPM","Serial No."
        ] if k in selection]
        cleaned_df = data_df[selected_actual].copy()
        cleaned_df = cleaned_df.rename(columns={v: k for k, v in selection.items()})
    else:
        # Headerless CSV: select by fixed positions and assign canonical names directly
        positions = [pos for pos in RAW_CSV_COLUMN_MAPPING.keys() if pos < df.shape[1]]
        if not positions:
            raise ValueError(f"Invalid CSV structure in {csv_path}")

        # Very final fallback: find the first row that has values in all selected positions
        def _present(v: object) -> bool:
            if pd.isna(v):
                return False
            s = str(v).strip()
            return s != "" and s.lower() != "nan"

        start_idx = None
        for r in range(len(df)):
            try:
                if all(_present(df.iat[r, p]) for p in positions):
                    start_idx = r
                    break
            except Exception:
                continue

        src = df.iloc[start_idx:, :] if start_idx is not None else df
        cleaned_df = src.iloc[:, positions].copy()
        rename_map = {i: RAW_CSV_COLUMN_MAPPING[i] for i in positions}
        cleaned_df.columns = [rename_map[i] for i in positions]

        # As an extra safeguard for headerless files, try to augment with
        # pattern-based inference if any of the key columns are still missing.
        missing = [c for c in ["Time Stamp", "Longitude", "Latitude", "Temperature", "PPM", "Serial No."] if c not in cleaned_df.columns]
        if missing:
            work = cleaned_df.copy()
            # Re-run a subset of the inference on the raw df to discover
            # additional candidates without headers.
        def _to_numeric(series) -> pd.Series:
            try:
                import pandas as _pd
                if isinstance(series, _pd.DataFrame):
                    series = series.iloc[:, 0]
            except Exception:
                pass
            return pd.to_numeric(series, errors="coerce")
            def _ratio(mask: pd.Series) -> float:
                denom = mask.shape[0] if mask is not None else 0
                if denom == 0:
                    return 0.0
                return float(mask.sum()) / float(denom)
            # Timestamp
            if "Time Stamp" in missing and df.shape[1] > 0:
                best, score = None, 0.0
                for i in range(min(df.shape[1], 20)):
                    s = df.iloc[:, i].astype(str).str.strip()
                    parsed = pd.to_datetime(s, errors="coerce")
                    sc = parsed.notna().mean()
                    if sc > 0.8 and sc >= score:
                        best, score = i, sc
                if best is not None:
                    work["Time Stamp"] = pd.to_datetime(df.iloc[:, best], errors="coerce")
            # Lat/Lon by range
            if any(c in missing for c in ("Longitude", "Latitude")):
                lat_best, lon_best = None, None
                for i in range(min(df.shape[1], 20)):
                    x = _to_numeric(df.iloc[:, i])
                    if _ratio(x.between(-90, 90)) > 0.95:
                        lat_best = i if lat_best is None else lat_best
                    if _ratio(x.between(-180, 180)) > 0.95:
                        lon_best = i if lon_best is None else lon_best
                if lat_best is not None and "Latitude" in missing:
                    work["Latitude"] = _to_numeric(df.iloc[:, lat_best])
                if lon_best is not None and "Longitude" in missing and lon_best != lat_best:
                    work["Longitude"] = _to_numeric(df.iloc[:, lon_best])
            # PPM small values
            if "PPM" in missing:
                best, best_score = None, -1.0
                for i in range(min(df.shape[1], 20)):
                    if i in positions:
                        continue
                    x = _to_numeric(df.iloc[:, i])
                    p_lt1 = _ratio((x.abs() < 1.0))
                    p_lt10 = _ratio((x.abs() < 10.0))
                    q95 = x.quantile(0.95)
                    score = p_lt1 * 2.0 + p_lt10
                    if q95 <= 1000 and score > best_score:
                        best, best_score = i, score
                if best is not None:
                    work["PPM"] = _to_numeric(df.iloc[:, best])
            cleaned_df = work

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

    # Remove rows with invalid coordinates (only if at least one coord is present)
    coord_subset = [c for c in [schema.longitude_col, schema.latitude_col] if c in out.columns]
    if coord_subset:
        out = out.dropna(subset=coord_subset)
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
        }
        for kmz_field, df_col in field_mapping.items():
            if kmz_field in res:
                df.at[idx, df_col] = res[kmz_field]

        # Append diagnostics at end if present
        if 'Match_Method' in res:
            df.at[idx, 'KMZ_Match_Method'] = res['Match_Method']
        if 'Nearest_Distance_Meters' in res:
            df.at[idx, 'Nearest_Distance_Meters'] = res['Nearest_Distance_Meters']

    # Exclude rows outside the upper bound distance (drop rows without a match or with distance > max_distance)
    if 'Nearest_Distance_Meters' in df.columns:
        try:
            mask = pd.to_numeric(df['Nearest_Distance_Meters'], errors='coerce') <= float(max_distance)
            df = df.loc[mask.fillna(False)].copy()
        except Exception:
            pass

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
