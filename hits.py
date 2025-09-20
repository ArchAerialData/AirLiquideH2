from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from modules.io_utils import resolve_input_paths, derive_output_path
from modules.file_discovery import find_csv_files
from modules.logging_utils import get_logger
from modules.output_schema import to_output_df, OUTPUT_COLUMNS, TRAILING_EXTRAS


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
        ("numpy", "numpy"),
        ("PIL", "pillow"),
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

    # Lazy imports after environment check
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
    from modules.config import (
        SeparatorStyle,
        DEFAULT_KMZ_PATH,
        DEFAULT_KMZ_DISTANCE_THRESHOLD,
        DEFAULT_PIN_SPACING_METERS,
        load_default_config,
    )

    HITS_THRESHOLD = 5.0  # inclusive (can be overridden by config)

    input_dirs = resolve_input_paths(argv)
    if not input_dirs:
        logger.warning("No input directories provided. Exiting.")
        return

    # Load optional YAML config (config.yaml / config.yml)
    cfg = load_default_config()

    logger.info("Input directories:")
    for p in input_dirs:
        logger.info(f"  - {p}")

    csv_files = find_csv_files(input_dirs)
    # Ignore previously generated exports to avoid double-processing
    csv_files = [p for p in csv_files if not p.name.lower().startswith("combined_extracted")]
    if not csv_files:
        logger.warning("No CSV files found under the provided directories.")
        return

    logger.info(f"Discovered {len(csv_files)} CSV files (recursive).")

    # Load KMZ index when present
    kmz_path = Path(cfg.kmz_path or DEFAULT_KMZ_PATH)
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
    total_before, total_after = 0, 0

    for csv_path in csv_files:
        try:
            logger.info(f"Processing: {csv_path}")

            # Parse + metadata
            cleaned_df, project_name = parse_raw_csv(csv_path)
            source_info = infer_source_info(csv_path)
            cleaned_df = attach_source_metadata(cleaned_df, csv_path, project_name)

            parsed_csv = ParsedCSV(
                df=cleaned_df,
                source_info=source_info,
                project_name=project_name,
            )

            # Optional KMZ enrichment
            if kmz_index is not None:
                max_dist = float(getattr(cfg, "kmz_distance_threshold", DEFAULT_KMZ_DISTANCE_THRESHOLD) or DEFAULT_KMZ_DISTANCE_THRESHOLD)
                parsed_csv = enrich_with_kmz(parsed_csv, kmz_index, max_distance=max_dist)

            # Project to output schema, then filter for hits
            projected_df = to_output_df(parsed_csv.df)
            total_before += len(projected_df)
            ppm = pd.to_numeric(projected_df.get("PPM"), errors="coerce")
            hits_thresh = float(getattr(cfg, "hits_threshold", HITS_THRESHOLD) or HITS_THRESHOLD)
            hits_df = projected_df.loc[ppm >= hits_thresh].copy()
            kept = len(hits_df)
            total_after += kept
            logger.info(f"  Hits kept: {kept} of {len(projected_df)} (PPM >= {hits_thresh})")

            parsed_csvs.append(ParsedCSV(
                df=hits_df,
                source_info=parsed_csv.source_info,
                project_name=parsed_csv.project_name,
                enriched=parsed_csv.enriched,
            ))

        except Exception as e:
            logger.error(f"Failed to process {csv_path}: {e}")

    if not parsed_csvs:
        logger.error("No CSVs were successfully processed.")
        return

    logger.info(f"Combining HITS data... (total hits kept {total_after} of {total_before})")
    combined_df, _ = combine(parsed_csvs)

    out_path = derive_output_path(input_dirs, preferred_name="Excel-Reports/Combined_Extracted_Hits.xlsx")
    logger.info(f"Writing output to: {out_path}")

    separator_style = SeparatorStyle()
    try:
        write_all_and_date_sheets(combined_df, out_path, separator_style)
    except Exception as e:
        logger.warning(f"Failed to write XLSX (continuing to CSV): {e}")

    # Also emit a CSV for Google Earth import (ALL rows only)
    try:
        headers = OUTPUT_COLUMNS + [c for c in TRAILING_EXTRAS if c in combined_df.columns]
        csv_df = combined_df.reindex(columns=headers)
        csv_path = derive_output_path(input_dirs, preferred_name="Excel-Reports/Combined_Extracted_Hits.csv")
        logger.info(f"Writing CSV (GE import) to: {csv_path}")
        # Use UTF-8 with BOM for Excel friendliness; empty string for missing values
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig", na_rep="")
    except Exception as e:
        logger.warning(f"Failed to write CSV export: {e}")

    # Build KMZs: Only two outputs requested — All bundle (heatmap + placemarks) and Placemarks-only
    try:
        _write_view_placemarks_kmz(
            Path("temp_kml"),
            derive_output_path(input_dirs, preferred_name="KMZ/Combined_Extracted_Hits_Placemarks.kmz"),
        )
        _write_combined_heatmap_and_placemarks(
            combined_df,
            Path("temp_kml"),
            derive_output_path(input_dirs, preferred_name="KMZ/Combined_Extracted_Hits_All.kmz"),
            spacing_m=float(getattr(cfg, "pin_spacing_meters", DEFAULT_PIN_SPACING_METERS) or DEFAULT_PIN_SPACING_METERS),
        )
    except Exception as e:
        logger.warning(f"Failed to write KMZ exports: {e}")

    logger.info("Hits-only processing complete!")


# ------------------- KMZ helpers -------------------

def _ppm_style_id(ppm: float) -> str:
    # Bucket by ranges for colorized icons
    if ppm >= 7.0:
        return "ppm_red"
    elif ppm >= 5.3:
        return "ppm_orange"
    else:
        return "ppm_yellow"


def _point_styles_block() -> str:
    # Use hosted Google icon circles; LabelStyle scaled for legible numbers
    styles = [
        ("ppm_yellow", "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png"),
        ("ppm_orange", "http://maps.google.com/mapfiles/kml/paddle/orange-circle.png"),
        ("ppm_red", "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"),
    ]
    blocks = []
    for sid, href in styles:
        blocks.append(f"""
  <Style id="{sid}">
    <IconStyle>
      <scale>0.9</scale>
      <Icon><href>{href}</href></Icon>
    </IconStyle>
    <LabelStyle>
      <scale>0.9</scale>
    </LabelStyle>
  </Style>
""")
    return "\n".join(blocks)


def _kmz_write(zip_path: Path, kml_text: str, assets: dict[str, bytes] | None = None) -> None:
    from zipfile import ZipFile, ZIP_DEFLATED
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml_text)
        if assets:
            for rel, data in assets.items():
                z.writestr(rel, data)


def _write_view_placemarks_kmz(temp_dir: Path, kmz_path: Path) -> None:
    """Zip all per-row KMLs in temp_kml into a KMZ alongside a doc.kml that links each.

    This preserves the individual, styled info balloons created for each row by
    the spreadsheet writer, and packages them for easy loading in Google Earth.
    """
    try:
        if not temp_dir.exists():
            return
        kml_files = sorted([p for p in temp_dir.glob("*.kml") if p.is_file()])
        if not kml_files:
            return
        # Construct a doc.kml that links all individual KMLs by relative path
        links = []
        for p in kml_files:
            # Display name: "5.49 PPM" when filename looks like "PPM_5.49[_...] .kml"
            stem = p.stem
            name = stem
            if stem.upper().startswith("PPM_"):
                rest = stem[4:]
                # Allow suffix like _r12; take the first token as the value
                token = rest.split("_", 1)[0]
                try:
                    val = float(token)
                    name = f"{val:.2f} PPM"
                except Exception:
                    name = token + " PPM"
            href = p.name  # relative within KMZ
            links.append(
                f"""
  <NetworkLink>
    <name>{name}</name>
    <Link><href>{href}</href></Link>
  </NetworkLink>"""
            )
        doc = f"""
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>View Placemark Files</name>
  <Folder>
    <name>Placemarks</name>
{''.join(links)}
  </Folder>
</Document>
</kml>
"""
        # Read all assets as bytes to embed into KMZ
        assets: dict[str, bytes] = {}
        for p in kml_files:
            try:
                assets[p.name] = p.read_bytes()
            except Exception:
                # Skip unreadable files but keep going
                pass
        _kmz_write(Path(kmz_path), doc, assets=assets)
    except Exception:
        # Do not crash the whole run on packaging errors
        pass


def _heatmap_snippet_and_assets(df: pd.DataFrame):
    """Return (overlay_snippet_xml, assets_dict) or (None, None) if no heatmap can be built."""
    import io
    import numpy as np
    from PIL import Image
    import pandas as pd

    if df.empty:
        return None, None

    lat = pd.to_numeric(df.get("Latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce")
    ppm = pd.to_numeric(df.get("PPM"), errors="coerce")
    mask = lat.notna() & lon.notna() & ppm.notna()
    sub = df.loc[mask, ["Latitude", "Longitude", "PPM"]].copy()
    if sub.empty:
        return None, None

    lats = sub["Latitude"].to_numpy(float)
    lons = sub["Longitude"].to_numpy(float)
    vals = sub["PPM"].to_numpy(float)

    # Bounds
    pad = 0.0003
    south, north = float(lats.min() - pad), float(lats.max() + pad)
    west, east = float(lons.min() - pad), float(lons.max() + pad)

    # Grid
    W = 512
    H = 512
    xs = np.linspace(west, east, W)
    ys = np.linspace(south, north, H)
    X, Y = np.meshgrid(xs, ys)

    # IDW
    lat0 = float(sub["Latitude"].mean())
    coslat = np.cos(np.deg2rad(lat0))
    def _dist(a_lon, a_lat):
        dx = (X - a_lon) * (111320.0 * coslat)
        dy = (Y - a_lat) * 110540.0
        return np.hypot(dx, dy)
    power = 2.0
    eps = 1.0
    num = np.zeros_like(X, dtype=float)
    den = np.zeros_like(X, dtype=float)
    for la, lo, v in zip(lats, lons, vals):
        d = _dist(lo, la)
        w = 1.0 / np.maximum(d, eps) ** power
        num += w * v
        den += w
    Z = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    single_value_mode = not (np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) > 1e-9)

    img = Image.new("RGBA", (W, H))
    px = img.load()

    if single_value_mode:
        sigma_m = 35.0
        def _dists(lon0, lat0_):
            dx = (X - lon0) * (111320.0 * coslat)
            dy = (Y - lat0_) * 110540.0
            return np.hypot(dx, dy)
        K = np.zeros_like(X)
        two_sig2 = 2.0 * (sigma_m ** 2)
        for la, lo in zip(lats, lons):
            d = _dists(lo, la)
            K += np.exp(-(d * d) / two_sig2)
        K = K / np.max(K)
        hot_stops = [
            (0.0, (255, 127, 0)),
            (0.5, (220, 0, 0)),
            (1.0, (120, 0, 140)),
        ]
        for j in range(H):
            for i in range(W):
                t = float(np.clip(K[j, i], 0.0, 1.0))
                r, g, b = _lerp_color(hot_stops, t)
                a = _alpha_from_t(t)
                px[i, H - 1 - j] = (r, g, b, a)
    else:
        stops = [
            (0.00, (0, 200, 0)),
            (0.33, (255, 255, 0)),
            (0.55, (255, 127, 0)),
            (0.80, (220, 0, 0)),
            (1.00, (120, 0, 140)),
        ]
        span = vmax - vmin
        for j in range(H):
            for i in range(W):
                t = 0.0 if span <= 0 else float(np.clip((Z[j, i] - vmin) / span, 0.0, 1.0))
                r, g, b = _lerp_color(stops, t)
                a = _alpha_from_t(t)
                px[i, H - 1 - j] = (r, g, b, a)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    snippet = f"""
  <GroundOverlay>
    <name>PPM Heatmap</name>
    <Icon><href>overlay.png</href></Icon>
    <LatLonBox>
      <north>{north}</north>
      <south>{south}</south>
      <east>{east}</east>
      <west>{west}</west>
    </LatLonBox>
  </GroundOverlay>
"""
    return snippet, {"overlay.png": png_bytes}


def _select_spaced_kml_names(df: pd.DataFrame, spacing_m: float) -> set[str]:
    """Return the set of temp_kml filenames (relative names) to include using spacing.

    Spacing logic:
    - Pick the row with the highest PPM as the center. If ties, choose the one
      closest to the centroid of all valid points.
    - Project all points to meters using local equirectangular approximation.
    - Compute principal axis via PCA; project points to a 1D coordinate s along
      that axis relative to the chosen center.
    - Bin by k = round(s / spacing_m). For k=0, force-select the center row.
      For other bins, pick the candidate with the highest PPM (tie -> closest
      to bin center |s - k*spacing|).
    - Convert selected row indices to expected temp KML filenames generated by
      the Excel writer: PPM_{ppm:.2f}_r{row+1}.kml
    """
    import numpy as np
    import pandas as pd

    if df.empty:
        return set()
    lat = pd.to_numeric(df.get("Latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce")
    ppm = pd.to_numeric(df.get("PPM"), errors="coerce")
    mask = lat.notna() & lon.notna() & ppm.notna()
    if not mask.any():
        return set()

    idxs = np.nonzero(mask.to_numpy())[0]
    lats = lat.to_numpy(float)[idxs]
    lons = lon.to_numpy(float)[idxs]
    vals = ppm.to_numpy(float)[idxs]

    # Heatmap center approximation: centroid
    latc = float(np.mean(lats))
    lonc = float(np.mean(lons))
    coslat = float(np.cos(np.deg2rad(latc)))
    def to_xy(lat_arr, lon_arr):
        dx = (lon_arr - lonc) * (111320.0 * coslat)
        dy = (lat_arr - latc) * 110540.0
        return np.stack([dx, dy], axis=1)
    XY = to_xy(lats, lons)

    # Choose center row: highest PPM; tie by distance to centroid
    vmax = float(np.max(vals))
    tmask = (vals >= vmax - 1e-12)
    if np.count_nonzero(tmask) > 1:
        d2 = np.sum(XY[tmask] ** 2, axis=1)
        j = np.argmin(d2)
        center_idx_local = np.nonzero(tmask)[0][j]
    else:
        center_idx_local = int(np.argmax(vals))
    center_global = int(idxs[center_idx_local])
    center_xy = XY[center_idx_local]

    # PCA for axis
    A = XY - center_xy
    try:
        cov = A.T @ A
        w, V = np.linalg.eigh(cov)
        v = V[:, int(np.argmax(w))]
        v = v / (np.linalg.norm(v) + 1e-12)
    except Exception:
        v = np.array([0.0, 1.0])

    s = (A @ v)  # meters along axis, centered at highest PPM

    # Bin by rounding to multiples of spacing
    k = np.round(s / max(spacing_m, 1e-6)).astype(int)

    # Build selection per bin
    selected = {}
    for local_idx, kk in enumerate(k):
        gi = int(idxs[local_idx])
        if kk == 0 and gi != center_global:
            # Ensure center bin contains the chosen center
            continue
        key = int(kk)
        # Candidate ranking: highest PPM, then closest to bin center
        rank = (vals[local_idx], -abs(s[local_idx] - key * spacing_m))
        prev = selected.get(key)
        if prev is None or rank > prev[0]:
            selected[key] = ((rank), gi)

    # Force insert center
    selected[0] = ((float(vmax), 0.0), center_global)

    # Convert to filenames
    chosen = sorted({gi for (_, gi) in selected.values()})
    names: set[str] = set()
    for gi in chosen:
        try:
            pval = float(pd.to_numeric(df.iloc[gi].get("PPM"), errors="coerce"))
            label = f"{pval:.2f}"
        except Exception:
            label = str(df.iloc[gi].get("PPM"))
        names.add(f"PPM_{label}_r{gi+1}.kml")
    return names


def _write_combined_heatmap_and_placemarks(df: pd.DataFrame, temp_dir: Path, kmz_path: Path, spacing_m: float | None = None) -> None:
    try:
        overlay_snippet, assets = _heatmap_snippet_and_assets(df)
        kml_files = []
        if temp_dir.exists():
            kml_files = sorted([p for p in temp_dir.glob("*.kml") if p.is_file()])
        if not kml_files and overlay_snippet is None:
            return

        # Optionally filter placemarks using spacing
        if spacing_m is not None:
            try:
                allowed = _select_spaced_kml_names(df, spacing_m)
                kml_files = [p for p in kml_files if p.name in allowed]
            except Exception:
                pass

        links = []
        for p in kml_files:
            stem = p.stem
            name = stem
            if stem.upper().startswith("PPM_"):
                rest = stem[4:]
                token = rest.split("_", 1)[0]
                try:
                    val = float(token)
                    name = f"{val:.2f} PPM"
                except Exception:
                    name = token + " PPM"
            links.append(
                f"""
  <NetworkLink>
    <name>{name}</name>
    <Link><href>{p.name}</href></Link>
  </NetworkLink>"""
            )

        placemark_folder = ""
        if links:
            placemark_folder = f"""
  <Folder>
    <name>Placemarks</name>
{''.join(links)}
  </Folder>
"""
        overlay_block = overlay_snippet or ""
        doc = f"""
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Hits Bundle (Heatmap + Placemarks)</name>
{overlay_block}{placemark_folder}
</Document>
</kml>
"""
        bundle_assets: dict[str, bytes] = {}
        if assets:
            bundle_assets.update(assets)
        for p in kml_files:
            try:
                bundle_assets[p.name] = p.read_bytes()
            except Exception:
                pass
        _kmz_write(Path(kmz_path), doc, assets=bundle_assets)
    except Exception:
        pass


def _write_hits_kmz_points(df: pd.DataFrame, kmz_path: Path) -> None:
    # Expect df already in output schema
    import pandas as pd
    if df.empty:
        return
    lat = pd.to_numeric(df.get("Latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce")
    ppm = pd.to_numeric(df.get("PPM"), errors="coerce")
    mask = lat.notna() & lon.notna() & ppm.notna()
    sub = df.loc[mask].copy()
    placemarks = []
    for _, row in sub.iterrows():
        p = float(row["PPM"]) if pd.notna(row["PPM"]) else 0.0
        s_id = _ppm_style_id(p)
        name = f"{p:.2f}"
        descr = ""  # could reuse the HTML table here if desired
        placemarks.append(f"""
  <Placemark>
    <name>{name}</name>
    <styleUrl>#{s_id}</styleUrl>
    <Point><coordinates>{float(row['Longitude'])},{float(row['Latitude'])},0</coordinates></Point>
  </Placemark>""")
    kml = f"""
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Hits (PPM ≥ 5)</name>
{_point_styles_block()}
{''.join(placemarks)}
</Document>
</kml>
"""
    _kmz_write(Path(kmz_path), kml)


def _lerp_color(stops: list[tuple[float, tuple[int, int, int]]], t: float) -> tuple[int, int, int]:
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r = int(c0[0] + (c1[0] - c0[0]) * w)
            g = int(c0[1] + (c1[1] - c0[1]) * w)
            b = int(c0[2] + (c1[2] - c0[2]) * w)
            return (r, g, b)
    return stops[-1][1]


def _alpha_from_t(t: float) -> int:
    # Softer edges and more opaque cores
    import numpy as np
    return int(30 + 200 * (np.clip(t, 0.0, 1.0) ** 0.75))


def _write_hits_kmz_heatmap(df: pd.DataFrame, kmz_path: Path) -> None:
    import io
    import numpy as np
    from PIL import Image
    import pandas as pd

    if df.empty:
        return

    lat = pd.to_numeric(df.get("Latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("Longitude"), errors="coerce")
    ppm = pd.to_numeric(df.get("PPM"), errors="coerce")
    mask = lat.notna() & lon.notna() & ppm.notna()
    sub = df.loc[mask, ["Latitude", "Longitude", "PPM"]].copy()
    if sub.empty:
        return

    lats = sub["Latitude"].to_numpy(float)
    lons = sub["Longitude"].to_numpy(float)
    vals = sub["PPM"].to_numpy(float)

    # Grid bounds with small padding
    # Tight overlay bounds with small padding
    pad = 0.0003
    south, north = float(lats.min() - pad), float(lats.max() + pad)
    west, east = float(lons.min() - pad), float(lons.max() + pad)

    # Grid resolution
    W = 512
    H = 512
    xs = np.linspace(west, east, W)
    ys = np.linspace(south, north, H)
    X, Y = np.meshgrid(xs, ys)

    # IDW interpolation
    lat0 = float(sub["Latitude"].mean())
    coslat = np.cos(np.deg2rad(lat0))
    def _dist(a_lon, a_lat):
        dx = (X - a_lon) * (111320.0 * coslat)
        dy = (Y - a_lat) * 110540.0
        return np.hypot(dx, dy)
    power = 2.0
    eps = 1.0
    num = np.zeros_like(X, dtype=float)
    den = np.zeros_like(X, dtype=float)
    for la, lo, v in zip(lats, lons, vals):
        d = _dist(lo, la)
        w = 1.0 / np.maximum(d, eps) ** power
        num += w * v
        den += w
    Z = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    # Choose colormap depending on span. If there is no spread, build a
    # distance-based kernel intensity so cores show as purple and fade to red/orange.
    single_value_mode = not (np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) > 1e-9)

    # Create RGBA image
    img = Image.new("RGBA", (W, H))
    px = img.load()

    if single_value_mode:
        # Kernel density field
        sigma_m = 35.0  # ~35 meters radius
        lat0 = float(sub["Latitude"].mean())
        coslat = np.cos(np.deg2rad(lat0))
        def _dists(lon0, lat0_):
            dx = (X - lon0) * (111320.0 * coslat)
            dy = (Y - lat0_) * 110540.0
            return np.hypot(dx, dy)
        K = np.zeros_like(X)
        two_sig2 = 2.0 * (sigma_m ** 2)
        for la, lo in zip(lats, lons):
            d = _dists(lo, la)
            K += np.exp(-(d * d) / two_sig2)
        K = K / np.max(K)
        hot_stops = [
            (0.0, (255, 127, 0)),   # orange at edges
            (0.5, (220, 0, 0)),     # red mid
            (1.0, (120, 0, 140)),   # purple core
        ]
        for j in range(H):
            for i in range(W):
                t = float(np.clip(K[j, i], 0.0, 1.0))
                r, g, b = _lerp_color(hot_stops, t)
                a = _alpha_from_t(t)
                px[i, H - 1 - j] = (r, g, b, a)
    else:
        # Value-driven gradient across [vmin, vmax]: green->yellow->orange->red->purple
        stops = [
            (0.00, (0, 200, 0)),
            (0.33, (255, 255, 0)),
            (0.55, (255, 127, 0)),
            (0.80, (220, 0, 0)),
            (1.00, (120, 0, 140)),
        ]
        span = vmax - vmin
        for j in range(H):
            for i in range(W):
                t = 0.0 if span <= 0 else float(np.clip((Z[j, i] - vmin) / span, 0.0, 1.0))
                r, g, b = _lerp_color(stops, t)
                a = _alpha_from_t(t)
                px[i, H - 1 - j] = (r, g, b, a)

    # Save PNG into memory for KMZ
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    overlay_kml = f"""
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Hits Heatmap (PPM ≥ 5)</name>
  <GroundOverlay>
    <name>PPM Heatmap</name>
    <Icon><href>overlay.png</href></Icon>
    <LatLonBox>
      <north>{north}</north>
      <south>{south}</south>
      <east>{east}</east>
      <west>{west}</west>
    </LatLonBox>
  </GroundOverlay>
</Document>
</kml>
"""
    _kmz_write(Path(kmz_path), overlay_kml, assets={"overlay.png": png_bytes})


if __name__ == "__main__":
    run(sys.argv[1:])

