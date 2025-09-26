"""
KMZ/KML Pipeline Image Renamer
Renames images based on proximity to pipeline linework and generates KMLs
"""

from __future__ import annotations

import os
import re
import json
import zipfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import xml.etree.ElementTree as ET

try:
    import piexif
    from shapely.geometry import LineString, Point
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    GEOSPATIAL_AVAILABLE = True
except Exception:
    GEOSPATIAL_AVAILABLE = False

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
logger = logging.getLogger(__name__)


class KMZParser:
    def __init__(self, client: str = "HGS") -> None:
        self.client = client
        if GEOSPATIAL_AVAILABLE:
            self._transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

    def _sanitize(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())

    def _extract_kml_from_kmz(self, kmz_path: str | Path) -> bytes:
        with zipfile.ZipFile(kmz_path) as z:
            for name in z.namelist():
                if name.lower().endswith(".kml"):
                    with z.open(name) as f:
                        return f.read()
        raise RuntimeError("No KML in KMZ")

    def load_pipelines(self, kmz_path: str | Path) -> List[Tuple[str, LineString]]:
        if not GEOSPATIAL_AVAILABLE:
            return []
        kml_data = self._extract_kml_from_kmz(kmz_path)
        root = ET.fromstring(kml_data)
        pipelines: List[Tuple[str, LineString]] = []
        for placemark in root.findall('.//kml:Placemark', KML_NS):
            ls = placemark.find('.//kml:LineString', KML_NS)
            if ls is None:
                continue
            name_el = placemark.find('kml:name', KML_NS)
            name = name_el.text.strip() if name_el is not None else 'pipeline'
            coord_el = ls.find('kml:coordinates', KML_NS)
            if coord_el is None or not coord_el.text:
                continue
            coords = []
            for part in coord_el.text.strip().split():
                pieces = part.split(',')
                if len(pieces) >= 2:
                    lon, lat = map(float, pieces[:2])
                    coords.append((lon, lat))
            if len(coords) >= 2:
                pipelines.append((name, LineString(coords)))
        return pipelines

    def _read_photo_coords(self, photo_path: str) -> Optional[Tuple[float, float]]:
        try:
            exif_data = piexif.load(photo_path)
            gps = exif_data.get("GPS", {})
            lat = gps.get(piexif.GPSIFD.GPSLatitude)
            lon = gps.get(piexif.GPSIFD.GPSLongitude)
            lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
            lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)
            if None in (lat, lon, lat_ref, lon_ref):
                return None

            def _to_deg(value):
                d, m, s = value
                return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

            lat_deg = _to_deg(lat)
            lon_deg = _to_deg(lon)
            if lat_ref in [b'S', b's']:
                lat_deg = -lat_deg
            if lon_ref in [b'W', b'w']:
                lon_deg = -lon_deg
            return lat_deg, lon_deg
        except Exception as e:
            logger.error(f"EXIF error in {photo_path}: {e}")
            return None

    def _create_kml(self, lat: float, lon: float, name: str, img_filename: str) -> str:
        return f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<kml xmlns=\"http://www.opengis.net/kml/2.2\">
  <Document>
    <Placemark>
      <name>{name}</name>
      <description><![CDATA[<img src=\"{img_filename}\" width=\"300\"/>]]></description>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>
"""

    def rename_images(self,
                      folder: str,
                      settings: dict,
                      pipeline: Optional[str] = None,
                      progress_callback=None,
                      status_callback=None,
                      pause_event=None) -> None:
        """Rename images based on KMZ proximity or override prefix; write KMLs."""
        override_prefix = settings.get("override_JPG_prefix")
        client_abbr = settings.get("client_abbreviation", self.client)

        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg"))]
        total = len(images)
        if total == 0:
            return

        kml_folder = os.path.join(folder, "KMLs")
        os.makedirs(kml_folder, exist_ok=True)

        if override_prefix:
            for idx, img in enumerate(images):
                if pause_event:
                    pause_event.wait()
                if status_callback:
                    status_callback(f"Renaming images ({idx+1}/{total})")
                img_path = os.path.join(folder, img)
                coords = self._read_photo_coords(img_path)
                new_name = re.sub(r'__+', '_', f"{override_prefix}_{img}")
                new_path = os.path.join(folder, new_name)
                os.rename(img_path, new_path)
                if coords:
                    lat, lon = coords
                    kml_name = os.path.splitext(new_name)[0]
                    kml_content = self._create_kml(lat, lon, kml_name, new_name)
                    with open(os.path.join(kml_folder, f"{kml_name}.kml"), "w", encoding="utf-8") as f:
                        f.write(kml_content)
                if progress_callback:
                    progress_callback(idx + 1, total)
            return

        if not GEOSPATIAL_AVAILABLE:
            logger.warning("Geospatial libraries not available; skipping KMZ-based renaming")
            return

        kmz_path = settings.get("kmz_path")
        if not kmz_path or not os.path.exists(kmz_path):
            logger.error(f"No valid KMZ path for {self.client}")
            return

        pipelines = self.load_pipelines(kmz_path)
        if not pipelines:
            logger.error("No pipelines loaded")
            return

        buffered = [
            (self._sanitize(name), shp_transform(self._transformer.transform, line).buffer(15.24))
            for name, line in pipelines
        ]

        for idx, img in enumerate(images):
            if pause_event:
                pause_event.wait()
            if status_callback:
                status_callback(f"Renaming images ({idx+1}/{total})")
            img_path = os.path.join(folder, img)
            coords = self._read_photo_coords(img_path)
            if not coords:
                logger.warning(f"No GPS data for {img}")
                continue
            x, y = self._transformer.transform(coords[1], coords[0])
            point = Point(x, y)
            hit_names = [name for name, buf in buffered if buf.contains(point)]

            # Fallback to closest pipeline if no matches
            if not hit_names:
                distances = []
                for name, line in pipelines:
                    line_3857 = shp_transform(self._transformer.transform, line)
                    distance = line_3857.distance(point)
                    distances.append((distance, name))
                if distances:
                    _, closest_name = min(distances)
                    hit_names = [self._sanitize(closest_name)]
                else:
                    continue

            img_base = os.path.splitext(img)[0]
            client_present = client_abbr and f"_{client_abbr}_" in f"_{img_base}_"

            def emit(new_name: str) -> None:
                new_path = os.path.join(folder, new_name)
                lat, lon = coords
                kml_name = os.path.splitext(new_name)[0]
                kml_content = self._create_kml(lat, lon, kml_name, new_name)
                with open(os.path.join(kml_folder, f"{kml_name}.kml"), "w", encoding="utf-8") as f:
                    f.write(kml_content)
                return new_path

            if client_present:
                if len(hit_names) == 1:
                    combined_name = f"{hit_names[0]}_{img}"
                    new_name = re.sub(r'__+', '_', combined_name)
                    new_path = emit(new_name)
                    os.rename(img_path, new_path)
                else:
                    for name in hit_names:
                        combined_name = f"{name}_{img}"
                        new_name = re.sub(r'__+', '_', combined_name)
                        new_path = emit(new_name)
                        shutil.copy2(img_path, new_path)
                    os.remove(img_path)
            else:
                if len(hit_names) == 1:
                    combined_name = f"{hit_names[0]}_{client_abbr}_{img}" if client_abbr else f"{hit_names[0]}_{img}"
                    new_name = re.sub(r'__+', '_', combined_name)
                    new_path = emit(new_name)
                    os.rename(img_path, new_path)
                else:
                    for name in hit_names:
                        combined_name = f"{name}_{client_abbr}_{img}" if client_abbr else f"{name}_{img}"
                        new_name = re.sub(r'__+', '_', combined_name)
                        new_path = emit(new_name)
                        shutil.copy2(img_path, new_path)
                    os.remove(img_path)

            if progress_callback:
                progress_callback(idx + 1, total)

