from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import zipfile

try:
    from fastkml import kml as fastkml
except Exception:  # pragma: no cover - optional at import time
    fastkml = None

try:
    from shapely.geometry import LineString, Point, MultiLineString
    from shapely.ops import transform as shp_transform
except Exception:  # pragma: no cover
    LineString = None  # type: ignore
    Point = None  # type: ignore
    MultiLineString = None  # type: ignore
    shp_transform = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    from rtree import index as rtree_index
except Exception:  # pragma: no cover
    rtree_index = None  # type: ignore

try:
    from lxml import etree
except Exception:  # pragma: no cover
    etree = None  # type: ignore

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore


@dataclass
class PolylineFeature:
    geometry: Any  # WGS84 LineString
    attributes: Dict[str, str]
    feature_id: str
    name: str = ""
    proj_line: Any = None  # Projected LineString (e.g., EPSG:3857)
    proj_buffer: Any = None  # Projected buffer polygon


class KMZIndex:
    def __init__(self, kmz_path: Path, buffer_meters: float = 15.24, candidate_count: int = 50, search_radius_m: float = 200.0):
        self.kmz_path = Path(kmz_path)
        self.features: List[PolylineFeature] = []
        self._spatial_index = None
        self.buffer_meters = float(buffer_meters)
        self.candidate_count = int(candidate_count)
        self.search_radius_m = float(search_radius_m)
        self._transformer = None
        self._load_kmz()

    def _load_kmz(self) -> None:
        if fastkml is None or LineString is None or BeautifulSoup is None:
            raise RuntimeError(
                "Missing dependencies for KMZ parsing. Please install fastkml, shapely, beautifulsoup4."
            )

        if not self.kmz_path.exists():
            raise FileNotFoundError(f"KMZ file not found: {self.kmz_path}")

        # Extract KML from KMZ
        with zipfile.ZipFile(self.kmz_path, "r") as kmz:
            kml_bytes = kmz.read("doc.kml")

        # Parse KML (fastkml expects bytes if XML has encoding declaration)
        k = fastkml.KML()
        k.from_string(kml_bytes)

        # Extract features recursively
        self._extract_features(k)

        # If nothing parsed, fallback to a simple lxml-based parser
        if not self.features and etree is not None and BeautifulSoup is not None and LineString is not None:
            self._fallback_parse_kml(kml_bytes)

        # Prepare projection and projected geometries
        if Transformer is not None and self.features:
            self._prepare_projected_geometries()

        # Build spatial index if rtree available
        if rtree_index is not None and self.features:
            self._build_spatial_index()

    def _iter_features(self, node: Any) -> Iterable[Any]:
        children: List[Any] = []
        if hasattr(node, "features"):
            try:
                attr = getattr(node, "features")
                if callable(attr):
                    children = list(attr())
                else:
                    children = list(attr) if attr is not None else []
            except Exception:
                children = []
        if children:
            for child in children:
                yield from self._iter_features(child)
        else:
            yield node

    def _extract_features(self, k: Any) -> None:
        for placemark in self._iter_features(k):
            try:
                geom = getattr(placemark, "geometry", None)
            except Exception:
                geom = None

            if geom is None:
                continue

            # Handle both shapely geometry or fastkml Geometry wrapper
            line = None
            # Accept any shapely geometry that exposes bounds/distance
            if hasattr(geom, "bounds") and hasattr(geom, "distance"):
                line = geom
            else:
                try:
                    inner = getattr(geom, "geom", None)
                    if inner is not None and hasattr(inner, "bounds") and hasattr(inner, "distance"):
                        line = inner
                except Exception:
                    line = None

            if line is None:
                continue

            attrs = self._parse_description_table(getattr(placemark, "description", ""))
            fid = getattr(placemark, "id", str(len(self.features)))
            name = getattr(placemark, "name", "") or ""
            self.features.append(PolylineFeature(geometry=line, attributes=attrs, feature_id=fid, name=str(name)))

    def _parse_description_table(self, description: str) -> Dict[str, str]:
        if not description or BeautifulSoup is None:
            return {}
        soup = BeautifulSoup(description, "html.parser")
        out: Dict[str, str] = {}
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                out[key] = val
        return out

    def _build_spatial_index(self) -> None:
        idx = rtree_index.Index()
        for i, feat in enumerate(self.features):
            try:
                bounds = (feat.proj_line or feat.geometry).bounds
            except Exception:
                continue
            idx.insert(i, bounds)
        self._spatial_index = idx

    def _fallback_parse_kml(self, kml_bytes: bytes) -> None:
        """Basic KML parser using lxml to extract Placemark LineStrings and attributes."""
        try:
            root = etree.fromstring(kml_bytes)
        except Exception:
            return

        ns = {"kml": "http://www.opengis.net/kml/2.2", "gx": "http://www.google.com/kml/ext/2.2"}

        for pm in root.findall(".//kml:Placemark", namespaces=ns):
            # Description attributes (HTML table)
            desc = pm.findtext("kml:description", default="", namespaces=ns) or ""
            attrs = self._parse_description_table(desc)
            name = pm.findtext("kml:name", default="", namespaces=ns) or ""

            # Collect all LineStrings under this Placemark (direct or within MultiGeometry)
            line_geoms: List[LineString] = []
            for ln in pm.findall(".//kml:LineString", namespaces=ns):
                coords_text = (ln.findtext("kml:coordinates", default="", namespaces=ns) or "").strip()
                if not coords_text:
                    continue
                pts = []
                for token in coords_text.split():
                    parts = token.split(",")
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0]); y = float(parts[1])
                            pts.append((x, y))
                        except Exception:
                            pass
                if pts and LineString is not None:
                    try:
                        line_geoms.append(LineString(pts))
                    except Exception:
                        pass

            if not line_geoms:
                continue

            geom = (
                line_geoms[0]
                if len(line_geoms) == 1 or MultiLineString is None
                else MultiLineString(line_geoms)
            )
            fid = pm.get("id") or str(len(self.features))
            self.features.append(PolylineFeature(geometry=geom, attributes=attrs, feature_id=fid, name=name))

    def _prepare_projected_geometries(self) -> None:
        if Transformer is None or shp_transform is None:
            return
        self._transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        def _tx(x, y, z=None):
            x2, y2 = self._transformer.transform(x, y)
            return (x2, y2) if z is None else (x2, y2, z)
        for f in self.features:
            try:
                proj_line = shp_transform(_tx, f.geometry)
                f.proj_line = proj_line
                f.proj_buffer = proj_line.buffer(self.buffer_meters)
            except Exception:
                f.proj_line = None
                f.proj_buffer = None

    def lookup(self, lat: float, lon: float, max_distance_meters: float = 100.0) -> Optional[Dict[str, str]]:
        if Point is None:
            raise RuntimeError("shapely is required for spatial lookup")

        # Project the point
        if self._transformer is not None:
            x, y = self._transformer.transform(lon, lat)
            pt_proj = Point(x, y)
        else:
            # fallback: no projection
            pt_proj = Point(lon, lat)

        # Gather candidates from spatial index
        def all_indices() -> List[int]:
            return list(range(len(self.features)))

        candidates: List[int] = all_indices()
        if self._spatial_index is not None:
            try:
                r = self.search_radius_m
                bbox = (pt_proj.x - r, pt_proj.y - r, pt_proj.x + r, pt_proj.y + r)
                candidates = list(self._spatial_index.intersection(bbox))
                if not candidates:
                    # fallback to nearest with larger candidate_count
                    candidates = list(self._spatial_index.nearest((pt_proj.x, pt_proj.y, pt_proj.x, pt_proj.y), self.candidate_count))
            except Exception:
                candidates = all_indices()

        # Buffer hits first
        hits: List[Tuple[float, int]] = []  # (distance, idx)
        for idx in candidates:
            feat = self.features[idx]
            geom = feat.proj_line or feat.geometry
            buf = feat.proj_buffer
            if buf is not None:
                try:
                    if buf.contains(pt_proj):
                        d = pt_proj.distance(geom)
                        hits.append((d, idx))
                except Exception:
                    pass

        if hits:
            hits.sort(key=lambda t: t[0])
            min_d, best_idx = hits[0]
            best = self.features[best_idx]
            method = "buffer_hit"
        else:
            # Nearest fallback among candidates
            min_d = float("inf")
            best_idx = -1
            for idx in candidates:
                feat = self.features[idx]
                geom = feat.proj_line or feat.geometry
                try:
                    d = pt_proj.distance(geom)
                except Exception:
                    continue
                if d < min_d:
                    min_d = d
                    best_idx = idx
            if best_idx < 0:
                return None
            best = self.features[best_idx]
            method = "nearest_fallback"

        result = dict(best.attributes)
        # Fallbacks for missing keys
        if 'Route_Name' not in result and best.name:
            result['Route_Name'] = best.name
        result['Match_Method'] = method
        result['Nearest_Distance_Meters'] = round(float(min_d), 2)
        # Enforce maximum allowed fallback distance
        if max_distance_meters is not None and float(min_d) > float(max_distance_meters):
            return None
        return result
