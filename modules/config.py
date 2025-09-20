from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional at this stage
    yaml = None

@dataclass
class SeparatorStyle:
    fill_rgb: str = "000000"   # black row
    height: int = 1

@dataclass
class Config:
    columns_to_extract: List[str] = field(default_factory=list)
    # Map input CSV headers -> output column names (optional)
    header_renames: dict[str, str] = field(default_factory=dict)

    # (Later) KMZ settings
    kmz_path: Optional[str] = None
    kmz_buffer_meters: float = 2.0
    kmz_distance_threshold: float = 150.0

    # Output
    output_format: str = "xlsx"
    include_separator_rows: bool = True
    separator_style: SeparatorStyle = field(default_factory=SeparatorStyle)

    # App behavior
    hits_threshold: float = 5.0
    pin_spacing_meters: float = 25.0

def load_config_from_yaml(path: str | None) -> Config:
    """Load a Config from YAML. If path is None or PyYAML is missing, return defaults."""
    if path is None or yaml is None:
        return Config()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Shallow mapping; expand as needed
    cfg = Config(
        columns_to_extract = list(data.get("columns_to_extract", [])),
        header_renames = dict(data.get("header_renames", {})),
        kmz_path = data.get("kmz_path"),
        kmz_buffer_meters = float(data.get("kmz_buffer_meters", 2.0)),
        kmz_distance_threshold = float(data.get("kmz_distance_threshold", 150.0)),
        output_format = str(data.get("output_format", "xlsx")),
        include_separator_rows = bool(data.get("include_separator_rows", True)),
        separator_style = SeparatorStyle(**(data.get("separator_style", {}) or {})),
        hits_threshold = float(data.get("hits_threshold", 5.0)),
        pin_spacing_meters = float(data.get("pin_spacing_meters", 25.0)),
    )
    return cfg

# Simple module-level defaults (can be used without YAML)
DEFAULT_KMZ_DISTANCE_THRESHOLD: float = 150.0
DEFAULT_KMZ_PATH: str = "kmz/AirLiquideH2.kmz"
OUTPUT_FORMAT: str = "xlsx"
INCLUDE_CSV_OUTPUT: bool = False
DEFAULT_PIN_SPACING_METERS: float = 25.0


def load_default_config() -> Config:
    """Load `config.yaml` or `config.yml` from CWD or project root.

    Falls back to defaults when no file is found or PyYAML is missing.
    """
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path(__file__).resolve().parent.parent / "config.yaml",
        Path(__file__).resolve().parent.parent / "config.yml",
    ]
    for p in candidates:
        try:
            if p.exists():
                return load_config_from_yaml(str(p))
        except Exception:
            # Fall through to next candidate
            pass
    return Config()
