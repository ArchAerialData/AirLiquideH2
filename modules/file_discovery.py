from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

CSV_EXTS = {".csv", ".CSV"}

def find_csv_files(roots: Iterable[Path]) -> List[Path]:
    """Recursively find CSV files under the given root directory(ies)."""
    out: List[Path] = []
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in CSV_EXTS:
                # Ignore Mac resource fork files like ._AL-Flight 1.CSV
                if p.name.startswith("._"):
                    continue
                out.append(p.resolve())
    # Stable sort by path to keep grouping predictable
    return sorted(out)
