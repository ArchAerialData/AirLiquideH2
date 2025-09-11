from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

def resolve_input_paths(argv: List[str]) -> List[Path]:
    """Return a list of folder paths.
    - If paths are passed in (drag-and-drop on Windows -> sys.argv[1:]), use those.
    - Otherwise, prompt the user to paste one or more paths separated by commas.
    """
    paths: List[Path] = []

    raw = argv or []
    if not raw:
        try:
            raw_text = input("Enter one or more folder paths (comma-separated): ").strip()
            if raw_text:
                raw = [p.strip().strip('"') for p in raw_text.split(',') if p.strip()]
        except EOFError:
            raw = []

    for p in raw:
        pp = Path(p).expanduser().resolve()
        if pp.is_dir():
            paths.append(pp)

    # Deduplicate while preserving order
    seen = set()
    unique_paths: List[Path] = []
    for p in paths:
        if p not in seen:
            unique_paths.append(p)
            seen.add(p)

    return unique_paths

def derive_output_path(input_dirs: Iterable[Path], preferred_name: str = "Combined_Extracted.xlsx") -> Path:
    """Compute a reasonable output path.
    - If one folder: write inside that folder.
    - If multiple: use their common parent.
    """
    input_dirs = list(input_dirs)
    if not input_dirs:
        raise ValueError("No input directories provided")

    if len(input_dirs) == 1:
        base = input_dirs[0]
    else:
        # common parent; fallback to the first one's parent if needed
        try:
            base = Path(Path(*Path.commonpath([str(p) for p in input_dirs])))
        except Exception:
            base = input_dirs[0].parent

    return (base / preferred_name).resolve()
