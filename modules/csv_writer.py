from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .aggregator import GroupBoundary

def write_with_markers(df: pd.DataFrame, groups: List[GroupBoundary], out_path: Path) -> None:
    """CSV fallback: write combined CSV and insert a minimal separator marker row
    between groups (e.g., a line of dashes). This is only for visibility in CSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        # Write header
        f.write(",".join(df.columns) + "\n")
        start = 0
        for i, g in enumerate(groups):
            end = g.end_row
            block = df.iloc[start:end+1]
            block.to_csv(f, header=False, index=False)
            start = end + 1
            if i < len(groups) - 1:
                f.write("# ---- separator between files ----\n")
