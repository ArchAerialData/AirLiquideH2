from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd

from .csv_parser import ParsedCSV

@dataclass
class GroupBoundary:
    csv_name: str
    start_row: int  # inclusive (in combined df)
    end_row: int    # inclusive
    row_count: int
    project_name: str

def combine(items: Iterable[ParsedCSV]) -> tuple[pd.DataFrame, List[GroupBoundary]]:
    """Combine per-CSV DataFrames, tracking boundaries so we can insert
    separator rows in the XLSX output later.

    This function is intentionally simple for now.
    """
    frames: List[pd.DataFrame] = []
    boundaries: List[GroupBoundary] = []
    cursor = 0

    for parsed in items:
        n = len(parsed.df)
        if n == 0:
            continue
        frames.append(parsed.df)
        boundaries.append(GroupBoundary(
            csv_name=parsed.source_info.filename,
            start_row=cursor,
            end_row=cursor + n - 1,
            row_count=n,
            project_name=parsed.project_name,
        ))
        cursor += n

    combined = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    return combined, boundaries
