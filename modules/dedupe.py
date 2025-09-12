from __future__ import annotations

import pandas as pd


def dedupe_by_measure(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deduplicated view keeping one row per (BeginMeasu, EndMeasure)
    with the highest PPM. Zeros are valid; only NaNs are considered missing.

    Precondition: `df` is already in the final output schema produced by
    modules.output_schema.to_output_df, so it contains at least:
    - 'BeginMeasu', 'EndMeasure', 'PPM'

    Implementation detail: We compute idxmax on a numeric-coerced copy to find
    the rows to keep, then return those rows from the original `df` to preserve
    all other columns and types as-is.
    """
    if not {'BeginMeasu', 'EndMeasure', 'PPM'}.issubset(df.columns):
        # Nothing to dedupe against; return as-is
        return df.copy()

    work = df[['BeginMeasu', 'EndMeasure', 'PPM']].copy()
    work['BeginMeasu'] = pd.to_numeric(work['BeginMeasu'], errors='coerce')
    work['EndMeasure'] = pd.to_numeric(work['EndMeasure'], errors='coerce')
    work['PPM'] = pd.to_numeric(work['PPM'], errors='coerce')

    # Only NaNs are treated as missing; zeros are valid and retained
    valid = work.dropna(subset=['BeginMeasu', 'EndMeasure', 'PPM'])
    if valid.empty:
        return df.iloc[0:0].copy()

    keep_idx = valid.groupby(['BeginMeasu', 'EndMeasure'])['PPM'].idxmax()
    kept = df.loc[keep_idx].copy()
    # Sort for readability and stable output
    kept = kept.sort_values(['BeginMeasu', 'EndMeasure'], kind='mergesort').reset_index(drop=True)
    return kept

