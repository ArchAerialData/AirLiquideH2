from pathlib import Path
import sys
import pandas as pd

# Ensure repo root is on sys.path so 'modules' imports work when running from .codex/tmp_scripts
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.csv_parser import parse_raw_csv, attach_source_metadata
from modules.output_schema import to_output_df

p = Path('test_folders/single_csv/Flight1/AL-Flight 1.CSV')
df, proj = parse_raw_csv(p)
df2 = attach_source_metadata(df, p, proj)
out = to_output_df(df2)
print('project:', proj)
print('columns:', list(out.columns))
print(out.head(3).to_string())

