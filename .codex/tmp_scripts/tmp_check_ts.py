from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.csv_parser import parse_raw_csv

p = Path('test_folders/single_csv/Flight1/AL-Flight 1.CSV')
df, proj = parse_raw_csv(p)
print('rows', len(df))
print('ts non-null', df['Time Stamp'].notna().sum())
print(df['Time Stamp'].head(5))

