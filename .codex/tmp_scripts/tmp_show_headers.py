import pandas as pd
from pathlib import Path

p=Path(r'test_folders/.example_output/HIT-AOI_2.CSV')
# Replicate the header detection from csv_parser.parse_raw_csv succinctly
from modules.csv_schema import fix_csv_headers

df=pd.read_csv(p, header=None, encoding='utf-8-sig')
probe_rows=min(6,len(df))
header_row_idx=None
for r in range(probe_rows):
    row_vals=df.iloc[r].astype(str).str.strip()
    if any(tok in set(row_vals) for tok in {'Time Stamp','Longitude','Latitude','Serial No.','PPM'}) or (('Longitude' in set(row_vals)) and ('Latitude' in set(row_vals))):
        header_row_idx=r
        break
print('header_row_idx =', header_row_idx)
raw_headers=df.iloc[header_row_idx].copy()
headers=fix_csv_headers(raw_headers)
print('headers length', len(headers))
print(list(headers))
