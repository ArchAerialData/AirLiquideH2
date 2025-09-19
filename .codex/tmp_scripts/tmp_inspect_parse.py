from pathlib import Path
from modules.csv_parser import parse_raw_csv

p=Path(r'test_folders/.example_output/HIT-AOI_2.CSV')
df, proj = parse_raw_csv(p)
print('Project:', proj)
print(df.columns.tolist())
print(df.head(5))
