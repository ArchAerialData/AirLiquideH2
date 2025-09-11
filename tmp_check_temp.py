from pathlib import Path
import pandas as pd
from modules.csv_parser import parse_raw_csv, attach_source_metadata
from modules.output_schema import to_output_df
p = Path('test_folders/single_csv/Flight1/AL-Flight 1.CSV')
df, proj = parse_raw_csv(p)
df2 = attach_source_metadata(df, p, proj)
out = to_output_df(df2)
print('project:', proj)
print('columns:', list(out.columns))
print(out.head(3).to_string())
