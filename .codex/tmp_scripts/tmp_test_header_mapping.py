from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.csv_parser import parse_raw_csv, attach_source_metadata
from modules.output_schema import to_output_df
p = Path(r'.codex/misc/AL-Flight 45.CSV')
df, proj = parse_raw_csv(p)
print('project:', proj)
print('columns after parse:', list(df.columns))
print(df.head(3).to_string())
out = to_output_df(attach_source_metadata(df, p, proj))
print('output columns:', list(out.columns))
print(out.head(3).to_string())
