from pathlib import Path
import pandas as pd
p = Path('test_folders/single_csv/Flight1/AL-Flight 1.CSV')
df = pd.read_csv(p, header=None, encoding='utf-8-sig')
raw = df.iloc[3:, :].copy()
headers = df.iloc[2].copy()
headers.iloc[11] = 'PPM'
raw.columns = headers
s = raw['Time Stamp'].astype(str).head(5).tolist()
print('raw_ts_head:', s)
