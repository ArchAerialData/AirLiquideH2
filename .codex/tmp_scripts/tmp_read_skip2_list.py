import pandas as pd
from pathlib import Path
p=Path(r'test_folders/.example_output/HIT-AOI_2.CSV')
df=pd.read_csv(p, header=None, encoding='utf-8-sig', skiprows=2)
print('first row column values:')
print(df.iloc[0].tolist())
