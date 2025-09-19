import pandas as pd
from pathlib import Path
p=Path(r'test_folders/.example_output/HIT-AOI_2.CSV')
try:
    df=pd.read_csv(p, header=None, encoding='latin-1', engine='python', on_bad_lines='skip')
    print('rows, cols:', df.shape)
    print(df.iloc[0].tolist())
    print(df.iloc[1].tolist())
    print(df.iloc[2].tolist())
except Exception as e:
    print('latin-1 engine failed', e)
