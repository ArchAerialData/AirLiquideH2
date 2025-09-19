import pandas as pd
from pathlib import Path
p=Path(r'test_folders/.example_output/HIT-AOI_2.CSV')
try:
    df=pd.read_csv(p, header=None, encoding='utf-8-sig', engine='python')
    print('rows, cols:', df.shape)
    print(df.head(5).to_string())
except Exception as e:
    print('python engine failed', e)
