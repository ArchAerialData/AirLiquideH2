import pandas as pd
from pathlib import Path
p=Path(r'test_folders/.example_output/HIT-AOI_2.CSV')
df=pd.read_csv(p, header=None, encoding='utf-8-sig', skiprows=2)
# Print first row len list length only to avoid encoding issues
print('cols:', df.shape[1])
print('sample types:', [type(x).__name__ for x in df.iloc[0].tolist()])
