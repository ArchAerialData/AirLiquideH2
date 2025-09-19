import pandas as pd
p=r'test_folders/.example_output/Combined_Extracted_Clean.xlsx'
df=pd.read_excel(p, sheet_name='ALL')
print('rows:', len(df))
print('min_ppm:', pd.to_numeric(df['PPM'], errors='coerce').min())
print('max_ppm:', pd.to_numeric(df['PPM'], errors='coerce').max())
print('>=5ppm count:', (pd.to_numeric(df['PPM'], errors='coerce')>=5).sum())
print('<5ppm count:', (pd.to_numeric(df['PPM'], errors='coerce')<5).sum())
print(df.to_string(index=False))
