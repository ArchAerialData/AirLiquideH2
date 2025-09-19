import pandas as pd
p=r'test_folders/.example_output/Combined_Extracted_Hits.xlsx'
df=pd.read_excel(p, sheet_name='ALL')
print('rows:', len(df))
print(df[['Date','PPM','Longitude','Latitude','TEMP (F)','Serial No.','Source_File']].head(10).to_string(index=False))
