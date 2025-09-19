import pandas as pd
p=r'test_folders/2025_Q1/2025_05/2025_05_05/Flight 86/Combined_Extracted.xlsx'
df=pd.read_excel(p, sheet_name='ALL')
print(df[['Date','PPM','Longitude','Latitude','TEMP (F)','Serial No.','Source_File']].head(5).to_string(index=False))
