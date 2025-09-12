import pandas as pd
from pathlib import Path
p = Path(r'test_folders/Combined_Extracted.xlsx')
if not p.exists():
    print('MISSING_FILE')
    raise SystemExit(0)
# Use row 2 as header (index 1), skip first merged project row
try:
    df = pd.read_excel(p, engine='openpyxl', header=1)
except Exception as e:
    print('READ_FAIL', e)
    raise SystemExit(0)

# Drop repeated header rows (they appear in data if multiple groups)
if 'PLID' in df.columns:
    df = df[df['PLID'].astype(str) != 'PLID']

# Ensure numeric types
for col in ['BeginMeasu','EndMeasure','PPM']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows without key fields
df = df.dropna(subset=['BeginMeasu','EndMeasure','PPM'])

# For each (BeginMeasu, EndMeasure), keep row(s) with max PPM; ties -> first
idx = df.groupby(['BeginMeasu','EndMeasure'])['PPM'].idxmax()
kept = df.loc[idx].copy()

# Sort for readability
kept = kept.sort_values(['BeginMeasu','EndMeasure']).reset_index(drop=True)
print('TOTAL_ROWS', len(df), 'KEPT', len(kept), 'GROUPS', kept.shape[0])
# Print as CSV to stdout with key columns first
cols = ['PLID','BeginMeasu','EndMeasure','PPM','Route_Desc','Date','Longitude','Latitude','TEMP (F)','Class_Loca','Diameter','Product','Serial No.','Source_File']
cols = [c for c in cols if c in kept.columns]
print('COLUMNS', '|'.join(cols))
for _, r in kept.iterrows():
    vals = []
    for c in cols:
        v = r[c]
        if pd.isna(v):
            vals.append('')
        elif hasattr(v,'strftime'):
            vals.append(v.strftime('%m/%d/%Y'))
        else:
            vals.append(str(v))
    print('|'.join(vals))
