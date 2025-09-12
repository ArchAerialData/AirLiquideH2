import zipfile
from pathlib import Path
p = Path(r'.codex/misc/example-style.kmz')
with zipfile.ZipFile(p,'r') as z:
    names = z.namelist()
    print('FILES', names)
    for name in names:
        if name.lower().endswith('.kml'):
            data = z.read(name).decode('utf-8', errors='replace')
            print('KML_NAME', name)
            lines = data.splitlines()
            # show a snippet around description and table styling
            desc_idx = None
            for i,line in enumerate(lines):
                if '<description>' in line:
                    desc_idx = i
                    break
            if desc_idx is not None:
                start = max(0, desc_idx-20)
                end = min(len(lines), desc_idx+60)
                for j in range(start, end):
                    print(f'{j+1:03d}:', lines[j])
            else:
                print('NO_DESCRIPTION_FOUND')
            break
