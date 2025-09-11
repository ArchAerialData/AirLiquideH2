from fastkml import kml
from pathlib import Path
p = Path('.codex/test-extract/test-kmz/doc.kml')
s = p.read_bytes()
k = kml.KML()
k.from_string(s)
root = list(k.features)
print('root count:', len(root))
if root:
    f1 = root[0]
    children = list(getattr(f1, 'features', []))
    print('children count:', len(children))
    for c in children:
        print('child:', type(c).__name__, 'features?', hasattr(c, 'features'), 'geom?', hasattr(c, 'geometry'))
