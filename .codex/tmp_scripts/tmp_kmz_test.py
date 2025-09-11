from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules.kmz_lookup import KMZIndex
from modules.config import DEFAULT_KMZ_PATH

kmz = KMZIndex(Path(DEFAULT_KMZ_PATH))
print('Features:', len(kmz.features))
print('Lookup:', kmz.lookup(29.6105, -95.0835, 200))
