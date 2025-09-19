from pathlib import Path
import sys
from pathlib import Path

# Ensure repository root is on sys.path when run from inside .codex/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.csv_parser import parse_raw_csv

def run_one(path_str: str) -> None:
    p = Path(path_str)
    df, proj = parse_raw_csv(p)
    print("project:", proj)
    print("columns:", list(df.columns))
    print("shape:", df.shape)
    print(df.head(5))

if __name__ == "__main__":
    run_one(sys.argv[1])
