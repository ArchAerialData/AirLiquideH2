from pathlib import Path
import pandas as pd
import sys

def show(path: str, rows: int = 6, cols: int = 25):
    p = Path(path)
    df = pd.read_csv(p, header=None, encoding='utf-8-sig')
    pd.set_option('display.max_columns', cols)
    s = df.head(rows).to_string()
    # Replace symbols that the console may not encode
    s = s.replace('\u2103', 'C').replace('°', 'deg')
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
    except Exception:
        pass
    print(s)

if __name__ == '__main__':
    show(sys.argv[1])
