from pathlib import Path
import pandas as pd
import sys

def analyze(path: str):
    try:
        import sys as _sys
        _sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
    except Exception:
        pass
    p = Path(path)
    df = pd.read_csv(p, header=None, encoding='utf-8-sig')
    expected_tokens = {"Time Stamp", "Longitude", "Latitude", "Serial No.", "PPM"}
    header_row_idx = None
    for r in range(min(6, len(df))):
        row_vals = df.iloc[r].astype(str).str.strip()
        if any(tok in set(row_vals) for tok in expected_tokens) or (
            ("Longitude" in set(row_vals)) and ("Latitude" in set(row_vals))
        ):
            header_row_idx = r
            break
    print("header_row_idx:", header_row_idx)
    raw_headers = df.iloc[header_row_idx].copy()
    def safe_list(vals):
        return [str(v).replace('\u2103', 'C').replace('°', 'deg') for v in vals]
    print("raw_headers(0..14):", safe_list(raw_headers.iloc[:15]))
    headers = raw_headers.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    if not (headers == "PPM").any():
        aliases = {"h2%", "h2 %", "h2_ppm", "h2ppm", "ppm(h2)", "ppm"}
        start, end = 9, min(14, len(headers) - 1)
        renamed = False
        for i in range(start, end + 1):
            label = str(headers.iloc[i]).strip().lower()
            if label in aliases or label.replace(" ", "") in aliases or label == "":
                headers.iloc[i] = "PPM"
                renamed = True
                break
        if not renamed and len(headers) > 11:
            headers.iloc[11] = "PPM"
    print("fixed headers(0..14):", safe_list(headers.iloc[:15]))
    data_df = df.iloc[header_row_idx + 1 :].copy()
    data_df.columns = headers
    print("data_df first row cols 0..14:")
    print(safe_list(data_df.columns[:15]))
    print("values:", safe_list(data_df.iloc[0, :15]))

if __name__ == "__main__":
    analyze(sys.argv[1])
