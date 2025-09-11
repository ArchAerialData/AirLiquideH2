# Air Liquide H2 Flight Data Processor

A Python tool that processes drone flight CSV files, fixes header alignment issues, enriches data with pipeline spatial information from KMZ files, and outputs a combined Excel spreadsheet with visual separation between flights.

## What It Does

**Input**: 
- 📁 Folders containing flight CSV files with misaligned headers
- 🗺️ KMZ file with pipeline route data

**Output**: 
- 📊 Single Excel file with cleaned, enriched flight data
- ⚫ Black separator rows between different flights
- 🚰 Pipeline route information added to each data point

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install pandas openpyxl pyyaml fastkml shapely beautifulsoup4 lxml rtree
   ```

2. **Run the processor**:
   ```bash
   # Drag folders onto main.py (Windows)
   # OR run directly:
   python main.py "C:\path\to\flight\folders"
   ```

3. **Output**: Find `Combined_Extracted.xlsx` in the parent directory of your input folders

## How It Works

### 1. CSV Correction
- Fixes header alignment issues in drone-generated CSVs
- Extracts only the needed columns: Time, Location, Temperature, PPM, Serial
- Adds missing "PPM" header to column L

### 2. Spatial Enrichment  
- Matches flight coordinates against pipeline KMZ data
- Finds nearest pipeline route for each data point
- Adds pipeline info: Route Name, Description, Diameter, Product Type, etc.

### 3. Combined Output
- Merges all flights into one Excel file
- Inserts black separator rows between different flights
- Maintains source file tracking for data traceability

## Example Output Structure

```
Combined_Extracted.xlsx:
┌─────────────────────────────────────────────────────────────────┐
│ Flight 1: AL-Flight 10 (150 rows)                              │
├─────────────────────────────────────────────────────────────────┤
│ Time Stamp │ Lon │ Lat │ Temp │ PPM │ Route │ Pipeline Info... │
│ 2025-09-03 │ ... │ ... │ 20.9 │ 3.28│ 100272│ Bayport-Webster │
│ ...        │ ... │ ... │ ...  │ ... │ ...   │ ...             │
├█████████████████████████████████████████████████████████████████┤ ← Separator
│ Flight 2: AL-Flight 15 (200 rows)                              │
├─────────────────────────────────────────────────────────────────┤
│ Time Stamp │ Lon │ Lat │ Temp │ PPM │ Route │ Pipeline Info... │
│ 2025-09-04 │ ... │ ... │ 21.5 │ 2.95│ 100272│ Webster-Houston │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
AirLiquideH2/
├── main.py                    # Main entry point
├── modules/                   # Core processing modules
│   ├── csv_parser.py         # CSV correction and parsing
│   ├── kmz_lookup.py         # Spatial pipeline matching
│   ├── aggregator.py         # Data combination
│   └── xlsx_writer.py        # Excel output with formatting
└── task.md                   # Detailed implementation guide
```

## Current Status

⚠️ **Implementation Required**: This is currently a skeleton project. See `task.md` for detailed implementation instructions.

The skeleton demonstrates the architecture but requires implementation of:
- CSV header correction logic
- KMZ spatial parsing and indexing  
- Data enrichment pipeline
- Excel formatting with separators

## For Developers

See `task.md` for comprehensive step-by-step implementation instructions including:
- Detailed code examples for each module
- KMZ field extraction strategies
- CSV alignment correction methods
- XLSX formatting requirements
- Testing and validation procedures

## Dependencies

- **pandas**: Data processing and CSV handling
- **openpyxl**: Excel file creation and formatting
- **fastkml**: KMZ/KML spatial data parsing
- **shapely**: Geometric operations and spatial indexing
- **beautifulsoup4**: HTML parsing for KMZ attributes
- **rtree**: Spatial indexing for performance

## Notes

- Designed for Windows drag-and-drop workflow
- Optimized for large datasets with spatial indexing
- Maintains full data traceability back to source files
- Professional Excel formatting with visual separation
- Configurable distance thresholds for pipeline matching