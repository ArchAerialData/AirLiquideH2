# Air Liquide H2 CSV Processing - Complete Implementation Tasks

This document provides detailed step-by-step instructions for implementing the complete Air Liquide H2 CSV processing pipeline. Follow these tasks sequentially to build a fully functional system.

## Project Overview

The system processes flight CSV files with misaligned headers, corrects them, extracts specific columns, enriches data with spatial KMZ pipeline information, and outputs a combined XLSX with visual separation between flight data.

### Input Requirements
- **Raw CSV Files**: Drone-generated flight data with header alignment issues
- **KMZ File**: Pipeline spatial data with polyline attributes
- **Output**: Single XLSX with enriched data and visual separators

### Key Data Flow
```
Raw CSVs → Header Correction → Column Extraction → KMZ Spatial Enrichment → Combined XLSX
```

## Known CSV Structure Issues

**Raw CSV Problems to Fix:**
- Row 1: Metadata with misaligned "H2 %" in column L
- Row 2: Project name in cell A2, random values in columns L-M
- Row 3: Actual headers start here, but column L missing "PPM" header
- Column L: Contains PPM values but missing proper header

**Required Column Extraction (only these 6 columns):**
- Column A: `Time Stamp`
- Column C: `Longitude` (for KMZ spatial matching)
- Column D: `Latitude` (for KMZ spatial matching)  
- Column E: `Temperature ℃`
- Column L: PPM values (needs "PPM" header added)
- Column N: `Serial No.`

**Ignore These Columns:** B, F, G, H, I, J, K, M, P, Q, R, S

## KMZ Structure Analysis

Based on analysis of `example-polyline.kmz`, the KMZ files contain:

### Available KMZ Fields to Extract
| Field Name | Example Value | Data Type | Description |
|------------|---------------|-----------|-------------|
| **FID** | `0` | Integer | Feature ID |
| **BeginMeasu** | `13239.57` | Float | Begin Measure |
| **EndMeasure** | `14681.05` | Float | End Measure |
| **Route_Name** | `100272` | String | Route identifier |
| **Route_Desc** | `Bayport to Webster` | String | Route description |
| **Class_Loca** | `3` | Integer | Class Location |
| **Diameter** | `14` | Integer | Pipeline diameter |
| **Product** | `H` | String | Product type (H = Hydrogen) |

### KMZ Structure
- KMZ = ZIP archive containing `doc.kml`
- KML contains `<Placemark>` elements with geometry and attributes
- Attributes stored in HTML table within `<description>` CDATA section
- Geometry stored as `<LineString>` with coordinate arrays

## TASK 1: Implement CSV Schema Definition

**File**: `modules/csv_schema.py`

**Purpose**: Define the canonical schema for corrected CSV data and KMZ enrichment.

### Implementation Steps:

1. **Define Raw CSV Column Mapping**:
```python
# Map raw CSV column positions to target headers
RAW_CSV_COLUMN_MAPPING = {
    0: 'Time Stamp',        # Column A
    2: 'Longitude',         # Column C  
    3: 'Latitude',          # Column D
    4: 'Temperature ℃',     # Column E
    11: 'PPM',              # Column L (needs header fix)
    13: 'Serial No.'        # Column N
}
```

2. **Define Target Schema with Data Types**:
```python
@dataclass
class CSVSchema:
    # Core extracted columns
    timestamp_col: str = 'Time Stamp'
    longitude_col: str = 'Longitude'
    latitude_col: str = 'Latitude'
    temperature_col: str = 'Temperature ℃'
    ppm_col: str = 'PPM'
    serial_col: str = 'Serial No.'
    
    # KMZ enrichment columns (to be added)
    kmz_route_name: str = 'KMZ_Route_Name'
    kmz_route_desc: str = 'KMZ_Route_Desc'
    kmz_diameter: str = 'KMZ_Diameter'
    kmz_product: str = 'KMZ_Product'
    kmz_class_loca: str = 'KMZ_Class_Loca'
    kmz_begin_measu: str = 'KMZ_BeginMeasu'
    kmz_end_measure: str = 'KMZ_EndMeasure'
    kmz_distance_meters: str = 'KMZ_Distance_Meters'
```

3. **Define Data Type Validations**:
```python
COLUMN_DTYPES = {
    'Time Stamp': 'datetime64[ns]',
    'Longitude': 'float64',
    'Latitude': 'float64', 
    'Temperature ℃': 'float64',
    'PPM': 'float64',
    'Serial No.': 'string'
}
```

4. **Implement Header Correction Function**:
```python
def fix_csv_headers(raw_headers: pd.Series) -> pd.Series:
    """Fix missing PPM header in column L (index 11)"""
    corrected = raw_headers.copy()
    corrected.iloc[11] = 'PPM'  # Fix missing header
    return corrected
```

## TASK 2: Implement CSV Parser with Header Correction

**File**: `modules/csv_parser.py`

**Purpose**: Parse raw CSV files, fix alignment issues, extract target columns, and prepare for KMZ enrichment.

### Implementation Steps:

1. **Implement Raw CSV Reading with Header Fix**:
```python
def parse_raw_csv(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Parse raw CSV with header correction and column extraction.
    
    Returns:
        - Cleaned DataFrame with target columns only
        - Project name extracted from row 2
    """
    # Read without parsing headers initially
    df = pd.read_csv(csv_path, header=None, encoding='utf-8-sig')
    
    # Extract project name from row 1 (0-based index)
    project_name = str(df.iloc[1, 0]) if len(df) > 1 else "Unknown"
    
    # Get headers from row 2 (0-based index) and fix missing PPM header
    if len(df) > 2:
        headers = df.iloc[2].copy()
        headers.iloc[11] = 'PPM'  # Fix column L header
        
        # Extract data starting from row 3
        data_df = df.iloc[3:].copy()
        data_df.columns = headers
        
        # Select only target columns using RAW_CSV_COLUMN_MAPPING
        target_cols = [headers.iloc[pos] for pos in RAW_CSV_COLUMN_MAPPING.keys()]
        cleaned_df = data_df[target_cols].copy()
        
        # Apply data type conversions and validations
        cleaned_df = validate_and_convert_types(cleaned_df)
        
        return cleaned_df, project_name
    else:
        raise ValueError(f"Invalid CSV structure in {csv_path}")
```

2. **Implement Data Type Validation**:
```python
def validate_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to proper data types and handle errors."""
    df = df.copy()
    
    # Convert timestamp
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['Longitude', 'Latitude', 'Temperature ℃', 'PPM']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert serial to string
    df['Serial No.'] = df['Serial No.'].astype(str)
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['Longitude', 'Latitude'])
    
    return df
```

3. **Implement Source Metadata Attachment**:
```python
def attach_source_metadata(df: pd.DataFrame, csv_path: Path, project_name: str) -> pd.DataFrame:
    """Add source tracking columns."""
    df = df.copy()
    df['Source_File'] = csv_path.name
    df['Source_Path'] = str(csv_path)
    df['Project_Name'] = project_name
    df['Date_Folder'] = csv_path.parent.name  # Extract from folder structure
    return df
```

## TASK 3: Implement KMZ Lookup with Spatial Indexing

**File**: `modules/kmz_lookup.py`

**Purpose**: Parse KMZ files, extract polyline attributes, build spatial index, and provide proximity-based lookup.

### Implementation Steps:

1. **Install Required Dependencies**:
```bash
pip install fastkml shapely beautifulsoup4 lxml rtree
```

2. **Implement KMZ Parsing**:
```python
from fastkml import kml
from shapely.geometry import LineString, Point
from bs4 import BeautifulSoup
from rtree import index
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class PolylineFeature:
    geometry: LineString
    attributes: Dict[str, str]
    feature_id: str

class KMZIndex:
    def __init__(self, kmz_path: Path):
        self.kmz_path = kmz_path
        self.features: List[PolylineFeature] = []
        self.spatial_index = index.Index()
        self._load_kmz()
    
    def _load_kmz(self):
        """Extract and parse KMZ file."""
        # Extract KML from KMZ
        with zipfile.ZipFile(self.kmz_path, 'r') as kmz:
            kml_content = kmz.read('doc.kml').decode('utf-8')
        
        # Parse KML
        k = kml.KML()
        k.from_string(kml_content)
        
        # Extract features
        self._extract_features(k)
        
        # Build spatial index
        self._build_spatial_index()
    
    def _extract_features(self, k):
        """Extract polyline features with attributes."""
        for feature in k.features():
            for placemark in feature.features():
                if hasattr(placemark, 'geometry') and placemark.geometry:
                    # Extract geometry
                    geometry = placemark.geometry
                    
                    # Extract attributes from HTML description
                    attributes = self._parse_description_table(placemark.description)
                    
                    # Create feature
                    feature_obj = PolylineFeature(
                        geometry=geometry,
                        attributes=attributes,
                        feature_id=getattr(placemark, 'id', str(len(self.features)))
                    )
                    self.features.append(feature_obj)
    
    def _parse_description_table(self, description: str) -> Dict[str, str]:
        """Parse HTML table in KML description to extract attributes."""
        if not description:
            return {}
        
        soup = BeautifulSoup(description, 'html.parser')
        attributes = {}
        
        # Find all table rows
        rows = soup.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                attributes[key] = value
        
        return attributes
    
    def _build_spatial_index(self):
        """Build rtree spatial index for fast proximity queries."""
        for i, feature in enumerate(self.features):
            bounds = feature.geometry.bounds
            self.spatial_index.insert(i, bounds)
```

3. **Implement Proximity Lookup**:
```python
    def lookup(self, lat: float, lon: float, max_distance_meters: float = 100.0) -> Optional[Dict[str, str]]:
        """
        Find nearest polyline within max_distance and return its attributes.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate  
            max_distance_meters: Maximum search distance in meters
            
        Returns:
            Dictionary with polyline attributes + distance, or None if no match
        """
        point = Point(lon, lat)  # Note: shapely uses (x, y) = (lon, lat)
        
        # Get candidate features from spatial index
        candidates = list(self.spatial_index.nearest((lon, lat, lon, lat), 5))
        
        best_feature = None
        min_distance = float('inf')
        
        for idx in candidates:
            feature = self.features[idx]
            distance_meters = self._calculate_distance_meters(point, feature.geometry)
            
            if distance_meters <= max_distance_meters and distance_meters < min_distance:
                min_distance = distance_meters
                best_feature = feature
        
        if best_feature:
            result = best_feature.attributes.copy()
            result['Distance_Meters'] = str(round(min_distance, 2))
            return result
        
        return None
    
    def _calculate_distance_meters(self, point: Point, line: LineString) -> float:
        """Calculate distance from point to line in meters."""
        # Project to approximate UTM for meter-based distance
        # For Texas area, rough approximation: 1 degree ≈ 111,000 meters
        distance_degrees = point.distance(line)
        distance_meters = distance_degrees * 111000  # Rough conversion
        return distance_meters
```

## TASK 4: Implement CSV Enrichment Pipeline

**File**: `modules/csv_parser.py` (extend existing functions)

**Purpose**: Integrate KMZ spatial lookup into CSV processing pipeline.

### Implementation Steps:

1. **Extend ParsedCSV Class**:
```python
@dataclass
class ParsedCSV:
    df: pd.DataFrame
    source_info: SourceInfo
    project_name: str
    enriched: bool = False  # Track if KMZ enrichment applied
```

2. **Implement KMZ Enrichment Function**:
```python
def enrich_with_kmz(parsed_csv: ParsedCSV, kmz_index: KMZIndex, max_distance: float = 100.0) -> ParsedCSV:
    """
    Enrich CSV data with KMZ polyline attributes.
    
    Args:
        parsed_csv: Parsed CSV object
        kmz_index: Loaded KMZ spatial index
        max_distance: Maximum distance for valid matches (meters)
        
    Returns:
        Enriched ParsedCSV with additional KMZ columns
    """
    df = parsed_csv.df.copy()
    
    # Initialize KMZ columns
    kmz_columns = {
        'KMZ_Route_Name': '',
        'KMZ_Route_Desc': '',
        'KMZ_Diameter': '',
        'KMZ_Product': '',
        'KMZ_Class_Loca': '',
        'KMZ_BeginMeasu': '',
        'KMZ_EndMeasure': '',
        'KMZ_Distance_Meters': ''
    }
    
    for col in kmz_columns:
        df[col] = kmz_columns[col]
    
    # Process each row
    for idx, row in df.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        
        # Skip if coordinates are invalid
        if pd.isna(lat) or pd.isna(lon):
            continue
        
        # Lookup nearest polyline
        result = kmz_index.lookup(lat, lon, max_distance)
        
        if result:
            # Map KMZ fields to DataFrame columns
            field_mapping = {
                'Route_Name': 'KMZ_Route_Name',
                'Route_Desc': 'KMZ_Route_Desc', 
                'Diameter': 'KMZ_Diameter',
                'Product': 'KMZ_Product',
                'Class_Loca': 'KMZ_Class_Loca',
                'BeginMeasu': 'KMZ_BeginMeasu',
                'EndMeasure': 'KMZ_EndMeasure',
                'Distance_Meters': 'KMZ_Distance_Meters'
            }
            
            for kmz_field, df_col in field_mapping.items():
                if kmz_field in result:
                    df.at[idx, df_col] = result[kmz_field]
    
    # Return enriched ParsedCSV
    return ParsedCSV(
        df=df,
        source_info=parsed_csv.source_info,
        project_name=parsed_csv.project_name,
        enriched=True
    )
```

## TASK 5: Implement XLSX Writer with Visual Separation

**File**: `modules/xlsx_writer.py`

**Purpose**: Write combined XLSX with black separator rows between CSV groups and professional formatting.

### Implementation Steps:

1. **Implement Group Boundary Tracking**:
```python
@dataclass
class GroupBoundary:
    csv_name: str
    start_row: int
    end_row: int
    row_count: int
    project_name: str

def write_with_separators(
    combined_df: pd.DataFrame, 
    boundaries: List[GroupBoundary], 
    output_path: Path,
    separator_style: SeparatorStyle
) -> None:
    """
    Write XLSX with black separator rows between CSV groups.
    
    Args:
        combined_df: Combined DataFrame with all CSV data
        boundaries: List of group boundaries for each CSV
        output_path: Output XLSX file path
        separator_style: Formatting for separator rows
    """
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Combined Flight Data"
    
    # Write headers
    headers = list(combined_df.columns)
    ws.append(headers)
    
    # Style headers
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    
    for col in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    
    current_row = 2
    
    # Write data with separators
    for boundary in boundaries:
        # Add CSV group header comment
        comment_text = f"--- {boundary.project_name} ({boundary.csv_name}) - {boundary.row_count} rows ---"
        ws.cell(row=current_row, column=1).value = comment_text
        ws.cell(row=current_row, column=1).font = Font(italic=True, color="666666")
        current_row += 1
        
        # Write CSV data
        csv_data = combined_df.iloc[boundary.start_row:boundary.end_row + 1]
        for row_data in dataframe_to_rows(csv_data, index=False, header=False):
            ws.append(row_data)
            current_row += 1
        
        # Add black separator row (except after last group)
        if boundary != boundaries[-1]:
            separator_row = [""] * len(headers)
            ws.append(separator_row)
            
            # Style separator row as black
            black_fill = PatternFill(start_color="000000", end_color="000000", fill_type="solid")
            for col in range(1, len(headers) + 1):
                ws.cell(row=current_row, column=col).fill = black_fill
            
            current_row += 1
    
    # Auto-size columns
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Save workbook
    wb.save(output_path)
```

## TASK 6: Update Main Pipeline Integration

**File**: `main.py`

**Purpose**: Integrate all components into complete processing pipeline.

### Implementation Steps:

1. **Update Main Function**:
```python
def run(argv: list[str]) -> None:
    """Complete processing pipeline."""
    # Setup
    input_dirs = resolve_input_paths(argv)
    if not input_dirs:
        logger.warning("No input directories provided. Exiting.")
        return
    
    logger.info("Input directories:")
    for p in input_dirs:
        logger.info(f"  - {p}")
    
    # Discover CSV files
    csv_files = find_csv_files(input_dirs)
    if not csv_files:
        logger.warning("No CSV files found under the provided directories.")
        return
    
    logger.info(f"Discovered {len(csv_files)} CSV files (recursive).")
    
    # Load KMZ index (you'll need to provide KMZ path)
    kmz_path = Path("path/to/your/pipeline.kmz")  # TODO: Make configurable
    if kmz_path.exists():
        logger.info(f"Loading KMZ spatial index from: {kmz_path}")
        kmz_index = KMZIndex(kmz_path)
    else:
        logger.warning("KMZ file not found. Proceeding without spatial enrichment.")
        kmz_index = None
    
    # Process each CSV
    parsed_csvs = []
    for csv_path in csv_files:
        try:
            logger.info(f"Processing: {csv_path}")
            
            # Parse and clean CSV
            cleaned_df, project_name = parse_raw_csv(csv_path)
            source_info = infer_source_info(csv_path)
            cleaned_df = attach_source_metadata(cleaned_df, csv_path, project_name)
            
            parsed_csv = ParsedCSV(
                df=cleaned_df,
                source_info=source_info,
                project_name=project_name
            )
            
            # Enrich with KMZ if available
            if kmz_index:
                parsed_csv = enrich_with_kmz(parsed_csv, kmz_index)
            
            parsed_csvs.append(parsed_csv)
            
        except Exception as e:
            logger.error(f"Failed to process {csv_path}: {e}")
    
    if not parsed_csvs:
        logger.error("No CSVs were successfully processed.")
        return
    
    # Combine all CSVs
    logger.info("Combining CSV data...")
    combined_df, boundaries = combine(parsed_csvs)
    
    # Write output
    out_path = derive_output_path(input_dirs, preferred_name="Combined_Extracted.xlsx")
    logger.info(f"Writing output to: {out_path}")
    
    separator_style = SeparatorStyle()  # Use default black separator
    write_with_separators(combined_df, boundaries, out_path, separator_style)
    
    logger.info("Processing complete!")
```

## TASK 7: Testing and Validation

### Implementation Steps:

1. **Test with Example Data**:
   - Use `.codex/test-extract/raw-csv/AL-Flight 10.CSV`
   - Use `.codex/test-extract/test-kmz/example-polyline.kmz`
   - Verify output matches `.codex/test-extract/desired-fields/desired-csv-example.csv`

2. **Validation Checklist**:
   - ✅ CSV headers correctly fixed (PPM header added to column L)
   - ✅ Only 6 target columns extracted (A, C, D, E, L, N)
   - ✅ Data types properly converted
   - ✅ KMZ attributes successfully extracted and matched
   - ✅ XLSX output includes black separator rows
   - ✅ Source file tracking works correctly
   - ✅ Performance acceptable for large datasets

3. **Error Handling**:
   - Invalid CSV structures
   - Missing KMZ files  
   - Coordinate validation failures
   - File I/O errors

## Configuration Requirements

Create these configuration constants:

```python
# Distance threshold for KMZ matching (meters)
DEFAULT_KMZ_DISTANCE_THRESHOLD = 100.0

# KMZ file path (make configurable)
DEFAULT_KMZ_PATH = "pipeline_data.kmz"

# Output preferences
OUTPUT_FORMAT = "xlsx"  # Only XLSX as requested
INCLUDE_CSV_OUTPUT = False
```

## Success Criteria

The implementation is complete when:

1. ✅ Raw CSVs with header issues are correctly parsed and cleaned
2. ✅ Only the 6 specified columns are extracted
3. ✅ KMZ spatial enrichment adds 7 new columns with pipeline data
4. ✅ Combined XLSX output has black separator rows between CSV groups
5. ✅ Source file tracking enables traceability
6. ✅ Performance handles multiple large CSV files efficiently
7. ✅ Error handling is robust for edge cases

Follow these tasks sequentially, testing each component before proceeding to the next. The modular design allows for independent development and testing of each component.