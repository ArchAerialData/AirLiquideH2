# KMZ Field Extraction Documentation

This document explains how to extract structured data from KMZ/KML files for spatial enrichment of CSV flight data.

## KMZ File Structure

KMZ files are ZIP archives containing KML (Keyhole Markup Language) files. The spatial data and attributes are embedded within the KML structure.

## Extraction Process

### Step 1: Extract KML from KMZ Archive

```bash
# KMZ is a ZIP file containing doc.kml
unzip -o "example-polyline.kmz"
# This creates: doc.kml
```

### Step 2: Parse KML Structure

The KML contains XML with the following key elements:

```xml
<Document>
    <Placemark id="ID_00000">
        <name>100272 13239.57 - 14681.05</name>
        <description><![CDATA[...HTML table with attributes...]]></description>
        <MultiGeometry>
            <LineString>
                <coordinates>lon,lat,elev lon,lat,elev ...</coordinates>
            </LineString>
        </MultiGeometry>
    </Placemark>
</Document>
```

### Step 3: Extract Attribute Data from HTML Description

The `<description>` element contains an HTML table with structured attributes:

```html
<table>
    <tr><td>FID</td><td>0</td></tr>
    <tr><td>BeginMeasu</td><td>13239.57</td></tr>
    <tr><td>EndMeasure</td><td>14681.05</td></tr>
    <tr><td>Route_Name</td><td>100272</td></tr>
    <tr><td>Route_Desc</td><td>Bayport to Webster</td></tr>
    <tr><td>Class_Loca</td><td>3</td></tr>
    <tr><td>Diameter</td><td>14</td></tr>
    <tr><td>Product</td><td>H</td></tr>
</table>
```

## Available Fields from Example KMZ

Based on analysis of `example-polyline.kmz`:

| Field Name | Example Value | Data Type | Description |
|------------|---------------|-----------|-------------|
| **FID** | `0` | Integer | Feature ID |
| **BeginMeasu** | `13239.57` | Float | Begin Measure (distance/stationing) |
| **EndMeasure** | `14681.05` | Float | End Measure (distance/stationing) |
| **Route_Name** | `100272` | String | Route identifier/number |
| **Route_Desc** | `Bayport to Webster` | String | Human-readable route description |
| **Class_Loca** | `3` | Integer | Class Location (regulatory classification) |
| **Diameter** | `14` | Integer | Pipeline diameter (inches) |
| **Product** | `H` | String | Product type (H = Hydrogen) |

## Geometric Data

- **Coordinates**: Array of `longitude,latitude,elevation` points defining the polyline
- **Example coordinate**: `-95.08383498999994,29.60802714000005,14.63920000000508`
- **Coordinate count**: 32 points in the example polyline

## Implementation Strategy

### 1. KML Parsing (using fastkml)
```python
from fastkml import kml
from shapely.geometry import LineString

# Parse KML and extract geometries
k = kml.KML()
k.from_string(kml_content)
features = list(k.features())
```

### 2. HTML Table Parsing (using BeautifulSoup)
```python
from bs4 import BeautifulSoup

# Extract attributes from HTML description
soup = BeautifulSoup(description_html, 'html.parser')
rows = soup.find_all('tr')
attributes = {}
for row in rows:
    cells = row.find_all('td')
    if len(cells) == 2:
        attributes[cells[0].text.strip()] = cells[1].text.strip()
```

### 3. Spatial Index (using rtree)
```python
from rtree import index

# Build spatial index for fast proximity queries
idx = index.Index()
for i, geometry in enumerate(geometries):
    idx.insert(i, geometry.bounds)
```

## Integration with CSV Processing

### Workflow:
1. **Load KMZ**: Parse all polylines and build spatial index
2. **Process CSV**: For each row with lat/lon coordinates:
   - Find nearest polyline using spatial index
   - Extract polyline attributes (Route_Name, Route_Desc, etc.)
   - Append attributes as new columns to CSV row
3. **Output**: Combined XLSX with original CSV data + KMZ-enriched fields

### New Columns Added to Output:
- `KMZ_Route_Name`: Pipeline route identifier
- `KMZ_Route_Desc`: Route description  
- `KMZ_Diameter`: Pipeline diameter
- `KMZ_Product`: Product type
- `KMZ_Class_Loca`: Class location
- `KMZ_BeginMeasu`: Begin measure
- `KMZ_EndMeasure`: End measure
- `KMZ_Distance_Meters`: Distance from point to nearest polyline

## Notes for Implementation

1. **Multiple Polylines**: A single KMZ may contain multiple `<Placemark>` elements
2. **Proximity Threshold**: Consider setting a maximum distance threshold (e.g., 100m) for valid matches
3. **Performance**: Use spatial indexing (rtree) for large KMZ files with many polylines
4. **Error Handling**: Handle cases where no nearby polylines are found
5. **Coordinate Systems**: Ensure consistent coordinate reference system (WGS84)

## Dependencies Required

```bash
pip install fastkml shapely beautifulsoup4 lxml rtree
```

## Future Enhancements

- Support for multiple KMZ files
- Configurable proximity thresholds
- Alternative spatial matching algorithms (buffer zones, etc.)
- Support for polygon geometries in addition to polylines