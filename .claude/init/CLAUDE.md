# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Air Liquide H2 CSV processing project - a modular Python application designed to merge multiple flight CSV files with spatial enrichment capabilities. The project is currently in skeleton form with intentional TODO markers throughout the codebase.

## Key Commands

Since this is a skeleton project with no dependency management files, install required packages manually:

```bash
# Core dependencies
pip install pandas openpyxl

# Optional (for future KMZ functionality)
pip install pyyaml fastkml shapely Rtree
```

**Run the application:**
```bash
# Drag and drop folders onto main.py (Windows)
# Or run directly:
python main.py "path/to/folder"

# For multiple folders:
python main.py "folder1" "folder2"
```

**Testing:**
No test framework is currently configured. Tests should be added before completing the skeleton.

## Architecture

### Entry Point
- `main.py`: Single entry point that orchestrates the entire pipeline but currently only shows what it would do

### Core Modules (in `modules/` directory)

**Configuration & Schema:**
- `config.py`: Dataclass-based configuration with YAML support
- `csv_schema.py`: Defines column extraction rules and source metadata tracking

**Data Processing Pipeline:**
1. `io_utils.py`: Input path resolution and output path derivation  
2. `file_discovery.py`: Recursive CSV file discovery
3. `csv_parser.py`: Individual CSV parsing with metadata attachment
4. `aggregator.py`: Combines parsed CSVs while tracking group boundaries
5. `xlsx_writer.py`: XLSX output with black separator rows between groups

**Future Components:**
- `kmz_lookup.py`: Spatial enrichment (placeholder - not implemented)
- `csv_writer.py`: Alternative CSV output format
- `validators.py`: Data validation utilities
- `logging_utils.py`: Logging configuration

### Data Flow

```
Input Folders → CSV Discovery → Parse Individual CSVs → Aggregate with Boundaries → Write XLSX with Separators
```

Each CSV gets source metadata (date folder, flight folder, filename) automatically inferred from the path structure.

## Import Structure

The project uses relative imports with the pattern `from .module_name import ...` within the modules package. The main entry point imports from `modules.*`.

## Current State

This is an intentional skeleton. Key areas needing implementation:
- Column extraction rules in `csv_schema.py`
- Actual CSV parsing logic in `csv_parser.py` 
- KMZ spatial lookup functionality in `kmz_lookup.py`
- Data validation in `validators.py`

The application currently only discovers files and shows where output would be written - no actual processing occurs until TODOs are completed.