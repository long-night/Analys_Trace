# AGENTS.md - OpenCode Agent Instructions

## Project Overview

This repository contains a standalone Python script for analyzing PyTorch profiler Chrome trace JSON files. It extracts CPU operator performance statistics (execution time, shapes, strides) and outputs structured reports.

**Tech Stack**: Python 3.12, standard library + optional `openpyxl` for Excel export  
**Domain**: PyTorch performance profiling / trace analysis  
**Language**: Chinese UI and comments, English code

## Repository Structure

```
Analys_Trace/
├── analys_trace_v4.py          # Main analysis script (standalone, no package)
├── profiler_trace.json         # Small example trace (~7MB, for testing)
├── George_*.pt.trace.json      # Full-size trace files (150MB+ each)
└── .git/
```

**No formal project structure**: no `pyproject.toml`, no `requirements.txt`, no tests, no CI/CD.

## Usage

### Basic Run
```bash
# Analyze a trace file (auto-detects Excel vs CSV based on openpyxl availability)
python analys_trace_v4.py profiler_trace.json

# Force CSV output
python analys_trace_v4.py profiler_trace.json --csv

# Force Excel output (requires openpyxl)
python analys_trace_v4.py profiler_trace.json --excel -o report.xlsx

# Skip summary display
python analys_trace_v4.py profiler_trace.json --no-summary
```

### Output Files
- **Excel mode** (default if `openpyxl` installed): `operator_analysis.xlsx` with two sheets:
  - `算子总表` — Aggregated operator stats (name, shapes, strides, call count, total/avg time)
  - `Shape统计表` — Per-shape breakdown (min/max/avg time per shape variant)
- **CSV mode** (fallback): Two separate files
  - `cpu_operators.csv` — Operator summary
  - `shape_statistics.csv` — Shape-level stats

### Installing Optional Dependency
```bash
pip install openpyxl
```

## Key Technical Conventions

### Time Units
- **Internal storage**: microseconds (from JSON `ts`/`dur` fields)
- **Output display**: milliseconds (divide by 1000)

### Data Format
Input is Chrome trace JSON from `torch.profiler`, containing `traceEvents` array with:
- `ph`: Phase — `"B"` (begin), `"E"` (end), or `"X"` (complete event)
- `cat`: Category — CPU ops have `cpu_op` or contain `"cpu"`
- `name`: Operator name — filters for `aten::` or `torch::` prefix
- `args`: Contains `Input Dims`, `Input Strides`, `input_dims`, `input_strides`

### Operator Matching
Only CPU operators are analyzed. Matching logic:
```python
name.startswith(('aten::', 'torch::')) and 'cpu' in cat.lower()
```

## Development Notes

- **Single-file script**: `analys_trace_v4.py` is self-contained. Edit directly.
- **No tests**: Verify by running against `profiler_trace.json` and inspecting output.
- **Large files**: Trace files are 100MB–200MB. Don't load entire files into memory unnecessarily (current implementation does `json.load()` — acceptable for this scale).
- **Encoding**: Output files use UTF-8 with Chinese headers.

## Agent Guidance

- When adding features, maintain Chinese UI strings and comments.
- Prefer standard library; only add dependencies if essential.
- If modifying output format, update both Excel (`export_to_excel`) and CSV (`export_to_csv`) paths.
- Run `python analys_trace_v4.py profiler_trace.json` to verify changes.
