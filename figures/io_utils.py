"""Small, generic utilities shared by the figures scripts: output-directory layout."""

from pathlib import Path

# Every script's output, grouped by file type under figures/output/ so a directory
# listing groups like with like instead of mixing csv/png/pdf/json/txt together.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CSV_DIR = OUTPUT_DIR / "csv"
PNG_DIR = OUTPUT_DIR / "png"
PDF_DIR = OUTPUT_DIR / "pdf"
JSON_DIR = OUTPUT_DIR / "json"
TXT_DIR = OUTPUT_DIR / "txt"
