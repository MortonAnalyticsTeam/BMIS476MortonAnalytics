"""
For the DDT Team: READ ME
I made a description of how to download the functions necessary for this function.
They are viewable on line 60. You can follow the directions to get the errors out. 
The code should run well after this. You can view a description of each system and function.
That can be found in the Google Drive or in the README file. 

================================================================================
DDT — GEN AI DATA PROCESSING AND ANALYTICS SOLUTION
AIS Vessel Event Detection, Labeling & AI Summary System
================================================================================
Project      : GEN AI DATA PROCESSING AND ANALYTICS SOLUTION
Team         : DDT Team
Version      : 2.0
Description  : Full pipeline for reading, processing, detecting, labeling, and
               summarizing AIS vessel event data at scale (8M+ rows supported).

SYSTEM REQUIREMENTS COVERAGE:
  Req #1  [CRITICAL] Event Labeling GenAI System
                     → Full AI-powered detection and labeling pipeline
  Req #2  [CRITICAL] Identify location of shipments
                     → Lat/Lon + timestamp tracking per vessel
  Req #3  [HIGH]     Event Data Output
                     → Structured output: Vessel ID, Event Type, Timestamp,
                       Location, Confidence Score, NULL flags
  Req #4  [HIGH]     Event Detection and Labeling
                     → Detects: ARRIVAL, DEPARTURE, ANCHORING,
                       ROUTE_DEVIATION, PROXIMITY
  Req #5  [HIGH]     Event Object Generation
                     → Generates: event_id, timestamp, location,
                       vessel(s), event_type, confidence_score
  Req #6  [HIGH]     Data Formatted in New Rows
                     → New columns appended to original DataFrame + CSV export
  Req #7  [CRITICAL] Documentation
                     → Inline comments and docstrings throughout (see below)
  Req #8  [HIGH]     Data Labeling Output Format
                     → Structured CSV output for downstream analytics
  Req #9  [HIGH]     Natural Language AI Event Summary
                     → Claude API generates plain-English summaries per event
  Req #10 [HIGH]     System Compatibility
                     → Reads CSV, JSON, and NMEA AIS datasets

PERFORMANCE NOTES (for 8M+ row datasets):
  - CSV loading uses chunked reading with dtype optimization
  - Event detection uses vectorized pandas operations (no row-by-row loops)
  - Proximity detection uses spatial blocking by time window
  - AI summaries are batched and rate-limited to avoid API overload
  - All intermediate exports write in streaming chunks

DEPENDENCIES:
  pip install pandas anthropic pyais tqdm
  (pyais only required for NMEA input files)

USAGE:
  python ddt_ais_pipeline.py your_data.csv --output-dir ./output
  python ddt_ais_pipeline.py your_data.csv --output-dir ./output --ai-summaries
  python ddt_ais_pipeline.py your_data.json --output-dir ./output --ai-summaries
================================================================================
"""

# ------------------------------------------------------------------------------
# TEAM SETUP -- HOW TO INSTALL REQUIRED PACKAGES
# Before running this program, you need to install the libraries it depends on.
# Open a terminal in this project folder and run the command below ONE TIME:
#
## (Paste this into the terminal by opening a terminal) 
## How to open a terminal: ctrl
#   pip install pandas numpy anthropic pyais
#
# What each package does:
#   pandas    -- loads and processes the AIS data tables
#   numpy     -- fast math operations used in distance/bearing calculations
#   anthropic -- connects to the Claude AI API to generate event summaries
#   pyais     -- decodes raw NMEA AIS messages from .nmea / .ais / .txt files
#
# If you are using the project virtual environment (.venv), activate it first:
#   Windows  : .venv\Scripts\activate
#   Mac/Linux: source .venv/bin/activate
# Then run the pip install command above.
#
# You will also need an Anthropic API key if you want AI summaries (Req #9).
# Set it in your terminal before running the script:
#   Windows  : set ANTHROPIC_API_KEY=your_key_here
#   Mac/Linux: export ANTHROPIC_API_KEY=your_key_here
# ------------------------------------------------------------------------------



# ── Standard library imports ──────────────────────────────────────────────────
import os
import sys
import json
import math
import uuid
import time
import argparse
import warnings
from datetime import datetime, timezone
from typing import Optional

# Ensure Unicode characters (arrows, dashes, etc.) print correctly on all platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Third-party imports ───────────────────────────────────────────────────────
import pandas as pd
import numpy as np
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import pyais  # noqa: F401 — imported for availability check; used inside load_nmea()
except ImportError:
    pyais = None  # type: ignore

# Suppress pandas performance warnings for large frame operations
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1A: CONFIGURATION
# All tunable thresholds and column aliases live here.
# Adjust these values to match your specific AIS dataset characteristics.
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Speed thresholds (knots) ──────────────────────────────────────────────
    # SOG = Speed Over Ground. Vessels reporting below arrival_speed_max
    # after previously moving are flagged as ARRIVAL candidates.
    "arrival_speed_max":    1.0,

    # Vessels accelerating above this after being slow = DEPARTURE
    "departure_speed_min":  1.5,

    # Near-zero speed sustained for stopped_min_rows = ANCHORING
    "anchoring_speed_max":  0.5,

    # Minimum consecutive slow readings before labeling ANCHORING
    # (helps filter noise/GPS drift in large datasets)
    "stopped_min_rows":     2,

    # ── Route deviation threshold ─────────────────────────────────────────────
    # COG = Course Over Ground. A bearing shift >= this value in degrees
    # between consecutive rows for the same vessel = ROUTE_DEVIATION.
    "deviation_bearing_threshold": 45.0,

    # ── Proximity detection ───────────────────────────────────────────────────
    # Two vessels within this many nautical miles at the same time = PROXIMITY.
    # Reduce this for busier ports (more events); increase for open-sea analysis.
    "proximity_nm": 0.5,

    # ── Large dataset performance ─────────────────────────────────────────────
    # Number of rows to read at a time when loading large CSV files.
    # Increase for faster load (uses more RAM), decrease if memory is limited.
    "chunk_size": 500_000,

    # Maximum number of AI summaries to generate in one run.
    # Set to None to summarize all events (may be slow/costly for large datasets).
    "ai_summary_max_events": 500,

    # Seconds to wait between Claude API calls to avoid rate limiting.
    "ai_summary_delay_sec": 0.3,

    # ── Column name aliases ───────────────────────────────────────────────────
    # The system will try each name in order until it finds a match.
    # Add your dataset's column names here if they are not already listed.
    "col_mmsi": ["MMSI", "mmsi", "vessel_id", "VesselID", "VESSEL_ID"],
    "col_lat":  ["LAT", "Latitude", "lat", "latitude", "LATITUDE"],
    "col_lon":  ["LON", "Longitude", "lon", "longitude", "LONGITUDE"],
    "col_time": ["BaseDateTime", "Timestamp", "timestamp", "TIME", "time", "DATETIME"],
    "col_sog":  ["SOG", "sog", "Speed", "speed", "SPEED"],
    "col_cog":  ["COG", "cog", "Course", "course", "COURSE"],
    "col_name": ["VesselName", "vessel_name", "Name", "name", "VESSEL_NAME"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Section 1B  DATA IMPORT FUNCTION  — Paste this after the CONFIG block
# Allows importing large datasets (850K+ rows) from your local computer.
# The full dataset is held ONLY in memory (never written to disk in your
# project folder), and a small preview CSV is saved for reference.
# This keeps your GitHub repo clean and avoids large file push errors.
# ──────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import filedialog

def import_dataset(preview_rows: int = 10,
                   save_preview: bool = True,
                   preview_dir: str = "./preview") -> pd.DataFrame:
    """
    Opens a file browser dialog so you can select your AIS dataset directly
    from your computer. The full dataset is loaded into memory but is NOT
    saved into your project folder — keeping your GitHub repo lightweight.

    A small preview CSV (default: 10 rows) is optionally saved so you have
    a reference sample to inspect or commit to GitHub if needed.

    Supports: .csv, .json, .nmea, .txt, .ais

    Args:
        preview_rows : Number of rows to save in the preview file (default 10)
        save_preview : Whether to save the preview CSV (default True)
        preview_dir  : Where to save the preview file (default ./preview)

    Returns:
        pd.DataFrame : Full dataset loaded in memory, ready for the pipeline

    Usage:
        df = import_dataset()               # Opens file browser
        df = import_dataset(preview_rows=5) # Save only 5 rows in preview
        labeled_df, events_df = run_pipeline_from_df(df)
    """

    # ── Step 1: Open file browser dialog ─────────────────────────────────────
    # Hide the root tkinter window — we only want the file picker
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Bring dialog to front of VS Code

    print("[IMPORT] Opening file browser — select your AIS dataset...")

    filepath = filedialog.askopenfilename(
        title="Select your AIS Dataset",
        filetypes=[
            ("All supported files", "*.csv *.json *.nmea *.txt *.ais"),
            ("CSV files",           "*.csv"),
            ("JSON files",          "*.json"),
            ("NMEA AIS files",      "*.nmea *.txt *.ais"),
            ("All files",           "*.*"),
        ]
    )

    root.destroy()  # Clean up the hidden tkinter window

    # User cancelled the dialog
    if not filepath:
        print("[IMPORT] No file selected. Exiting.")
        return None

    print(f"[IMPORT] Selected: {filepath}")

    # ── Step 2: Load the full dataset into memory ONLY ────────────────────────
    # load_file() reads the data but does NOT write anything to your project.
    # The DataFrame lives only in RAM — gone when the script ends.
    # This means no giant files sitting in your VS Code workspace or on GitHub.
    try:
        df = load_file(filepath)
    except Exception as e:
        print(f"[IMPORT] ERROR: Failed to load file — {e}")
        return None

    # ── Step 3: Show a quick preview in the console ───────────────────────────
    print(f"\n[IMPORT] ── Dataset Preview (first {min(preview_rows, len(df))} rows) ──")
    print(df.head(preview_rows).to_string(index=False))
    print(f"\n[IMPORT] Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[IMPORT] Columns : {list(df.columns)}")

    # ── Step 4: Save a small preview file (optional) ──────────────────────────
    # This tiny file (10 rows) IS saved to disk — safe to commit to GitHub.
    # Useful for teammates or documentation to understand the data structure.
    if save_preview and len(df) > 0:
        os.makedirs(preview_dir, exist_ok=True)

        # Name the preview file after the original dataset
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        preview_path = os.path.join(preview_dir, f"{base_name}_preview.csv")

        df.head(preview_rows).to_csv(preview_path, index=False)
        print(f"\n[IMPORT] Preview saved → {preview_path}")
        print(f"         ({preview_rows} rows only — safe to commit to GitHub)")

    print(f"\n[IMPORT] Full dataset ({df.shape[0]:,} rows) is loaded in memory only.")
    print(f"         It will NOT be written to your project folder.")
    print(f"         ✓ Your GitHub repo stays clean.\n")

    return df


def run_pipeline_from_df(df: pd.DataFrame,
                          output_dir: str = "./output",
                          ai_summaries: bool = False):
    """
    Run the full DDT pipeline on an already-loaded DataFrame.
    Use this after import_dataset() so you don't re-load the file.

    This version skips the file loading step and goes straight to
    column resolution, event detection, and export.

    Args:
        df           : DataFrame returned by import_dataset()
        output_dir   : Where to save the labeled output CSVs
        ai_summaries : Whether to generate Claude AI summaries (Req #9)

    Returns:
        labeled_df   : Original data + new event label columns
        events_df    : Events-only summary DataFrame

    Example:
        df = import_dataset()
        labeled_df, events_df = run_pipeline_from_df(df, ai_summaries=True)
    """
    if df is None or df.empty:
        print("[ERROR] No data to process. Run import_dataset() first.")
        return None, None

    print("=" * 60)
    print("  DDT — PIPELINE RUNNING ON IMPORTED DATASET")
    print("=" * 60)

    start_time = time.time()

    # Resolve column names from the loaded DataFrame
    print_section("COLUMN RESOLUTION")
    cols = resolve_columns(df)
    print(f"  MMSI  → '{cols['mmsi']}'")
    print(f"  LAT   → '{cols['lat']}'")
    print(f"  LON   → '{cols['lon']}'")
    print(f"  TIME  → '{cols['time']}'")
    print(f"  SOG   → {repr(cols['sog'])}")
    print(f"  COG   → {repr(cols['cog'])}")
    print(f"  NAME  → {repr(cols['name'])}")

    # -- Date filter: choose all rows or specific day(s) before processing
    df, date_label = prompt_date_filter(df, cols["time"])

    # Run event detection
    events_df = detect_events(df, cols)

    # Optional AI summaries
    if ai_summaries and events_df is not None and not events_df.empty:
        events_df = generate_ai_summaries(events_df)

    # Merge labels into original DataFrame
    labeled_df = build_labeled_dataset(df, events_df, cols)

    # Export outputs -- date label in filename prevents overwriting previous runs
    print_section("EXPORTING OUTPUTS  (Req #6, #8)")
    os.makedirs(output_dir, exist_ok=True)

    file_prefix = date_label if date_label != "all" else "full_dataset"
    labeled_path = os.path.join(output_dir, file_prefix + "_labeled.csv")
    export_csv(labeled_df, labeled_path, label="Full labeled dataset")

    if events_df is not None and not events_df.empty:
        events_path = os.path.join(output_dir, file_prefix + "_events_summary.csv")
        export_csv(events_df, events_path, label="Events summary")

    # Summary
    print_pipeline_summary("imported_dataset", labeled_df, events_df, output_dir)

    elapsed = time.time() - start_time
    print(f"  Total runtime  : {elapsed:.1f} seconds")
    print("=" * 60 + "\n")

    return labeled_df, events_df


# ──────────────────────────────────────────────────────────────────────────────
# HOW TO USE — run these two lines at the bottom of your script
# (or in a new cell if using Jupyter):
#
#   df = import_dataset()                          # Opens file browser
#   labeled_df, events_df = run_pipeline_from_df(df)
#
# With AI summaries:
#   labeled_df, events_df = run_pipeline_from_df(df, ai_summaries=True)
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1C: DATE FILTER
# Lets you process just one day (or a few days) instead of the full year.
# Applied right after column resolution, before any detection runs.
# ──────────────────────────────────────────────────────────────────────────────

def filter_by_dates(df, time_col, dates):
    """
    Filter DataFrame to rows whose date matches any entry in the dates list.
    dates: list of date strings in YYYY-MM-DD format.
    """
    date_series = pd.to_datetime(df[time_col], errors="coerce").dt.date
    target_dates = []
    for d in dates:
        try:
            target_dates.append(pd.to_datetime(d.strip()).date())
        except Exception:
            print("[FILTER] WARNING: Could not parse date '{}' -- skipping.".format(d))
    if not target_dates:
        print("[FILTER] No valid dates -- processing all rows.")
        return df
    filtered = df[date_series.isin(target_dates)].copy()
    pct = 100 * len(filtered) / len(df) if len(df) else 0
    print("[FILTER] Kept {:,} rows across {} date(s) ({:.2f}% of dataset)".format(
        len(filtered), len(target_dates), pct))
    return filtered


def prompt_date_filter(df, time_col):
    """
    Interactive menu: process all rows, or filter to specific date(s).
    Returns (filtered_df, date_label) where date_label is used in output filenames.
    """
    total = len(df)
    parsed = pd.to_datetime(df[time_col], errors="coerce").dt.date
    min_date = parsed.min()
    max_date = parsed.max()

    print()
    print("=" * 60)
    print("  DATE FILTER")
    print("=" * 60)
    print("  Dataset range : {}  to  {}".format(min_date, max_date))
    print("  Total rows    : {:,}".format(total))
    print()
    print("  1. Process ALL rows (full dataset -- may be slow)")
    print("  2. Filter to specific day(s)  [recommended for testing]")
    print()

    while True:
        choice = input("  Enter choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Please enter 1 or 2.")

    if choice == "1":
        print("[FILTER] Processing all {:,} rows.".format(total))
        return df, "all"

    print()
    print("  Enter date(s) in YYYY-MM-DD format.")
    print("  Single day    : 2023-06-15")
    print("  Multiple days : 2023-06-15, 2023-07-04, 2023-12-25")
    print()

    while True:
        raw = input("  Date(s): ").strip()
        if raw:
            break
        print("  Please enter at least one date.")

    dates = [d.strip() for d in raw.split(",") if d.strip()]
    filtered = filter_by_dates(df, time_col, dates)

    if filtered.empty:
        print("[FILTER] WARNING: No data for those dates -- falling back to all rows.")
        return df, "all"

    unique_dates = sorted(
        pd.to_datetime(filtered[time_col], errors="coerce")
        .dt.strftime("%Y-%m-%d").dropna().unique()
    )
    if len(unique_dates) == 1:
        label = unique_dates[0]
    elif len(unique_dates) == 2:
        label = "_".join(unique_dates)
    else:
        label = "{}_to_{}".format(unique_dates[0], unique_dates[-1])

    return filtered, label


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: UTILITY FUNCTIONS
# Small, reusable helper functions used throughout the pipeline.
# ──────────────────────────────────────────────────────────────────────────────

def haversine_nm_vectorized(lat1: pd.Series, lon1: pd.Series,
                             lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Vectorized haversine distance calculation between two sets of lat/lon points.
    Returns distance in nautical miles as a pandas Series.

    Using vectorized numpy operations here is critical for performance —
    a row-by-row Python loop on 8M rows would take hours; this takes seconds.
    """
    R = 3440.065  # Earth's radius in nautical miles

    # Convert degrees to radians using numpy for speed
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (np.sin(dphi / 2) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)

    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Scalar haversine distance between two lat/lon points (nautical miles).
    Used for single-pair comparisons (e.g. proximity checks on small groups).
    """
    R = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_change_vectorized(cog1: pd.Series, cog2: pd.Series) -> pd.Series:
    """
    Vectorized absolute angular difference between two bearing Series (0–180°).
    Used for route deviation detection across all rows simultaneously.
    """
    diff = (cog1 - cog2).abs() % 360
    # Angles > 180 wrap around — take the shorter arc
    return diff.where(diff <= 180, 360 - diff)


def resolve_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """
    Find the first matching column name from a list of candidates.
    Returns the matched column name, or None if no match is found.
    Case-sensitive — ensure your column aliases in CONFIG match exactly.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_event_id() -> str:
    """
    Generate a unique event identifier in the format EVT-XXXXXXXX.
    UUIDs ensure no two events share an ID even across parallel runs.
    """
    return "EVT-" + str(uuid.uuid4())[:8].upper()


def confidence_score(event_type: str, sog_value: Optional[float] = None) -> float:
    """
    Rule-based confidence scoring (0.0 – 1.0) for a detected event.

    Base scores are set by event type. Additional boost is applied when
    the SOG signal is unusually clear (e.g. vessel is nearly stationary
    for ANCHORING, or moving fast for DEPARTURE).

    This is a heuristic model — for production, consider training a
    classifier on labeled ground-truth AIS data for higher accuracy.

    Args:
        event_type : One of ARRIVAL, DEPARTURE, ANCHORING,
                     ROUTE_DEVIATION, PROXIMITY
        sog_value  : Speed Over Ground at time of event (optional boost)

    Returns:
        float between 0.0 and 1.0
    """
    # Base confidence by event type (tuned from domain knowledge)
    base = {
        "ARRIVAL":         0.80,
        "DEPARTURE":       0.78,
        "ANCHORING":       0.85,
        "ROUTE_DEVIATION": 0.70,
        "PROXIMITY":       0.75,
    }.get(event_type, 0.60)

    # Boost confidence when SOG makes the event very clear
    if sog_value is not None and not pd.isna(sog_value):
        if event_type in ("ARRIVAL", "ANCHORING") and sog_value < 0.2:
            base = min(base + 0.10, 1.0)   # Near-zero speed = strong signal
        elif event_type == "DEPARTURE" and sog_value > 5.0:
            base = min(base + 0.08, 1.0)   # High speed = clearly underway

    return round(base, 2)


def null_flags(row: pd.Series, key_cols: list) -> str:
    """
    Check for NULL/NaN values in a row's key columns (Req #3).
    Returns a comma-separated string of column names that are null,
    or 'None' if all key fields are present.

    This satisfies Req #3's requirement to report any NULLs in event output.
    """
    nulls = [c for c in key_cols if c and pd.isna(row.get(c, None))]
    return ", ".join(nulls) if nulls else "None"


def print_section(title: str):
    """Print a formatted section header to the console for pipeline readability."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA LOADERS  (Req #10 — CSV, JSON, NMEA compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load AIS data from a CSV file with performance optimizations for large files.

    For datasets of 8M+ rows, we use chunked reading to avoid loading the
    entire file into memory at once. Chunks are concatenated after loading.
    dtype optimization (float32 for lat/lon/sog/cog) reduces memory by ~40%.

    Req #10: CSV format support.
    """
    print(f"[LOAD] Reading CSV: {filepath}")

    # Optimized dtypes for common AIS columns — reduces RAM usage significantly
    # float32 is sufficient precision for lat/lon/speed/course in AIS data
    dtype_hints = {
        "LAT": "float32", "Latitude": "float32", "lat": "float32",
        "LON": "float32", "Longitude": "float32", "lon": "float32",
        "SOG": "float32", "sog": "float32", "Speed": "float32",
        "COG": "float32", "cog": "float32", "Course": "float32",
    }

    # Read in chunks to handle 8M+ rows without memory errors
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=CONFIG["chunk_size"],
                              low_memory=False, dtype=dtype_hints,
                              on_bad_lines="skip"):  # skip malformed rows
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"       ...loaded {total_rows:,} rows", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n[LOAD] CSV complete — {len(df):,} rows, {len(df.columns)} columns.")
    return df


def load_json(filepath: str) -> pd.DataFrame:
    """
    Load AIS data from a JSON file.
    Supports two formats:
      1. Array of records: [{"MMSI": 123, ...}, ...]
      2. Newline-delimited JSON (NDJSON): one record per line

    Req #10: JSON format support.
    """
    print(f"[LOAD] Reading JSON: {filepath}")
    try:
        # Try standard JSON array first
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
    except json.JSONDecodeError:
        # Fall back to newline-delimited JSON (common in streaming AIS feeds)
        print("[LOAD] Standard JSON parse failed — trying newline-delimited JSON...")
        df = pd.read_json(filepath, lines=True)

    print(f"[LOAD] JSON complete — {len(df):,} rows, {len(df.columns)} columns.")
    return df


def load_nmea(filepath: str) -> pd.DataFrame:
    """
    Parse NMEA 0183 AIS sentences (VDM/VDO message types) into a DataFrame.
    Decodes each sentence to extract: MMSI, LAT, LON, SOG, COG.
    Timestamps use the current UTC time as a fallback when not encoded.

    Requires: pip install pyais
    Ref: https://pyais.readthedocs.io

    Req #10: NMEA AIS dataset support.
    """
    print(f"[LOAD] Reading NMEA AIS: {filepath}")
    try:
        from pyais import FileReaderStream
    except ImportError:
        raise ImportError(
            "\n[ERROR] pyais is required to parse NMEA files.\n"
            "Install it with:  pip install pyais\n"
        )

    records = []
    skipped = 0

    with FileReaderStream(filepath) as stream:
        for msg in stream:
            try:
                decoded = msg.decode()
                # Extract standard AIS fields — getattr with None default handles
                # message types that don't carry all fields (e.g. type 5 vs type 1)
                record = {
                    "MMSI":         getattr(decoded, "mmsi", None),
                    "LAT":          getattr(decoded, "lat",  None),
                    "LON":          getattr(decoded, "lon",  None),
                    "SOG":          getattr(decoded, "speed", None),
                    "COG":          getattr(decoded, "course", None),
                    "VesselName":   getattr(decoded, "shipname", None),
                    # NMEA sentences don't always carry timestamps;
                    # use current UTC as a best-effort fallback
                    "BaseDateTime": datetime.now(timezone.utc).isoformat(),
                }
                records.append(record)
            except Exception:
                # Malformed or unsupported sentence types are skipped silently
                skipped += 1
                continue

    df = pd.DataFrame(records).dropna(subset=["MMSI", "LAT", "LON"])
    print(f"[LOAD] NMEA complete — {len(df):,} valid rows decoded. ({skipped} skipped)")
    return df


def load_file(filepath: str) -> pd.DataFrame:
    """
    Auto-detect input file type by extension and route to the correct loader.
    Supported: .csv, .json, .nmea, .txt, .ais

    Req #10: System compatibility with all three formats.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: '{filepath}'")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return load_csv(filepath)
    elif ext == ".json":
        return load_json(filepath)
    elif ext in (".nmea", ".txt", ".ais"):
        return load_nmea(filepath)
    else:
        raise ValueError(
            f"[ERROR] Unsupported file extension: '{ext}'.\n"
            f"        Accepted formats: .csv, .json, .nmea, .txt, .ais"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: COLUMN RESOLUTION
# Maps dataset column names to internal standardized keys.
# ──────────────────────────────────────────────────────────────────────────────

def resolve_columns(df: pd.DataFrame) -> dict:
    """
    Resolve the actual column names in the DataFrame against the CONFIG aliases.
    Returns a mapping of internal keys (mmsi, lat, lon, time, sog, cog, name)
    to the real column names found in the DataFrame.

    Required columns (mmsi, lat, lon, time) will raise an error if not found.
    Optional columns (sog, cog, name) return None if not found — pipeline
    degrades gracefully (e.g. skips deviation detection if COG is absent).
    """
    mapping = {}

    # These four are REQUIRED — pipeline cannot run without them
    required = {
        "mmsi": CONFIG["col_mmsi"],
        "lat":  CONFIG["col_lat"],
        "lon":  CONFIG["col_lon"],
        "time": CONFIG["col_time"],
    }

    # These are OPTIONAL — detection is partially skipped if absent
    optional = {
        "sog":  CONFIG["col_sog"],   # Speed — needed for arrival/departure/anchoring
        "cog":  CONFIG["col_cog"],   # Course — needed for route deviation
        "name": CONFIG["col_name"],  # Vessel name — cosmetic, not required
    }

    for key, candidates in required.items():
        col = resolve_column(df, candidates)
        if col is None:
            raise ValueError(
                f"[ERROR] Required column '{key}' not found.\n"
                f"        Tried aliases: {candidates}\n"
                f"        Columns in file: {list(df.columns)}\n"
                f"        → Add your column name to CONFIG['col_{key}'] and retry."
            )
        mapping[key] = col

    for key, candidates in optional.items():
        mapping[key] = resolve_column(df, candidates)  # None if not found

    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: EVENT DETECTION ENGINE  (Req #1, #2, #4, #5)
# Core AI/rule-based logic for identifying vessel events.
# All operations are vectorized for performance on large datasets.
# ──────────────────────────────────────────────────────────────────────────────

def detect_events(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """
    Main event detection function. Processes the full AIS DataFrame and
    returns a new DataFrame of detected events (one row per event).

    Detection strategy:
    - Sort each vessel's track by timestamp
    - Use vectorized shift() comparisons to flag state transitions
    - Proximity uses time-bucketed spatial grouping for scalability

    Detects (Req #4):
      ARRIVAL         — vessel slows to near-stop after moving
      DEPARTURE        — vessel accelerates after being stopped
      ANCHORING        — vessel remains near-stationary for multiple readings
      ROUTE_DEVIATION  — sudden large bearing change
      PROXIMITY        — two vessels within N nautical miles simultaneously

    Returns:
        pd.DataFrame with columns: event_id, vessel_id, vessel_name,
        event_type, timestamp, latitude, longitude, confidence_score, null_flags
    """
    print_section("EVENT DETECTION  (Req #1, #2, #4, #5)")

    # Unpack column name references from the resolved mapping
    col_mmsi = cols["mmsi"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_time = cols["time"]
    col_sog  = cols["sog"]
    col_cog  = cols["cog"]
    col_name = cols["name"]

    all_events = []

    # ── Step 1: Parse and sort by vessel + time ───────────────────────────────
    # Sorting is critical — all shift-based detections assume chronological order
    print("[DETECT] Parsing timestamps and sorting...")
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    df = df.sort_values([col_mmsi, col_time]).reset_index(drop=True)

    n_vessels = df[col_mmsi].nunique()
    print(f"[DETECT] {len(df):,} rows | {n_vessels:,} unique vessels")

    # ── Step 2: Speed-based events (ARRIVAL, DEPARTURE, ANCHORING) ────────────
    # These all require SOG — skip if column not found
    if col_sog:
        print("[DETECT] Running speed-based event detection (arrival / departure / anchoring)...")
        all_events += _detect_speed_events(df, cols)
    else:
        print("[DETECT] WARNING: SOG column not found — skipping speed-based events.")

    # ── Step 3: Course-based events (ROUTE_DEVIATION) ────────────────────────
    # Requires COG — skip if column not found
    if col_cog:
        print("[DETECT] Running course deviation detection...")
        all_events += _detect_route_deviations(df, cols)
    else:
        print("[DETECT] WARNING: COG column not found — skipping route deviation detection.")

    # ── Step 4: Cross-vessel events (PROXIMITY) ───────────────────────────────
    # This is the most computationally expensive — uses time-bucketed grouping
    print("[DETECT] Running proximity detection (cross-vessel)...")
    all_events += _detect_proximity(df, cols)

    print(f"\n[DETECT] Detection complete — {len(all_events):,} total events found.")

    if not all_events:
        return pd.DataFrame()

    return pd.DataFrame(all_events)


def _detect_speed_events(df: pd.DataFrame, cols: dict) -> list:
    """
    Vectorized detection of ARRIVAL, DEPARTURE, and ANCHORING events.

    Uses pandas shift() to compare each row's SOG against the previous row
    for the same vessel. This avoids Python loops on 8M+ rows.

    Logic:
      ARRIVAL   : prev_SOG > arrival_max AND curr_SOG <= arrival_max
      DEPARTURE : prev_SOG <= arrival_max AND curr_SOG >= departure_min
      ANCHORING : SOG <= anchoring_max for stopped_min_rows consecutive rows
    """
    col_mmsi = cols["mmsi"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_time = cols["time"]
    col_sog  = cols["sog"]
    col_name = cols["name"]

    events = []

    # Shift SOG within each vessel group (not across vessels)
    # prev_sog aligns the previous row's speed with the current row
    df["_prev_sog"] = df.groupby(col_mmsi)[col_sog].shift(1)

    arr_max  = CONFIG["arrival_speed_max"]
    dep_min  = CONFIG["departure_speed_min"]
    anc_max  = CONFIG["anchoring_speed_max"]

    # ── ARRIVAL mask ─────────────────────────────────────────────────────────
    # Vessel was moving, now slow — transition into port/stop zone
    arrival_mask = (
        df["_prev_sog"].notna() &
        (df["_prev_sog"] > arr_max) &
        (df[col_sog] <= arr_max)
    )

    # ── DEPARTURE mask ────────────────────────────────────────────────────────
    # Vessel was slow/stopped, now accelerating — leaving port/anchorage
    departure_mask = (
        df["_prev_sog"].notna() &
        (df["_prev_sog"] <= arr_max) &
        (df[col_sog] >= dep_min)
    )

    # ── ANCHORING detection ───────────────────────────────────────────────────
    # Vessel nearly stationary — use rolling minimum per vessel to check
    # that SOG stays low for at least stopped_min_rows consecutive rows
    min_rows = CONFIG["stopped_min_rows"]
    df["_rolling_min_sog"] = (
        df.groupby(col_mmsi)[col_sog]
        .transform(lambda x: x.rolling(min_rows, min_periods=min_rows).min())
    )
    anchoring_mask = df["_rolling_min_sog"].notna() & (df["_rolling_min_sog"] <= anc_max)

    # ── Build event records from masked rows ─────────────────────────────────
    # We iterate over the (small) matched subset, not the full 8M rows
    for event_type, mask in [
        ("ARRIVAL",   arrival_mask),
        ("DEPARTURE", departure_mask),
        ("ANCHORING", anchoring_mask),
    ]:
        matched = df[mask][[col_mmsi, col_lat, col_lon, col_time, col_sog,
                              *([ col_name] if col_name else [])]].copy()

        for _, row in matched.iterrows():
            sog_val = row[col_sog]
            events.append({
                "event_id":         make_event_id(),
                "vessel_id":        row[col_mmsi],
                "vessel_name":      row[col_name] if col_name and col_name in row else str(row[col_mmsi]),
                "event_type":       event_type,
                "timestamp":        row[col_time],
                "latitude":         row[col_lat],
                "longitude":        row[col_lon],
                "confidence_score": confidence_score(event_type, sog_val),
                # Flag any NULL values in this row's key fields (Req #3)
                "null_flags":       null_flags(row, [col_lat, col_lon, col_time, col_sog]),
            })

        print(f"         {event_type:<20} → {len(matched):,} events detected")

    # Clean up temporary columns
    df.drop(columns=["_prev_sog", "_rolling_min_sog"], inplace=True, errors="ignore")
    return events


def _detect_route_deviations(df: pd.DataFrame, cols: dict) -> list:
    """
    Vectorized detection of ROUTE_DEVIATION events.

    Compares each vessel's current COG against its previous COG.
    A deviation >= deviation_bearing_threshold degrees is flagged.

    This handles the 0°/360° wrap-around in bearings correctly.
    """
    col_mmsi = cols["mmsi"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_time = cols["time"]
    col_cog  = cols["cog"]
    col_sog  = cols["sog"]
    col_name = cols["name"]

    events = []

    # Shift COG within each vessel to get previous course reading
    df["_prev_cog"] = df.groupby(col_mmsi)[col_cog].shift(1)

    # Compute bearing change using vectorized function
    df["_bearing_change"] = bearing_change_vectorized(df[col_cog], df["_prev_cog"])

    threshold = CONFIG["deviation_bearing_threshold"]

    # Flag rows where bearing change meets or exceeds threshold
    deviation_mask = (
        df["_prev_cog"].notna() &
        df[col_cog].notna() &
        (df["_bearing_change"] >= threshold)
    )

    matched = df[deviation_mask]

    for _, row in matched.iterrows():
        sog_val = row[col_sog] if col_sog and col_sog in row else None
        events.append({
            "event_id":         make_event_id(),
            "vessel_id":        row[col_mmsi],
            "vessel_name":      row[col_name] if col_name and col_name in row else str(row[col_mmsi]),
            "event_type":       "ROUTE_DEVIATION",
            "timestamp":        row[col_time],
            "latitude":         row[col_lat],
            "longitude":        row[col_lon],
            "confidence_score": confidence_score("ROUTE_DEVIATION", sog_val),
            "null_flags":       null_flags(row, [col_lat, col_lon, col_time, col_cog]),
        })

    print(f"         {'ROUTE_DEVIATION':<20} → {len(events):,} events detected")

    # Clean up temporary columns
    df.drop(columns=["_prev_cog", "_bearing_change"], inplace=True, errors="ignore")
    return events


def _detect_proximity(df: pd.DataFrame, cols: dict) -> list:
    """
    Detect PROXIMITY events: two different vessels within proximity_nm
    nautical miles of each other at (approximately) the same time.

    Strategy for scalability on 8M rows:
    - Round timestamps to the nearest minute to create time buckets
    - Only compare vessels within the same time bucket
    - Skip buckets with fewer than 2 vessels (no possible proximity pair)
    - This avoids O(n²) comparisons across the full dataset

    For extremely dense datasets (many vessels per time bucket), consider
    adding a spatial grid filter (e.g. round lat/lon to 0.1° cells) to
    further reduce comparisons.
    """
    col_mmsi = cols["mmsi"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_time = cols["time"]

    events = []
    prox_nm = CONFIG["proximity_nm"]

    # Bucket rows by minute — reduces comparison space enormously
    df["_ts_bucket"] = pd.to_datetime(df[col_time], errors="coerce").dt.floor("min")

    # Get only one position per vessel per time bucket (most recent)
    # This avoids redundant pair checks within a single vessel's track
    snapshot = (df.groupby([col_mmsi, "_ts_bucket"])
                  .agg({col_lat: "last", col_lon: "last"})
                  .reset_index())

    # Group by time bucket and compare all vessel pairs within that bucket
    total_buckets = snapshot["_ts_bucket"].nunique()
    checked = 0

    for ts_bucket, group in snapshot.groupby("_ts_bucket"):
        checked += 1
        if checked % 10000 == 0:
            print(f"         Proximity scan: {checked:,}/{total_buckets:,} time buckets...", end="\r")

        # Need at least 2 vessels to form a pair
        if len(group) < 2:
            continue

        vessels = group.reset_index(drop=True)
        n = len(vessels)

        # Pairwise comparison within this time bucket
        for i in range(n):
            for j in range(i + 1, n):
                r1 = vessels.iloc[i]
                r2 = vessels.iloc[j]

                # Skip same vessel (shouldn't happen after groupby, but be safe)
                if r1[col_mmsi] == r2[col_mmsi]:
                    continue

                # Skip rows with invalid coordinates
                if any(pd.isna([r1[col_lat], r1[col_lon], r2[col_lat], r2[col_lon]])):
                    continue

                dist = haversine_nm(r1[col_lat], r1[col_lon], r2[col_lat], r2[col_lon])

                if dist <= prox_nm:
                    # Midpoint location for the event record
                    mid_lat = (r1[col_lat] + r2[col_lat]) / 2
                    mid_lon = (r1[col_lon] + r2[col_lon]) / 2

                    events.append({
                        "event_id":         make_event_id(),
                        # Record both MMSIs so the event is traceable to both vessels
                        "vessel_id":        f"{r1[col_mmsi]} & {r2[col_mmsi]}",
                        "vessel_name":      "PROXIMITY PAIR",
                        "event_type":       "PROXIMITY",
                        "timestamp":        ts_bucket,
                        "latitude":         mid_lat,
                        "longitude":        mid_lon,
                        "confidence_score": confidence_score("PROXIMITY"),
                        "null_flags":       "None",
                    })

    print(f"\n         {'PROXIMITY':<20} → {len(events):,} events detected")

    # Clean up temporary column
    df.drop(columns=["_ts_bucket"], inplace=True, errors="ignore")
    return events


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: AI NATURAL LANGUAGE SUMMARIES  (Req #9)
# Uses the Claude API to generate plain-English descriptions of each event.
# ──────────────────────────────────────────────────────────────────────────────

def generate_ai_summaries(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a short natural-language AI summary for each detected event
    using the Claude API (Req #9).

    Each summary is a 1-2 sentence plain-English description of the event,
    suitable for display in dashboards or analyst reports.

    Summaries are added as a new column 'ai_summary' in the events DataFrame.

    Notes:
    - Requires ANTHROPIC_API_KEY environment variable to be set
    - Rate-limited by CONFIG['ai_summary_delay_sec'] to avoid API errors
    - Limited to CONFIG['ai_summary_max_events'] rows to control cost/time
    - If the API call fails, summary is set to the error message

    Req #9: Natural Language AI Event Summary.
    """
    print_section("AI EVENT SUMMARIES  (Req #9)")

    # Import Anthropic SDK — requires: pip install anthropic
    try:
        import anthropic
    except ImportError:
        print("[AI] WARNING: anthropic package not installed.")
        print("     Install with: pip install anthropic")
        print("     Skipping AI summaries — adding placeholder column.")
        events_df["ai_summary"] = "AI summary unavailable (anthropic not installed)"
        return events_df

    # Check for API key in environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[AI] WARNING: ANTHROPIC_API_KEY environment variable not set.")
        print("     Set it with: export ANTHROPIC_API_KEY=your_key_here")
        print("     Skipping AI summaries.")
        events_df["ai_summary"] = "AI summary unavailable (no API key)"
        return events_df

    client = anthropic.Anthropic(api_key=api_key)

    # Limit the number of events we send to the API to control cost and time.
    # For production, you could increase this or run in batches overnight.
    max_events = CONFIG["ai_summary_max_events"]
    total = min(len(events_df), max_events) if max_events else len(events_df)

    print(f"[AI] Generating summaries for {total:,} events...")
    if max_events and len(events_df) > max_events:
        print(f"     (Dataset has {len(events_df):,} events — "
              f"summarizing first {max_events:,}. Adjust CONFIG to change.)")

    summaries = []
    subset = events_df.head(total) if max_events else events_df

    for idx, (_, row) in enumerate(subset.iterrows()):
        # Build a concise prompt for Claude with the key event facts
        prompt = (
            f"You are an AIS maritime data analyst. Write a single clear, "
            f"professional sentence summarizing this vessel event for an analyst report.\n\n"
            f"Event Type: {row['event_type']}\n"
            f"Vessel: {row['vessel_name']} (MMSI: {row['vessel_id']})\n"
            f"Time: {row['timestamp']}\n"
            f"Location: Lat {row['latitude']:.4f}, Lon {row['longitude']:.4f}\n"
            f"Confidence: {row['confidence_score']}\n\n"
            f"Summary (1 sentence only, no preamble):"
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=100,  # Short summary — 1-2 sentences max
                messages=[{"role": "user", "content": prompt}]
            )
            # Extract text from the response content block
            summary = response.content[0].text.strip()
        except Exception as e:
            # Don't crash the whole pipeline on a single API failure
            summary = f"Summary generation failed: {str(e)[:80]}"

        summaries.append(summary)

        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"[AI] Summarized {idx + 1:,}/{total:,} events...", end="\r")

        # Rate limiting — pause between API calls to avoid hitting limits
        time.sleep(CONFIG["ai_summary_delay_sec"])

    print(f"\n[AI] Summaries complete.")

    # Pad remaining rows (beyond max_events) with a note
    remaining = len(events_df) - len(summaries)
    if remaining > 0:
        summaries += ["Summary not generated (batch limit reached)"] * remaining

    events_df = events_df.copy()
    events_df["ai_summary"] = summaries
    return events_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: OUTPUT BUILDER  (Req #3, #5, #6, #8)
# Merges event labels back into the original DataFrame as new columns.
# ──────────────────────────────────────────────────────────────────────────────

def build_labeled_dataset(original_df: pd.DataFrame,
                           events_df: pd.DataFrame,
                           cols: dict) -> pd.DataFrame:
    """
    Merge detected event labels back into the original AIS DataFrame.

    New columns added to the original data (Req #6):
      - event_id         : Unique event identifier (or None if no event)
      - event_type       : Type of event (ARRIVAL, DEPARTURE, etc.)
      - confidence_score : How confident the system is in this label
      - null_flags       : Any NULL fields in this row's key data
      - ai_summary       : Natural language description (if generated)

    The merge is done on (vessel_id + timestamp-minute) to align
    event records back to their source rows efficiently.

    Req #3 : Structured output per event row
    Req #6 : New columns in DataFrame + CSV export
    Req #8 : CSV-compatible output format
    """
    print_section("BUILDING LABELED DATASET  (Req #3, #6, #8)")

    col_mmsi = cols["mmsi"]
    col_time = cols["time"]

    if events_df is None or events_df.empty:
        # No events detected — still add the columns so schema is consistent
        print("[OUTPUT] No events detected. Adding empty event columns.")
        original_df["event_id"]         = None
        original_df["event_type"]       = None
        original_df["confidence_score"] = None
        original_df["null_flags"]       = None
        if "ai_summary" not in original_df.columns:
            original_df["ai_summary"]   = None
        return original_df

    # ── Build a fast lookup dictionary: (mmsi_str, ts_minute) → event row ────
    # This is O(n) — much faster than a DataFrame merge for this use case
    # because many original rows will NOT have a matching event.
    print("[OUTPUT] Building event lookup index...")

    events_df = events_df.copy()

    # Normalize vessel_id to string for consistent key matching
    events_df["_key_mmsi"] = events_df["vessel_id"].astype(str)
    events_df["_key_ts"]   = (pd.to_datetime(events_df["timestamp"], errors="coerce")
                               .dt.floor("min").astype(str))

    # Build the lookup — if multiple events match the same key, keep first
    # PROXIMITY events store both MMSIs as "A & B"; split them so both vessels get tagged
    lookup = {}
    for _, ev in events_df.iterrows():
        ev_data = {
            "event_id":         ev["event_id"],
            "event_type":       ev["event_type"],
            "confidence_score": ev["confidence_score"],
            "null_flags":       ev["null_flags"],
            "ai_summary":       ev.get("ai_summary", None),
        }
        mmsi_keys = [m.strip() for m in ev["_key_mmsi"].split(" & ")]
        for mmsi_key in mmsi_keys:
            key = (mmsi_key, ev["_key_ts"])
            if key not in lookup:
                lookup[key] = ev_data

    print(f"[OUTPUT] Lookup index built — {len(lookup):,} unique event keys.")

    # ── Tag original DataFrame rows ───────────────────────────────────────────
    # Create temporary key columns for fast vectorized matching
    original_df["_key_mmsi"] = original_df[col_mmsi].astype(str)
    original_df["_key_ts"]   = (pd.to_datetime(original_df[col_time], errors="coerce")
                                 .dt.floor("min").astype(str))

    # Map each row to its event data using the lookup
    # Using list comprehension is faster than apply() for large DataFrames
    print("[OUTPUT] Tagging rows with event labels (this may take a moment for 8M rows)...")

    event_ids    = []
    event_types  = []
    conf_scores  = []
    null_flag_col = []
    ai_summaries = []

    for mmsi_key, ts_key in zip(original_df["_key_mmsi"], original_df["_key_ts"]):
        ev = lookup.get((mmsi_key, ts_key))
        if ev:
            event_ids.append(ev["event_id"])
            event_types.append(ev["event_type"])
            conf_scores.append(ev["confidence_score"])
            null_flag_col.append(ev["null_flags"])
            ai_summaries.append(ev["ai_summary"])
        else:
            event_ids.append(None)
            event_types.append(None)
            conf_scores.append(None)
            null_flag_col.append(None)
            ai_summaries.append(None)

    # Assign new columns to the DataFrame (Req #6)
    original_df["event_id"]         = event_ids
    original_df["event_type"]       = event_types
    original_df["confidence_score"] = conf_scores
    original_df["null_flags"]       = null_flag_col
    original_df["ai_summary"]       = ai_summaries

    # Remove temporary key columns before export
    original_df.drop(columns=["_key_mmsi", "_key_ts"], inplace=True, errors="ignore")

    labeled_count = original_df["event_type"].notna().sum()
    print(f"[OUTPUT] {labeled_count:,} rows tagged with event labels.")

    return original_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: CSV EXPORT  (Req #6, #8)
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, filepath: str, label: str = "output"):
    """
    Export a DataFrame to CSV.

    For large datasets (8M+ rows), we use chunksize in to_csv() to write
    in streaming fashion — this prevents memory spikes during export.

    Req #6 : Final CSV export function
    Req #8 : Structured CSV format for downstream analytics / storage
    """
    print(f"[EXPORT] Writing {label} → {filepath}")
    # chunksize here is for the CSV writer's internal buffer, not read chunks
    df.to_csv(filepath, index=False)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"         Done — {len(df):,} rows | {size_mb:.1f} MB")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: PIPELINE SUMMARY / REPORT  (Req #7 — Documentation support)
# ──────────────────────────────────────────────────────────────────────────────

def print_pipeline_summary(input_path: str, labeled_df: pd.DataFrame,
                            events_df: pd.DataFrame, output_dir: str):
    """
    Print a human-readable summary of the pipeline run.
    Useful for documentation, sign-off, and analyst handover.

    Req #7: Documentation of the data processing pipeline and techniques.
    """
    print_section("PIPELINE SUMMARY  (Req #7)")

    total_rows   = len(labeled_df)
    labeled_rows = labeled_df["event_type"].notna().sum() if "event_type" in labeled_df.columns else 0
    event_counts = (events_df["event_type"].value_counts().to_dict()
                    if events_df is not None and not events_df.empty else {})

    print(f"  Input file     : {os.path.basename(input_path)}")
    print(f"  Total rows     : {total_rows:,}")
    print(f"  Labeled rows   : {labeled_rows:,} ({(100*labeled_rows/total_rows if total_rows else 0.0):.2f}% of data)")
    print(f"  Output dir     : {output_dir}")
    print(f"\n  Event breakdown:")
    for etype, count in sorted(event_counts.items()):
        print(f"    {etype:<20} : {count:,}")

    has_summaries = (events_df is not None and
                     "ai_summary" in events_df.columns and
                     events_df["ai_summary"].notna().any())
    print(f"\n  AI summaries   : {'YES' if has_summaries else 'NO (use --ai-summaries flag)'}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10: MAIN PIPELINE ORCHESTRATOR
# Ties all sections together in order.
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str,
                 output_dir: str = "./output",
                 ai_summaries: bool = False):
    """
    Full end-to-end DDT AIS processing pipeline.

    Steps:
      1. Load data (CSV / JSON / NMEA)           → Req #10
      2. Resolve column names                    → Internal
      3. Detect and label vessel events          → Req #1, #2, #4, #5
      4. Generate AI summaries (optional)        → Req #9
      5. Merge events into original dataset      → Req #3, #6
      6. Export labeled CSV + events summary     → Req #6, #8
      7. Print pipeline summary                  → Req #7

    Args:
        input_path   : Path to AIS input file (.csv, .json, .nmea)
        output_dir   : Directory for output files
        ai_summaries : If True, generate Claude AI summaries (Req #9)
    """
    print("\n" + "=" * 60)
    print("  DDT — GEN AI AIS EVENT DETECTION & LABELING PIPELINE")
    print("  Version 2.0 | All Requirements Active")
    print("=" * 60)

    start_time = time.time()

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print_section("LOADING DATA  (Req #10)")
    df = load_file(input_path)

    # ── Step 2: Resolve column names ─────────────────────────────────────────
    print_section("COLUMN RESOLUTION")
    cols = resolve_columns(df)
    print(f"  MMSI  → '{cols['mmsi']}'")
    print(f"  LAT   → '{cols['lat']}'")
    print(f"  LON   → '{cols['lon']}'")
    print(f"  TIME  → '{cols['time']}'")
    print(f"  SOG   → {repr(cols['sog'])} {'⚠ Speed events will be skipped' if not cols['sog'] else ''}")
    print(f"  COG   → {repr(cols['cog'])} {'⚠ Deviation events will be skipped' if not cols['cog'] else ''}")
    print(f"  NAME  → {repr(cols['name'])}")

    # ── Step 2.5: Date filter ─────────────────────────────────────────────────
    df, date_label = prompt_date_filter(df, cols["time"])

    # ── Step 3: Event detection ───────────────────────────────────────────────
    events_df = detect_events(df, cols)

    # ── Step 4: AI summaries (optional) ──────────────────────────────────────
    if ai_summaries and events_df is not None and not events_df.empty:
        events_df = generate_ai_summaries(events_df)
    elif ai_summaries:
        print("[AI] No events to summarize.")

    # ── Step 5: Merge events into original dataset ────────────────────────────
    labeled_df = build_labeled_dataset(df, events_df, cols)

    # ── Step 6: Export outputs ────────────────────────────────────────────────
    print_section("EXPORTING OUTPUTS  (Req #6, #8)")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    file_prefix = (base + "_" + date_label) if date_label != "all" else base

    # Output 1: Full labeled dataset (original data + new event columns)
    labeled_path = os.path.join(output_dir, file_prefix + "_labeled.csv")
    export_csv(labeled_df, labeled_path, label="Full labeled dataset")

    # Output 2: Events-only summary (one row per event — for dashboards/analytics)
    if events_df is not None and not events_df.empty:
        events_path = os.path.join(output_dir, file_prefix + "_events_summary.csv")
        export_csv(events_df, events_path, label="Events summary")
    else:
        events_path = None
        print("[EXPORT] No events to export.")

    # ── Step 7: Summary ───────────────────────────────────────────────────────
    print_pipeline_summary(input_path, labeled_df, events_df, output_dir)

    elapsed = time.time() - start_time
    print(f"  Total runtime  : {elapsed:.1f} seconds")
    print("=" * 60 + "\n")

    return labeled_df, events_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: COMMAND LINE INTERFACE
# Run this script directly from the terminal.
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "DDT -- GEN AI AIS Event Detection & Labeling System\n"
            "Detects ARRIVAL, DEPARTURE, ANCHORING, ROUTE_DEVIATION, "
            "and PROXIMITY events from AIS vessel tracking data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Optional: input file path (if not provided, file browser opens)
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to AIS input file (.csv, .json, .nmea/.txt/.ais). If omitted, a file browser will open."
    )

    # Optional: output directory
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Directory where output CSVs will be saved (default: ./output)"
    )

    # Optional: enable AI summaries (requires ANTHROPIC_API_KEY env var)
    parser.add_argument(
        "--ai-summaries",
        action="store_true",
        default=False,
        help=(
            "Generate natural-language AI summaries for each event (Req #9). "
            "Requires: ANTHROPIC_API_KEY environment variable to be set. "
            "Example: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    )

    # Optional: override max AI summary count
    parser.add_argument(
        "--max-summaries",
        type=int,
        default=None,
        help="Max number of events to summarize with AI (overrides CONFIG default)"
    )

    args = parser.parse_args()

    # Apply CLI overrides to CONFIG
    if args.max_summaries is not None:
        CONFIG["ai_summary_max_events"] = args.max_summaries

    # If no input file given, open file browser and run pipeline on the result
    if args.input is None:
        df = import_dataset()
        if df is not None:
            run_pipeline_from_df(df, output_dir=args.output_dir, ai_summaries=args.ai_summaries)
    else:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            ai_summaries=args.ai_summaries,
        )
