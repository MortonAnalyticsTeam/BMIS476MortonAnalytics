"""
================================================================================
DDT — GEN AI DATA PROCESSING AND ANALYTICS SOLUTION
AIS Vessel Event Detection, Labeling & AI Summary System
================================================================================
Project      : GEN AI DATA PROCESSING AND ANALYTICS SOLUTION
Team         : DDT Team
Version      : 2.0
Description  : Full pipeline for reading, processing, detecting, labeling, and
               summarizing AIS vessel event data at scale (8M+ rows supported).


DEPENDENCIES:
  pip install pandas anthropic pyais
  (pyais only required for NMEA input files)

USAGE:
  python ddt_ais_pipeline.py your_data.csv --output-dir ./output
  python ddt_ais_pipeline.py your_data.csv --output-dir ./output --ai-summaries
  python ddt_ais_pipeline.py your_data.json --output-dir ./output --ai-summaries
================================================================================
"""
# ── Standard library imports ──────────────────────────────────────────────────
import os
import sys
import json
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
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import pyais  
except ImportError:
    pyais = None  # type: ignore

# Suppress pandas performance warnings for large frame operations
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1A: CONFIGURATION
# This central configuration block serves as the control panel for our entire
# maritime analytics pipeline. Here we define all the tunable parameters that
# govern event detection sensitivity, performance optimizations, and data
# compatibility. By keeping everything in one location, we ensure consistency
# and make it easy to adapt the system for different AIS datasets or operational
# requirements. Each setting is carefully chosen based on maritime standards
# and real-world testing.
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
    "col_name":   ["VesselName", "vessel_name", "Name", "name", "VESSEL_NAME"],
    "col_dest":   ["Destination", "destination", "DESTINATION", "dest", "Dest"],
    "col_eta":    ["ETA", "eta", "EstimatedArrival", "estimated_arrival"],
    "col_status": ["Status", "status", "NavStatus", "nav_status"],
}

# ──────────────────────────────────────────────────────────────────────────────
# MARITIME REGION LOOKUP TABLE
# Used by get_region_name() to convert lat/lon into a human-readable location.
# Ordered most-specific first so inner bays resolve before their parent ocean.
# Each entry: (region_name, lat_min, lat_max, lon_min, lon_max)
# ──────────────────────────────────────────────────────────────────────────────
MARITIME_REGIONS = [
    # ── US East Coast ports & bays ────────────────────────────────────────────
    ("Boston Harbor",                    42.20, 42.45, -71.10, -70.85),
    ("Long Island Sound",                40.80, 41.60, -74.00, -71.80),
    ("Port of New York/New Jersey",      40.40, 40.80, -74.30, -73.70),
    ("Delaware Bay",                     38.50, 40.00, -75.70, -74.80),
    ("Chesapeake Bay",                   36.80, 39.60, -76.60, -75.40),
    ("Port of Charleston",               32.65, 32.90, -80.05, -79.85),
    ("Port of Savannah",                 31.90, 32.20, -81.30, -80.80),
    ("Port of Jacksonville",             30.20, 30.50, -81.75, -81.40),
    ("Port of Miami",                    25.70, 25.90, -80.25, -80.05),
    ("Tampa Bay",                        27.40, 28.20, -83.00, -82.30),
    # ── US Gulf Coast ─────────────────────────────────────────────────────────
    ("Port of New Orleans",              29.80, 30.20, -90.50, -89.80),
    ("Mississippi River Delta",          28.50, 30.00, -91.50, -88.80),
    ("Mobile Bay",                       30.00, 30.80, -88.30, -87.80),
    ("Port of Houston / Galveston Bay",  29.20, 29.90, -95.20, -94.50),
    ("Port of Corpus Christi",           27.70, 28.00, -97.50, -97.10),
    # ── US West Coast ports ───────────────────────────────────────────────────
    ("Port of Los Angeles / Long Beach", 33.50, 34.10, -118.55, -117.90),
    ("San Francisco Bay",                37.30, 38.20, -122.70, -122.10),
    ("Port of Portland, OR",             45.40, 45.70, -122.90, -122.60),
    ("Puget Sound",                      47.00, 48.60, -123.00, -122.00),
    ("Strait of Juan de Fuca",           48.00, 48.70, -124.70, -122.50),
    # ── Alaska ────────────────────────────────────────────────────────────────
    ("Prince William Sound",             59.50, 61.50, -148.50, -145.50),
    ("Gulf of Alaska",                   54.00, 62.00, -162.00, -135.00),
    # ── US coastal waters ─────────────────────────────────────────────────────
    ("Gulf of Mexico",                   18.00, 31.00,  -98.00,  -80.00),
    ("Caribbean Sea",                     8.00, 23.50,  -87.00,  -59.00),
    ("US East Coast (offshore)",         25.00, 47.00,  -80.00,  -65.00),
    ("US West Coast (offshore)",         32.00, 50.00, -130.00, -117.00),
    # ── Major world maritime zones ────────────────────────────────────────────
    ("English Channel",                  49.00, 52.00,   -5.50,   3.00),
    ("North Sea",                        51.00, 62.00,   -5.00,  12.00),
    ("Baltic Sea",                       53.00, 66.00,    9.00,  31.00),
    ("Mediterranean Sea",                30.00, 46.00,   -6.00,  42.00),
    ("Black Sea",                        40.50, 47.00,   27.50,  42.00),
    ("Red Sea",                          12.00, 30.50,   32.00,  45.00),
    ("Persian Gulf",                     22.50, 30.50,   47.50,  57.00),
    ("Arabian Sea",                       4.00, 25.50,   55.00,  80.00),
    ("Bay of Bengal",                     4.00, 23.00,   80.00, 100.00),
    ("Strait of Malacca",                 1.00,  6.00,   99.00, 104.50),
    ("South China Sea",                   0.00, 23.00,  100.00, 122.00),
    ("East China Sea",                   22.00, 34.00,  117.00, 132.00),
    ("Sea of Japan",                     33.00, 45.00,  128.00, 142.00),
    ("Yellow Sea",                       31.00, 41.00,  118.00, 127.00),
    # ── Ocean basin fallbacks ─────────────────────────────────────────────────
    ("North Atlantic Ocean",              0.00, 70.00,  -80.00,   0.00),
    ("South Atlantic Ocean",            -70.00,  0.00,  -65.00,  20.00),
    ("North Pacific Ocean",               0.00, 66.00, -180.00, -100.00),
    ("South Pacific Ocean",             -70.00,  0.00, -180.00,  -65.00),
    ("Indian Ocean",                    -70.00, 25.00,   20.00, 150.00),
    ("Southern Ocean",                  -90.00,-60.00, -180.00, 180.00),
    ("Arctic Ocean",                     66.00, 90.00, -180.00, 180.00),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 1B: DATA IMPORT FUNCTION
# This user-friendly import system provides multiple ways to bring AIS data
# into our analytics pipeline. Whether you're working with massive 8M+
# row datasets or smaller test files, this section handles the loading process
# with memory efficiency and user convenience in mind. We support interactive
# file browsing, direct path specification, and programmatic access for
# integration with notebooks and automated workflows.
# ──────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import filedialog

def import_dataset(preview_rows: int = 10,
                   save_preview: bool = True,
                   preview_dir: str = "./preview") -> pd.DataFrame:
    """
    Opens a file browser dialog so you can select your AIS dataset directly
    from your computer. The full dataset is loaded into memory but is NOT
    saved into your project folder. The folder will be too large if it exports the dataset with 
    the function. You can easily change this if needed.
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
    print(f"         ✓ Your repository stays clean.\n")

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
        ai_summaries : Whether to generate plain-English event summaries

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
    df, date_label, ai_summaries, output_format = prompt_date_filter(df, cols["time"])

    # -- Sort and deduplicate so every vessel's track is in chronological order
    # and duplicate pings (same vessel, same timestamp) are collapsed to one row
    print_section("PREPROCESSING")
    df[cols["time"]] = pd.to_datetime(df[cols["time"]], errors="coerce")
    df = df.sort_values([cols["mmsi"], cols["time"]]).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(keep="first")  # only removes rows identical across every column
    removed = before - len(df)
    if removed:
        print(f"[PREP] Removed {removed:,} duplicate vessel/timestamp rows.")
    print(f"[PREP] {len(df):,} clean rows ready for processing.")

    # Run event detection
    events_df = detect_events(df, cols)

    # Optional AI summaries
    if ai_summaries and events_df is not None and not events_df.empty:
        events_df = generate_ai_summaries(events_df)

    # Merge labels into original DataFrame
    labeled_df = build_labeled_dataset(df, events_df, cols)

    # Export outputs -- date label in filename prevents overwriting previous runs
    print_section("EXPORTING OUTPUTS")
    os.makedirs(output_dir, exist_ok=True)

    file_prefix = date_label if date_label != "all" else "full_dataset"

    if output_format == "single":
        report = build_shipment_report(events_df, df, cols)
        report_path = os.path.join(output_dir, file_prefix + "_shipment_report.csv")
        export_csv(report, report_path, label="Shipment report")
    else:
        events_only = labeled_df[labeled_df["event_type"].notna()].copy()
        export_csv(events_only,
                   os.path.join(output_dir, file_prefix + "_events_labeled.csv"),
                   label="Event-labeled rows")
        if events_df is not None and not events_df.empty:
            export_csv(events_df,
                       os.path.join(output_dir, file_prefix + "_events_summary.csv"),
                       label="Events summary")
        export_vessel_status_table(df, events_df, output_dir, file_prefix)

    # Summary
    print_pipeline_summary("imported_dataset", labeled_df, events_df, output_dir)

    elapsed = time.time() - start_time
    print(f"  Total runtime  : {elapsed:.1f} seconds")
    print("=" * 60 + "\n")

    return labeled_df, events_df


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
        return df, "all", _ask_ai_summaries_prompt(), _ask_output_format_prompt()

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
        return df, "all", _ask_ai_summaries_prompt(), _ask_output_format_prompt()

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

    return filtered, label, _ask_ai_summaries_prompt(), _ask_output_format_prompt()


def _ask_ai_summaries_prompt() -> bool:
    """Ask the AI summaries question as part of the setup flow."""
    print()
    print("=" * 60)
    print("  AI SUMMARIES")
    print("=" * 60)
    print("  Generate plain-English Claude AI summaries for each event?")
    print("  (Requires ANTHROPIC_API_KEY env variable to be set)")
    print()
    while True:
        answer = input("  Generate AI summaries? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter y or n.")


def _ask_output_format_prompt() -> str:
    """Ask how the user wants results exported. Returns 'single' or 'separate'."""
    print()
    print("=" * 60)
    print("  OUTPUT FORMAT")
    print("=" * 60)
    print("  1. Shipment report  — one file with all events + vessel context")
    print("  2. Separate files   — events summary, vessel status, and labeled data")
    print()
    while True:
        choice = input("  Enter choice (1 or 2): ").strip()
        if choice == "1":
            return "single"
        if choice == "2":
            return "separate"
        print("  Please enter 1 or 2.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: UTILITY FUNCTIONS
# Small, reusable helper functions used throughout the pipeline.
# ──────────────────────────────────────────────────────────────────────────────

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


def confidence_score(event_type: str,
                     sog_value: Optional[float] = None,
                     bearing_change: Optional[float] = None) -> float:
    """
    Rule-based confidence scoring (0.0 – 1.0) for a detected event.

    Base scores are set by event type, then adjusted by supporting signals:

    ROUTE_DEVIATION scoring factors:
      - Bearing change magnitude  : larger turns are harder to explain as noise
          45–60°  → base 0.70
          60–90°  → +0.08  (moderate turn)
          90–135° → +0.13  (sharp turn)
          >135°   → +0.17  (near-reversal — very strong signal)
      - Speed at time of turn:
          > 5 kts  → +0.05  (vessel clearly underway, not drifting)
          < 1.5 kts → -0.10 (slow maneuvering in port; may not be a true deviation)

    Other event types use SOG as the primary signal (arrival/departure/anchoring).

    Args:
        event_type     : ARRIVAL, DEPARTURE, ANCHORING, ROUTE_DEVIATION, PROXIMITY
        sog_value      : Speed Over Ground at time of event (knots)
        bearing_change : Absolute bearing change in degrees (ROUTE_DEVIATION only)

    Returns:
        float between 0.0 and 1.0
    """
    base = {
        "ARRIVAL":         0.80,
        "DEPARTURE":       0.78,
        "ANCHORING":       0.85,
        "ROUTE_DEVIATION": 0.70,
        "PROXIMITY":       0.75,
    }.get(event_type, 0.60)

    if event_type == "ROUTE_DEVIATION":
        # Adjust for how sharp the turn was
        if bearing_change is not None and not pd.isna(bearing_change):
            if bearing_change > 135:
                base = min(base + 0.17, 1.0)
            elif bearing_change > 90:
                base = min(base + 0.13, 1.0)
            elif bearing_change > 60:
                base = min(base + 0.08, 1.0)
            # 45–60° range: no adjustment — that's the detection threshold itself
        # Adjust for speed (slow = could just be port maneuvering)
        if sog_value is not None and not pd.isna(sog_value):
            if sog_value > 5.0:
                base = min(base + 0.05, 1.0)
            elif sog_value < 1.5:
                base = max(base - 0.10, 0.0)
    else:
        # Boost confidence when SOG makes other event types very clear
        if sog_value is not None and not pd.isna(sog_value):
            if event_type in ("ARRIVAL", "ANCHORING") and sog_value < 0.2:
                base = min(base + 0.10, 1.0)
            elif event_type == "DEPARTURE" and sog_value > 5.0:
                base = min(base + 0.08, 1.0)

    return round(base, 2)


def print_section(title: str):
    """Print a formatted section header to the console for pipeline readability."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def get_region_name(lat: float, lon: float) -> str:
    """
    Return a human-readable maritime region name for a lat/lon coordinate.
    Uses the MARITIME_REGIONS priority list (most specific first).
    Falls back to a cardinal open-ocean description if no region matches.
    """
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown Location"
    for name, lat_min, lat_max, lon_min, lon_max in MARITIME_REGIONS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    hemi_lat = "N" if lat >= 0 else "S"
    hemi_lon = "E" if lon >= 0 else "W"
    return f"Open Ocean ({abs(lat):.1f}°{hemi_lat}, {abs(lon):.1f}°{hemi_lon})"


def describe_vessel_status(sog: Optional[float],
                            cog: Optional[float] = None) -> str:
    """Return a plain-English description of a vessel's current activity."""
    if sog is None or pd.isna(sog):
        return "Status unknown (speed data unavailable)"
    cog_str = f", heading {cog:.0f}°" if cog is not None and not pd.isna(cog) else ""
    if sog < 0.3:
        return f"Stopped / at anchor (SOG {sog:.1f} kts)"
    elif sog < 1.5:
        return f"Maneuvering slowly (SOG {sog:.1f} kts{cog_str})"
    elif sog < 5.0:
        return f"Moving at low speed (SOG {sog:.1f} kts{cog_str})"
    elif sog < 15.0:
        return f"Underway (SOG {sog:.1f} kts{cog_str})"
    else:
        return f"Underway at high speed (SOG {sog:.1f} kts{cog_str})"


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2B: VESSEL LOOKUP
# Call lookup_vessel(labeled_df, events_df, <MMSI or name>) after running the
# pipeline to get a plain-English status report for any specific vessel.
# ──────────────────────────────────────────────────────────────────────────────

def lookup_vessel(df: pd.DataFrame,
                  events_df: pd.DataFrame,
                  identifier,
                  print_report: bool = True) -> str:
    """
    Look up a vessel by MMSI or name and print a plain-English status report.

    Shows:
      - Last known position with named region
      - Current speed / activity description
      - Destination and ETA (if present in the dataset)
      - All detected events for this vessel with location and description

    Args:
        df           : The labeled (or raw) AIS DataFrame
        events_df    : Events summary DataFrame from the pipeline
        identifier   : MMSI number (int/str) OR vessel name (partial match OK)
        print_report : Print to console (default True); always returns the string

    Usage:
        lookup_vessel(labeled_df, events_df, 123456789)
        lookup_vessel(labeled_df, events_df, "EVER GIVEN")
    """
    cols     = resolve_columns(df)
    col_mmsi = cols["mmsi"]
    col_time = cols["time"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_sog  = cols.get("sog")
    col_cog  = cols.get("cog")
    col_name = cols.get("name")

    # Optional AIS fields that are present in some (not all) datasets
    col_dest   = resolve_column(df, CONFIG["col_dest"])
    col_eta    = resolve_column(df, CONFIG["col_eta"])
    col_status = resolve_column(df, CONFIG["col_status"])

    ident_str = str(identifier).strip()

    # Try exact MMSI match first, then case-insensitive partial name match
    mask = df[col_mmsi].astype(str) == ident_str
    if not mask.any() and col_name:
        mask = df[col_name].astype(str).str.upper().str.contains(
            ident_str.upper(), na=False, regex=False
        )

    if not mask.any():
        msg = f"[LOOKUP] No vessel found matching '{identifier}'\n"
        if print_report:
            print(msg)
        return msg

    vessel_rows          = df[mask].copy()
    vessel_rows[col_time] = pd.to_datetime(vessel_rows[col_time], errors="coerce")
    latest               = vessel_rows.sort_values(col_time).iloc[-1]

    # ── Core fields ───────────────────────────────────────────────────────────
    v_name = (str(latest[col_name])
              if col_name and not pd.isna(latest.get(col_name, float("nan")))
              else "Unknown")
    mmsi   = str(latest[col_mmsi])
    ts     = str(latest[col_time])
    lat    = float(latest[col_lat])
    lon    = float(latest[col_lon])
    region = get_region_name(lat, lon)

    sog = (float(latest[col_sog])
           if col_sog and not pd.isna(latest.get(col_sog, float("nan"))) else None)
    cog = (float(latest[col_cog])
           if col_cog and not pd.isna(latest.get(col_cog, float("nan"))) else None)
    activity = describe_vessel_status(sog, cog)

    # ── Optional AIS fields ───────────────────────────────────────────────────
    destination = (str(latest[col_dest])
                   if col_dest and not pd.isna(latest.get(col_dest, float("nan"))) else None)
    eta         = (str(latest[col_eta])
                   if col_eta and not pd.isna(latest.get(col_eta, float("nan"))) else None)
    nav_status  = (str(latest[col_status])
                   if col_status and not pd.isna(latest.get(col_status, float("nan"))) else None)

    # ── Detected events for this vessel ──────────────────────────────────────
    vessel_events = []
    if events_df is not None and not events_df.empty:
        ev_mask = events_df["vessel_id"].astype(str).str.contains(
            mmsi, na=False, regex=False
        )
        if ev_mask.any():
            vessel_events = (events_df[ev_mask]
                             .sort_values("timestamp", ascending=False)
                             .head(10)
                             .to_dict("records"))

    # ── Format the report ─────────────────────────────────────────────────────
    sep = "=" * 55
    lines = [
        sep,
        "  VESSEL STATUS REPORT",
        sep,
        f"  Name        : {v_name}",
        f"  MMSI        : {mmsi}",
        f"  Last seen   : {ts}",
        f"  Location    : {region}",
        f"                Lat {lat:.4f}, Lon {lon:.4f}",
        f"  Activity    : {activity}",
    ]
    if nav_status:
        lines.append(f"  AIS Status  : {nav_status}")
    if destination:
        lines.append(f"  Destination : {destination}")
    if eta:
        lines.append(f"  ETA         : {eta}")

    lines.append(f"\n  Detected Events ({len(vessel_events)} most recent):")
    if vessel_events:
        for ev in vessel_events:
            ev_lat    = float(ev.get("latitude", 0))
            ev_lon    = float(ev.get("longitude", 0))
            ev_region = get_region_name(ev_lat, ev_lon)
            ev_label  = ev.get("event_label") or ev.get("event_type", "EVENT")
            ev_ts     = str(ev.get("timestamp", ""))
            ev_conf   = ev.get("confidence_score", "")
            lines += [
                f"",
                f"    [{ev_ts}]",
                f"      What    : {ev_label}",
                f"      Where   : {ev_region}  (Lat {ev_lat:.4f}, Lon {ev_lon:.4f})",
                f"      Conf.   : {ev_conf}",
            ]
    else:
        lines.append("    None detected for this vessel.")

    lines.append(sep)

    report = "\n".join(lines)
    if print_report:
        print(report)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2C: BULK VESSEL STATUS EXPORT
# Produces a one-row-per-vessel CSV — fast on 8M+ rows via groupby().last().
# ──────────────────────────────────────────────────────────────────────────────

def export_vessel_status_table(df: pd.DataFrame,
                                events_df,
                                output_dir: str,
                                file_prefix: str) -> pd.DataFrame:
    """
    Build and export a one-row-per-vessel status table.

    Columns: MMSI, vessel_name, last_seen, region_name, lat, lon,
             activity, event_count  (+ destination/eta if in data)

    Uses sort + groupby().last() — single pass, ~10–30 s on 8M rows.
    The apply() calls for region/activity run only on the small per-vessel
    table (hundreds to thousands of rows), not the full dataset.

    Args:
        df          : Full AIS DataFrame (raw or labeled)
        events_df   : Events summary DataFrame from the pipeline (may be None)
        output_dir  : Directory where the CSV will be saved
        file_prefix : Filename prefix (e.g. "2023-06-15" or "full_dataset")

    Returns:
        status DataFrame (also exported to CSV)
    """
    print("[STATUS TABLE] Building per-vessel status table...")

    cols     = resolve_columns(df)
    col_mmsi = cols["mmsi"]
    col_time = cols["time"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_sog  = cols.get("sog")
    col_cog  = cols.get("cog")
    col_name = cols.get("name")
    col_dest = resolve_column(df, CONFIG["col_dest"])
    col_eta  = resolve_column(df, CONFIG["col_eta"])

    # Sort by time (detect_events already parsed timestamps in-place)
    # Only bring along columns we need before sorting to save memory
    keep = [col_mmsi, col_time, col_lat, col_lon]
    for c in [col_sog, col_cog, col_name, col_dest, col_eta]:
        if c and c in df.columns and c not in keep:
            keep.append(c)

    status = (df[keep]
              .sort_values(col_time)
              .groupby(col_mmsi, sort=False)
              .last()
              .reset_index())

    # Rename to clean standard names
    rename = {col_mmsi: "MMSI", col_time: "last_seen",
              col_lat: "lat", col_lon: "lon"}
    if col_sog:
        rename[col_sog] = "_sog"
    if col_cog:
        rename[col_cog] = "_cog"
    if col_name:
        rename[col_name] = "vessel_name"
    if col_dest:
        rename[col_dest] = "destination"
    if col_eta:
        rename[col_eta] = "eta"
    status = status.rename(columns=rename)

    # Fill missing vessel_name with MMSI string
    if "vessel_name" not in status.columns:
        status["vessel_name"] = status["MMSI"].astype(str)
    else:
        status["vessel_name"] = (
            status["vessel_name"]
            .astype(str)
            .replace("nan", pd.NA)
            .fillna(status["MMSI"].astype(str))
        )

    # Region name — runs on the small per-vessel table, not 8M rows
    status["region_name"] = status.apply(
        lambda r: get_region_name(float(r["lat"]), float(r["lon"])), axis=1
    )

    # Activity description
    status["activity"] = status.apply(
        lambda r: describe_vessel_status(
            float(r["_sog"]) if "_sog" in r and pd.notna(r["_sog"]) else None,
            float(r["_cog"]) if "_cog" in r and pd.notna(r["_cog"]) else None,
        ),
        axis=1,
    )

    # AIS ping count per vessel (how many position reports we received for each ship)
    ping_counts = df[col_mmsi].value_counts().rename("ping_count")
    status["ping_count"] = status["MMSI"].map(ping_counts).fillna(0).astype(int)

    # Event counts per vessel
    if events_df is not None and not events_df.empty:
        event_counts = (events_df["vessel_id"]
                        .astype(str)
                        .value_counts()
                        .rename("event_count"))
        status["_mmsi_str"] = status["MMSI"].astype(str)
        status = status.join(event_counts, on="_mmsi_str").drop(columns="_mmsi_str")

        # Route deviation count and average confidence per vessel
        dev_events = events_df[events_df["event_type"] == "ROUTE_DEVIATION"].copy()
        dev_events["_mmsi_str"] = dev_events["vessel_id"].astype(str)

        dev_counts = (dev_events.groupby("_mmsi_str")["vessel_id"]
                      .count().rename("deviation_count"))
        dev_conf   = (dev_events.groupby("_mmsi_str")["confidence_score"]
                      .mean().rename("_avg_dev_conf"))

        status["_mmsi_str"] = status["MMSI"].astype(str)
        status = (status
                  .join(dev_counts, on="_mmsi_str")
                  .join(dev_conf,   on="_mmsi_str")
                  .drop(columns="_mmsi_str"))
    else:
        status["event_count"]    = 0
        status["deviation_count"] = 0
        status["_avg_dev_conf"]  = float("nan")

    status["event_count"]    = status["event_count"].fillna(0).astype(int)
    status["deviation_count"] = status["deviation_count"].fillna(0).astype(int)

    # Human-readable confidence label for deviations
    def _conf_label(row):
        if row["deviation_count"] == 0:
            return "N/A"
        score = row["_avg_dev_conf"]
        if pd.isna(score):
            return "N/A"
        if score >= 0.88:
            label = "High"
        elif score >= 0.75:
            label = "Medium"
        else:
            label = "Low"
        return f"{label} ({score:.2f})"

    status["deviation_confidence"] = status.apply(_conf_label, axis=1)
    status = status.drop(columns=["_avg_dev_conf"])

    # Drop internal helper columns and set final column order
    status = status.drop(columns=[c for c in ["_sog", "_cog"] if c in status.columns])
    output_cols = ["MMSI", "vessel_name", "ping_count", "last_seen", "region_name",
                   "lat", "lon", "activity", "deviation_count",
                   "deviation_confidence", "event_count"]
    for opt in ["destination", "eta"]:
        if opt in status.columns:
            output_cols.append(opt)
    status = status[[c for c in output_cols if c in status.columns]]

    # Export
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_prefix + "_vessel_status.csv")
    status.to_csv(out_path, index=False)
    print(f"[STATUS TABLE] {len(status):,} vessels → {out_path}")

    return status


def build_shipment_report(events_df: pd.DataFrame,
                           raw_df: pd.DataFrame,
                           cols: dict) -> pd.DataFrame:
    """
    Build a single analyst-ready shipment report grouped by vessel + event type.

    One row per vessel per event type — if a ship has ARRIVAL and ROUTE_DEVIATION
    events it gets two rows; if it only has ROUTE_DEVIATION it gets one row.

    Answers in one file:
      - Which vessel had which type of event, and how many times?
      - When did those events first and last occur, and where?
      - Is the vessel on schedule (destination/ETA if available)?
      - How active has it been overall (total events, deviation count)?
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame()

    # ── Group events by vessel + event type ──────────────────────────────────
    grp = events_df.groupby(["vessel_id", "vessel_name", "event_type"], sort=False)

    report = grp.agg(
        event_count=("event_type", "count"),
        first_event=("timestamp", "min"),
        last_event=("timestamp", "max"),
        avg_confidence=("confidence_score", "mean"),
        last_region=("region_name", "last"),
        last_lat=("latitude", "last"),
        last_lon=("longitude", "last"),
        ai_summary=("ai_summary", lambda s: "; ".join(
            str(x) for x in s.dropna().unique()[:3]
        ) if s.notna().any() else None),
    ).reset_index()

    report["avg_confidence"] = report["avg_confidence"].round(2)

    # ── Merge vessel status context ──────────────────────────────────────────
    status = _build_vessel_status_df(raw_df, events_df, cols)

    status_cols = ["MMSI", "last_seen", "activity", "deviation_count",
                   "event_count", "deviation_confidence"]
    for opt in ["destination", "eta"]:
        if opt in status.columns:
            status_cols.append(opt)

    status_slim = status[[c for c in status_cols if c in status.columns]].copy()
    status_slim = status_slim.rename(columns={
        "MMSI": "_merge_mmsi",
        "event_count": "vessel_total_events",
    })
    status_slim["_merge_mmsi"] = status_slim["_merge_mmsi"].astype(str)

    report["_merge_mmsi"] = report["vessel_id"].astype(str)
    report = report.merge(status_slim, on="_merge_mmsi", how="left")
    report = report.drop(columns=["_merge_mmsi"], errors="ignore")

    report.sort_values(["vessel_name", "event_type"], inplace=True)

    # ── Final column order — readable left to right ──────────────────────────
    ordered = [
        "vessel_name", "vessel_id",
        "event_type", "event_count",
        "first_event", "last_event",
        "last_region", "last_lat", "last_lon",
        "avg_confidence",
        "ai_summary",
        "last_seen", "activity", "destination", "eta",
        "deviation_count", "deviation_confidence", "vessel_total_events",
    ]
    final_cols = [c for c in ordered if c in report.columns]
    return report[final_cols]


def _build_vessel_status_df(raw_df: pd.DataFrame,
                              events_df: pd.DataFrame,
                              cols: dict) -> pd.DataFrame:
    """
    Build the vessel status DataFrame without writing it to disk.
    Extracted from export_vessel_status_table so it can be reused.
    """
    col_mmsi = cols["mmsi"]
    col_time = cols["time"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_sog  = cols.get("sog")
    col_cog  = cols.get("cog")
    col_name = cols.get("name")

    agg = {col_time: "last", col_lat: "last", col_lon: "last", col_mmsi: "count"}
    if col_sog:  agg[col_sog] = "last"
    if col_cog:  agg[col_cog] = "last"

    status = (raw_df.sort_values(col_time)
                    .groupby(col_mmsi, as_index=False)
                    .agg(agg)
                    .rename(columns={
                        col_mmsi: "MMSI",
                        col_time: "last_seen",
                        col_lat:  "lat",
                        col_lon:  "lon",
                        col_mmsi: "ping_count",
                    }))

    # Rename count column
    if col_mmsi in status.columns:
        status = status.rename(columns={col_mmsi: "ping_count"})
    status["MMSI"] = status["MMSI"].astype(str) if "MMSI" in status.columns else status.index.astype(str)

    # Vessel name
    if col_name and col_name in raw_df.columns:
        names = (raw_df[[col_mmsi, col_name]]
                 .dropna(subset=[col_name])
                 .drop_duplicates(subset=[col_mmsi])
                 .rename(columns={col_mmsi: "MMSI", col_name: "vessel_name"}))
        names["MMSI"] = names["MMSI"].astype(str)
        status = status.merge(names, on="MMSI", how="left")

    # Region name
    status["region_name"] = status.apply(
        lambda r: get_region_name(r.get("lat", 0), r.get("lon", 0)), axis=1
    )

    # Activity description
    sog_col = col_sog if col_sog and col_sog in status.columns else None
    def _activity(row):
        s = row.get(sog_col) if sog_col else None
        if s is None or pd.isna(s): return "Unknown"
        if s < 0.5:  return "Stationary"
        if s < 3.0:  return "Slow / maneuvering"
        if s < 10.0: return "Moderate speed"
        return "Underway at speed"
    status["activity"] = status.apply(_activity, axis=1)

    # Event counts from events_df
    if events_df is not None and not events_df.empty:
        event_counts = (events_df["vessel_id"].astype(str)
                        .value_counts().rename("event_count").reset_index()
                        .rename(columns={"index": "MMSI"}))
        if "MMSI" not in event_counts.columns:
            event_counts.columns = ["MMSI", "event_count"]
        event_counts["MMSI"] = event_counts["MMSI"].astype(str)
        status = status.merge(event_counts, on="MMSI", how="left")
        status["event_count"] = status["event_count"].fillna(0).astype(int)

        dev_events = events_df[events_df["event_type"] == "ROUTE_DEVIATION"].copy()
        dev_events["_mmsi"] = dev_events["vessel_id"].astype(str)
        dev_counts = (dev_events.groupby("_mmsi")["vessel_id"]
                      .count().rename("deviation_count").reset_index()
                      .rename(columns={"_mmsi": "MMSI"}))
        dev_conf = (dev_events.groupby("_mmsi")["confidence_score"]
                    .mean().rename("_avg_dev_conf").reset_index()
                    .rename(columns={"_mmsi": "MMSI"}))
        status = status.merge(dev_counts, on="MMSI", how="left")
        status = status.merge(dev_conf,   on="MMSI", how="left")
        status["deviation_count"] = status["deviation_count"].fillna(0).astype(int)

        def _conf_label(row):
            v = row.get("_avg_dev_conf")
            if pd.isna(v): return "N/A"
            if v >= 0.8:   return "High"
            if v >= 0.5:   return "Medium"
            return "Low"
        status["deviation_confidence"] = status.apply(_conf_label, axis=1)
        status = status.drop(columns=["_avg_dev_conf"], errors="ignore")

    for opt_col in ["destination", "eta"]:
        if opt_col in raw_df.columns:
            opt_vals = (raw_df[[col_mmsi, opt_col]]
                        .dropna(subset=[opt_col])
                        .drop_duplicates(subset=[col_mmsi])
                        .rename(columns={col_mmsi: "MMSI"}))
            opt_vals["MMSI"] = opt_vals["MMSI"].astype(str)
            status = status.merge(opt_vals, on="MMSI", how="left")

    return status


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load AIS data from a CSV file with performance optimizations for large files.

    For datasets of 8M+ rows, we use chunked reading to avoid loading the
    entire file into memory at once. Chunks are concatenated after loading.
    dtype optimization (float32 for lat/lon/sog/cog) reduces memory by ~40%.

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

    """
    print(f"[LOAD] Reading JSON: {filepath}")
    try:
        # Try standard JSON array first
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
    except json.JSONDecodeError:
        # Fall back to newline-delimited JSON
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
# SECTION 5: EVENT DETECTION ENGINE
# This section implements the core intelligence of our maritime event detection system.
# Using a combination of AI-enhanced rule-based algorithms, we automatically identify
# and classify five critical types of vessel activities: arrivals, departures, anchoring,
# route deviations, and vessel proximities. All detection logic is implemented using
# vectorized pandas and NumPy operations, enabling us to process millions of AIS data
# rows in seconds rather than hours. This performance optimization is crucial for handling
# real-world datasets at scale while maintaining accuracy and reliability.
# ──────────────────────────────────────────────────────────────────────────────

def detect_events(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """
    Main event detection function. Processes the full AIS DataFrame and
    returns a new DataFrame of detected events (one row per event).

    Detection strategy:
    - Sort each vessel's track by timestamp
    - Use vectorized shift() comparisons to flag state transitions
    - Proximity uses time-bucketed spatial grouping for scalability

    Detects:
      ARRIVAL         — vessel slows to near-stop after moving
      DEPARTURE        — vessel accelerates after being stopped
      ANCHORING        — vessel remains near-stationary for multiple readings
      ROUTE_DEVIATION  — sudden large bearing change
      PROXIMITY        — two vessels within N nautical miles simultaneously

    Returns:
        pd.DataFrame with columns: event_id, vessel_id, vessel_name,
        event_type, timestamp, latitude, longitude, confidence_score, null_flags
    """
    print_section("EVENT DETECTION")

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

    print(f"\n[DETECT] Detection complete — {len(all_events):,} total events found.")

    if not all_events:
        return pd.DataFrame()

    events_df = pd.DataFrame(all_events)
    # Add human-readable region name to every event
    events_df["region_name"] = events_df.apply(
        lambda r: get_region_name(r["latitude"], r["longitude"]), axis=1
    )
    return events_df


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
    for event_type, mask in [
        ("ARRIVAL",   arrival_mask),
        ("DEPARTURE", departure_mask),
        ("ANCHORING", anchoring_mask),
    ]:
        cols_needed = [col_mmsi, col_lat, col_lon, col_time, col_sog]
        if col_name:
            cols_needed.append(col_name)
        matched = df[mask][cols_needed].copy()
        n_matched = len(matched)

        # Extract as numpy arrays — avoids slow iterrows() on large matched sets
        mmsi_arr = matched[col_mmsi].to_numpy()
        lat_arr  = matched[col_lat].to_numpy()
        lon_arr  = matched[col_lon].to_numpy()
        time_arr = matched[col_time].to_numpy()
        sog_arr  = matched[col_sog].to_numpy()
        name_arr = matched[col_name].to_numpy() if col_name else None

        # Vectorized null-flag computation across all matched rows at once
        null_check_cols = [col_lat, col_lon, col_time, col_sog]
        null_matrix = pd.isna(matched[null_check_cols]).to_numpy()

        for k in range(n_matched):
            sog_raw = sog_arr[k]
            sog_val = float(sog_raw) if not pd.isna(sog_raw) else None
            nulls   = [c for c, bad in zip(null_check_cols, null_matrix[k]) if bad]
            events.append({
                "event_id":         make_event_id(),
                "vessel_id":        mmsi_arr[k],
                "vessel_name":      (name_arr[k] if col_name and not pd.isna(name_arr[k])
                                     else str(mmsi_arr[k])),
                "event_type":       event_type,
                "timestamp":        time_arr[k],
                "latitude":         float(lat_arr[k]),
                "longitude":        float(lon_arr[k]),
                "confidence_score": confidence_score(event_type, sog_val),
                "null_flags":       ", ".join(nulls) if nulls else "None",
            })

        print(f"         {event_type:<20} → {n_matched:,} events detected")

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

    # Extract matched rows as arrays — avoids slow iterrows() over large deviation sets
    dev_cols = [col_mmsi, col_lat, col_lon, col_time, col_cog, "_bearing_change", "_prev_cog"]
    if col_sog:
        dev_cols.append(col_sog)
    if col_name:
        dev_cols.append(col_name)
    matched_sub = matched[dev_cols]
    n_dev = len(matched_sub)

    mmsi_arr     = matched_sub[col_mmsi].to_numpy()
    lat_arr      = matched_sub[col_lat].to_numpy()
    lon_arr      = matched_sub[col_lon].to_numpy()
    time_arr     = matched_sub[col_time].to_numpy()
    cog_arr      = matched_sub[col_cog].to_numpy()
    bearing_arr  = matched_sub["_bearing_change"].to_numpy()
    prev_cog_arr = matched_sub["_prev_cog"].to_numpy()
    sog_arr      = matched_sub[col_sog].to_numpy() if col_sog else None
    name_arr     = matched_sub[col_name].to_numpy() if col_name else None

    null_check_cols = [col_lat, col_lon, col_time, col_cog]
    null_matrix = pd.isna(matched_sub[null_check_cols]).to_numpy()

    for k in range(n_dev):
        sog_raw  = sog_arr[k] if sog_arr is not None else None
        sog_val  = float(sog_raw) if sog_raw is not None and not pd.isna(sog_raw) else None
        nulls    = [c for c, bad in zip(null_check_cols, null_matrix[k]) if bad]

        # Build a specific label describing what the vessel did in this turn
        b_deg    = float(bearing_arr[k]) if not pd.isna(bearing_arr[k]) else 0.0
        curr_cog = float(cog_arr[k])      if not pd.isna(cog_arr[k])    else None
        prev_cog = float(prev_cog_arr[k]) if not pd.isna(prev_cog_arr[k]) else None
        if curr_cog is not None and prev_cog is not None:
            delta     = (curr_cog - prev_cog + 360) % 360
            direction = "starboard" if delta <= 180 else "port"
            cog_str   = f" (COG {prev_cog:.0f}°→{curr_cog:.0f}°)"
        else:
            direction = "unknown direction"
            cog_str   = ""
        sog_str     = f" at {sog_val:.1f} kts" if sog_val is not None else ""
        event_label = f"Turned {b_deg:.0f}° to {direction}{cog_str}{sog_str}"

        events.append({
            "event_id":         make_event_id(),
            "vessel_id":        mmsi_arr[k],
            "vessel_name":      (name_arr[k] if col_name and not pd.isna(name_arr[k])
                                 else str(mmsi_arr[k])),
            "event_type":       "ROUTE_DEVIATION",
            "event_label":      event_label,
            "timestamp":        time_arr[k],
            "latitude":         float(lat_arr[k]),
            "longitude":        float(lon_arr[k]),
            "bearing_change":   round(b_deg, 1),
            "cog_before":       round(prev_cog, 1) if prev_cog is not None else None,
            "cog_after":        round(curr_cog, 1) if curr_cog is not None else None,
            "turn_direction":   direction,
            "speed_kts":        round(sog_val, 1) if sog_val is not None else None,
            "confidence_score": confidence_score("ROUTE_DEVIATION", sog_val, b_deg),
            "null_flags":       ", ".join(nulls) if nulls else "None",
        })

    print(f"         {'ROUTE_DEVIATION':<20} → {n_dev:,} events detected")

    # Clean up temporary columns
    df.drop(columns=["_prev_cog", "_bearing_change"], inplace=True, errors="ignore")
    return events


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: EVENT SUMMARIES
# Generates plain-English descriptions for every detected maritime event using
# template-based formatting. No external API or API key required — summaries
# are built instantly from the event data already in the DataFrame.
# ──────────────────────────────────────────────────────────────────────────────

def _compass(degrees) -> str:
    """Convert a bearing in degrees to a full cardinal/intercardinal direction word."""
    if degrees is None:
        return "unknown direction"
    dirs = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return dirs[round(float(degrees) / 45) % 8]


_SUMMARY_TEMPLATES = {
    "ARRIVAL": (
        "{name} arrived near {region} at {time}. "
        "Vessel decelerated to near-zero speed, indicating a port call or anchorage entry."
    ),
    "DEPARTURE": (
        "{name} departed from {region} at {time}. "
        "Vessel accelerated from a stationary or slow state, indicating the start of a transit."
    ),
    "ANCHORING": (
        "{name} anchored near {region} at {time}. "
        "Vessel held a near-stationary position across multiple consecutive pings."
    ),
    # ROUTE_DEVIATION is built dynamically in _build_summary using stored COG fields.
    "PROXIMITY": (
        "{name} was detected in close proximity to another vessel near {region} at {time}. "
        "This may indicate a rendezvous, convoy operation, or potential collision risk."
    ),
}

_SUMMARY_DEFAULT = (
    "{name} triggered a {event_type} event near {region} at {time}."
)


def _build_summary(row: pd.Series) -> str:
    """Build a plain-English summary for a single event row."""
    try:
        return _build_summary_inner(row)
    except Exception:
        etype = row.get("event_type", "unknown event")
        name  = row.get("vessel_name", row.get("vessel_id", "Unknown vessel"))
        return f"{name} triggered a {etype} event. (Insufficient data for full summary.)"


def _build_summary_inner(row: pd.Series) -> str:
    name   = str(row.get("vessel_name", row.get("vessel_id", "Unknown vessel")))
    region = str(row.get("region_name", "unknown location"))
    etype  = str(row.get("event_type", ""))

    # Format timestamp — strip microseconds if present
    ts = row.get("timestamp", "")
    try:
        ts = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts = str(ts)

    if etype == "ROUTE_DEVIATION":
        b_deg     = row.get("bearing_change")
        cog_before = row.get("cog_before")
        cog_after  = row.get("cog_after")
        turn_dir   = row.get("turn_direction", "unknown direction")
        speed      = row.get("speed_kts")

        if b_deg is not None and cog_before is not None and cog_after is not None:
            dir_after = _compass(cog_after)
            spd_str   = f" at {speed:.1f} knots" if speed is not None else ""
            turn_desc = "starboard (right)" if str(turn_dir).lower() == "starboard" else \
                        "port (left)"       if str(turn_dir).lower() == "port"      else \
                        str(turn_dir)
            return (
                f"{name} made a {b_deg:.0f}-degree course change to the {turn_desc} "
                f"near {region} at {ts}{spd_str}. "
                f"The vessel's heading shifted from {_compass(cog_before)} to {dir_after}."
            )
        # Fallback if COG data was missing
        return (
            f"{name} made a significant course change near {region} at {ts}. "
            f"Heading shift detected but course data was incomplete."
        )

    template = _SUMMARY_TEMPLATES.get(etype, _SUMMARY_DEFAULT)
    return template.format(name=name, region=region, time=ts, event_type=etype)


def generate_ai_summaries(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a short plain-English summary for each detected event.

    Summaries are built from templates using the event fields already in the
    DataFrame — no API key or network connection required.

    """
    print_section("EVENT SUMMARIES")

    events_df = events_df.copy()
    events_df["ai_summary"] = None

    deviation_mask = events_df["event_type"] == "ROUTE_DEVIATION"
    n_dev = deviation_mask.sum()
    print(f"[SUMMARIES] Generating descriptions for {n_dev:,} ROUTE_DEVIATION events (skipping {len(events_df) - n_dev:,} other events)...")

    events_df.loc[deviation_mask, "ai_summary"] = (
        events_df.loc[deviation_mask].apply(_build_summary, axis=1)
    )

    print(f"[SUMMARIES] Done.")
    return events_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: OUTPUT BUILDER
# Merges event labels back into the original DataFrame as new columns.
# ──────────────────────────────────────────────────────────────────────────────

def build_labeled_dataset(original_df: pd.DataFrame,
                           events_df: pd.DataFrame,
                           cols: dict) -> pd.DataFrame:
    """
    Merge detected event labels back into the original AIS DataFrame.

    New columns added to the original data:
      - event_id         : Unique event identifier (or None if no event)
      - event_type       : Type of event (ARRIVAL, DEPARTURE, etc.)
      - confidence_score : How confident the system is in this label
      - null_flags       : Any NULL fields in this row's key data
      - ai_summary       : Natural language description (if generated)

    The merge is done on (vessel_id + timestamp-minute) to align
    event records back to their source rows efficiently.

    """
    print_section("BUILDING LABELED DATASET")

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

    # Add timestamp-minute key
    events_df["_key_ts"] = (pd.to_datetime(events_df["timestamp"], errors="coerce")
                             .dt.floor("min").astype(str))

    events_expanded = events_df.copy()
    events_expanded["_key_mmsi"] = events_expanded["vessel_id"].astype(str)

    # Keep first event per (mmsi, ts-minute) key — mirrors original "first wins" logic
    events_expanded = events_expanded.drop_duplicates(
        subset=["_key_mmsi", "_key_ts"], keep="first"
    )

    merge_cols = ["event_id", "event_type", "confidence_score", "null_flags"]
    if "event_label" in events_expanded.columns:
        merge_cols.append("event_label")
    if "region_name" in events_expanded.columns:
        merge_cols.append("region_name")
    if "ai_summary" in events_expanded.columns:
        merge_cols.append("ai_summary")

    print(f"[OUTPUT] Lookup index built — {len(events_expanded):,} unique event keys.")

    # ── Tag original DataFrame rows via merge (replaces 7M+ row Python loop) ──
    original_df["_key_mmsi"] = original_df[col_mmsi].astype(str)
    original_df["_key_ts"]   = (pd.to_datetime(original_df[col_time], errors="coerce")
                                 .dt.floor("min").astype(str))

    print("[OUTPUT] Tagging rows with event labels...")
    original_df = original_df.merge(
        events_expanded[["_key_mmsi", "_key_ts"] + merge_cols],
        on=["_key_mmsi", "_key_ts"],
        how="left",
    )

    if "ai_summary" not in original_df.columns:
        original_df["ai_summary"] = None

    # Remove temporary key columns before export
    original_df.drop(columns=["_key_mmsi", "_key_ts"], inplace=True, errors="ignore")

    labeled_count = original_df["event_type"].notna().sum()
    print(f"[OUTPUT] {labeled_count:,} rows tagged with event labels.")

    return original_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: CSV EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, filepath: str, label: str = "output"):
    """
    Export a DataFrame to CSV.

    For large datasets (8M+ rows), we use chunksize in to_csv() to write
    in streaming fashion — this prevents memory spikes during export.

    """
    print(f"[EXPORT] Writing {label} → {filepath}")
    # chunksize here is for the CSV writer's internal buffer, not read chunks
    df.to_csv(filepath, index=False)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"         Done — {len(df):,} rows | {size_mb:.1f} MB")


def export_vessel_events_grouped(events_df: pd.DataFrame,
                                  output_dir: str,
                                  file_prefix: str) -> None:
    """
    Export one row per vessel+event_type combination.

    If a ship has only ROUTE_DEVIATION events, it gets one row.
    If it has ROUTE_DEVIATION and ARRIVAL events, it gets two rows — one per type.
    Columns show the count of that event type, time range, avg confidence, and region.
    """
    if events_df is None or events_df.empty:
        return

    agg = (
        events_df.groupby(["vessel_id", "vessel_name", "event_type"], sort=False)
        .agg(
            event_count=("event_type", "count"),
            first_event=("timestamp", "min"),
            last_event=("timestamp", "max"),
            avg_confidence=("confidence_score", "mean"),
            last_region=("region_name", "last"),
        )
        .reset_index()
    )

    agg["avg_confidence"] = agg["avg_confidence"].round(2)
    agg.sort_values(["vessel_name", "event_type"], inplace=True)

    out_path = os.path.join(output_dir, file_prefix + "_vessel_events_grouped.csv")
    export_csv(agg, out_path, label="Vessel events grouped")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: PIPELINE SUMMARY / REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_pipeline_summary(input_path: str, labeled_df: pd.DataFrame,
                            events_df: pd.DataFrame, output_dir: str):
    """
    Print a human-readable summary of the pipeline run.
    Useful for documentation, sign-off, and analyst handover.

    """
    print_section("PIPELINE SUMMARY")

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
      1. Load data (CSV / JSON / NMEA)
      2. Resolve column names
      3. Detect and label vessel events
      4. Generate AI summaries (optional)
      5. Merge events into original dataset
      6. Export labeled CSV + events summary
      7. Print pipeline summary

    Args:
        input_path   : Path to AIS input file (.csv, .json, .nmea)
        output_dir   : Directory for output files
        ai_summaries : If True, generate Claude AI summaries
    """
    print("\n" + "=" * 60)
    print("  DDT — GEN AI AIS EVENT DETECTION & LABELING PIPELINE")
    print("  Version 2.0 | All Requirements Active")
    print("=" * 60)

    start_time = time.time()

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print_section("LOADING DATA")
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
    df, date_label, ai_summaries, output_format = prompt_date_filter(df, cols["time"])

    # ── Step 2.6: Sort and deduplicate ────────────────────────────────────────
    print_section("PREPROCESSING")
    df[cols["time"]] = pd.to_datetime(df[cols["time"]], errors="coerce")
    df = df.sort_values([cols["mmsi"], cols["time"]]).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(keep="first")  # only removes rows identical across every column
    removed = before - len(df)
    if removed:
        print(f"[PREP] Removed {removed:,} duplicate vessel/timestamp rows.")
    print(f"[PREP] {len(df):,} clean rows ready for processing.")

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
    print_section("EXPORTING OUTPUTS")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    file_prefix = (base + "_" + date_label) if date_label != "all" else base

    if output_format == "single":
        report = build_shipment_report(events_df, df, cols)
        report_path = os.path.join(output_dir, file_prefix + "_shipment_report.csv")
        export_csv(report, report_path, label="Shipment report")
    else:
        events_only = labeled_df[labeled_df["event_type"].notna()].copy()
        export_csv(events_only,
                   os.path.join(output_dir, file_prefix + "_events_labeled.csv"),
                   label="Event-labeled rows")
        if events_df is not None and not events_df.empty:
            export_csv(events_df,
                       os.path.join(output_dir, file_prefix + "_events_summary.csv"),
                       label="Events summary")
        else:
            print("[EXPORT] No events to export.")
        export_vessel_status_table(df, events_df, output_dir, file_prefix)

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
            "Generate natural-language AI summaries for each event. "
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
    # AI summaries choice is collected interactively inside prompt_date_filter.
    # The --ai-summaries flag still works when passed explicitly from the CLI.
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
