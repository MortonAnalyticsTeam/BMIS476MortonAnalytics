"""
DDT — GEN AI DATA PROCESSING AND ANALYTICS SOLUTION
AIS Vessel Event Detection, Labeling & AI Summary Pipeline
Team: DDT | Version: 2.0

Install dependencies:
    pip install pandas numpy anthropic pyais
Set API key:
    Windows : set ANTHROPIC_API_KEY=your_key_here
    Mac/Linux: export ANTHROPIC_API_KEY=your_key_here
"""

import os
import sys
import json
import uuid
import time
import argparse
import warnings
from datetime import datetime, timezone
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import pyais  # noqa: F401
except ImportError:
    pyais = None  # type: ignore

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1A: CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "arrival_speed_max":    1.0,
    "departure_speed_min":  1.5,
    "anchoring_speed_max":  0.5,
    "stopped_min_rows":     2,
    "deviation_bearing_threshold": 45.0,
    "chunk_size": 500_000,
    "ai_summary_max_events": 500,
    "ai_summary_delay_sec": 0.3,
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
# MARITIME REGION LOOKUP TABLE — ordered most-specific first
# ──────────────────────────────────────────────────────────────────────────────
MARITIME_REGIONS = [
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
    ("Port of New Orleans",              29.80, 30.20, -90.50, -89.80),
    ("Mississippi River Delta",          28.50, 30.00, -91.50, -88.80),
    ("Mobile Bay",                       30.00, 30.80, -88.30, -87.80),
    ("Port of Houston / Galveston Bay",  29.20, 29.90, -95.20, -94.50),
    ("Port of Corpus Christi",           27.70, 28.00, -97.50, -97.10),
    ("Port of Los Angeles / Long Beach", 33.50, 34.10, -118.55, -117.90),
    ("San Francisco Bay",                37.30, 38.20, -122.70, -122.10),
    ("Port of Portland, OR",             45.40, 45.70, -122.90, -122.60),
    ("Puget Sound",                      47.00, 48.60, -123.00, -122.00),
    ("Strait of Juan de Fuca",           48.00, 48.70, -124.70, -122.50),
    ("Prince William Sound",             59.50, 61.50, -148.50, -145.50),
    ("Gulf of Alaska",                   54.00, 62.00, -162.00, -135.00),
    ("Gulf of Mexico",                   18.00, 31.00,  -98.00,  -80.00),
    ("Caribbean Sea",                     8.00, 23.50,  -87.00,  -59.00),
    ("US East Coast (offshore)",         25.00, 47.00,  -80.00,  -65.00),
    ("US West Coast (offshore)",         32.00, 50.00, -130.00, -117.00),
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
    ("North Atlantic Ocean",              0.00, 70.00,  -80.00,   0.00),
    ("South Atlantic Ocean",            -70.00,  0.00,  -65.00,  20.00),
    ("North Pacific Ocean",               0.00, 66.00, -180.00, -100.00),
    ("South Pacific Ocean",             -70.00,  0.00, -180.00,  -65.00),
    ("Indian Ocean",                    -70.00, 25.00,   20.00, 150.00),
    ("Southern Ocean",                  -90.00,-60.00, -180.00, 180.00),
    ("Arctic Ocean",                     66.00, 90.00, -180.00, 180.00),
]

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1B: DATA IMPORT
# Opens a file browser, loads the dataset into memory, and saves a small preview.
# Usage:
#   df = import_dataset()
#   labeled_df, events_df = run_pipeline_from_df(df)
#   labeled_df, events_df = run_pipeline_from_df(df, ai_summaries=True)
# ──────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import filedialog

def import_dataset(preview_rows: int = 10,
                   save_preview: bool = True,
                   preview_dir: str = "./preview") -> pd.DataFrame:
    """
    Opens a file browser to select an AIS dataset. The full dataset is loaded
    into memory only — it is NOT written to the project folder.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

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

    root.destroy()

    if not filepath:
        print("[IMPORT] No file selected. Exiting.")
        return None

    print(f"[IMPORT] Selected: {filepath}")

    try:
        df = load_file(filepath)
    except Exception as e:
        print(f"[IMPORT] ERROR: Failed to load file — {e}")
        return None

    print(f"\n[IMPORT] ── Dataset Preview (first {min(preview_rows, len(df))} rows) ──")
    print(df.head(preview_rows).to_string(index=False))
    print(f"\n[IMPORT] Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[IMPORT] Columns : {list(df.columns)}")

    if save_preview and len(df) > 0:
        os.makedirs(preview_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        preview_path = os.path.join(preview_dir, f"{base_name}_preview.csv")
        df.head(preview_rows).to_csv(preview_path, index=False)
        print(f"\n[IMPORT] Preview saved → {preview_path}")

    print(f"\n[IMPORT] Full dataset ({df.shape[0]:,} rows) is loaded in memory only.\n")

    return df


def run_pipeline_from_df(df: pd.DataFrame,
                          output_dir: str = "./output",
                          ai_summaries: bool = False):
    """
    Run the full DDT pipeline on an already-loaded DataFrame.
    Skips the file-loading step and goes straight to event detection and export.

    Args:
        df           : DataFrame returned by import_dataset()
        output_dir   : Where to save the labeled output CSVs
        ai_summaries : Whether to generate Claude AI summaries
    """
    if df is None or df.empty:
        print("[ERROR] No data to process. Run import_dataset() first.")
        return None, None

    print("=" * 60)
    print("  DDT — PIPELINE RUNNING ON IMPORTED DATASET")
    print("=" * 60)

    start_time = time.time()

    print_section("COLUMN RESOLUTION")
    cols = resolve_columns(df)
    print(f"  MMSI  → '{cols['mmsi']}'")
    print(f"  LAT   → '{cols['lat']}'")
    print(f"  LON   → '{cols['lon']}'")
    print(f"  TIME  → '{cols['time']}'")
    print(f"  SOG   → {repr(cols['sog'])}")
    print(f"  COG   → {repr(cols['cog'])}")
    print(f"  NAME  → {repr(cols['name'])}")

    df, date_label = prompt_date_filter(df, cols["time"])

    print_section("PREPROCESSING")
    df[cols["time"]] = pd.to_datetime(df[cols["time"]], errors="coerce")
    df = df.sort_values([cols["mmsi"], cols["time"]]).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    if removed:
        print(f"[PREP] Removed {removed:,} duplicate vessel/timestamp rows.")
    print(f"[PREP] {len(df):,} clean rows ready for processing.")

    events_df = detect_events(df, cols)

    if ai_summaries and events_df is not None and not events_df.empty:
        events_df = generate_ai_summaries(events_df)

    labeled_df = build_labeled_dataset(df, events_df, cols)

    print_section("EXPORTING OUTPUTS")
    os.makedirs(output_dir, exist_ok=True)

    file_prefix = date_label if date_label != "all" else "full_dataset"

    events_only = labeled_df[labeled_df["event_type"].notna()].copy()
    labeled_path = os.path.join(output_dir, file_prefix + "_events_labeled.csv")
    export_csv(events_only, labeled_path, label="Event-labeled rows")

    if events_df is not None and not events_df.empty:
        events_path = os.path.join(output_dir, file_prefix + "_events_summary.csv")
        export_csv(events_df, events_path, label="Events summary")

    export_vessel_status_table(df, events_df, output_dir, file_prefix)

    print_pipeline_summary("imported_dataset", labeled_df, events_df, output_dir)

    elapsed = time.time() - start_time
    print(f"  Total runtime  : {elapsed:.1f} seconds")
    print("=" * 60 + "\n")

    return labeled_df, events_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1C: DATE FILTER
# ──────────────────────────────────────────────────────────────────────────────

def filter_by_dates(df, time_col, dates):
    """Filter DataFrame to rows matching any date in the provided YYYY-MM-DD list."""
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
    Interactive menu to process all rows or filter to specific date(s).
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
# ──────────────────────────────────────────────────────────────────────────────

def bearing_change_vectorized(cog1: pd.Series, cog2: pd.Series) -> pd.Series:
    """Vectorized absolute angular difference between two bearing Series (0–180°)."""
    diff = (cog1 - cog2).abs() % 360
    return diff.where(diff <= 180, 360 - diff)


def resolve_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first matching column name from candidates, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_event_id() -> str:
    """Generate a unique event identifier in the format EVT-XXXXXXXX."""
    return "EVT-" + str(uuid.uuid4())[:8].upper()


def confidence_score(event_type: str,
                     sog_value: Optional[float] = None,
                     bearing_change: Optional[float] = None) -> float:
    """
    Rule-based confidence scoring (0.0–1.0) for a detected event.
    Base scores are set by event type, then adjusted by SOG and bearing magnitude.
    """
    base = {
        "ARRIVAL":         0.80,
        "DEPARTURE":       0.78,
        "ANCHORING":       0.85,
        "ROUTE_DEVIATION": 0.70,
        "PROXIMITY":       0.75,
    }.get(event_type, 0.60)

    if event_type == "ROUTE_DEVIATION":
        if bearing_change is not None and not pd.isna(bearing_change):
            if bearing_change > 135:
                base = min(base + 0.17, 1.0)
            elif bearing_change > 90:
                base = min(base + 0.13, 1.0)
            elif bearing_change > 60:
                base = min(base + 0.08, 1.0)
        if sog_value is not None and not pd.isna(sog_value):
            if sog_value > 5.0:
                base = min(base + 0.05, 1.0)
            elif sog_value < 1.5:
                base = max(base - 0.10, 0.0)
    else:
        if sog_value is not None and not pd.isna(sog_value):
            if event_type in ("ARRIVAL", "ANCHORING") and sog_value < 0.2:
                base = min(base + 0.10, 1.0)
            elif event_type == "DEPARTURE" and sog_value > 5.0:
                base = min(base + 0.08, 1.0)

    return round(base, 2)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def get_region_name(lat: float, lon: float) -> str:
    """Return a human-readable maritime region name for a lat/lon coordinate."""
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
# Call lookup_vessel(labeled_df, events_df, <MMSI or name>) for a status report.
# ──────────────────────────────────────────────────────────────────────────────

def lookup_vessel(df: pd.DataFrame,
                  events_df: pd.DataFrame,
                  identifier,
                  print_report: bool = True) -> str:
    """
    Look up a vessel by MMSI or name and return a plain-English status report.
    Accepts exact MMSI or partial case-insensitive vessel name.
    """
    cols     = resolve_columns(df)
    col_mmsi = cols["mmsi"]
    col_time = cols["time"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_sog  = cols.get("sog")
    col_cog  = cols.get("cog")
    col_name = cols.get("name")

    col_dest   = resolve_column(df, CONFIG["col_dest"])
    col_eta    = resolve_column(df, CONFIG["col_eta"])
    col_status = resolve_column(df, CONFIG["col_status"])

    ident_str = str(identifier).strip()

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

    destination = (str(latest[col_dest])
                   if col_dest and not pd.isna(latest.get(col_dest, float("nan"))) else None)
    eta         = (str(latest[col_eta])
                   if col_eta and not pd.isna(latest.get(col_eta, float("nan"))) else None)
    nav_status  = (str(latest[col_status])
                   if col_status and not pd.isna(latest.get(col_status, float("nan"))) else None)

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
# Produces a one-row-per-vessel CSV via groupby().last() — efficient on 8M+ rows.
# ──────────────────────────────────────────────────────────────────────────────

def export_vessel_status_table(df: pd.DataFrame,
                                events_df,
                                output_dir: str,
                                file_prefix: str) -> pd.DataFrame:
    """
    Build and export a one-row-per-vessel status table to CSV.
    Columns: MMSI, vessel_name, last_seen, region_name, lat, lon,
             activity, event_count (+ destination/eta if present).
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

    keep = [col_mmsi, col_time, col_lat, col_lon]
    for c in [col_sog, col_cog, col_name, col_dest, col_eta]:
        if c and c in df.columns and c not in keep:
            keep.append(c)

    status = (df[keep]
              .sort_values(col_time)
              .groupby(col_mmsi, sort=False)
              .last()
              .reset_index())

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

    if "vessel_name" not in status.columns:
        status["vessel_name"] = status["MMSI"].astype(str)
    else:
        status["vessel_name"] = (
            status["vessel_name"]
            .astype(str)
            .replace("nan", pd.NA)
            .fillna(status["MMSI"].astype(str))
        )

    status["region_name"] = status.apply(
        lambda r: get_region_name(float(r["lat"]), float(r["lon"])), axis=1
    )

    status["activity"] = status.apply(
        lambda r: describe_vessel_status(
            float(r["_sog"]) if "_sog" in r and pd.notna(r["_sog"]) else None,
            float(r["_cog"]) if "_cog" in r and pd.notna(r["_cog"]) else None,
        ),
        axis=1,
    )

    ping_counts = df[col_mmsi].value_counts().rename("ping_count")
    status["ping_count"] = status["MMSI"].map(ping_counts).fillna(0).astype(int)

    if events_df is not None and not events_df.empty:
        event_counts = (events_df["vessel_id"]
                        .astype(str)
                        .value_counts()
                        .rename("event_count"))
        status["_mmsi_str"] = status["MMSI"].astype(str)
        status = status.join(event_counts, on="_mmsi_str").drop(columns="_mmsi_str")

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

    status = status.drop(columns=[c for c in ["_sog", "_cog"] if c in status.columns])
    output_cols = ["MMSI", "vessel_name", "ping_count", "last_seen", "region_name",
                   "lat", "lon", "activity", "deviation_count",
                   "deviation_confidence", "event_count"]
    for opt in ["destination", "eta"]:
        if opt in status.columns:
            output_cols.append(opt)
    status = status[[c for c in output_cols if c in status.columns]]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_prefix + "_vessel_status.csv")
    status.to_csv(out_path, index=False)
    print(f"[STATUS TABLE] {len(status):,} vessels → {out_path}")

    return status


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load AIS data from a CSV using chunked reading and dtype optimization
    to handle 8M+ row datasets with reduced memory usage.
    """
    print(f"[LOAD] Reading CSV: {filepath}")

    dtype_hints = {
        "LAT": "float32", "Latitude": "float32", "lat": "float32",
        "LON": "float32", "Longitude": "float32", "lon": "float32",
        "SOG": "float32", "sog": "float32", "Speed": "float32",
        "COG": "float32", "cog": "float32", "Course": "float32",
    }

    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=CONFIG["chunk_size"],
                              low_memory=False, dtype=dtype_hints,
                              on_bad_lines="skip"):
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"       ...loaded {total_rows:,} rows", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n[LOAD] CSV complete — {len(df):,} rows, {len(df.columns)} columns.")
    return df


def load_json(filepath: str) -> pd.DataFrame:
    """
    Load AIS data from a JSON file.
    Supports standard JSON arrays and newline-delimited JSON (NDJSON).
    """
    print(f"[LOAD] Reading JSON: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
    except json.JSONDecodeError:
        print("[LOAD] Standard JSON parse failed — trying newline-delimited JSON...")
        df = pd.read_json(filepath, lines=True)

    print(f"[LOAD] JSON complete — {len(df):,} rows, {len(df.columns)} columns.")
    return df


def load_nmea(filepath: str) -> pd.DataFrame:
    """
    Parse NMEA 0183 AIS sentences into a DataFrame.
    Extracts MMSI, LAT, LON, SOG, COG. Requires: pip install pyais
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
                record = {
                    "MMSI":         getattr(decoded, "mmsi", None),
                    "LAT":          getattr(decoded, "lat",  None),
                    "LON":          getattr(decoded, "lon",  None),
                    "SOG":          getattr(decoded, "speed", None),
                    "COG":          getattr(decoded, "course", None),
                    "VesselName":   getattr(decoded, "shipname", None),
                    "BaseDateTime": datetime.now(timezone.utc).isoformat(),
                }
                records.append(record)
            except Exception:
                skipped += 1
                continue

    df = pd.DataFrame(records).dropna(subset=["MMSI", "LAT", "LON"])
    print(f"[LOAD] NMEA complete — {len(df):,} valid rows decoded. ({skipped} skipped)")
    return df


def load_file(filepath: str) -> pd.DataFrame:
    """Auto-detect input file type and route to the correct loader (.csv, .json, .nmea/.txt/.ais)."""
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
# Maps dataset column names to internal standardized keys using CONFIG aliases.
# ──────────────────────────────────────────────────────────────────────────────

def resolve_columns(df: pd.DataFrame) -> dict:
    """
    Resolve actual DataFrame column names against CONFIG aliases.
    Required columns (mmsi, lat, lon, time) raise an error if not found.
    Optional columns (sog, cog, name) return None — pipeline degrades gracefully.
    """
    mapping = {}

    required = {
        "mmsi": CONFIG["col_mmsi"],
        "lat":  CONFIG["col_lat"],
        "lon":  CONFIG["col_lon"],
        "time": CONFIG["col_time"],
    }

    optional = {
        "sog":  CONFIG["col_sog"],
        "cog":  CONFIG["col_cog"],
        "name": CONFIG["col_name"],
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
        mapping[key] = resolve_column(df, candidates)

    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: EVENT DETECTION ENGINE
# Vectorized detection of ARRIVAL, DEPARTURE, ANCHORING, and ROUTE_DEVIATION.
# ──────────────────────────────────────────────────────────────────────────────

def detect_events(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """
    Main event detection function. Returns a DataFrame of detected events
    (one row per event) with columns: event_id, vessel_id, vessel_name,
    event_type, timestamp, latitude, longitude, confidence_score, null_flags.
    """
    print_section("EVENT DETECTION")

    col_mmsi = cols["mmsi"]
    col_time = cols["time"]
    col_sog  = cols["sog"]
    col_cog  = cols["cog"]

    all_events = []

    print("[DETECT] Parsing timestamps and sorting...")
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    df = df.sort_values([col_mmsi, col_time]).reset_index(drop=True)

    n_vessels = df[col_mmsi].nunique()
    print(f"[DETECT] {len(df):,} rows | {n_vessels:,} unique vessels")

    if col_sog:
        print("[DETECT] Running speed-based event detection (arrival / departure / anchoring)...")
        all_events += _detect_speed_events(df, cols)
    else:
        print("[DETECT] WARNING: SOG column not found — skipping speed-based events.")

    if col_cog:
        print("[DETECT] Running course deviation detection...")
        all_events += _detect_route_deviations(df, cols)
    else:
        print("[DETECT] WARNING: COG column not found — skipping route deviation detection.")

    print(f"\n[DETECT] Detection complete — {len(all_events):,} total events found.")

    if not all_events:
        return pd.DataFrame()

    events_df = pd.DataFrame(all_events)
    events_df["region_name"] = events_df.apply(
        lambda r: get_region_name(r["latitude"], r["longitude"]), axis=1
    )
    return events_df


def _detect_speed_events(df: pd.DataFrame, cols: dict) -> list:
    """
    Vectorized detection of ARRIVAL, DEPARTURE, and ANCHORING using shift()
    comparisons per vessel. Anchoring uses a rolling minimum SOG window.
    """
    col_mmsi = cols["mmsi"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_time = cols["time"]
    col_sog  = cols["sog"]
    col_name = cols["name"]

    events = []

    df["_prev_sog"] = df.groupby(col_mmsi)[col_sog].shift(1)

    arr_max  = CONFIG["arrival_speed_max"]
    dep_min  = CONFIG["departure_speed_min"]
    anc_max  = CONFIG["anchoring_speed_max"]

    arrival_mask = (
        df["_prev_sog"].notna() &
        (df["_prev_sog"] > arr_max) &
        (df[col_sog] <= arr_max)
    )

    departure_mask = (
        df["_prev_sog"].notna() &
        (df["_prev_sog"] <= arr_max) &
        (df[col_sog] >= dep_min)
    )

    min_rows = CONFIG["stopped_min_rows"]
    df["_rolling_min_sog"] = (
        df.groupby(col_mmsi)[col_sog]
        .transform(lambda x: x.rolling(min_rows, min_periods=min_rows).min())
    )
    anchoring_mask = df["_rolling_min_sog"].notna() & (df["_rolling_min_sog"] <= anc_max)

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

        mmsi_arr = matched[col_mmsi].to_numpy()
        lat_arr  = matched[col_lat].to_numpy()
        lon_arr  = matched[col_lon].to_numpy()
        time_arr = matched[col_time].to_numpy()
        sog_arr  = matched[col_sog].to_numpy()
        name_arr = matched[col_name].to_numpy() if col_name else None

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

    df.drop(columns=["_prev_sog", "_rolling_min_sog"], inplace=True, errors="ignore")
    return events


def _detect_route_deviations(df: pd.DataFrame, cols: dict) -> list:
    """
    Vectorized detection of ROUTE_DEVIATION events by comparing each vessel's
    current COG against its previous COG. Handles 0°/360° wrap-around correctly.
    """
    col_mmsi = cols["mmsi"]
    col_lat  = cols["lat"]
    col_lon  = cols["lon"]
    col_time = cols["time"]
    col_cog  = cols["cog"]
    col_sog  = cols["sog"]
    col_name = cols["name"]

    events = []

    df["_prev_cog"] = df.groupby(col_mmsi)[col_cog].shift(1)
    df["_bearing_change"] = bearing_change_vectorized(df[col_cog], df["_prev_cog"])

    threshold = CONFIG["deviation_bearing_threshold"]

    deviation_mask = (
        df["_prev_cog"].notna() &
        df[col_cog].notna() &
        (df["_bearing_change"] >= threshold)
    )

    matched = df[deviation_mask]

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
            "confidence_score": confidence_score("ROUTE_DEVIATION", sog_val, b_deg),
            "null_flags":       ", ".join(nulls) if nulls else "None",
        })

    print(f"         {'ROUTE_DEVIATION':<20} → {n_dev:,} events detected")

    df.drop(columns=["_prev_cog", "_bearing_change"], inplace=True, errors="ignore")
    return events


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: AI NATURAL LANGUAGE SUMMARIES
# Calls the Claude API to generate a plain-English summary for each event.
# ──────────────────────────────────────────────────────────────────────────────

def generate_ai_summaries(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a 1-sentence AI summary per event using the Claude API.
    Adds an 'ai_summary' column to the events DataFrame.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    print_section("AI EVENT SUMMARIES")

    if anthropic is None:
        print("[AI] WARNING: anthropic package not installed. Run: pip install anthropic")
        events_df["ai_summary"] = "AI summary unavailable (anthropic not installed)"
        return events_df

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[AI] WARNING: ANTHROPIC_API_KEY not set. Skipping AI summaries.")
        events_df["ai_summary"] = "AI summary unavailable (no API key)"
        return events_df

    client = anthropic.Anthropic(api_key=api_key)

    max_events = CONFIG["ai_summary_max_events"]
    total = min(len(events_df), max_events) if max_events else len(events_df)

    print(f"[AI] Generating summaries for {total:,} events...")
    if max_events and len(events_df) > max_events:
        print(f"     (Dataset has {len(events_df):,} events — "
              f"summarizing first {max_events:,}. Adjust CONFIG to change.)")

    summaries = []
    subset = events_df.head(total) if max_events else events_df

    for idx, (_, row) in enumerate(subset.iterrows()):
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
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.content[0].text.strip()
        except Exception as e:
            summary = f"Summary generation failed: {str(e)[:80]}"

        summaries.append(summary)

        if (idx + 1) % 50 == 0:
            print(f"[AI] Summarized {idx + 1:,}/{total:,} events...", end="\r")

        time.sleep(CONFIG["ai_summary_delay_sec"])

    print(f"\n[AI] Summaries complete.")

    remaining = len(events_df) - len(summaries)
    if remaining > 0:
        summaries += ["Summary not generated (batch limit reached)"] * remaining

    events_df = events_df.copy()
    events_df["ai_summary"] = summaries
    return events_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7: OUTPUT BUILDER
# Merges event labels back into the original DataFrame as new columns.
# ──────────────────────────────────────────────────────────────────────────────

def build_labeled_dataset(original_df: pd.DataFrame,
                           events_df: pd.DataFrame,
                           cols: dict) -> pd.DataFrame:
    """
    Merge detected event labels into the original AIS DataFrame.
    Adds columns: event_id, event_type, confidence_score, null_flags, ai_summary.
    Merge key is (vessel_id + timestamp-minute) for efficient row alignment.
    """
    print_section("BUILDING LABELED DATASET")

    col_mmsi = cols["mmsi"]
    col_time = cols["time"]

    if events_df is None or events_df.empty:
        print("[OUTPUT] No events detected. Adding empty event columns.")
        original_df["event_id"]         = None
        original_df["event_type"]       = None
        original_df["confidence_score"] = None
        original_df["null_flags"]       = None
        if "ai_summary" not in original_df.columns:
            original_df["ai_summary"]   = None
        return original_df

    print("[OUTPUT] Building event lookup index...")

    events_df = events_df.copy()
    events_df["_key_ts"] = (pd.to_datetime(events_df["timestamp"], errors="coerce")
                             .dt.floor("min").astype(str))

    events_expanded = events_df.copy()
    events_expanded["_key_mmsi"] = events_expanded["vessel_id"].astype(str)
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

    original_df.drop(columns=["_key_mmsi", "_key_ts"], inplace=True, errors="ignore")

    labeled_count = original_df["event_type"].notna().sum()
    print(f"[OUTPUT] {labeled_count:,} rows tagged with event labels.")

    return original_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8: CSV EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, filepath: str, label: str = "output"):
    """Export a DataFrame to CSV and print row count and file size."""
    print(f"[EXPORT] Writing {label} → {filepath}")
    df.to_csv(filepath, index=False)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"         Done — {len(df):,} rows | {size_mb:.1f} MB")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9: PIPELINE SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def print_pipeline_summary(input_path: str, labeled_df: pd.DataFrame,
                            events_df: pd.DataFrame, output_dir: str):
    """Print a human-readable summary of the pipeline run."""
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
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str,
                 output_dir: str = "./output",
                 ai_summaries: bool = False):
    """
    Full end-to-end DDT AIS processing pipeline.
    Loads data, resolves columns, detects events, optionally generates AI summaries,
    merges labels, and exports results to CSV.
    """
    print("\n" + "=" * 60)
    print("  DDT — GEN AI AIS EVENT DETECTION & LABELING PIPELINE")
    print("  Version 2.0 | All Requirements Active")
    print("=" * 60)

    start_time = time.time()

    print_section("LOADING DATA")
    df = load_file(input_path)

    print_section("COLUMN RESOLUTION")
    cols = resolve_columns(df)
    print(f"  MMSI  → '{cols['mmsi']}'")
    print(f"  LAT   → '{cols['lat']}'")
    print(f"  LON   → '{cols['lon']}'")
    print(f"  TIME  → '{cols['time']}'")
    print(f"  SOG   → {repr(cols['sog'])} {'⚠ Speed events will be skipped' if not cols['sog'] else ''}")
    print(f"  COG   → {repr(cols['cog'])} {'⚠ Deviation events will be skipped' if not cols['cog'] else ''}")
    print(f"  NAME  → {repr(cols['name'])}")

    df, date_label = prompt_date_filter(df, cols["time"])

    print_section("PREPROCESSING")
    df[cols["time"]] = pd.to_datetime(df[cols["time"]], errors="coerce")
    df = df.sort_values([cols["mmsi"], cols["time"]]).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    if removed:
        print(f"[PREP] Removed {removed:,} duplicate vessel/timestamp rows.")
    print(f"[PREP] {len(df):,} clean rows ready for processing.")

    events_df = detect_events(df, cols)

    if ai_summaries and events_df is not None and not events_df.empty:
        events_df = generate_ai_summaries(events_df)
    elif ai_summaries:
        print("[AI] No events to summarize.")

    labeled_df = build_labeled_dataset(df, events_df, cols)

    print_section("EXPORTING OUTPUTS")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    file_prefix = (base + "_" + date_label) if date_label != "all" else base

    events_only = labeled_df[labeled_df["event_type"].notna()].copy()
    labeled_path = os.path.join(output_dir, file_prefix + "_events_labeled.csv")
    export_csv(events_only, labeled_path, label="Event-labeled rows")

    if events_df is not None and not events_df.empty:
        events_path = os.path.join(output_dir, file_prefix + "_events_summary.csv")
        export_csv(events_df, events_path, label="Events summary")
    else:
        print("[EXPORT] No events to export.")

    export_vessel_status_table(df, events_df, output_dir, file_prefix)

    print_pipeline_summary(input_path, labeled_df, events_df, output_dir)

    elapsed = time.time() - start_time
    print(f"  Total runtime  : {elapsed:.1f} seconds")
    print("=" * 60 + "\n")

    return labeled_df, events_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11: COMMAND LINE INTERFACE
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

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to AIS input file (.csv, .json, .nmea/.txt/.ais). If omitted, a file browser will open."
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Directory where output CSVs will be saved (default: ./output)"
    )

    parser.add_argument(
        "--ai-summaries",
        action="store_true",
        default=False,
        help=(
            "Generate natural-language AI summaries for each event. "
            "Requires: ANTHROPIC_API_KEY environment variable."
        )
    )

    parser.add_argument(
        "--max-summaries",
        type=int,
        default=None,
        help="Max number of events to summarize with AI (overrides CONFIG default)"
    )

    args = parser.parse_args()

    if args.max_summaries is not None:
        CONFIG["ai_summary_max_events"] = args.max_summaries

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
