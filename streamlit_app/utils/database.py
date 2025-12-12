"""
Database Module
===============
SQLite database operations for storing user profiles and BP predictions.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict, List

# Database path
DB_DIR = Path(__file__).parent.parent / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "health_data.db"


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # User profile table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            sex TEXT,
            height_cm REAL,
            weight_kg REAL,
            medical_conditions TEXT,
            medications TEXT,
            notes TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sbp_mean REAL NOT NULL,
            sbp_std REAL NOT NULL,
            sbp_min REAL NOT NULL,
            sbp_max REAL NOT NULL,
            dbp_mean REAL NOT NULL,
            dbp_std REAL NOT NULL,
            dbp_min REAL NOT NULL,
            dbp_max REAL NOT NULL,
            num_windows INTEGER,
            duration REAL,
            mean_hr REAL,
            hrv_sdnn REAL,
            category TEXT,
            cuff_sbp REAL,
            cuff_dbp REAL,
            window_sbp TEXT,
            window_dbp TEXT,
            position TEXT,
            post_exercise TEXT,
            time_of_day TEXT,
            notes TEXT
        )
    """)

    # Migration: Add cuff_sbp and cuff_dbp columns if they don't exist
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN cuff_sbp REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN cuff_dbp REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Migration: Add window prediction columns
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN window_sbp TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN window_dbp TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: Add metadata columns
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN position TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN post_exercise TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN time_of_day TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN notes TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: Add calibration columns
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN calibrated_sbp REAL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN calibrated_dbp REAL")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN calibration_model TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: Add source column to distinguish live vs uploaded
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN source TEXT DEFAULT 'uploaded'")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


# User Profile Functions

def save_user_profile(
    name: str,
    age: int,
    sex: Optional[str] = None,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    medical_conditions: Optional[str] = None,
    medications: Optional[str] = None,
    notes: Optional[str] = None
) -> int:
    """Save or update user profile."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if profile exists
    cursor.execute("SELECT id FROM user_profile LIMIT 1")
    existing = cursor.fetchone()

    if existing:
        # Update existing profile
        cursor.execute("""
            UPDATE user_profile
            SET name=?, age=?, sex=?, height_cm=?, weight_kg=?,
                medical_conditions=?, medications=?, notes=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (name, age, sex, height_cm, weight_kg, medical_conditions,
              medications, notes, existing[0]))
        profile_id = existing[0]
    else:
        # Insert new profile
        cursor.execute("""
            INSERT INTO user_profile
            (name, age, sex, height_cm, weight_kg, medical_conditions, medications, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, age, sex, height_cm, weight_kg, medical_conditions, medications, notes))
        profile_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return profile_id


def get_user_profile() -> Optional[Dict]:
    """Retrieve user profile."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM user_profile LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


# Prediction Functions

def save_prediction(
    filename: str,
    timestamp: datetime,
    sbp_mean: float,
    sbp_std: float,
    sbp_min: float,
    sbp_max: float,
    dbp_mean: float,
    dbp_std: float,
    dbp_min: float,
    dbp_max: float,
    num_windows: int,
    duration: float,
    mean_hr: float,
    hrv_sdnn: float,
    category: str,
    cuff_sbp: Optional[float] = None,
    cuff_dbp: Optional[float] = None,
    window_sbp: Optional[list] = None,
    window_dbp: Optional[list] = None,
    position: Optional[str] = None,
    post_exercise: Optional[str] = None,
    notes: Optional[str] = None,
    calibrated_sbp: Optional[float] = None,
    calibrated_dbp: Optional[float] = None,
    calibration_model: Optional[str] = None,
    source: Optional[str] = 'uploaded'
) -> int:
    """Save a new BP prediction."""
    import json

    conn = get_connection()
    cursor = conn.cursor()

    # Convert window predictions to JSON strings
    window_sbp_json = json.dumps(window_sbp) if window_sbp else None
    window_dbp_json = json.dumps(window_dbp) if window_dbp else None

    # Determine time of day from timestamp
    hour = timestamp.hour
    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    cursor.execute("""
        INSERT INTO predictions
        (filename, timestamp, sbp_mean, sbp_std, sbp_min, sbp_max,
         dbp_mean, dbp_std, dbp_min, dbp_max,
         num_windows, duration, mean_hr, hrv_sdnn, category, cuff_sbp, cuff_dbp,
         window_sbp, window_dbp, position, post_exercise, time_of_day, notes,
         calibrated_sbp, calibrated_dbp, calibration_model, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (filename, timestamp, sbp_mean, sbp_std, sbp_min, sbp_max,
          dbp_mean, dbp_std, dbp_min, dbp_max,
          num_windows, duration, mean_hr, hrv_sdnn, category, cuff_sbp, cuff_dbp,
          window_sbp_json, window_dbp_json, position, post_exercise, time_of_day, notes,
          calibrated_sbp, calibrated_dbp, calibration_model, source))

    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return prediction_id


def get_all_predictions() -> List[Dict]:
    """Retrieve all predictions."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_prediction_by_id(prediction_id: int) -> Optional[Dict]:
    """Retrieve a specific prediction by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def get_latest_prediction() -> Optional[Dict]:
    """Retrieve the most recent prediction."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY timestamp DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def get_prediction_count() -> int:
    """Get total number of predictions."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()

    return count


def get_average_bp() -> Optional[Dict]:
    """Calculate average BP across all predictions."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            AVG(sbp_mean) as avg_sbp,
            AVG(dbp_mean) as avg_dbp,
            AVG(mean_hr) as avg_hr,
            AVG(hrv_sdnn) as avg_hrv
        FROM predictions
    """)

    row = cursor.fetchone()
    conn.close()

    if row and row[0] is not None:
        return {
            'avg_sbp': row[0],
            'avg_dbp': row[1],
            'avg_hr': row[2],
            'avg_hrv': row[3]
        }
    return None


def get_bp_trend(days: int = 30) -> List[Dict]:
    """Get BP trend for the last N days."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            DATE(timestamp) as date,
            AVG(sbp_mean) as avg_sbp,
            AVG(dbp_mean) as avg_dbp,
            COUNT(*) as count
        FROM predictions
        WHERE timestamp >= datetime('now', '-' || ? || ' days')
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, (days,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_prediction(prediction_id: int):
    """Delete a prediction by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))

    conn.commit()
    conn.close()


def update_cuff_bp(prediction_id: int, cuff_sbp: Optional[float], cuff_dbp: Optional[float]):
    """Update cuff BP reference values for an existing prediction."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE predictions
        SET cuff_sbp=?, cuff_dbp=?
        WHERE id=?
    """, (cuff_sbp, cuff_dbp, prediction_id))

    conn.commit()
    conn.close()


def update_calibrated_bp(prediction_id: int, calibrated_sbp: float, calibrated_dbp: float, calibration_model: str):
    """Update calibrated BP values for an existing prediction."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE predictions
        SET calibrated_sbp=?, calibrated_dbp=?, calibration_model=?
        WHERE id=?
    """, (calibrated_sbp, calibrated_dbp, calibration_model, prediction_id))

    conn.commit()
    conn.close()


def get_predictions_with_cuff() -> List[Dict]:
    """Retrieve all predictions that have cuff BP measurements."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        WHERE cuff_sbp IS NOT NULL AND cuff_dbp IS NOT NULL
        ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_predictions_by_source(source: str) -> List[Dict]:
    """Retrieve predictions filtered by source (live_monitor or uploaded)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        WHERE source=?
        ORDER BY timestamp DESC
    """, (source,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_all_predictions():
    """Delete all predictions (use with caution)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()


# Statistics Functions

def get_category_distribution() -> Dict[str, int]:
    """Get distribution of BP categories."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM predictions
        GROUP BY category
        ORDER BY count DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return {row[0]: row[1] for row in rows}


def get_statistics_summary() -> Dict:
    """Get comprehensive statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_recordings,
            AVG(sbp_mean) as avg_sbp,
            MIN(sbp_mean) as min_sbp,
            MAX(sbp_mean) as max_sbp,
            AVG(dbp_mean) as avg_dbp,
            MIN(dbp_mean) as min_dbp,
            MAX(dbp_mean) as max_dbp,
            AVG(mean_hr) as avg_hr,
            AVG(hrv_sdnn) as avg_hrv,
            MIN(timestamp) as first_recording,
            MAX(timestamp) as last_recording
        FROM predictions
    """)

    row = cursor.fetchone()
    conn.close()

    if row and row[0] > 0:
        return {
            'total_recordings': row[0],
            'avg_sbp': row[1],
            'min_sbp': row[2],
            'max_sbp': row[3],
            'avg_dbp': row[4],
            'min_dbp': row[5],
            'max_dbp': row[6],
            'avg_hr': row[7],
            'avg_hrv': row[8],
            'first_recording': row[9],
            'last_recording': row[10]
        }
    return None
