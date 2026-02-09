"""
HemaVision Database Layer
━━━━━━━━━━━━━━━━━━━━━━━━━
SQLite-backed persistent storage for analysis records.

Tables:
  analyses  — stores every prediction with patient context, results, and timestamps.

Usage:
  db = AnalysisDatabase()           # connect (creates tables if needed)
  db.save_analysis(record)          # insert a record
  records = db.get_all_analyses()   # fetch all
  record  = db.get_analysis(id)     # fetch one
  stats   = db.get_statistics()     # aggregate stats
"""

import sqlite3
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "hemavision.db"


class AnalysisRecord:
    """Represents a single analysis record."""

    def __init__(
        self,
        prediction: str,
        probability: float,
        confidence: float,
        risk_level: str,
        risk_color: str,
        inference_time_ms: float,
        patient_age: int,
        patient_sex: str,
        npm1_mutated: bool,
        flt3_mutated: bool,
        genetic_other: bool,
        image_filename: Optional[str] = None,
        gradcam_base64: Optional[str] = None,
        id: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.prediction = prediction
        self.probability = probability
        self.confidence = confidence
        self.risk_level = risk_level
        self.risk_color = risk_color
        self.inference_time_ms = inference_time_ms
        self.patient_age = patient_age
        self.patient_sex = patient_sex
        self.npm1_mutated = npm1_mutated
        self.flt3_mutated = flt3_mutated
        self.genetic_other = genetic_other
        self.image_filename = image_filename
        self.gradcam_base64 = gradcam_base64
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "risk_color": self.risk_color,
            "inference_time_ms": self.inference_time_ms,
            "patient_age": self.patient_age,
            "patient_sex": self.patient_sex,
            "npm1_mutated": self.npm1_mutated,
            "flt3_mutated": self.flt3_mutated,
            "genetic_other": self.genetic_other,
            "image_filename": self.image_filename,
            "gradcam_base64": self.gradcam_base64,
            "created_at": self.created_at,
        }


class UserRecord:
    """Represents a registered user."""

    def __init__(
        self,
        username: str,
        email: str,
        password_hash: str,
        display_name: str = "",
        sex: str = "Male",
        id: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.display_name = display_name or username
        self.sex = sex  # "Male" or "Female" — determines avatar
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "display_name": self.display_name,
            "sex": self.sex,
            "created_at": self.created_at,
        }


class AnalysisDatabase:
    """SQLite database for storing analysis records and users."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              TEXT PRIMARY KEY,
                    username        TEXT NOT NULL UNIQUE,
                    email           TEXT NOT NULL UNIQUE,
                    password_hash   TEXT NOT NULL,
                    display_name    TEXT NOT NULL,
                    sex             TEXT NOT NULL DEFAULT 'Male',
                    created_at      TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_username
                ON users(username)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email
                ON users(email)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id              TEXT PRIMARY KEY,
                    prediction      TEXT NOT NULL,
                    probability     REAL NOT NULL,
                    confidence      REAL NOT NULL,
                    risk_level      TEXT NOT NULL,
                    risk_color      TEXT NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    patient_age     INTEGER NOT NULL,
                    patient_sex     TEXT NOT NULL,
                    npm1_mutated    INTEGER NOT NULL DEFAULT 0,
                    flt3_mutated    INTEGER NOT NULL DEFAULT 0,
                    genetic_other   INTEGER NOT NULL DEFAULT 0,
                    image_filename  TEXT,
                    gradcam_base64  TEXT,
                    created_at      TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyses_created
                ON analyses(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyses_risk
                ON analyses(risk_level)
            """)
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        finally:
            conn.close()

    def save_analysis(self, record: AnalysisRecord) -> str:
        """Save an analysis record. Returns the record ID."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO analyses (
                    id, prediction, probability, confidence, risk_level,
                    risk_color, inference_time_ms, patient_age, patient_sex,
                    npm1_mutated, flt3_mutated, genetic_other,
                    image_filename, gradcam_base64, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.prediction,
                    record.probability,
                    record.confidence,
                    record.risk_level,
                    record.risk_color,
                    record.inference_time_ms,
                    record.patient_age,
                    record.patient_sex,
                    int(record.npm1_mutated),
                    int(record.flt3_mutated),
                    int(record.genetic_other),
                    record.image_filename,
                    record.gradcam_base64,
                    record.created_at,
                ),
            )
            conn.commit()
            logger.info(f"Analysis saved: {record.id}")
            return record.id
        finally:
            conn.close()

    def get_analysis(self, analysis_id: str) -> Optional[dict[str, Any]]:
        """Get a single analysis by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)
        finally:
            conn.close()

    def get_all_analyses(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get all analyses, most recent first."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregate statistics across all analyses."""
        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            if total == 0:
                return {
                    "total_analyses": 0,
                    "aml_detected": 0,
                    "normal_detected": 0,
                    "avg_confidence": 0,
                    "avg_inference_ms": 0,
                    "risk_distribution": {},
                }

            aml = conn.execute(
                "SELECT COUNT(*) FROM analyses WHERE prediction LIKE '%AML%'"
            ).fetchone()[0]

            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM analyses"
            ).fetchone()[0]

            avg_time = conn.execute(
                "SELECT AVG(inference_time_ms) FROM analyses"
            ).fetchone()[0]

            risk_rows = conn.execute(
                "SELECT risk_level, COUNT(*) as cnt FROM analyses GROUP BY risk_level"
            ).fetchall()
            risk_dist = {row["risk_level"]: row["cnt"] for row in risk_rows}

            return {
                "total_analyses": total,
                "aml_detected": aml,
                "normal_detected": total - aml,
                "avg_confidence": round(avg_conf, 4),
                "avg_inference_ms": round(avg_time, 2),
                "risk_distribution": risk_dist,
            }
        finally:
            conn.close()

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis by ID. Returns True if deleted."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM analyses WHERE id = ?", (analysis_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def clear_all(self) -> int:
        """Delete all analyses. Returns count of deleted records."""
        conn = self._get_connection()
        try:
            count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            conn.execute("DELETE FROM analyses")
            conn.commit()
            logger.info(f"Cleared {count} analysis records")
            return count
        finally:
            conn.close()

    # ── User Methods ────────────────────────────────────────────

    def create_user(self, user: UserRecord) -> str:
        """Create a new user. Returns the user ID."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, display_name, sex, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user.id, user.username, user.email, user.password_hash,
                 user.display_name, user.sex, user.created_at),
            )
            conn.commit()
            logger.info(f"User created: {user.username}")
            return user.id
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        """Get a user by username."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        """Get a user by email."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_user_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get a user by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["npm1_mutated"] = bool(d["npm1_mutated"])
        d["flt3_mutated"] = bool(d["flt3_mutated"])
        d["genetic_other"] = bool(d["genetic_other"])
        return d
