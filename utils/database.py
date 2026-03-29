"""
HemaVision Database Layer
━━━━━━━━━━━━━━━━━━━━━━━━━
PostgreSQL-backed persistent storage for analysis records.
"""

import os
import json
import uuid
import logging
import psycopg
from psycopg.rows import dict_row
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class AnalysisRecord:
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
        source_image_base64: Optional[str] = None,
        gradcam_base64: Optional[str] = None,
        user_id: Optional[str] = None,
        id: Optional[str] = None,
        created_at: Optional[str] = None,
        cells_data: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.prediction = prediction
        self.probability = float(probability)
        self.confidence = float(confidence)
        self.risk_level = risk_level
        self.risk_color = risk_color
        self.inference_time_ms = float(inference_time_ms)
        self.patient_age = int(patient_age)
        self.patient_sex = patient_sex
        self.npm1_mutated = npm1_mutated
        self.flt3_mutated = flt3_mutated
        self.genetic_other = genetic_other
        self.image_filename = image_filename
        self.source_image_base64 = source_image_base64
        self.gradcam_base64 = gradcam_base64
        self.user_id = user_id
        self.created_at = created_at or datetime.now().isoformat()
        self.cells_data = cells_data

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
            "source_image_base64": self.source_image_base64,
            "gradcam_base64": self.gradcam_base64,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "cells_data": self.cells_data,
        }

class UserRecord:
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
        self.sex = sex
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
    """PostgreSQL database for storing analysis records and users."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        self._init_db()

    def _get_connection(self):
        return psycopg.connect(self.db_url, row_factory=dict_row)

    def _init_db(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
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
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_username
                    ON users(username)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_email
                    ON users(email)
                """)
                cur.execute("""
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
                        source_image_base64 TEXT,
                        gradcam_base64  TEXT,
                        cells_data      TEXT,
                        user_id         TEXT,
                        created_at      TEXT NOT NULL
                    )
                """)
                # Handle lightweight migration for user_id, cells_data, source_image_base64
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='analyses'
                """)
                columns = [row["column_name"] for row in cur.fetchall()]
                if "user_id" not in columns:
                    cur.execute("ALTER TABLE analyses ADD COLUMN user_id TEXT")
                if "cells_data" not in columns:
                    cur.execute("ALTER TABLE analyses ADD COLUMN cells_data TEXT")
                if "source_image_base64" not in columns:
                    cur.execute("ALTER TABLE analyses ADD COLUMN source_image_base64 TEXT")
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_analyses_created
                    ON analyses(created_at DESC)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_analyses_risk
                    ON analyses(risk_level)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_analyses_user_created
                    ON analyses(user_id, created_at DESC)
                """)
                conn.commit()
                logger.info("PostgreSQL Database initialized successfully.")

    def save_analysis(self, record: AnalysisRecord) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analyses (
                        id, prediction, probability, confidence, risk_level,
                        risk_color, inference_time_ms, patient_age, patient_sex,
                        npm1_mutated, flt3_mutated, genetic_other,
                        image_filename, source_image_base64, gradcam_base64, cells_data, user_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.id,
                        record.prediction,
                        float(record.probability),
                        float(record.confidence),
                        record.risk_level,
                        record.risk_color,
                        float(record.inference_time_ms),
                        int(record.patient_age),
                        record.patient_sex,
                        int(record.npm1_mutated),
                        int(record.flt3_mutated),
                        int(record.genetic_other),
                        record.image_filename,
                        record.source_image_base64,
                        record.gradcam_base64,
                        record.cells_data, # Use the object's cells_data directly
                        record.user_id,
                        record.created_at,
                    ),
                )
                conn.commit()
                logger.info(f"Analysis saved: {record.id}")
                return record.id

    def get_analysis(self, analysis_id: str, user_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if user_id is None:
                    cur.execute(
                        "SELECT * FROM analyses WHERE id = %s", (analysis_id,)
                    )
                else:
                    cur.execute(
                        "SELECT * FROM analyses WHERE id = %s AND user_id = %s",
                        (analysis_id, user_id),
                    )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    def get_all_analyses(
        self, limit: int = 50, offset: int = 0, user_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if user_id is None:
                    cur.execute(
                        "SELECT * FROM analyses ORDER BY created_at DESC LIMIT %s OFFSET %s",
                        (limit, offset),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM analyses WHERE user_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
                        (user_id, limit, offset),
                    )
                rows = cur.fetchall()
                return [self._row_to_dict(row) for row in rows]

    def get_statistics(self, user_id: Optional[str] = None) -> dict[str, Any]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                where = " WHERE user_id = %s" if user_id is not None else ""
                params: tuple[Any, ...] = (user_id,) if user_id is not None else tuple()

                cur.execute(f"SELECT COUNT(*) as count FROM analyses{where}", params)
                total_res = cur.fetchone()
                total = total_res["count"] if total_res else 0
                if total == 0:
                    return {
                        "total_analyses": 0,
                        "aml_detected": 0,
                        "normal_detected": 0,
                        "avg_confidence": 0,
                        "avg_inference_ms": 0,
                        "risk_distribution": {},
                    }

                if user_id is None:
                    cur.execute("SELECT COUNT(*) as count FROM analyses WHERE prediction LIKE %s", ("%AML%",))
                    aml_res = cur.fetchone()
                    aml = aml_res["count"] if aml_res else 0
                    cur.execute("SELECT AVG(confidence) as avg_conf FROM analyses")
                    avg_conf_res = cur.fetchone()
                    avg_conf = avg_conf_res["avg_conf"] if avg_conf_res and avg_conf_res["avg_conf"] else 0
                    cur.execute("SELECT AVG(inference_time_ms) as avg_time FROM analyses")
                    avg_time_res = cur.fetchone()
                    avg_time = avg_time_res["avg_time"] if avg_time_res and avg_time_res["avg_time"] else 0
                    cur.execute("SELECT risk_level, COUNT(*) as cnt FROM analyses GROUP BY risk_level")
                    risk_rows = cur.fetchall()
                else:
                    cur.execute("SELECT COUNT(*) as count FROM analyses WHERE user_id = %s AND prediction LIKE %s", (user_id, "%AML%"))
                    aml_res = cur.fetchone()
                    aml = aml_res["count"] if aml_res else 0
                    cur.execute("SELECT AVG(confidence) as avg_conf FROM analyses WHERE user_id = %s", (user_id,))
                    avg_conf_res = cur.fetchone()
                    avg_conf = avg_conf_res["avg_conf"] if avg_conf_res and avg_conf_res["avg_conf"] else 0
                    cur.execute("SELECT AVG(inference_time_ms) as avg_time FROM analyses WHERE user_id = %s", (user_id,))
                    avg_time_res = cur.fetchone()
                    avg_time = avg_time_res["avg_time"] if avg_time_res and avg_time_res["avg_time"] else 0
                    cur.execute("SELECT risk_level, COUNT(*) as cnt FROM analyses WHERE user_id = %s GROUP BY risk_level", (user_id,))
                    risk_rows = cur.fetchall()

                risk_dist = {row["risk_level"]: row["cnt"] for row in risk_rows}

                return {
                    "total_analyses": total,
                    "aml_detected": aml,
                    "normal_detected": total - aml,
                    "avg_confidence": round(float(avg_conf), 4),
                    "avg_inference_ms": round(float(avg_time), 2),
                    "risk_distribution": risk_dist,
                }

    def delete_analysis(self, analysis_id: str, user_id: Optional[str] = None) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if user_id is None:
                    cur.execute("DELETE FROM analyses WHERE id = %s", (analysis_id,))
                else:
                    cur.execute("DELETE FROM analyses WHERE id = %s AND user_id = %s", (analysis_id, user_id))
                conn.commit()
                return cur.rowcount > 0

    def clear_all(self) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM analyses")
                res = cur.fetchone()
                count = res["count"] if res else 0
                cur.execute("DELETE FROM analyses")
                conn.commit()
                logger.info(f"Cleared {count} analysis records")
                return count

    def create_user(self, user: UserRecord) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, username, email, password_hash, display_name, sex, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (user.id, user.username, user.email, user.password_hash,
                     user.display_name, user.sex, user.created_at),
                )
                conn.commit()
                logger.info(f"User created: {user.username}")
                return user.id

    def get_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username = %s", (username,))
                row = cur.fetchone()
                return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                row = cur.fetchone()
                return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: dict) -> dict[str, Any]:
        d = dict(row)
        d["npm1_mutated"] = bool(d["npm1_mutated"])
        d["flt3_mutated"] = bool(d["flt3_mutated"])
        d["genetic_other"] = bool(d["genetic_other"])
        return d
