import time
import json
import os
from datetime import datetime

from sqlalchemy import create_engine, text

# ============================================================
# Load config (A: global config.py)
# ============================================================
from src.config import (
    USE_AWS,
    RDS_USER,
    RDS_PASS,
    RDS_HOST,
    RDS_DB,
)

# If not using AWS → disable DB logging
if USE_AWS:
    engine = create_engine(
        f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}/{RDS_DB}"
    )
else:
    engine = None


# ------------------------------------------------------------
# Safe DB existence check (AWS only)
# ------------------------------------------------------------
def ensure_database_exists():
    if not USE_AWS:
        return

    # SQLAlchemy 2.x requires exec_driver_sql
    with engine.connect() as conn:
        result = conn.exec_driver_sql(
            f"SHOW DATABASES LIKE '{RDS_DB}';"
        ).fetchone()

        if not result:
            conn.exec_driver_sql(f"CREATE DATABASE {RDS_DB};")


# Ensure DB exists only once at import time
if USE_AWS:
    ensure_database_exists()


# ------------------------------------------------------------
# Provenance Logging
# ------------------------------------------------------------
def log_provenance(
    stage: str,
    status: str,
    input_source: str = None,
    output_target: str = None,
    records_in: int = None,
    records_out: int = None,
    duration_seconds: float = None,
    extra: dict = None
):
    """
    Insert provenance into RDS only when USE_AWS=true.
    Otherwise, print locally.
    """

    if not USE_AWS:
        print(f"[LOCAL MODE] Provenance skipped: {stage} ({status})")
        return

    payload = {
        "stage": stage,
        "status": status,
        "timestamp": datetime.utcnow(),
        "input_source": input_source,
        "output_target": output_target,
        "records_in": records_in,
        "records_out": records_out,
        "duration_seconds": duration_seconds,
        "extra": json.dumps(extra) if extra else None,
    }

    sql = text("""
        INSERT INTO provenance_log
        (stage, status, timestamp, input_source, output_target,
         records_in, records_out, duration_seconds, extra)
        VALUES
        (:stage, :status, :timestamp, :input_source, :output_target,
         :records_in, :records_out, :duration_seconds, :extra)
    """)

    with engine.begin() as conn:
        conn.execute(sql, payload)


# ------------------------------------------------------------
# Context Manager
# ------------------------------------------------------------
class ProvenanceTimer:
    def __init__(self, stage, input_source=None, output_target=None):
        self.stage = stage
        self.input_source = input_source
        self.output_target = output_target

    def __enter__(self):
        self.start = time.time()
        return self

    def commit(self, status="SUCCESS", records_in=None, records_out=None, extra=None):
        duration = time.time() - self.start
        log_provenance(
            stage=self.stage,
            status=status,
            input_source=self.input_source,
            output_target=self.output_target,
            records_in=records_in,
            records_out=records_out,
            duration_seconds=duration,
            extra=extra,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            duration = time.time() - self.start
            log_provenance(
                stage=self.stage,
                status="FAILURE",
                input_source=self.input_source,
                output_target=self.output_target,
                duration_seconds=duration,
                extra={"error": str(exc_value)},
            )
