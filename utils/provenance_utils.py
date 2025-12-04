import time
import json
from sqlalchemy import create_engine
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# Load Amazon RDS credentials from environment
# ---------------------------------------------------------------------------
RDS_USER = os.getenv("RDS_USER")
RDS_PASS = os.getenv("RDS_PASS")
RDS_HOST = os.getenv("RDS_HOST")
RDS_DB   = os.getenv("RDS_DB", "weatherdb")   # default to weatherdb if unset

# Validate required variables
missing = []

if not RDS_USER:
    missing.append("RDS_USER")
if not RDS_PASS:
    missing.append("RDS_PASS")
if not RDS_HOST:
    missing.append("RDS_HOST")

if missing:
    raise ValueError(f"Missing required RDS environment variables: {', '.join(missing)}")


engine = create_engine(
    f"mysql+pymysql://{RDS_USER}:{RDS_PASS}@{RDS_HOST}/{RDS_DB}"
)

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
    """Insert a provenance entry into RDS."""
    with engine.begin() as conn:
        conn.execute(
            """
            INSERT INTO provenance_log
            (stage, status, timestamp, input_source, output_target,
             records_in, records_out, duration_seconds, extra)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                stage,
                status,
                datetime.utcnow(),
                input_source,
                output_target,
                records_in,
                records_out,
                duration_seconds,
                json.dumps(extra) if extra else None
            )
        )

class ProvenanceTimer:
    """Context manager for timing a pipeline stage."""
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
            extra=extra
        )

    def __exit__(self, exc_type, exc_value, traceback):
        # Auto-log failure if exception occurs
        if exc_type is not None:
            duration = time.time() - self.start
            log_provenance(
                stage=self.stage,
                status="FAILURE",
                input_source=self.input_source,
                output_target=self.output_target,
                duration_seconds=duration,
                extra={"error": str(exc_value)}
            )
