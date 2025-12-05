#!/usr/bin/env python3
# Central configuration for AWS / local dual-mode operation

import os

# ----------------------------------------------------------------------
# MODE TOGGLE
# ----------------------------------------------------------------------
# USE_AWS = True:
#   - S3 and RDS are used
#   - All artifacts ALSO sync to local_store/
#
# USE_AWS = False:
#   - No AWS calls
#   - Everything runs from local_store/
# ----------------------------------------------------------------------

USE_AWS = os.getenv("USE_AWS", "true").lower() == "true"


# ----------------------------------------------------------------------
# LOCAL STORAGE (ALWAYS USED)
# ----------------------------------------------------------------------

LOCAL_ROOT = os.path.abspath("local_store")

LOCAL_SEQUENCES   = os.path.join(LOCAL_ROOT, "sequences")
LOCAL_MODELS      = os.path.join(LOCAL_ROOT, "models")
LOCAL_CLEAN       = os.path.join(LOCAL_ROOT, "clean")
LOCAL_CLEANED     = LOCAL_CLEAN  # alias for clarity
LOCAL_LIVE        = os.path.join(LOCAL_ROOT, "live_ingest")
LOCAL_TRANSFORMED = os.path.join(LOCAL_ROOT, "transformed")
LOCAL_HISTORICAL  = os.path.join(LOCAL_ROOT, "historical")

# Ensure all directories exist
for d in [
    LOCAL_ROOT,
    LOCAL_SEQUENCES,
    LOCAL_MODELS,
    LOCAL_CLEAN,
    LOCAL_LIVE,
    LOCAL_TRANSFORMED,
    LOCAL_HISTORICAL,
]:
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------
# AWS S3 PATHS (ONLY USED WHEN USE_AWS=True)
# ----------------------------------------------------------------------

S3_BUCKET_ROOT = "s3://ece5984-s3-rameyjm7/Project"

S3_TRANSFORMED = f"{S3_BUCKET_ROOT}/transformed"
S3_SEQUENCES   = f"{S3_BUCKET_ROOT}/sequences"
S3_MODELS      = f"{S3_BUCKET_ROOT}/models"
S3_LIVE        = f"{S3_BUCKET_ROOT}/live_ingest"
S3_BATCH       = f"{S3_BUCKET_ROOT}/batch_ingest"
S3_HISTORICAL  = f"{S3_BUCKET_ROOT}/historical"


# ----------------------------------------------------------------------
# AMAZON RDS CONFIGURATION
# ----------------------------------------------------------------------
# Local mode DOES NOT use RDS at all.
# AWS mode requires working credentials here.
# ----------------------------------------------------------------------

RDS_HOST = os.getenv("RDS_HOST", "")
RDS_USER = os.getenv("RDS_USER", "admin")
RDS_PASS = os.getenv("RDS_PASS", "")
RDS_DB   = os.getenv("RDS_DB",   "rameyjm7")


# ----------------------------------------------------------------------
# OPENWEATHER API KEY
# ----------------------------------------------------------------------

API_KEY = os.getenv("API_KEY", "CHANGE_ME")


# ----------------------------------------------------------------------
# FIXED CITY LIST FOR ALL INGESTION / TRAINING
# ----------------------------------------------------------------------

CITIES = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Miami",
    "Seattle",
]
