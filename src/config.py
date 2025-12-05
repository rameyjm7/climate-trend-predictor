#!/usr/bin/env python3
# Central configuration for AWS / local dual-mode operation

import os

# ----------------------------------------------------------------------
# Mode Toggle
# ----------------------------------------------------------------------
# When USE_AWS = True:
#   - S3 is used for reads/writes
#   - Files are ALSO written locally for offline reuse
#
# When USE_AWS = False:
#   - No S3 calls are made
#   - All I/O uses the local_store/ directory only
# ----------------------------------------------------------------------

USE_AWS = os.getenv("USE_AWS", "true").lower() == "true"

# ----------------------------------------------------------------------
# Local storage root (always used even in AWS mode)
# ----------------------------------------------------------------------
LOCAL_ROOT = os.path.abspath("local_store")

LOCAL_SEQUENCES   = os.path.join(LOCAL_ROOT, "sequences")
LOCAL_MODELS      = os.path.join(LOCAL_ROOT, "models")
LOCAL_CLEAN       = os.path.join(LOCAL_ROOT, "clean")
LOCAL_LIVE        = os.path.join(LOCAL_ROOT, "live_ingest")
LOCAL_TRANSFORMED = os.path.join(LOCAL_ROOT, "transformed")
LOCAL_CLEANED     = LOCAL_CLEAN 
# Directories to ensure exist
for d in [
    LOCAL_ROOT,
    LOCAL_SEQUENCES,
    LOCAL_MODELS,
    LOCAL_CLEAN,
    LOCAL_LIVE,
    LOCAL_TRANSFORMED,
]:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# AWS S3 configuration (only used when USE_AWS=True)
# ----------------------------------------------------------------------
S3_BUCKET_ROOT = "s3://ece5984-s3-rameyjm7/Project"

S3_TRANSFORMED = f"{S3_BUCKET_ROOT}/transformed"
S3_SEQUENCES   = f"{S3_BUCKET_ROOT}/sequences"
S3_MODELS      = f"{S3_BUCKET_ROOT}/models"
S3_LIVE        = f"{S3_BUCKET_ROOT}/live_ingest"
S3_BATCH       = f"{S3_BUCKET_ROOT}/batch_ingest"

# ----------------------------------------------------------------------
# Amazon RDS configuration (centralized here)
# ----------------------------------------------------------------------
# Only used when USE_AWS=True. Local mode skips RDS entirely.

RDS_HOST = os.getenv("RDS_HOST", "your-rds-host-goes-here")
RDS_USER = os.getenv("RDS_USER", "admin")
RDS_PASS = os.getenv("RDS_PASS", "")
RDS_DB   = os.getenv("RDS_DB",   "rameyjm7")

# ----------------------------------------------------------------------
# OpenWeather API Key (optional centralization)
# ----------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "CHANGE_ME")
