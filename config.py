import os
from enum import Enum

class DepthNullStrategy(str, Enum):
    NULL = "null"             # Sends "depth_cm": null (standard JSON nullable)
    SENTINEL = "sentinel"     # Sends "depth_cm": -1.0 (numeric flag for non-nullable float)
    OMIT = "omit"             # Drops the "depth_cm" key entirely from the JSON payload

# Default strategy (to be adjusted once OmniSight backend schema is confirmed):
DEPTH_NULL_STRATEGY = DepthNullStrategy.NULL

# Queue configuration
QUEUE_SENT_RETENTION_DAYS = 30
QUEUE_REJECTED_RETENTION_DAYS = 60
