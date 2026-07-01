"""Versioned model bundles on disk.

Layout::

    artifacts/
      LATEST                       # text file with the current version id
      20260105-142233-3f9a1c/
        bundle.joblib              # model, thresholds, reason coder, drift reference
        metadata.json              # human-readable summary of the same run
        model_card.md
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .explain import ReasonCoder
from .model import FraudModel
from .thresholds import Thresholds

BUNDLE_FILE = "bundle.joblib"
METADATA_FILE = "metadata.json"
LATEST_FILE = "LATEST"


@dataclass
class ModelBundle:
    model: FraudModel
    thresholds: Thresholds
    reason_coder: ReasonCoder
    reference_profile: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

