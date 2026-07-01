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

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unversioned"))


def make_version(fingerprint: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{hashlib.sha1(fingerprint.encode()).hexdigest()[:6]}"


def save_bundle(bundle: ModelBundle, root: str | Path) -> Path:
    root = Path(root)
    out = root / bundle.version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / BUNDLE_FILE)
    (out / METADATA_FILE).write_text(
        json.dumps(bundle.metadata, indent=2, default=str), encoding="utf-8"
    )
    (root / LATEST_FILE).write_text(bundle.version, encoding="utf-8")
    return out


def resolve_bundle_dir(path: str | Path) -> Path:
    """Accept a bundle directory, an artifacts root (uses LATEST) or 'artifacts/latest'."""
    path = Path(path)
    if path.name.lower() == "latest":
        path = path.parent
    if (path / BUNDLE_FILE).exists():
        return path
    latest = path / LATEST_FILE
    if latest.exists():
        return path / latest.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"No model bundle found at {path}. Run `fraud-detect train` first.")


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(resolve_bundle_dir(path) / BUNDLE_FILE)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("File does not contain a ModelBundle")
    return bundle
