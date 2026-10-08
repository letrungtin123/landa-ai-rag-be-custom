"""Canonical JSON hashing: the one implementation behind every contract/evidence hash (STD-7)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_hash(value: Any) -> str:
    """SHA-256 hex digest of canonical JSON: sorted keys, compact separators, UTF-8 (no ASCII escapes).

    Contract hashes are compared with the Node backend's mirror of this encoding; never change it.
    """
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
