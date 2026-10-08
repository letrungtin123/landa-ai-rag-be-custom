"""Small network helpers shared by the database and storage adapters."""

from __future__ import annotations

import ipaddress


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals (127.0.0.0/8, ::1)."""
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False
