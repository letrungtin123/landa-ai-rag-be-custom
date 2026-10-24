"""Guards for every pytest session, installed before any test module (or application module) is imported.

* ``AI_RAG_ENV=test``: ``app.core.config`` reads no dotenv file in the test profile, so a developer's ``.env``
  (keys, a database URL, limits) never changes what a test sees, on any machine.
* No outbound network (R9): a socket may only reach this machine's loopback interface. On Windows every asyncio
  event loop opens a loopback socket pair for its self-pipe; no test talks to anything else. A connection, a
  datagram or a name lookup for any other host raises ``OutboundNetworkBlockedError`` instead of reaching a real
  service (``tests/support/network_guard.py``).
"""

from __future__ import annotations

import os
import sys

os.environ["AI_RAG_ENV"] = "test"
if "app.core.config" in sys.modules:  # pragma: no cover - only when a plugin imports the application first
    raise RuntimeError("app.core.config was imported before tests/conftest.py forced AI_RAG_ENV=test")

# Imported after the environment is set: the guard imports app.infra.net.
from tests.support.network_guard import install_network_guard

install_network_guard()
