"""Process entrypoint: ``python -m app``.

All bind/runtime options come from validated Settings so the same command
works under PM2, a container or a plain shell on any OS.
"""

from __future__ import annotations

from typing import Any

import uvicorn

from app.core.config import Settings, settings


def uvicorn_options(value: Settings) -> dict[str, Any]:
    return {
        "host": value.host,
        "port": value.port,
        "workers": value.workers,
        "proxy_headers": value.proxy_headers,
        "forwarded_allow_ips": value.forwarded_allow_ips,
        "timeout_keep_alive": value.keep_alive_timeout_seconds,
        "timeout_graceful_shutdown": value.shutdown_grace_seconds,
        "log_level": "info",
        "access_log": False,
    }


def main() -> None:
    uvicorn.run("app.main:app", **uvicorn_options(settings))


if __name__ == "__main__":
    main()
