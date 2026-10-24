"""Loopback-only sockets for the test session (``tests/conftest.py``; R9: tests never reach the network).

The guard wraps the entry points every client in the dependency tree goes through: ``socket.socket.connect`` /
``connect_ex`` / ``sendto`` (sync HTTP clients, the selector event loop), ``socket.getaddrinfo`` (every name
lookup, so a host name never resolves, including inside asyncio and anyio) and the proactor event loop's
``sock_connect``, which connects through ``ConnectEx`` without calling ``socket.connect``. Loopback addresses and
local (``AF_UNIX``) sockets pass unchanged.
"""

from __future__ import annotations

import asyncio
import socket
from asyncio import proactor_events
from collections.abc import Callable
from typing import Any

from app.infra.net import is_loopback_host


class OutboundNetworkBlockedError(RuntimeError):
    """A test tried to reach a host other than this machine's loopback interface."""


def _host_of(address: Any) -> str | None:
    """The host of an ``AF_INET``/``AF_INET6`` address tuple; ``None`` for a local socket path."""

    if isinstance(address, tuple) and address:
        return str(address[0])
    return None


def check_address(address: Any) -> None:
    host = _host_of(address)
    if host is not None and not is_loopback_host(host):
        raise OutboundNetworkBlockedError(f"tests may not reach {host!r}: only loopback is allowed")


def _check_host(host: Any) -> None:
    name = host.decode() if isinstance(host, (bytes, bytearray)) else "" if host is None else str(host)
    if name and not is_loopback_host(name):  # None or "": a passive (bind) lookup
        raise OutboundNetworkBlockedError(f"tests may not look up {name!r}: only loopback is allowed")


def _guard_method(original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: socket.socket, address: Any, *args: Any) -> Any:
        check_address(address)
        return original(self, address, *args)

    return guarded


def _guard_sendto(original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: socket.socket, data: Any, *args: Any) -> Any:
        check_address(args[-1])  # sendto(data, address) or sendto(data, flags, address)
        return original(self, data, *args)

    return guarded


def _guard_getaddrinfo(original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(host: Any, *args: Any, **kwargs: Any) -> Any:
        _check_host(host)
        return original(host, *args, **kwargs)

    return guarded


def _guard_sock_connect(original: Callable[..., Any]) -> Callable[..., Any]:
    async def guarded(self: asyncio.AbstractEventLoop, sock: socket.socket, address: Any) -> Any:
        check_address(address)
        return await original(self, sock, address)

    return guarded


_installed: list[bool] = []


def install_network_guard() -> None:
    """Install the guard once per process (idempotent)."""

    if _installed:
        return
    socket.socket.connect = _guard_method(socket.socket.connect)
    socket.socket.connect_ex = _guard_method(socket.socket.connect_ex)
    socket.socket.sendto = _guard_sendto(socket.socket.sendto)
    socket.getaddrinfo = _guard_getaddrinfo(socket.getaddrinfo)
    loop_class = proactor_events.BaseProactorEventLoop
    loop_class.sock_connect = _guard_sock_connect(loop_class.sock_connect)
    _installed.append(True)
