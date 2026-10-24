"""The guards ``tests/conftest.py`` installs for every pytest session: the test settings profile and no network."""

from __future__ import annotations

import asyncio
import os
import socket
import unittest

from app.core.config import settings
from tests.support.network_guard import OutboundNetworkBlockedError, check_address

# TEST-NET-1 (RFC 5737): never routed, so even a guard failure could not reach a real host.
OUTSIDE = ("192.0.2.1", 9)


class TestSessionGuardTests(unittest.TestCase):
    def test_the_session_runs_the_test_profile(self) -> None:
        self.assertEqual(os.environ["AI_RAG_ENV"], "test")
        self.assertEqual(settings.environment, "test")

    def test_a_connection_outside_loopback_is_refused_before_it_is_made(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, self.assertRaises(OutboundNetworkBlockedError):
            sock.connect(OUTSIDE)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, self.assertRaises(OutboundNetworkBlockedError):
            sock.connect_ex(OUTSIDE)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, self.assertRaises(OutboundNetworkBlockedError):
            sock.sendto(b"x", OUTSIDE)
        with self.assertRaises(OutboundNetworkBlockedError):
            socket.create_connection(OUTSIDE, timeout=1)

    def test_a_host_name_never_resolves(self) -> None:
        with self.assertRaises(OutboundNetworkBlockedError):
            socket.getaddrinfo("generativelanguage.googleapis.com", 443)

    def test_asyncio_connections_outside_loopback_are_refused(self) -> None:
        async def connect(host: str) -> None:
            await asyncio.open_connection(host, OUTSIDE[1])

        for host in (OUTSIDE[0], "example.com"):
            with self.subTest(host=host), self.assertRaises(OutboundNetworkBlockedError):
                asyncio.run(connect(host))

    def test_loopback_and_local_sockets_still_work(self) -> None:
        # The event loop's self-pipe (a loopback socket pair on Windows) and a loopback server.
        left, right = socket.socketpair()
        with left, right:
            left.sendall(b"ping")
            self.assertEqual(right.recv(4), b"ping")
        with (socket.create_server(("127.0.0.1", 0)) as server,
              socket.create_connection(server.getsockname()[:2], timeout=5) as client):
            accepted, _address = server.accept()
            with accepted:
                client.sendall(b"ok")
                self.assertEqual(accepted.recv(2), b"ok")
        self.assertTrue(socket.getaddrinfo("localhost", 80))
        for address in (("127.0.0.1", 1), ("::1", 1), ("localhost", 1), "local.sock"):
            check_address(address)


if __name__ == "__main__":
    unittest.main()
