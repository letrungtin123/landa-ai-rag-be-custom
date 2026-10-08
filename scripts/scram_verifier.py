"""Print a PostgreSQL SCRAM-SHA-256 verifier for a role password (no plaintext in SQL history).

Usage:
    python scripts/scram_verifier.py              # prompts for the password (not echoed)
    python scripts/scram_verifier.py --generate   # creates a random password and prints both

Then, in a separate query per environment:
    ALTER ROLE landa_ai_rag WITH PASSWORD '<printed verifier>';
and store the plaintext only in the secret store used for DATABASE_URL.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import secrets
import sys

ITERATIONS = 4096  # PostgreSQL default scram_iterations
SALT_BYTES = 16
GENERATED_PASSWORD_BYTES = 32


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def scram_sha256_verifier(password: str, *, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    """Return the verifier PostgreSQL stores in pg_authid.rolpassword (RFC 5802 / RFC 7677)."""

    if not password.isascii():
        # PostgreSQL applies SASLprep to non-ASCII passwords; keep the generator to ASCII.
        raise ValueError("use an ASCII password (e.g. --generate)")
    salt = salt if salt is not None else secrets.token_bytes(SALT_BYTES)
    salted = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    return f"SCRAM-SHA-256${iterations}:{_b64(salt)}${_b64(stored_key)}:{_b64(server_key)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true", help="generate a random URL-safe password")
    args = parser.parse_args()
    if args.generate:
        password = secrets.token_urlsafe(GENERATED_PASSWORD_BYTES)
        sys.stdout.write(f"password (put in the secret store / DATABASE_URL): {password}\n")
    else:
        password = getpass.getpass("password: ")
        if password != getpass.getpass("repeat: "):
            sys.stderr.write("passwords differ\n")
            return 1
    sys.stdout.write(f"verifier (use in ALTER ROLE ... PASSWORD): {scram_sha256_verifier(password)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
