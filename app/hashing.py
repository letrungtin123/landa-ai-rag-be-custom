"""Canonical JSON hashing: the one implementation behind every contract/evidence hash (STD-7).

Two encodings, both sorted keys, compact separators and raw UTF-8:

- ``canonical_hash``: Python's own number and key formatting. Every Python-only hash (source evidence,
  snapshot structure, cache keys) uses it and is stored; it never changes.
- ``js_canonical_hash``: the Node backend's encoding byte for byte (``orchestrationV2Hash`` /
  ``idmHash``). The hashes one side writes and the other side checks use it: the unit contract hash
  (Node writes, Python checks), the unit brief hash (Node writes) and the design hash (Python writes,
  Node checks). The two encodings differ only for a float that is a whole number or very small or very
  large (``1.0``, ``5e-05``, ``1e+16`` here; ``1``, ``0.00005``, ``10000000000000000`` in JavaScript), for
  ``-0.0``, for an integer beyond 2**53, for a lone surrogate, and for the order of keys that mix
  astral-plane characters with characters from U+E000 to U+FFFF (JavaScript sorts UTF-16 code units).

Neither encoding normalises Unicode: NFC and NFD text hash differently, on both sides.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

# JavaScript numbers are doubles: every integer up to 2**53 prints exactly, beyond that the double does.
_JS_EXACT_INTEGER = 2**53
# ECMAScript Number::toString: a decimal point position n in (-6, 21] is written without an exponent.
_JS_FIXED_MIN_POINT = -6
_JS_FIXED_MAX_POINT = 21
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def canonical_hash(value: Any) -> str:
    """SHA-256 hex digest of canonical JSON: sorted keys, compact separators, UTF-8 (no ASCII escapes).

    NaN and Infinity are refused (they are not JSON). Hashes of this encoding are stored; never change it.
    """
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def js_canonical_hash(value: Any) -> str:
    """SHA-256 hex digest of ``js_canonical_json(value)``: the Node backend's contract hash."""
    return hashlib.sha256(js_canonical_json(value).encode("utf-8")).hexdigest()


def js_canonical_json(value: Any) -> str:
    """Canonical JSON exactly as the Node backend writes it (its ``canonicalJson``).

    Keys sorted by UTF-16 code unit, numbers as JavaScript's ``JSON.stringify`` writes them, strings as
    ``JSON.stringify`` writes them (a lone surrogate as a lowercase ``\\udxxx`` escape). Raises ValueError
    for NaN, Infinity or a cycle and TypeError for a value JSON cannot represent, like ``json.dumps``.
    """
    return _encode(value, set())


def _encode(value: Any, ancestors: set[int]) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _js_string(value)
    if isinstance(value, (int, float)):
        return _js_number(value)
    if isinstance(value, (dict, list, tuple)):
        marker = id(value)
        if marker in ancestors:
            raise ValueError("Circular reference detected")
        ancestors.add(marker)
        try:
            if isinstance(value, dict):
                items = sorted(((_js_key(key), item) for key, item in value.items()),
                               key=lambda pair: pair[0].encode("utf-16-be", "surrogatepass"))
                return "{" + ",".join(f"{_js_string(key)}:{_encode(item, ancestors)}" for key, item in items) + "}"
            return "[" + ",".join(_encode(item, ancestors) for item in value) + "]"
        finally:
            ancestors.discard(marker)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _js_key(key: Any) -> str:
    # json.dumps turns these key types into strings too; JSON objects only ever have string keys.
    if isinstance(key, str):
        return key
    if key is None:
        return "null"
    if key is True:
        return "true"
    if key is False:
        return "false"
    if isinstance(key, (int, float)):
        return _js_number(key)
    raise TypeError(f"keys must be str, int, float, bool or None, not {type(key).__name__}")


def _js_string(value: str) -> str:
    # json.dumps(ensure_ascii=False) escapes exactly what JSON.stringify escapes (quote, backslash,
    # control characters, \b \f \n \r \t short forms, lowercase \u00xx), except a lone surrogate.
    return _LONE_SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", json.dumps(value, ensure_ascii=False))


def _js_number(value: int | float) -> str:
    """``JSON.stringify(number)``: ECMAScript Number::toString for finite values."""
    if isinstance(value, int):
        if -_JS_EXACT_INTEGER <= value <= _JS_EXACT_INTEGER:
            return int.__repr__(value)
        try:
            value = float(value)
        except OverflowError:
            raise ValueError("Out of range float values are not JSON compliant") from None
    if not math.isfinite(value):
        raise ValueError("Out of range float values are not JSON compliant")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    # repr() gives the shortest digits that round-trip, as JavaScript does; only the layout differs.
    mantissa, _, exponent_text = float.__repr__(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    point = len(whole) + (int(exponent_text) if exponent_text else 0)
    significant = digits.lstrip("0")
    point -= len(digits) - len(significant)
    significant = significant.rstrip("0")
    count = len(significant)
    if count <= point <= _JS_FIXED_MAX_POINT:
        return sign + significant + "0" * (point - count)
    if 0 < point <= _JS_FIXED_MAX_POINT:
        return sign + significant[:point] + "." + significant[point:]
    if _JS_FIXED_MIN_POINT < point <= 0:
        return sign + "0." + "0" * -point + significant
    exponent = point - 1
    head = significant[0] + ("." + significant[1:] if count > 1 else "")
    return f"{sign}{head}e{'+' if exponent >= 0 else '-'}{abs(exponent)}"
