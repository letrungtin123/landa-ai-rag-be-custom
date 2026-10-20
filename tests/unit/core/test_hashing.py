"""app.hashing: canonical_hash keeps its bytes; js_canonical_json writes JSON exactly as the Node backend does.

The cross-language half (the same vectors hashed by Node's orchestrationV2Hash) is the backend test
``lesson-author-orchestration-v2-canonical-json.cross-language.test.ts``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unittest
from typing import Any

from app.contracts.orchestration_v2 import provider
from app.hashing import canonical_hash, js_canonical_hash, js_canonical_json
from app.idm import contracts as idm_contracts

# (Python value, what JavaScript's JSON.stringify writes for the same number)
NUMBERS: list[tuple[int | float, str]] = [
    (0, "0"), (-0.0, "0"), (0.0, "0"), (1, "1"), (1.0, "1"), (-1.0, "-1"), (100.0, "100"), (0.5, "0.5"),
    (-2.25, "-2.25"), (0.1, "0.1"), (123.456, "123.456"), (5e-05, "0.00005"), (0.000123, "0.000123"),
    (1e-06, "0.000001"), (1e-07, "1e-7"), (1.5e-07, "1.5e-7"), (1e16, "10000000000000000"),
    (1e20, "100000000000000000000"),
    (1.2345678901234567e20, "123456789012345670000"), (1e21, "1e+21"), (1.5e300, "1.5e+300"),
    (1.7976931348623157e308, "1.7976931348623157e+308"), (5e-324, "5e-324"), (2**53 - 1, "9007199254740991"),
    (-(2**53 - 1), "-9007199254740991"), (2**53, "9007199254740992"), (2**53 + 1, "9007199254740992"),
    (10**21, "1e+21"), (12345678901234567890, "12345678901234567000"),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _random_json(rng: random.Random, depth: int) -> Any:
    roll = rng.random()
    if depth <= 0 or roll < 0.45:
        scalar = rng.random()
        if scalar < 0.1:
            return None
        if scalar < 0.2:
            return rng.random() < 0.5
        if scalar < 0.4:
            return rng.choice([0, 1, -7, 2**53 - 1, 123456789, 42])
        if scalar < 0.55:
            return rng.choice(["", "a", "Việt Nam", "Vie\u0323\u0302t", "\u2028", "😀", "ctl\u0001\n\t", '"q" \\'])
        letters = [0x41, 0x7A, 0xE9, 0x111, 0x1EBF, 0x4E2D, 0x1F600]
        return "".join(chr(rng.choice(letters)) for _ in range(rng.randrange(5)))
    if roll < 0.7:
        return [_random_json(rng, depth - 1) for _ in range(rng.randrange(4))]
    keys = ["a", "b", "z", "A", "_", "0", "10", "2", "đ", "é", "key"]
    return {rng.choice(keys): _random_json(rng, depth - 1) for _ in range(rng.randrange(5))}


class CanonicalHashTest(unittest.TestCase):
    def test_canonical_hash_bytes_are_unchanged(self) -> None:
        # Python-only hashes (evidence, snapshot structure, cache keys) are stored with this encoding.
        self.assertEqual(canonical_hash({"b": [1, "đ"], "a": None}), _sha('{"a":null,"b":[1,"đ"]}'))
        self.assertEqual(canonical_hash({"bbox": [0.0, 1.0, 5e-05]}), _sha('{"bbox":[0.0,1.0,5e-05]}'))

    def test_canonical_hash_refuses_nan_and_infinity(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                canonical_hash({"x": [value]})


class JsCanonicalJsonTest(unittest.TestCase):
    def test_numbers_are_written_like_javascript(self) -> None:
        for value, expected in NUMBERS:
            self.assertEqual(js_canonical_json(value), expected, repr(value))
            self.assertEqual(js_canonical_json([value]), f"[{expected}]", repr(value))

    def test_random_floats_round_trip_through_their_javascript_text(self) -> None:
        rng = random.Random(20261020)  # noqa: S311 - seeded test data, not security
        for _ in range(5000):
            value = rng.choice([
                rng.uniform(-1, 1), rng.uniform(-1e6, 1e6), 10 ** rng.uniform(-30, 30), round(rng.random(), 6),
                rng.random() * 10 ** rng.randrange(-8, 23), float(rng.randrange(-(10**6), 10**6)),
            ])
            text = js_canonical_json(value)
            self.assertEqual(float(text), value, text)
            self.assertNotIn("+0", text)
            if "e" not in text and value == int(value) and abs(value) < 1e21:
                self.assertNotIn(".", text, text)

    def test_strings_keys_and_order_follow_json_stringify(self) -> None:
        self.assertEqual(js_canonical_json("\u2028\u2029"), '"\u2028\u2029"')
        self.assertEqual(js_canonical_json("\u0000\u0001\b\f\n\r\t\u001f\u007f"),
                         '"\\u0000\\u0001\\b\\f\\n\\r\\t\\u001f\u007f"')
        self.assertEqual(js_canonical_json("a\ud800b"), '"a\\ud800b"')
        self.assertEqual(js_canonical_json('"q" \\'), '"\\"q\\" \\\\"')
        # UTF-16 code unit order: the astral key sorts before U+FB01 (Python's code point order is the reverse).
        value = {"ﬁ": 1, "😀": 2, "z": 0}
        self.assertEqual(js_canonical_json(value), '{"z":0,"😀":2,"ﬁ":1}')
        self.assertEqual(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                         '{"z":0,"ﬁ":1,"😀":2}')
        # No Unicode normalisation: NFC and NFD hash differently.
        self.assertNotEqual(js_canonical_hash("Việt"), js_canonical_hash("Vie\u0323\u0302t"))
        # null and a missing key differ.
        self.assertNotEqual(js_canonical_hash({"a": None}), js_canonical_hash({}))

    def test_plain_json_without_number_layout_differences_hashes_like_canonical_hash(self) -> None:
        rng = random.Random(7)  # noqa: S311 - seeded test data, not security
        for _ in range(500):
            value = _random_json(rng, 5)
            self.assertEqual(js_canonical_hash(value), canonical_hash(value))
            self.assertEqual(js_canonical_json(value), js_canonical_json(json.loads(json.dumps(value))))

    def test_values_json_cannot_hold_are_refused(self) -> None:
        for value in (math.nan, math.inf, -math.inf, 10**400):
            with self.assertRaises(ValueError):
                js_canonical_json({"x": value})
        for value in ({1, 2}, object(), b"bytes"):
            with self.assertRaises(TypeError):
                js_canonical_json([value])
        cyclic: list[Any] = []
        cyclic.append(cyclic)
        with self.assertRaises(ValueError):
            js_canonical_json(cyclic)
        self.assertEqual(js_canonical_json({1: "a", None: "b", False: "c"}), '{"1":"a","false":"c","null":"b"}')

    def test_cross_language_hashes_use_the_node_encoding(self) -> None:
        # The design hash Python writes and the brief hash it checks are Node's encoding (Node writes or checks them).
        payload = {"design_hash": "x", "est_minutes": 1.0, "weights": [5e-05], "title": "Thiết kế"}
        self.assertEqual(idm_contracts.design_hash_of(payload),
                         _sha('{"est_minutes":1,"title":"Thiết kế","weights":[0.00005]}'))
        self.assertEqual(idm_contracts.brief_hash_of({"brief_hash": "x", "share": 0.5}), _sha('{"share":0.5}'))
        self.assertIs(provider.js_canonical_hash, js_canonical_hash)


if __name__ == "__main__":
    unittest.main()
