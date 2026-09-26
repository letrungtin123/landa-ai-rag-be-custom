"""Explicitly authorized synthetic probes only; private settings via stdin.

No DB, no embeddings, no retry, no returned content or credentials in stdout.
Invocation count must also be accounted against the user's overall approval.
"""
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
import requests
from tests.staged_schema_probe import capture_sdk_body, schema_cases, schema_metadata, isolated_bound_variant, BOUNDS


def run():
    config = json.load(sys.stdin)
    names = config["cases"]
    if not 1 <= len(names) <= 10 or not re.fullmatch(r"[a-zA-Z0-9._-]+", config["model"]):
        raise ValueError("INVALID_PROBE_REQUEST")
    cases = schema_cases()
    session = requests.Session()
    for index, name in enumerate(names, 1):
        base, _, variant = name.partition(":")
        if base not in cases or variant not in {"", "no_string_bounds", "no_value_bounds"}:
            raise ValueError("INVALID_PROBE_CASE")
        body = capture_sdk_body(cases[base], model=config["model"])
        if variant:
            body = isolated_bound_variant(body, {"min_length", "max_length"} if variant == "no_string_bounds" else BOUNDS)
        metadata = {"case": name, "correlation_id": config["correlation_id"], "model": config["model"],
                    "call_number": config["start_number"] + index - 1, **schema_metadata(body)}
        if metadata["call_number"] > 10:
            raise ValueError("AUTHORIZED_CALL_CAP")
        print(json.dumps({**metadata, "event": "dispatch"}), flush=True)
        started = time.perf_counter()
        try:
            result = session.post(
                "https://generativelanguage.googleapis.com/v1beta/models/" + config["model"] + ":generateContent",
                headers={"x-goog-api-key": config["api_key"], "Content-Type": "application/json"},
                data=json.dumps(body), timeout=180, allow_redirects=False)
            try:
                data = result.json()
            except ValueError:
                data = {}
            metadata.update(http_status=result.status_code, duration_ms=round((time.perf_counter()-started)*1000))
            if result.status_code == 200:
                candidate = (data.get("candidates") or [{}])[0]
                parts = candidate.get("content", {}).get("parts", [])
                output = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                try:
                    json.loads(output)
                    complete = True
                except (ValueError, TypeError):
                    complete = False
                usage = data.get("usageMetadata", {})
                metadata.update(provider_acceptance="ACCEPTED", finish_reason=candidate.get("finishReason", "unavailable"),
                                json_complete=complete, response_chars=len(output),
                                usage_source="provider" if usage else "unavailable",
                                usage={k:v for k,v in usage.items() if k in {"promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"} and type(v) is int})
            else:
                error = data.get("error", {})
                message = str(error.get("message", ""))
                metadata.update(provider_acceptance="REJECTED", provider_status=error.get("status"),
                                generic_invalid_argument=message.strip().casefold() == "request contains an invalid argument.",
                                error_message_chars=len(message), error_sha256=hashlib.sha256(message.encode()).hexdigest()[:16],
                                error_detail_count=len(error.get("details", [])), usage_source="unavailable")
            print(json.dumps({**metadata, "event": "result"}), flush=True)
            if result.status_code != 200 and result.status_code != 400:
                break  # quota/auth/transient failure is not permission to retry
        except requests.RequestException as error:
            print(json.dumps({**metadata, "event": "transport_failure", "error_type": type(error).__name__,
                              "duration_ms": round((time.perf_counter()-started)*1000)}), flush=True)
            break


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(json.dumps({"event": "local_failure", "error_type": type(error).__name__}), flush=True)
        sys.exit(1)
