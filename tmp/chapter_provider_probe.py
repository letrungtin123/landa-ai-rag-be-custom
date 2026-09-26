"""One authorized generation, read-only DB; stdout is a private parent-process pipe.

No prompts, request bodies or provider output are logged. Parent prints only safe
metadata and validates the private result in memory. Never use as a public API.
"""
import asyncio
import hashlib
import json
import logging
import sys
from time import perf_counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asyncpg
import requests
from app import main

logging.disable(logging.CRITICAL)
request = main.RagLessonAuthorCheckpointRequest.model_validate(json.load(sys.stdin))
events = []
dispatch_count = 0
original_http = requests.Session.request


def http_probe(session, *args, **kwargs):
    global dispatch_count
    url = kwargs.get("url", args[1] if len(args) > 1 else "")
    generating = ":generateContent" in url
    if generating:
        dispatch_count += 1
        if dispatch_count > 1:
            raise RuntimeError("AUTHORIZED_PROBE_CALL_CAP")
        body = json.loads(kwargs["data"])
        schema = body.get("generationConfig", {}).get("responseSchema", {})
        bounds = []

        def visit(node, path):
            if not isinstance(node, dict):
                return
            for key in ("minItems", "maxItems", "minLength", "maxLength", "min_items", "max_items", "min_length", "max_length"):
                if key in node:
                    bounds.append({"path": path, "constraint": key, "value": node[key]})
            for name, child in node.get("properties", {}).items():
                visit(child, path + "." + name)
            if "items" in node:
                visit(node["items"], path + ".items")

        visit(schema, "responseSchema")
        events.append({"event": "sdk_wire_schema", "bounds": bounds,
                       "fingerprint": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16],
                       "output_tokens": body["generationConfig"].get("maxOutputTokens")})
    started = perf_counter()
    response = original_http(session, *args, **kwargs)
    if generating and response.status_code == 200:
        parsed_response = response.json()
        usage = parsed_response.get("usageMetadata", {})
        candidates = parsed_response.get("candidates", [])
        finish = candidates[0].get("finishReason") if candidates else None
        events.append({"event": "provider_response", "http_status": 200,
                       "duration_ms": round((perf_counter() - started) * 1000),
                       "finish_reason": finish if finish in {"STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"} else "unavailable",
                       "provider_usage": {key: value for key, value in usage.items() if key in {"promptTokenCount", "candidatesTokenCount", "totalTokenCount", "thoughtsTokenCount"} and type(value) is int}})
    if generating and response.status_code >= 400:
        try:
            error = response.json().get("error", {})
        except ValueError:
            error = {}
        message = str(error.get("message", ""))
        known_generic = message.strip().casefold() in {
            "request contains an invalid argument.", "request contains an invalid argument", "invalid argument."
        }
        details = error.get("details", [])
        details = details if isinstance(details, list) else []
        events.append({"event": "provider_error_structure", "http_status": response.status_code,
                       "generic_invalid_argument_only": known_generic, "message_chars": len(message),
                       "message_sha256": hashlib.sha256(message.encode()).hexdigest()[:16],
                       "detail_count": len(details),
                       "bad_request_detail_count": sum(isinstance(d, dict) and d.get("@type") == "type.googleapis.com/google.rpc.BadRequest" for d in details),
                       "error_info_detail_count": sum(isinstance(d, dict) and d.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo" for d in details)})
    return response


async def run():
    pool = await asyncpg.create_pool(main.settings.database_url, min_size=1, max_size=2,
                                    server_settings={"default_transaction_read_only": "on"})
    try:
        with patch("requests.Session.request", http_probe):
            result = await main.lesson_author_chapter_checkpoint(request, pool)
            if result.get("status") == "unit_ready" and sum(len(lesson.units) for lesson in request.blueprint_architecture.lessons) == 1:
                final_request = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                    "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=result["unit"])]})
                result["probe_final"] = await main.lesson_author_chapter_checkpoint(final_request, pool)
        return {"safe": events, "status": 200, "private_result": result, "generation_http_calls": dispatch_count}
    except main.HTTPException as error:
        return {"safe": events, "status": error.status_code, "private_result": {"detail": error.detail},
                "generation_http_calls": dispatch_count}
    except Exception as error:
        return {"safe": events, "status": 0, "error_type": type(error).__name__, "generation_http_calls": dispatch_count}
    finally:
        await pool.close()


result = asyncio.run(run())
print("SAFE_PROBE_METADATA " + json.dumps({key: value for key, value in result.items() if key != "private_result"}))
print("PRIVATE_RESULT_JSON " + json.dumps(result.get("private_result", {}), ensure_ascii=False))
