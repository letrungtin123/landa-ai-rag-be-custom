from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import threading
import time
from typing import Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_network() -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("CP3A_NETWORK_FORBIDDEN")

    socket.create_connection = blocked  # type: ignore[assignment]
    original_socket = socket.socket

    class GuardedSocket(original_socket):
        def connect(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

        def connect_ex(self, *_args: Any, **_kwargs: Any) -> Any:
            return blocked()

    socket.socket = GuardedSocket  # type: ignore[assignment,misc]


def _environment_bytes(prefix: Path) -> int:
    total = 0
    for path in prefix.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _table_summary(box: dict[str, Any]) -> dict[str, Any]:
    table = box.get("table") if isinstance(box.get("table"), dict) else {}
    rows = table.get("extract") if isinstance(table.get("extract"), list) else []
    flattened = "\n".join(
        " | ".join(str(cell or "").strip() for cell in row)
        for row in rows
        if isinstance(row, list)
    )
    return {
        "bbox": [round(float(box[key]), 3) for key in ("x0", "y0", "x1", "y1")],
        "row_count": int(table.get("row_count") or len(rows)),
        "column_count": int(table.get("col_count") or max((len(row) for row in rows), default=0)),
        "text": flattened[:20_000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated PyMuPDF4LLM CP3A candidate")
    parser.add_argument("--source", required=True)
    parser.add_argument("--pages", default="")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"
    _block_network()

    import psutil
    import pymupdf4llm

    source = Path(args.source).resolve()
    selected_pages = [int(value) - 1 for value in args.pages.split(",") if value.strip()]
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    stop = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop.wait(0.01):
            peak_rss = max(peak_rss, process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started_cpu = process.cpu_times()
    started = time.perf_counter()
    raw = pymupdf4llm.to_json(
        str(source),
        pages=selected_pages or None,
        use_ocr=False,
        force_ocr=False,
        write_images=False,
        embed_images=False,
        show_progress=False,
    )
    elapsed = time.perf_counter() - started
    cpu = process.cpu_times()
    stop.set()
    sampler.join(timeout=1)
    parsed = json.loads(raw)

    pages: list[dict[str, Any]] = []
    for page in parsed.get("pages", []):
        boxes = [box for box in page.get("boxes", []) if isinstance(box, dict)]
        class_counts = Counter(str(box.get("boxclass") or "unknown") for box in boxes)
        tables = [_table_summary(box) for box in boxes if box.get("boxclass") == "table"]
        pictures = [
            {
                "bbox": [round(float(box[key]), 3) for key in ("x0", "y0", "x1", "y1")],
            }
            for box in boxes
            if box.get("boxclass") == "picture"
        ]
        pages.append({
            "page": int(page.get("page_number") or 0),
            "class_counts": dict(sorted(class_counts.items())),
            "table_count": len(tables),
            "tables": tables,
            "picture_count": len(pictures),
            "pictures": pictures,
        })

    result = {
        "candidate": "pymupdf4llm-layout",
        "candidate_version": str(getattr(pymupdf4llm, "__version__", "unknown")),
        "python_version": platform.python_version(),
        "source_sha256": _file_sha256(source),
        "page_count": int(parsed.get("page_count") or 0),
        "pages": pages,
        "performance": {
            "elapsed_seconds": round(elapsed, 4),
            "cpu_seconds": round(
                (cpu.user - started_cpu.user) + (cpu.system - started_cpu.system),
                4,
            ),
            "peak_rss_bytes": peak_rss,
            "isolated_environment_bytes": _environment_bytes(Path(sys.prefix)),
        },
        "controls": {
            "network_blocked": True,
            "ocr_used": False,
            "remote_service_used": False,
            "images_written": False,
        },
    }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
