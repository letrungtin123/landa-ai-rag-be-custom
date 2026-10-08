"""Text and layout extraction from uploaded documents (PDF, DOCX, PPTX, XLSX, XLS, DOC, text)."""

from __future__ import annotations

import io
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.document_limits import (
    INDEX_DOCUMENT_SUFFIXES,
    assert_document_size,
    run_limited_subprocess,
    validate_ooxml_archive,
)
from app.core.errors import DocumentLimitError
from app.infra.pdf_layout import column_reading_order
from app.services.text import clean_text
from app.source_readiness import collect_pdf_visual_regions


@dataclass
class ExtractedSection:
    text: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def decode_bytes(buffer: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return buffer.decode(encoding)
        except UnicodeDecodeError:
            continue
    return buffer.decode("utf-8", errors="ignore")


STRUCTURED_EXTRACTION_VERSION = "structured-source-v1"


def _clean_table_cell(value: Any) -> str:
    return clean_text("" if value is None else str(value)).replace("|", "¦")


def _render_structured_table(rows: list[list[Any]], *, coordinate_labels: bool = False) -> str:
    """Render a table without flattening row/column relationships into prose."""

    rendered: list[str] = ["[TABLE]"]
    for row_number, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_number, value in enumerate(row, start=1):
            cell = _clean_table_cell(value)
            if coordinate_labels:
                if not cell:
                    continue
                # Spreadsheet columns remain identifiable even when blank cells
                # exist between populated values.
                column_label = ""
                number = column_number
                while number:
                    number, remainder = divmod(number - 1, 26)
                    column_label = chr(65 + remainder) + column_label
                cells.append(f"{column_label}{row_number}={cell}")
            else:
                # A blank native-table cell is positional evidence. Keep it so
                # every later value remains under its original source column.
                cells.append(cell)
        if any(cells):
            rendered.append(f"Row {row_number}: " + " | ".join(cells))
    return "\n".join(rendered) if len(rendered) > 1 else ""


def _bbox_overlap_ratio(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return intersection / area if area else 0.0


def _pdf_column_fragments(
    text_entries: list[dict[str, Any]],
    table_entries: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Re-read a multi-column page line by line; ``None`` for a single-column page.

    PyMuPDF merges lines that share a baseline across columns into one block, so columns are
    detected on line boxes. Consecutive lines of the same block in the new order form one
    fragment (joined like a block); tables stay whole.
    """
    units: list[tuple[tuple[str, int], dict[str, Any]]] = []
    for block_number, entry in enumerate(text_entries):
        for line in entry.get("lines") or []:
            units.append((("text", block_number), line))
    for table_number, entry in enumerate(table_entries):
        units.append((("table", table_number), entry))
    permutation = column_reading_order([unit["bbox"] for _, unit in units])
    if permutation is None:
        return None
    fragments: list[dict[str, Any]] = []
    previous_group: tuple[str, int] | None = None
    for index in permutation:
        group, unit = units[index]
        if group == previous_group:
            fragment = fragments[-1]
            fragment["text"] = f"{fragment['text']}\n{unit['text']}"
            fragment["max_font_size"] = max(fragment["max_font_size"], float(unit.get("max_font_size") or 0.0))
        else:
            fragments.append({"kind": group[0], "text": unit["text"],
                              "max_font_size": float(unit.get("max_font_size") or 0.0)})
        previous_group = group
    return fragments


def _pdf_page_content(page: Any) -> tuple[str, dict[str, Any]]:
    page_dict = page.get_text("dict", sort=True) or {}
    span_sizes: list[float] = []
    text_entries: list[dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        lines: list[str] = []
        block_sizes: list[float] = []
        line_units: list[dict[str, Any]] = []
        bbox = block.get("bbox")
        for line in block.get("lines", []):
            spans = line.get("spans", []) if isinstance(line, dict) else []
            line_text = clean_text("".join(str(span.get("text") or "") for span in spans if isinstance(span, dict)))
            line_sizes: list[float] = []
            for span in spans:
                if isinstance(span, dict) and isinstance(span.get("size"), (int, float)):
                    size = float(span["size"])
                    block_sizes.append(size)
                    span_sizes.append(size)
                    line_sizes.append(size)
            if line_text:
                lines.append(line_text)
                line_bbox = line.get("bbox") if isinstance(line, dict) else None
                line_units.append({
                    "bbox": tuple(float(value) for value in (
                        line_bbox if isinstance(line_bbox, (list, tuple)) and len(line_bbox) == 4 else bbox
                    )),
                    "text": line_text,
                    "max_font_size": max(line_sizes, default=0.0),
                })
        text = "\n".join(lines).strip()
        if text and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            text_entries.append({
                "kind": "text",
                "bbox": tuple(float(value) for value in bbox),
                "text": text,
                "max_font_size": max(block_sizes, default=0.0),
                "lines": line_units,
            })

    table_entries: list[dict[str, Any]] = []
    try:
        finder = page.find_tables()
        for table in getattr(finder, "tables", []) or []:
            rows = table.extract() or []
            rendered = _render_structured_table(rows)
            bbox = getattr(table, "bbox", None)
            if rendered and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                table_entries.append({
                    "kind": "table",
                    "bbox": tuple(float(value) for value in bbox),
                    "text": rendered,
                })
    except (AttributeError, TypeError, ValueError):
        # Text extraction remains available for malformed or unsupported table
        # geometry. The diagnostic below makes the fallback observable.
        table_entries = []

    table_boxes = [entry["bbox"] for entry in table_entries]
    visible_text_entries = [
        entry for entry in text_entries
        if not any(_bbox_overlap_ratio(entry["bbox"], table_bbox) >= 0.8 for table_bbox in table_boxes)
    ]
    ordered_entries = sorted(
        [*visible_text_entries, *table_entries],
        key=lambda entry: (entry["bbox"][1], entry["bbox"][0]),
    )
    heading_entries = visible_text_entries
    reading_order = "top_to_bottom_left_to_right"
    if settings.pdf_column_aware:
        # Side-by-side columns are read column by column; single-column pages keep the order above.
        fragments = _pdf_column_fragments(visible_text_entries, table_entries)
        if fragments is not None:
            ordered_entries = fragments
            heading_entries = [fragment for fragment in fragments if fragment["kind"] == "text"]
            reading_order = "columns_left_to_right"

    body_font_size = sorted(span_sizes)[(len(span_sizes) - 1) // 2] if span_sizes else 0.0
    heading_candidates: list[dict[str, Any]] = []
    for entry in heading_entries:
        title = entry["text"].replace("\n", " ").strip()
        size = float(entry.get("max_font_size") or 0.0)
        if (
            body_font_size > 0
            and size >= max(body_font_size * 1.18, body_font_size + 1.5)
            and 3 <= len(title) <= 160
            and len(title.split()) <= 18
        ):
            heading_candidates.append({
                "title": title,
                "level": 1 if size >= body_font_size * 1.45 else 2,
            })

    return clean_text("\n\n".join(entry["text"] for entry in ordered_entries)), {
        "extraction_version": STRUCTURED_EXTRACTION_VERSION,
        "content_kinds": sorted({entry["kind"] for entry in ordered_entries}),
        "table_count": len(table_entries),
        "heading_candidates": heading_candidates,
        "reading_order": reading_order,
    }


def _visual_prompt_text(page_text: str) -> str | None:
    candidates = [
        re.sub(r"\s+", " ", line).strip()
        for line in page_text.splitlines()
        if re.search(
            r"\?|\b(?:hãy|quan sát|tình huống|thảo luận|yêu cầu|identify|observe|scenario|discuss)\b",
            line,
            flags=re.IGNORECASE,
        )
    ]
    value = " ".join(candidate for candidate in candidates[:4] if candidate).strip()
    return value[:600] or None


def extract_pdf(path: Path) -> list[ExtractedSection]:
    # Parse from memory: after a failed open PyMuPDF keeps the file handle alive on
    # Windows, which blocks temp-dir cleanup, leaves the upload on disk and masks
    # the real parser error. The size is already bounded by assert_document_size.
    data = path.read_bytes()
    try:
        import pymupdf

        sections: list[ExtractedSection] = []
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.page_count > settings.max_document_pages:
                raise DocumentLimitError()
            visual_regions_by_page = collect_pdf_visual_regions(document)
            for index, page in enumerate(document, start=1):
                text, metadata = _pdf_page_content(page)
                if text:
                    visual_regions = visual_regions_by_page.get(index, [])
                    prompt_text = _visual_prompt_text(text)
                    if visual_regions:
                        metadata["visual_regions"] = visual_regions[:4]
                        metadata["visual_region_count"] = len(visual_regions[:4])
                        if prompt_text:
                            metadata["visual_prompt_text"] = prompt_text
                    sections.append(ExtractedSection(text=text, page=index, metadata=metadata))
        if sections:
            return sections
    except ImportError:
        pass

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if len(reader.pages) > settings.max_document_pages:
        raise DocumentLimitError()
    sections: list[ExtractedSection] = []
    for index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            sections.append(ExtractedSection(text=text, page=index))
    if sections:
        return sections
    raise ValueError(
        "OCR_REQUIRED: PDF không có lớp văn bản để đọc. Hãy tải lên PDF có thể chọn/copy văn bản "
        "hoặc bổ sung bước OCR trước khi học tài liệu."
    )


def extract_docx(path: Path) -> list[ExtractedSection]:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    parts: list[str] = []
    heading_candidates: list[dict[str, Any]] = []
    table_count = 0
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, document)
            text = clean_text(paragraph.text)
            if not text:
                continue
            parts.append(text)
            style_name = str(getattr(paragraph.style, "name", "") or "")
            match = re.match(r"Heading\s+(\d+)$", style_name, flags=re.IGNORECASE)
            if match:
                heading_candidates.append({"title": text[:220], "level": max(1, min(6, int(match.group(1))))})
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            rendered = _render_structured_table([
                [cell.text for cell in row.cells]
                for row in table.rows
            ])
            if rendered:
                table_count += 1
                parts.append(rendered)
    return [ExtractedSection(
        text=clean_text("\n\n".join(parts)),
        metadata={
            "extraction_version": STRUCTURED_EXTRACTION_VERSION,
            "content_kinds": ["table", "text"] if table_count else ["text"],
            "table_count": table_count,
            "heading_candidates": heading_candidates,
            "reading_order": "document_order",
        },
    )]


def extract_pptx(path: Path) -> list[ExtractedSection]:
    from pptx import Presentation

    presentation = Presentation(str(path))
    if len(presentation.slides) > settings.max_document_pages:
        raise DocumentLimitError()
    sections: list[ExtractedSection] = []
    for index, slide in enumerate(presentation.slides, start=1):
        parts: list[str] = []
        heading_candidates: list[dict[str, Any]] = []
        table_count = 0
        for shape in sorted(slide.shapes, key=lambda item: (int(item.top), int(item.left))):
            if getattr(shape, "has_table", False):
                rendered = _render_structured_table([
                    [cell.text for cell in row.cells]
                    for row in shape.table.rows
                ])
                if rendered:
                    table_count += 1
                    parts.append(rendered)
                continue
            text = getattr(shape, "text", "")
            if text and text.strip():
                cleaned = clean_text(text)
                parts.append(cleaned)
                if shape == slide.shapes.title:
                    heading_candidates.append({"title": cleaned[:220], "level": 1})
        notes_included = False
        try:
            notes_text = clean_text(slide.notes_slide.notes_text_frame.text)
            if notes_text:
                parts.append("[SPEAKER NOTES]\n" + notes_text)
                notes_included = True
        except (AttributeError, KeyError):
            pass
        text = clean_text("\n\n".join(parts))
        if text:
            sections.append(ExtractedSection(
                text=text,
                page=index,
                section=f"Slide {index}",
                metadata={
                    "extraction_version": STRUCTURED_EXTRACTION_VERSION,
                    "content_kinds": sorted({"text", *( ["table"] if table_count else []), *( ["speaker_notes"] if notes_included else [])}),
                    "table_count": table_count,
                    "heading_candidates": heading_candidates,
                    "notes_included": notes_included,
                    "reading_order": "top_to_bottom_left_to_right",
                },
            ))
    return sections


def extract_xlsx(path: Path) -> list[ExtractedSection]:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=False)
    sections: list[ExtractedSection] = []
    cell_count = 0
    try:
        for sheet in workbook.worksheets:
            values: list[list[Any]] = []
            for row in sheet.iter_rows(values_only=True):
                row_values = list(row)
                cell_count += len(row_values)
                if cell_count > settings.max_xlsx_cells:
                    raise DocumentLimitError()
                values.append(row_values)
            text = clean_text(_render_structured_table(values, coordinate_labels=True))
            if text:
                sections.append(ExtractedSection(
                    text=text,
                    section=sheet.title,
                    metadata={
                        "extraction_version": STRUCTURED_EXTRACTION_VERSION,
                        "content_kinds": ["table"],
                        "table_count": 1,
                        "heading_candidates": [{"title": sheet.title[:220], "level": 1}],
                        "reading_order": "row_major_with_coordinates",
                    },
                ))
    finally:
        workbook.close()
    return sections


def extract_xls(path: Path) -> list[ExtractedSection]:
    import xlrd

    workbook = xlrd.open_workbook(str(path))
    sections: list[ExtractedSection] = []
    cell_count = 0
    for sheet in workbook.sheets():
        cell_count += sheet.nrows * sheet.ncols
        if cell_count > settings.max_xlsx_cells:
            raise DocumentLimitError()
        values = [sheet.row_values(row_index) for row_index in range(sheet.nrows)]
        text = clean_text(_render_structured_table(values, coordinate_labels=True))
        if text:
            sections.append(ExtractedSection(
                text=text,
                section=sheet.name,
                metadata={
                    "extraction_version": STRUCTURED_EXTRACTION_VERSION,
                    "content_kinds": ["table"],
                    "table_count": 1,
                    "heading_candidates": [{"title": sheet.name[:220], "level": 1}],
                    "reading_order": "row_major_with_coordinates",
                },
            ))
    return sections


def extract_doc(path: Path) -> list[ExtractedSection]:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile = (Path(temp_dir) / "libreoffice-profile").resolve()
            run_limited_subprocess(
                [
                    soffice,
                    "--headless",
                    f"-env:UserInstallation={profile.as_uri()}",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    temp_dir,
                    str(path),
                ],
                timeout_seconds=settings.libreoffice_timeout_seconds,
            )
            converted = Path(temp_dir) / f"{path.stem}.docx"
            if converted.exists():
                return extract_docx(converted)
    text = clean_text(re.sub(r"[^\x09\x0a\x0d\x20-\x7e\u00c0-\u1ef9]+", " ", decode_bytes(path.read_bytes())))
    if len(text) < 200:
        raise ValueError("File .doc quá cũ hoặc không trích xuất được đủ nội dung. Hãy chuyển file sang .docx hoặc PDF.")
    return [ExtractedSection(text=text)]


def extract_plain_text(path: Path) -> list[ExtractedSection]:
    return [ExtractedSection(text=clean_text(decode_bytes(path.read_bytes())))]


def index_document_temp_path(temp_dir: str, document_id: str, original_name: str | None) -> Path:
    """Return a server-owned temporary path while preserving extractor routing."""
    suffix = Path(original_name or "").suffix.lower()
    safe_suffix = suffix if suffix in INDEX_DOCUMENT_SUFFIXES else ".txt"
    safe_document_id = str(UUID(str(document_id)))
    root = Path(temp_dir).resolve()
    candidate = (root / f"{safe_document_id}{safe_suffix}").resolve()
    if candidate.parent != root:
        raise ValueError("Invalid temporary document path.")
    return candidate


def extract_sections(path: Path, file_name: str) -> list[ExtractedSection]:
    ext = Path(file_name).suffix.lower()
    assert_document_size(path.stat().st_size, maximum=settings.max_document_bytes)
    # The zip-bomb guard keys off the same suffix the extractor is routed by (F2).
    validate_ooxml_archive(
        path,
        suffix=ext,
        max_uncompressed_bytes=settings.max_ooxml_uncompressed_bytes,
        max_entries=settings.max_ooxml_entries,
        max_compression_ratio=settings.max_ooxml_compression_ratio,
    )
    if ext == ".pdf":
        sections = extract_pdf(path)
    elif ext == ".docx":
        sections = extract_docx(path)
    elif ext == ".doc":
        sections = extract_doc(path)
    elif ext == ".pptx":
        sections = extract_pptx(path)
    elif ext == ".xlsx":
        sections = extract_xlsx(path)
    elif ext == ".xls":
        sections = extract_xls(path)
    elif ext in {".txt", ".md", ".csv"}:
        sections = extract_plain_text(path)
    else:
        raise ValueError(f"Định dạng file chưa được hỗ trợ: {ext or file_name}")
    sections = [section for section in sections if clean_text(section.text)]
    if not sections:
        raise ValueError("Không trích xuất được nội dung từ tài liệu.")
    return sections
