"""Versioned display order; no fact allocation or semantic-entailment claims."""
from typing import Any
from copy import deepcopy

FIELDS = {"paragraph": "paragraphs", "bullets": "bullet_points", "steps": "ordered_steps", "warning": "warnings", "table": "comparison_rows", "task": "paragraphs"}
LEGACY_FIELDS = ("heading", "paragraphs", "bullet_points", "ordered_steps", "warnings", "comparison_rows", "bullets", "steps", "warning", "table_rows")


class ProviderSemanticVersionError(ValueError):
    pass


def semantic_shape_diagnostics(value: Any) -> dict:
    """Fixed field names/counts only, never values or arbitrary provider keys."""
    content = value if isinstance(value, dict) else {}
    sections = content.get("sections")
    return {
        "semantic_version": content.get("version") if type(content.get("version")) is int and content["version"] in (1, 2) else "missing_or_invalid",
        "legacy_fields_present": [key for key in LEGACY_FIELDS if key in content],
        "legacy_fields_populated": [key for key in LEGACY_FIELDS if content.get(key)],
        "unknown_field_count": len(set(content) - {"version", "sections", *LEGACY_FIELDS}),
        "section_count": len(sections) if isinstance(sections, list) else 0,
        "block_count": sum(len(s["blocks"]) for s in sections if isinstance(s, dict) and isinstance(s.get("blocks"), list)) if isinstance(sections, list) else 0,
    }


def bind_provider_semantic_versions(value: Any) -> Any:
    """Stamp the new writer's format, not its content or evidence declarations.

    Called only on freshly parsed Stage-2 responses, including SDK raw fallback.
    Do not invoke on persisted/legacy reads. Unknown and legacy fields survive
    intact so the ordinary component validation/repair path can reject them.
    """
    result = deepcopy(value)

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            semantic = node.get("semantic_content")
            if isinstance(semantic, dict):
                # A provider cannot select a legacy format on this writer path.
                # Preserve an invalid explicit version for rejection, never fix it.
                if "version" in semantic and (type(semantic["version"]) is not int or semantic["version"] != 2):
                    raise ProviderSemanticVersionError("HTML_SEMANTIC_WRITER_VERSION_INVALID")
                semantic.setdefault("version", 2)
            for key in ("components", "units", "lessons", "chapters", "chapter", "unit", "proposal"):
                children = node.get(key)
                if key == "components" and isinstance(children, dict):
                    for child in children.values():
                        visit(child)
                else:
                    visit(children)
    visit(result)
    return result


def ordered_content_fields(value: Any) -> list[tuple[str, str]]:
    """Preserve safe paths for observations; called after structural acceptance."""
    if not isinstance(value, dict) or value.get("version") != 2:
        return []
    result = []
    for si, section in enumerate(value.get("sections") or []):
        prefix = f"sections[{si}]"
        result.append((f"{prefix}.heading", section["heading"]))
        for bi, block in enumerate(section["blocks"]):
            path = f"{prefix}.blocks[{bi}]"
            if block["kind"] in {"paragraph", "warning", "task"}:
                result.append((f"{path}.text", block["text"]))
            elif block["kind"] == "table":
                result.extend((f"{path}.rows[{ri}]", row["label"] + " " + row["value"]) for ri, row in enumerate(block["rows"]))
            else:
                result.extend((f"{path}.items[{ii}]", item) for ii, item in enumerate(block["items"]))
    return result


def flatten_ordered_content(value: dict) -> dict:
    """Validate v2 shape then adapt for existing aggregate size/quality checks.

    This is not the renderer: ordered sections remain intact in persisted payloads.
    Reject mixed forms rather than silently dropping one representation.
    """
    if value.get("version") != 2:
        if value.get("version") not in (None, 1) or value.get("sections"):
            raise ValueError("Unsupported semantic content version.")
        return value
    if any(value.get(k) for k in ("heading", "paragraphs", "bullet_points", "ordered_steps", "warnings", "comparison_rows", "bullets", "steps", "warning", "table_rows")):
        raise ValueError("Mixed ordered and legacy semantic content.")
    if set(value) - {"version", "sections", "heading", "paragraphs", "bullet_points", "ordered_steps", "warnings", "comparison_rows"}:
        raise ValueError("Unknown ordered content field.")
    sections = value.get("sections")
    if not isinstance(sections, list) or not 1 <= len(sections) <= 12:
        raise ValueError("Ordered content requires 1-12 sections.")
    flat = {field: [] for field in set(FIELDS.values())}
    for section in sections:
        if not isinstance(section, dict) or set(section) - {"heading", "learning_block_ids", "blocks"}:
            raise ValueError("Invalid semantic section.")
        heading = section.get("heading")
        if not isinstance(heading, str) or not heading.strip() or len(heading.strip()) > 240:
            raise ValueError("Section heading must be 1-240 characters.")
        refs = section.get("learning_block_ids", [])
        if not isinstance(refs, list) or len(refs) > 24 or any(not isinstance(r, str) or not r.strip() or len(r) > 160 for r in refs) or len(set(refs)) != len(refs):
            raise ValueError("Invalid section learning block references.")
        blocks = section.get("blocks")
        if not isinstance(blocks, list) or not 1 <= len(blocks) <= 12:
            raise ValueError("Section requires 1-12 ordered blocks.")
        for block in blocks:
            if not isinstance(block, dict) or block.get("kind") not in FIELDS or set(block) - {"kind", "text", "items", "rows"}:
                raise ValueError("Unknown ordered block kind or field.")
            kind = block["kind"]
            field = "rows" if kind == "table" else "items" if kind in {"bullets", "steps"} else "text"
            if any(block.get(k) for k in {"text", "items", "rows"} - {field}):
                raise ValueError("Mixed ordered block fields.")
            data = block.get(field)
            if field == "text":
                if not isinstance(data, str) or not data.strip():
                    raise ValueError("Ordered block text required.")
                data = [data]
            elif not isinstance(data, list) or not data:
                raise ValueError("Ordered block items required.")
            flat[FIELDS[kind]].extend(data)
    return flat
