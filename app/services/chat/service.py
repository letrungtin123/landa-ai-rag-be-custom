"""Grounded chat answers over retrieved knowledge-base chunks."""

from __future__ import annotations

from typing import Any, Literal

import asyncpg

from app.core.config import settings
from app.core.logging import redact_secret_like_values
from app.prompt_safety import untrusted_block, untrusted_content_rule
from app.schemas.chat import RagChatMessage, RagChatRequest
from app.services import provider
from app.services.deadlines import run_with_deadline
from app.services.provider import combine_usage
from app.services.retrieval import search as retrieval_search
from app.services.retrieval.query import retrieval_limits
from app.services.retrieval.search import build_retrieval_diagnostics, format_sources


def format_history(history: list[RagChatMessage]) -> str:
    items = history[-12:]
    lines: list[str] = []
    for item in items:
        label = "Người dùng" if item.role == "user" else "Trợ lý"
        lines.append(f"{label}: {item.content[:1800]}")
    return "\n".join(lines)


def build_no_context_answer(locale: Literal["vi", "en"]) -> str:
    if locale == "en":
        return (
            "I could not find enough relevant information in the selected Knowledge Base to answer this accurately. "
            "Please check that the bot is linked to the correct Knowledge Base and that the related files have finished learning."
        )
    return (
        "Hiện tại tôi chưa tìm thấy đủ thông tin liên quan trong Kho tri thức đang chọn để trả lời chính xác. "
        "Vui lòng kiểm tra bot đã được gắn đúng Kho tri thức và các tài liệu liên quan đã học xong."
    )


CHAT_UNTRUSTED_TAGS = ("SOURCE_DOCUMENTS", "SOURCE_OUTLINE", "COURSE_CONTEXT", "CONVERSATION_HISTORY", "USER_QUESTION")
UNTRUSTED_CONTENT_RULE_VI = untrusted_content_rule(CHAT_UNTRUSTED_TAGS, "vi")
UNTRUSTED_CONTENT_RULE_EN = untrusted_content_rule(CHAT_UNTRUSTED_TAGS, "en")
SYSTEM_PROMPT_LEAK_MIN_CHARS = 80


def guard_chat_output(text: str, system_prompt: str) -> str:
    """Never return provider keys/secrets or a verbatim copy of the tenant system prompt."""
    guarded = redact_secret_like_values(text)
    persona = (system_prompt or "").strip()
    if len(persona) >= SYSTEM_PROMPT_LEAK_MIN_CHARS and persona in guarded:
        guarded = guarded.replace(persona, "[…]")
    return guarded


def build_chat_prompt(
    request: RagChatRequest,
    context: str,
    source_outline: str = "",
) -> str:
    locale_rule = "Trả lời bằng tiếng Việt có dấu." if request.locale == "vi" else "Answer in English."
    knowledge_rule = (
        "Nguyên tắc: ưu tiên tài liệu/kiến thức được cung cấp. Nếu tài liệu không đủ, nói rõ phần còn thiếu và không bịa dữ kiện."
        if request.locale == "vi"
        else "Principle: prioritize the provided documents/knowledge. If the material is insufficient, state what is missing and do not invent facts."
    )
    no_context = (
        "Chưa tìm thấy đoạn tài liệu liên quan trong kho kiến thức."
        if request.locale == "vi"
        else "No relevant document excerpt was found in the Knowledge Base."
    )
    return "\n\n".join(
        part
        for part in [
            request.system_prompt,
            locale_rule,
            knowledge_rule,
            "Trả lời đầy đủ theo yêu cầu. Với câu hỏi đơn giản, trả lời gọn; với câu hỏi cần giải thích, trình bày đủ các ý chính hoặc từng bước. Không cắt bỏ điều kiện quan trọng.",
            UNTRUSTED_CONTENT_RULE_VI if request.locale == "vi" else UNTRUSTED_CONTENT_RULE_EN,
            f"Lịch sử hội thoại gần đây:\n{untrusted_block('CONVERSATION_HISTORY', format_history(request.history))}"
            if request.history else "",
            f"Ngữ cảnh khóa học hiện tại:\n{untrusted_block('COURSE_CONTEXT', request.course_context)}"
            if request.course_context else "",
            f"Cấu trúc mục lục/tiêu đề của tài liệu nguồn:\n{untrusted_block('SOURCE_OUTLINE', source_outline)}"
            if source_outline else "",
            f"Tài liệu/kiến thức liên quan:\n{untrusted_block('SOURCE_DOCUMENTS', context)}" if context else no_context,
            f"Câu hỏi hiện tại:\n{untrusted_block('USER_QUESTION', request.user_message)}",
        ]
        if part
    )


async def chat(request: RagChatRequest, pool: asyncpg.Pool) -> dict[str, Any]:
    return await run_with_deadline("/v1/chat", settings.chat_deadline_ms, _chat(request, pool))


async def _chat(request: RagChatRequest, pool: asyncpg.Pool) -> dict[str, Any]:
    rows, retrieval_usage, structure_context = await retrieval_search.retrieve_chunks(pool, request)
    context, sources = format_sources(rows, max_context_chars=retrieval_limits(request)["max_context_chars"])
    retrieval = build_retrieval_diagnostics(request, rows, sources, structure_context)
    if not context:
        return {
            "text": build_no_context_answer(request.locale),
            "usage": retrieval_usage.model_dump(),
            "sources": sources,
            "retrieval": retrieval,
        }
    prompt = build_chat_prompt(request, context, structure_context.get("outline", ""))
    text, generation_usage = await provider.generate_content(
        request.api_key,
        request.model,
        prompt,
        max_output_tokens=request.max_output_tokens,
    )
    usage = combine_usage(retrieval_usage, generation_usage)
    text = guard_chat_output(text, request.system_prompt)
    return {"text": text, "usage": usage.model_dump(), "sources": sources, "retrieval": retrieval}
