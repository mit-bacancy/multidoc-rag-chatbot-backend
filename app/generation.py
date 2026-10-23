"""Query rewriting, prompt building and answer generation."""

import re
from collections.abc import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import DOCUMENTS
from app.llm import get_chat_llm, invoke_text, message_text
from app.retrieval import ContextBlock, get_retriever
from app.schemas import ChatMessage, ChatResponse, Source

REWRITE_PROMPT = (
    "Rewrite the user's latest message as a single standalone question that can be understood "
    "without the conversation. Keep names, numbers and years exactly. "
    "Return only the question."
)

SYSTEM_PROMPT = """You answer questions about the document "{title}" using ONLY the context below.

Rules:
- Use only the context. Do not use outside knowledge.
- Cite every fact as [p. X, Section] using the page and section shown on its context block. If a block spans pages, use the nearest preceding [p. N] marker inside the block for the exact page. If a block has no section, cite just [p. X]. Never invent a section name.
- Keep numbers, units and years exactly as written in the context.
- If the answer is not in the context, say clearly that the document does not contain it.
- Be concise; use lists or tables when they help.

Context:
{context}"""


# Words that make a message depend on earlier turns ("what about it", "compare that")
_FOLLOW_UP = re.compile(
    r"^\s*(and|but|so|or)\b|\b(it|its|this|that|these|those|they|them|their|there|above|"
    r"previous|same|else|compare|compared|versus|vs|instead|what about|how about)\b",
    re.I,
)


def needs_rewrite(message: str, history: list[ChatMessage]) -> bool:
    """A rewrite costs a full LLM round trip (~3s), so only pay it for real follow-ups."""
    return bool(history) and (len(message.split()) <= 6 or bool(_FOLLOW_UP.search(message)))


def rewrite_query(message: str, history: list[ChatMessage]) -> str:
    if not needs_rewrite(message, history):
        return message
    convo = "\n".join(f"{m.role}: {m.content}" for m in history[-6:])
    return invoke_text(
        [
            SystemMessage(REWRITE_PROMPT),
            HumanMessage(f"Conversation:\n{convo}\n\nLatest message: {message}"),
        ],
        fast=True,
    ).strip() or message


def format_context(blocks: list[ContextBlock]) -> str:
    parts = []
    for i, b in enumerate(blocks, 1):
        header = f"[{i}] p. {b.page_label}"
        if b.section_path:
            header += f", {b.section_path}"
        header += f" ({b.content_type})"
        parts.append(f"{header}\n{b.text}")
    return "\n\n---\n\n".join(parts)


def _prepare(doc_id: str, strategy: str, message: str, history: list[ChatMessage]):
    query = rewrite_query(message, history)
    blocks = get_retriever(doc_id, strategy).retrieve(query)

    messages = [
        SystemMessage(
            SYSTEM_PROMPT.format(
                title=DOCUMENTS[doc_id]["title"],
                context=format_context(blocks) or "(no relevant context found)",
            )
        )
    ]
    for m in history:
        messages.append(HumanMessage(m.content) if m.role == "user" else AIMessage(m.content))
    messages.append(HumanMessage(message))

    sources = [
        Source(page=b.page, section_path=b.section_path, content_type=b.content_type, snippet=b.snippet)
        for b in blocks
    ]
    return messages, sources


def answer(doc_id: str, strategy: str, message: str, history: list[ChatMessage]) -> ChatResponse:
    messages, sources = _prepare(doc_id, strategy, message, history)
    return ChatResponse(answer=invoke_text(messages), sources=sources)


def answer_stream(doc_id: str, strategy: str, message: str, history: list[ChatMessage]) -> Iterator[dict]:
    """Yields {"type": "sources"} first, then {"type": "token"} chunks as Gemini generates."""
    messages, sources = _prepare(doc_id, strategy, message, history)
    yield {"type": "sources", "sources": [s.model_dump() for s in sources]}
    for chunk in get_chat_llm().stream(messages):
        text = message_text(chunk)
        if text:
            yield {"type": "token", "text": text}
