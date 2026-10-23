import base64
from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GOOGLE_API_KEY, LLM_MODEL, VISION_MODEL


@lru_cache
def get_chat_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1, max_retries=3
    )


@lru_cache
def get_fast_llm() -> ChatGoogleGenerativeAI:
    """Minimal thinking for small utility calls (query rewrite): roughly halves their latency."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0, max_retries=3,
        thinking_level="minimal",
    )


@lru_cache
def get_vision_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=VISION_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0, max_retries=5
    )


def message_text(message: BaseMessage) -> str:
    """Gemini may return content as a string or a list of parts."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


def invoke_text(messages: list[BaseMessage], fast: bool = False) -> str:
    llm = get_fast_llm() if fast else get_chat_llm()
    return message_text(llm.invoke(messages))


def describe_image(png_bytes: bytes, prompt: str) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ]
    )
    return message_text(get_vision_llm().invoke([msg]))
