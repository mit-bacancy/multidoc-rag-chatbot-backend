from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    doc_id: str
    message: str = Field(min_length=1)
    history: list[ChatMessage] = []
    strategy: str | None = None  # evaluation override, e.g. "basic"


class Source(BaseModel):
    page: int | None
    section_path: str | None
    content_type: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    pages: int | None
    ready: bool
