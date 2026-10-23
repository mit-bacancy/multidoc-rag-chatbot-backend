from functools import lru_cache

import tiktoken
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from app.config import EMBEDDING_MODEL


@lru_cache
def get_embeddings() -> HuggingFaceBgeEmbeddings:
    # Local model: no API quota or cost
    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count, used for chunk/context budgets."""
    return len(_encoder().encode(text))
