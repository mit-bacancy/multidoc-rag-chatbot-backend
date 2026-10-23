import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import store
from app.config import DOCUMENTS, FRONTEND_ORIGIN, STRATEGIES
from app.generation import answer, answer_stream
from app.retrieval import load_retrievers
from app.schemas import ChatRequest, ChatResponse, DocumentInfo

logger = logging.getLogger("uvicorn.error")

# (doc_id, strategy) -> latest ingest run; refreshed at startup
READY: dict[tuple[str, str], dict] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        READY.update(store.latest_runs())
        load_retrievers([k for k in READY if k[0] in DOCUMENTS])
    except Exception:
        logger.exception("Startup: could not load ingested documents")
    yield


app = FastAPI(title="Multi-Document RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    db_ok = store.ping()
    return {"status": "ok" if db_ok else "degraded", "db": db_ok}


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    out = []
    for doc_id, doc in DOCUMENTS.items():
        run = READY.get((doc_id, doc["strategy"]))
        out.append(
            DocumentInfo(
                doc_id=doc_id,
                title=doc["title"],
                pages=run["page_count"] if run else None,
                ready=run is not None,
            )
        )
    return out


def _resolve_strategy(req: ChatRequest) -> str:
    doc = DOCUMENTS.get(req.doc_id)
    if not doc:
        raise HTTPException(400, f"Unknown doc_id '{req.doc_id}'")
    strategy = req.strategy or doc["strategy"]
    if strategy not in STRATEGIES:
        raise HTTPException(400, f"Unknown strategy '{strategy}'")
    if (req.doc_id, strategy) not in READY:
        raise HTTPException(400, f"Document '{req.doc_id}' is not ready ({strategy} not ingested)")
    return strategy


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    strategy = _resolve_strategy(req)
    return await run_in_threadpool(answer, req.doc_id, strategy, req.message, req.history)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-sent events: one `sources` event, then `token` events, then `done` (or `error`)."""
    strategy = _resolve_strategy(req)

    def events():
        try:
            for event in answer_stream(req.doc_id, strategy, req.message, req.history):
                yield f"data: {json.dumps(event)}\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception:
            logger.exception("chat stream failed")
            yield 'data: {"type": "error", "message": "Generation failed. Please try again."}\n\n'

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
