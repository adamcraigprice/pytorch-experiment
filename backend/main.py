"""
AI Productivity Assistant – FastAPI Application
================================================
REST API that ties together:
  • T5 meeting summarization
  • BERT NER action-item extraction
  • BERT embedding-based task prioritization
  • FAISS vector search over meeting transcripts

Run with:
    uvicorn backend.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models import (
    TextInput,
    SearchQuery,
    DocumentInput,
    BatchDocumentInput,
    SummaryResponse,
    ActionItemsResponse,
    PrioritizedResponse,
    SearchResponse,
    IndexResponse,
    HealthResponse,
    FullAnalysisResponse,
)
from backend.summarizer import summarize_text
from backend.action_items import extract_action_items
from backend.prioritizer import prioritize_tasks
from backend.vector_search import vector_db, search_similar

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy models at startup so the first request is fast."""
    logger.info("Starting %s …", settings.app_name)
    # Warm-up: importing the modules triggers @lru_cache model loads
    # You could also do explicit warm-up calls here
    yield
    logger.info("Shutting down %s.", settings.app_name)


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description=(
        "An AI productivity assistant powered by PyTorch and transformer "
        "models (BERT / T5).  Summarize meetings, extract action items, "
        "prioritize tasks, and perform semantic search over your meeting "
        "history."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# Health Check
# ===================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return service health and basic statistics."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_indexed_documents=vector_db.total_documents,
    )

# ===================================================================
# Summarization Endpoints
# ===================================================================

@app.post("/summarize", response_model=SummaryResponse, tags=["Summarization"])
async def summarize_from_json(body: TextInput):
    """Summarize a meeting transcript provided as JSON text."""
    summary = summarize_text(body.text)
    return SummaryResponse(summary=summary)


@app.post("/summarize/upload", response_model=SummaryResponse, tags=["Summarization"])
async def summarize_from_file(file: UploadFile = File(...)):
    """Summarize a meeting transcript uploaded as a text file."""
    text = (await file.read()).decode("utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    summary = summarize_text(text)
    return SummaryResponse(summary=summary)

# ===================================================================
# Action-Item Extraction Endpoints
# ===================================================================

@app.post("/action-items", response_model=ActionItemsResponse, tags=["Action Items"])
async def action_items_from_json(body: TextInput):
    """Extract action items from a meeting transcript (JSON body)."""
    items = extract_action_items(body.text)
    return ActionItemsResponse(action_items=items, count=len(items))


@app.post("/action-items/upload", response_model=ActionItemsResponse, tags=["Action Items"])
async def action_items_from_file(file: UploadFile = File(...)):
    """Extract action items from an uploaded text file."""
    text = (await file.read()).decode("utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    items = extract_action_items(text)
    return ActionItemsResponse(action_items=items, count=len(items))

# ===================================================================
# Task Prioritization Endpoints
# ===================================================================

@app.post("/prioritize", response_model=PrioritizedResponse, tags=["Prioritization"])
async def prioritize_from_json(body: TextInput):
    """Extract action items and prioritize them (JSON body)."""
    items = extract_action_items(body.text)
    priorities = prioritize_tasks(items)
    return PrioritizedResponse(prioritized_tasks=priorities, count=len(priorities))


@app.post("/prioritize/upload", response_model=PrioritizedResponse, tags=["Prioritization"])
async def prioritize_from_file(file: UploadFile = File(...)):
    """Extract and prioritize action items from an uploaded file."""
    text = (await file.read()).decode("utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    items = extract_action_items(text)
    priorities = prioritize_tasks(items)
    return PrioritizedResponse(prioritized_tasks=priorities, count=len(priorities))

# ===================================================================
# Vector Search Endpoints
# ===================================================================

@app.post("/search", response_model=SearchResponse, tags=["Vector Search"])
async def vector_search(body: SearchQuery):
    """Semantic search over indexed meeting transcripts."""
    results = search_similar(body.query, k=body.k)
    return SearchResponse(results=results, count=len(results))


@app.post("/index", response_model=IndexResponse, tags=["Vector Search"])
async def index_document(body: DocumentInput):
    """Add a single document to the vector store."""
    doc_id = vector_db.add_text(body.text, source=body.source, doc_type=body.doc_type)
    return IndexResponse(indexed_ids=[doc_id], total_documents=vector_db.total_documents)


@app.post("/index/batch", response_model=IndexResponse, tags=["Vector Search"])
async def index_batch(body: BatchDocumentInput):
    """Add multiple documents to the vector store in one call."""
    texts = [d.text for d in body.documents]
    source = body.documents[0].source if body.documents else ""
    doc_type = body.documents[0].doc_type if body.documents else "transcript"
    ids = vector_db.add_documents(texts, source=source, doc_type=doc_type)
    return IndexResponse(indexed_ids=ids, total_documents=vector_db.total_documents)


@app.post("/index/upload", response_model=IndexResponse, tags=["Vector Search"])
async def index_from_file(file: UploadFile = File(...)):
    """Index an uploaded text file into the vector store."""
    text = (await file.read()).decode("utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    doc_id = vector_db.add_text(text, source=file.filename or "upload")
    return IndexResponse(indexed_ids=[doc_id], total_documents=vector_db.total_documents)


@app.delete("/index", tags=["Vector Search"])
async def clear_index():
    """Clear all documents from the vector store."""
    vector_db.clear()
    return {"message": "Vector store cleared.", "total_documents": 0}

# ===================================================================
# Full Analysis (Pipeline)
# ===================================================================

@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Full Analysis"])
async def full_analysis(body: TextInput):
    """Run the full pipeline: summarize → extract action items → prioritize.

    This is the flagship endpoint that chains all three transformer-backed
    stages together in a single request.
    """
    summary = summarize_text(body.text)
    items = extract_action_items(body.text)
    priorities = prioritize_tasks(items)

    # Also index the transcript for future search
    vector_db.add_text(body.text, source="api-analyze", doc_type="transcript")

    return FullAnalysisResponse(
        summary=summary,
        action_items=items,
        prioritized_tasks=priorities,
    )


@app.post("/analyze/upload", response_model=FullAnalysisResponse, tags=["Full Analysis"])
async def full_analysis_upload(file: UploadFile = File(...)):
    """Full pipeline from an uploaded text file."""
    text = (await file.read()).decode("utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    summary = summarize_text(text)
    items = extract_action_items(text)
    priorities = prioritize_tasks(items)

    vector_db.add_text(text, source=file.filename or "upload", doc_type="transcript")

    return FullAnalysisResponse(
        summary=summary,
        action_items=items,
        prioritized_tasks=priorities,
    )
