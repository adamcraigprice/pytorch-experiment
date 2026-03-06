"""
Pydantic Models (Request / Response Schemas)
=============================================
Defines the data contracts between the FastAPI endpoints and callers.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Request Bodies
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    """Raw text input for summarization / extraction."""
    text: str = Field(..., min_length=1, description="Meeting transcript text")


class SearchQuery(BaseModel):
    """Semantic search query."""
    query: str = Field(..., min_length=1, description="Search query string")
    k: int = Field(5, ge=1, le=50, description="Number of results to return")


class DocumentInput(BaseModel):
    """Document to be indexed in the vector store."""
    text: str = Field(..., min_length=1, description="Document text")
    source: str = Field("", description="Source label (e.g. filename)")
    doc_type: str = Field("transcript", description="Document type")


class BatchDocumentInput(BaseModel):
    """Batch of documents to index."""
    documents: list[DocumentInput]


# ---------------------------------------------------------------------------
# Response Bodies
# ---------------------------------------------------------------------------

class SummaryResponse(BaseModel):
    summary: str


class ActionItemOut(BaseModel):
    text: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    source_line: int = 0
    confidence: float = 1.0
    extraction_method: str = "rule"


class ActionItemsResponse(BaseModel):
    action_items: list[ActionItemOut]
    count: int


class PrioritizedTaskOut(BaseModel):
    text: str
    priority_score: float
    priority_label: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None


class PrioritizedResponse(BaseModel):
    prioritized_tasks: list[PrioritizedTaskOut]
    count: int


class SearchResultOut(BaseModel):
    id: int
    text: str
    source: str
    doc_type: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultOut]
    count: int


class IndexResponse(BaseModel):
    indexed_ids: list[int]
    total_documents: int


class HealthResponse(BaseModel):
    status: str
    version: str
    total_indexed_documents: int


class FullAnalysisResponse(BaseModel):
    summary: str
    action_items: list[ActionItemOut]
    prioritized_tasks: list[PrioritizedTaskOut]
