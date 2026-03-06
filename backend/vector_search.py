"""
Vector Search Module (FAISS)
============================
Implements semantic search over meeting transcripts using BERT embeddings
stored in a FAISS (Facebook AI Similarity Search) index.

Key PyTorch / ML Concepts Demonstrated:
- BERT mean-pooled embeddings for dense retrieval
- FAISS IndexFlatIP (inner-product / cosine-similarity) for exact kNN
- L2-normalization of vectors so inner-product == cosine similarity
- Persistent index serialization to disk with `faiss.write_index`
- Thread-safe singleton pattern for the in-memory index
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_INDEX_DIR = Path(os.getenv("VECTOR_INDEX_DIR", "./data/faiss_index"))
_INDEX_FILE = _INDEX_DIR / "index.faiss"
_META_FILE = _INDEX_DIR / "metadata.json"
_EMBEDDING_DIM = 768  # BERT-base hidden size


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_bert():
    """Load and cache BERT tokenizer + model for embeddings."""
    logger.info("Loading BERT model for vector search...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def _embed(texts: list[str]) -> np.ndarray:
    """Compute L2-normalized BERT mean-pooled embeddings.

    L2 normalization ensures that FAISS inner-product search is equivalent
    to cosine similarity, which is the standard metric for semantic search.
    """
    tokenizer, model = _load_bert()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    hidden = outputs.last_hidden_state
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = (summed / counts).numpy().astype("float32")

    # L2 normalize → inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# Document Store
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Metadata stored alongside each indexed vector."""
    id: int
    text: str
    source: str = ""       # e.g. filename or meeting title
    doc_type: str = "transcript"  # transcript | summary | action_item

    def to_dict(self) -> dict:
        return asdict(self)


class VectorStore:
    """Thread-safe FAISS-backed vector store with persistence.

    Architecture Notes
    ------------------
    * The FAISS index lives in memory for fast search.
    * Metadata (text, source, type) is stored separately in a JSON sidecar
      because FAISS indices only hold float vectors.
    * `IndexFlatIP` performs exact (brute-force) inner-product search.  For
      datasets > ~100 k vectors, switch to `IndexIVFFlat` or `IndexHNSW`
      for approximate (but much faster) search.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._documents: list[Document] = []
        self._index: faiss.Index = faiss.IndexFlatIP(_EMBEDDING_DIM)
        self._next_id: int = 0
        self._load_from_disk()

    # -- Persistence --------------------------------------------------------

    def _load_from_disk(self):
        """Restore index + metadata from disk if available."""
        if _INDEX_FILE.exists() and _META_FILE.exists():
            logger.info("Loading persisted FAISS index from %s", _INDEX_DIR)
            self._index = faiss.read_index(str(_INDEX_FILE))
            with open(_META_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._documents = [Document(**d) for d in raw]
            self._next_id = max((d.id for d in self._documents), default=-1) + 1
            logger.info("Loaded %d documents.", len(self._documents))

    def _save_to_disk(self):
        """Persist the current index + metadata."""
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(_INDEX_FILE))
        with open(_META_FILE, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in self._documents], f, indent=2)

    # -- Write Operations ---------------------------------------------------

    def add_documents(self, texts: list[str], source: str = "",
                      doc_type: str = "transcript") -> list[int]:
        """Embed and index a batch of text documents.

        Returns the list of assigned document IDs.
        """
        if not texts:
            return []

        embeddings = _embed(texts)  # (N, 768) float32, L2-normed

        with self._lock:
            ids: list[int] = []
            for i, text in enumerate(texts):
                doc = Document(
                    id=self._next_id,
                    text=text,
                    source=source,
                    doc_type=doc_type,
                )
                self._documents.append(doc)
                ids.append(self._next_id)
                self._next_id += 1

            self._index.add(embeddings)
            self._save_to_disk()

        logger.info("Indexed %d new document(s) (total: %d).", len(texts), self._index.ntotal)
        return ids

    def add_text(self, text: str, source: str = "",
                 doc_type: str = "transcript") -> int:
        """Convenience wrapper to add a single document."""
        return self.add_documents([text], source=source, doc_type=doc_type)[0]

    # -- Read Operations ----------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search: return the *k* most similar documents.

        Returns a list of dicts with ``id``, ``text``, ``source``,
        ``doc_type``, and ``score`` (cosine similarity, 0–1).
        """
        if self._index.ntotal == 0:
            logger.warning("Search on empty index — returning empty results.")
            return []

        q_emb = _embed([query])  # (1, 768)
        actual_k = min(k, self._index.ntotal)

        with self._lock:
            scores, indices = self._index.search(q_emb, actual_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx]
            results.append({
                **doc.to_dict(),
                "score": round(float(score), 4),
            })

        return results

    # -- Utilities ----------------------------------------------------------

    @property
    def total_documents(self) -> int:
        return self._index.ntotal

    def clear(self):
        """Remove all documents and reset the index."""
        with self._lock:
            self._index = faiss.IndexFlatIP(_EMBEDDING_DIM)
            self._documents.clear()
            self._next_id = 0
            self._save_to_disk()
        logger.info("Vector store cleared.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

vector_db = VectorStore()


def search_similar(query: str, k: int = 5) -> list[dict]:
    """Public helper used by the FastAPI router."""
    return vector_db.search(query, k=k)
