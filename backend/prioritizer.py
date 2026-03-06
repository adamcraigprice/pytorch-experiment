"""
Task Prioritizer Module
=======================
Uses BERT sentence embeddings + keyword-based urgency scoring to rank
extracted action items by priority.

Key PyTorch / Transformer Concepts Demonstrated:
- Generating sentence-level embeddings with BERT (mean pooling)
- `torch.no_grad()` context manager for inference-only forward passes
- Cosine similarity via `torch.nn.functional.cosine_similarity`
- Combining neural similarity scores with hand-crafted feature weights
"""

import logging
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_bert():
    """Load and cache BERT tokenizer + model for embedding tasks."""
    logger.info("Loading BERT model for task prioritization...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def _embed_texts(texts: list[str]) -> torch.Tensor:
    """Compute BERT [CLS]-pooled embeddings for a list of texts.

    Mean-pooling over the last hidden state (with attention mask) produces
    richer sentence representations than using [CLS] alone.
    """
    tokenizer, model = _load_bert()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling: mask padded tokens before averaging
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
    hidden = outputs.last_hidden_state  # (B, T, H)
    summed = (hidden * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts  # (B, H)


# ---------------------------------------------------------------------------
# Urgency Scoring
# ---------------------------------------------------------------------------
# We compute urgency from three complementary signals:
#   1. **Keyword score** – presence of urgency keywords ("ASAP", "critical", …)
#   2. **Semantic similarity** – cosine similarity to a bank of high-urgency
#      reference sentences, computed with BERT embeddings.
#   3. **Length penalty** – very short items are often low-context; slightly
#      penalize them.
# ---------------------------------------------------------------------------

_URGENCY_KEYWORDS: dict[str, float] = {
    "asap": 0.30,
    "urgent": 0.30,
    "critical": 0.28,
    "immediately": 0.25,
    "blocker": 0.25,
    "high priority": 0.22,
    "deadline": 0.18,
    "important": 0.15,
    "soon": 0.12,
    "end of day": 0.20,
    "end of week": 0.15,
    "tomorrow": 0.18,
    "today": 0.22,
}

_URGENCY_REFERENCE_SENTENCES = [
    "This is an urgent and critical task that must be done immediately.",
    "High-priority blocker that needs resolution ASAP.",
    "Deadline is today, this cannot wait.",
]


def _keyword_score(text: str) -> float:
    """Return sum of matching urgency keyword weights (capped at 1.0)."""
    text_lower = text.lower()
    score = sum(
        weight for kw, weight in _URGENCY_KEYWORDS.items() if kw in text_lower
    )
    return min(score, 1.0)


def _semantic_urgency_score(text_embedding: torch.Tensor,
                            ref_embeddings: torch.Tensor) -> float:
    """Cosine similarity between the task embedding and urgency references."""
    similarities = F.cosine_similarity(
        text_embedding.unsqueeze(0), ref_embeddings, dim=1
    )
    return float(similarities.max().item())


def _length_score(text: str, max_len: int = 200) -> float:
    """Slight bonus for substantive items (more context → easier to act on)."""
    return min(len(text) / max_len, 1.0)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class PrioritizedTask:
    text: str
    priority_score: float
    priority_label: str  # "high" | "medium" | "low"
    assignee: Optional[str] = None
    deadline: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _label_from_score(score: float) -> str:
    if score >= 0.60:
        return "high"
    elif score >= 0.35:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prioritize_tasks(tasks: list[dict | str]) -> list[dict]:
    """Score and rank action items by priority.

    Parameters
    ----------
    tasks : list[dict | str]
        Action items – either plain strings or dicts with at least a ``text``
        key (as returned by `action_items.extract_action_items`).

    Returns
    -------
    list[dict]
        Tasks sorted from highest to lowest priority, each containing
        ``priority_score`` (0–1) and ``priority_label``.
    """
    if not tasks:
        return []

    # Normalize input
    normalized: list[dict] = []
    for task in tasks:
        if isinstance(task, str):
            normalized.append({"text": task})
        else:
            normalized.append(task)

    texts = [t["text"] for t in normalized]

    # Compute embeddings
    task_embeddings = _embed_texts(texts)
    ref_embeddings = _embed_texts(_URGENCY_REFERENCE_SENTENCES)

    scored: list[PrioritizedTask] = []
    for i, task_dict in enumerate(normalized):
        txt = task_dict["text"]
        kw = _keyword_score(txt)
        sem = _semantic_urgency_score(task_embeddings[i], ref_embeddings)
        ln = _length_score(txt)

        # Weighted combination
        score = 0.40 * kw + 0.45 * sem + 0.15 * ln
        score = round(float(score), 4)

        scored.append(
            PrioritizedTask(
                text=txt,
                priority_score=score,
                priority_label=_label_from_score(score),
                assignee=task_dict.get("assignee"),
                deadline=task_dict.get("deadline"),
            )
        )

    # Sort descending by score
    scored.sort(key=lambda t: t.priority_score, reverse=True)
    logger.info(
        "Prioritized %d task(s): %d high, %d medium, %d low.",
        len(scored),
        sum(1 for t in scored if t.priority_label == "high"),
        sum(1 for t in scored if t.priority_label == "medium"),
        sum(1 for t in scored if t.priority_label == "low"),
    )
    return [t.to_dict() for t in scored]
