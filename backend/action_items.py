"""
Action Item Extractor
=====================
Combines rule-based heuristics with a BERT-based Named-Entity-Recognition (NER)
model to extract actionable tasks from meeting transcripts.

Key PyTorch / Transformer Concepts Demonstrated:
- Using a pre-trained BERT token-classification model for NER
- Running inference with `torch.no_grad()` to save memory
- Combining ML confidence scores with deterministic rules
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class ActionItem:
    """Represents a single extracted action item."""
    text: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    source_line: int = 0
    confidence: float = 1.0
    extraction_method: str = "rule"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rule-Based Extraction
# ---------------------------------------------------------------------------
# Regex patterns capture common phrasing humans use to assign tasks during
# meetings: "Alice will …", "TODO: …", "Action item: …", "need to …", etc.
# ---------------------------------------------------------------------------

_ACTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?:action\s*item|todo|task|follow[- ]?up)\s*[:;-]?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\b\w+\b)\s+(?:will|should|needs? to|must|has to|is going to)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:please|pls)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:assigned to|owner)\s*[:;-]?\s*(\w+)\s*[:;-]?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:deadline|due|by)\s*[:;-]?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:make sure|ensure|remember to|don't forget to)\s+(.+)",
        re.IGNORECASE,
    ),
]


def _extract_rule_based(text: str) -> list[ActionItem]:
    """Extract action items using deterministic regex patterns."""
    items: list[ActionItem] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        for pattern in _ACTION_PATTERNS:
            match = pattern.search(line_stripped)
            if match:
                items.append(
                    ActionItem(
                        text=line_stripped,
                        source_line=line_no,
                        confidence=0.85,
                        extraction_method="rule",
                    )
                )
                break  # one match per line is enough
    return items


# ---------------------------------------------------------------------------
# NER-Based Extraction (BERT)
# ---------------------------------------------------------------------------
# We use a BERT NER model to identify person names (PER entities) inside
# action items so we can populate the *assignee* field automatically.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ner_pipeline():
    """Load a BERT-base NER pipeline (cached)."""
    logger.info("Loading BERT NER model...")
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        tokenizer="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )


def _extract_entities(text: str) -> list[dict]:
    """Run NER on *text* and return entity dicts."""
    ner = _load_ner_pipeline()
    return ner(text)


def _find_assignee(text: str) -> Optional[str]:
    """Return the first PER entity found in *text*, if any."""
    entities = _extract_entities(text)
    for ent in entities:
        if ent["entity_group"] == "PER":
            return ent["word"]
    return None


def _find_deadline(text: str) -> Optional[str]:
    """Heuristic: look for date-like phrases near deadline keywords."""
    pattern = re.compile(
        r"(?:by|before|due|deadline)\s*[:;]?\s*"
        r"((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|tomorrow|next week|end of (?:day|week|month|sprint)"
        r"|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2})"
        r"(?:\s*,?\s*\d{4})?)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_action_items(text: str) -> list[dict]:
    """Extract action items from a meeting transcript.

    Strategy
    --------
    1. Run rule-based extraction to capture candidate lines.
    2. Enrich each item with NER-based *assignee* detection.
    3. Attempt to extract *deadline* information.
    4. Deduplicate on normalized text.

    Returns a list of dicts suitable for JSON serialization.
    """
    if not text or not text.strip():
        return []

    raw_items = _extract_rule_based(text)

    # Enrich with NER
    enriched: list[ActionItem] = []
    seen_texts: set[str] = set()
    for item in raw_items:
        normalized = item.text.lower().strip()
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)

        item.assignee = _find_assignee(item.text)
        item.deadline = _find_deadline(item.text)
        enriched.append(item)

    logger.info("Extracted %d action item(s).", len(enriched))
    return [item.to_dict() for item in enriched]
