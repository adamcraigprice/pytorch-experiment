"""
Meeting Summarizer Module
=========================
Uses HuggingFace T5 (Text-to-Text Transfer Transformer) to generate abstractive
summaries of meeting transcripts. T5 frames every NLP task as a text-to-text
problem, making it ideal for summarization.

Key PyTorch / Transformer Concepts Demonstrated:
- Pipeline abstraction over tokenizer + model forward pass
- Handling long documents via chunked summarization
- Controlling generation with beam search parameters
"""

import logging
from functools import lru_cache
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Loading (cached singleton)
# ---------------------------------------------------------------------------
# Using @lru_cache ensures the model is loaded once and reused across requests.
# In production, this avoids the ~2-3 s overhead of reloading weights per call.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_summarization_pipeline():
    """Load and cache the T5-small summarization pipeline.

    Under the hood this:
      1. Downloads / loads T5-small weights (~242 MB)
      2. Wraps them in a GenerationPipeline that handles tokenization,
         forward pass through the encoder-decoder, and beam-search decoding.
    """
    device = 0 if torch.cuda.is_available() else -1
    logger.info("Loading T5-small summarization model (device=%s)...", device)
    return pipeline(
        "summarization",
        model="t5-small",
        tokenizer="t5-small",
        device=device,
    )


# ---------------------------------------------------------------------------
# Chunking Strategy
# ---------------------------------------------------------------------------
# T5-small has a 512-token context window.  Meeting transcripts are often much
# longer, so we split the text into overlapping chunks, summarize each one
# independently, and then (optionally) run a second pass to merge them.
# ---------------------------------------------------------------------------

_MAX_CHUNK_TOKENS = 450  # leave headroom for special tokens
_OVERLAP_TOKENS = 50     # overlap keeps context at boundaries


def _chunk_text(text: str, max_tokens: int = _MAX_CHUNK_TOKENS,
                overlap: int = _OVERLAP_TOKENS) -> list[str]:
    """Split *text* into token-level chunks with overlap."""
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        start += max_tokens - overlap

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_text(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """Summarize a meeting transcript.

    For short texts (<= 450 tokens) a single pass is used.
    For longer texts the document is chunked, each chunk summarized, and the
    chunk summaries are merged in a second summarization pass.

    Parameters
    ----------
    text : str
        Raw meeting transcript.
    max_length : int
        Maximum summary token length.
    min_length : int
        Minimum summary token length.

    Returns
    -------
    str
        The generated summary.
    """
    if not text or not text.strip():
        return ""

    pipe = _load_summarization_pipeline()
    chunks = _chunk_text(text)
    logger.info("Summarizing %d chunk(s)...", len(chunks))

    if len(chunks) == 1:
        result = pipe(
            f"summarize: {chunks[0]}",
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        return result[0]["summary_text"]

    # Multi-chunk: summarize each chunk, then merge summaries
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        result = pipe(
            f"summarize: {chunk}",
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        chunk_summaries.append(result[0]["summary_text"])
        logger.debug("Chunk %d/%d summarized.", i + 1, len(chunks))

    # Second-pass merge
    merged_input = " ".join(chunk_summaries)
    final = pipe(
        f"summarize: {merged_input}",
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return final[0]["summary_text"]
