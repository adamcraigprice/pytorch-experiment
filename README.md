# AI Productivity Assistant

> **An AI-powered meeting productivity tool built with PyTorch, Transformer models (BERT & T5), FastAPI, and FAISS vector search.**

This project demonstrates how to build a production-style NLP pipeline that automatically **summarizes meetings**, **extracts action items**, **prioritizes tasks**, and provides **semantic search** over your meeting history — all powered by PyTorch and HuggingFace Transformers.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Technologies & Why They Were Chosen](#key-technologies--why-they-were-chosen)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Running the Application](#running-the-application)
6. [API Reference](#api-reference)
7. [Sample Usage (cURL)](#sample-usage-curl)
8. [How Each Module Works — Key Learnings](#how-each-module-works--key-learnings)
   - [Meeting Summarization (T5)](#1-meeting-summarization-t5)
   - [Action Item Extraction (BERT NER)](#2-action-item-extraction-bert-ner)
   - [Task Prioritization (BERT Embeddings)](#3-task-prioritization-bert-embeddings)
   - [Vector Search (FAISS)](#4-vector-search-faiss)
9. [PyTorch Concepts Demonstrated](#pytorch-concepts-demonstrated)
10. [Testing](#testing)
11. [Configuration](#configuration)
12. [Future Improvements](#future-improvements)

---

## Architecture Overview

```
┌───────────────────── FastAPI Server ─────────────────────┐
│                                                          │
│  POST /analyze  ──────────────────────────────────────┐  │
│                                                       │  │
│    ┌──────────────┐   ┌──────────────┐   ┌─────────┐ │  │
│    │ Summarizer   │   │ Action Item  │   │ Priori- │ │  │
│    │ (T5-small)   │   │ Extractor    │   │ tizer   │ │  │
│    │              │   │ (BERT NER)   │   │ (BERT)  │ │  │
│    └──────┬───────┘   └──────┬───────┘   └────┬────┘ │  │
│           │                  │                │      │  │
│           ▼                  ▼                ▼      │  │
│    ┌─────────────────────────────────────────────────┐│  │
│    │           JSON Response to Client               ││  │
│    └─────────────────────────────────────────────────┘│  │
│                                                       │  │
│  POST /index ──▶ BERT Embedding ──▶ FAISS Index       │  │
│  POST /search ──▶ BERT Embedding ──▶ FAISS kNN Search │  │
│                                                       │  │
└───────────────────────────────────────────────────────┘  │
                                                           │
                    ┌──────────────┐                       │
                    │  FAISS Index │  (persisted to disk)   │
                    │  + Metadata  │                        │
                    └──────────────┘                        │
```

**Data Flow for `/analyze`:**
1. Raw meeting transcript → **T5 Summarizer** → Abstractive summary
2. Raw meeting transcript → **Regex + BERT NER** → Structured action items (with assignee, deadline)
3. Action items → **BERT Embeddings + Urgency Scoring** → Prioritized task list
4. Transcript → **BERT Embedding → FAISS** → Indexed for future semantic search

---

## Key Technologies & Why They Were Chosen

| Technology | Role | Why |
|---|---|---|
| **PyTorch** | Deep learning framework | Industry-standard, dynamic computation graphs make debugging easy, strong GPU support |
| **T5 (Text-to-Text Transfer Transformer)** | Meeting summarization | Frames all NLP tasks as text-to-text, excellent at abstractive summarization |
| **BERT (Bidirectional Encoder Representations from Transformers)** | Embeddings, NER, classification | Pre-trained bidirectional context understanding, produces rich sentence embeddings |
| **HuggingFace Transformers** | Model hub & pipeline API | Unified API for downloading/running thousands of pre-trained models |
| **FAISS** | Vector similarity search | Facebook's library for efficient similarity search over dense vectors, scales to billions of vectors |
| **FastAPI** | Web framework | Async, auto-generated OpenAPI docs, Pydantic validation, production-ready |

---

## Project Structure

```
pytorch-experiment/
├── backend/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI application & route definitions
│   ├── config.py            # Centralized settings (env-overridable)
│   ├── models.py            # Pydantic request/response schemas
│   ├── summarizer.py        # T5-based meeting summarization
│   ├── action_items.py      # Regex + BERT NER action item extraction
│   ├── prioritizer.py       # BERT embedding-based task prioritization
│   ├── vector_search.py     # FAISS vector store with BERT embeddings
│   └── requirements.txt     # Python dependencies
├── tests/
│   ├── __init__.py
│   └── test_app.py          # Unit + integration tests
├── samples/
│   ├── sprint_planning.txt      # Sample meeting transcript
│   ├── roadmap_review.txt       # Sample meeting transcript
│   └── incident_postmortem.txt  # Sample meeting transcript
├── data/                    # Created at runtime (FAISS index persistence)
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **Git**
- ~2 GB disk space (for model weights, downloaded on first run)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/adamcraigprice/pytorch-experiment.git
cd pytorch-experiment
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r backend/requirements.txt
```

> **Note:** The first run will download model weights (~242 MB for T5-small, ~440 MB for BERT-base). This happens once and is cached by HuggingFace in `~/.cache/huggingface/`.

### Step 4 — (Optional) Create a `.env` File

```bash
# .env (all optional — defaults are sensible)
LOG_LEVEL=INFO
VECTOR_INDEX_DIR=./data/faiss_index
DEBUG=false
```

---

## Running the Application

### Start the Server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000**.

### Interactive API Docs

FastAPI auto-generates interactive documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## API Reference

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |

### Summarization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/summarize` | Summarize text (JSON body) |
| `POST` | `/summarize/upload` | Summarize text (file upload) |

### Action Items

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/action-items` | Extract action items (JSON body) |
| `POST` | `/action-items/upload` | Extract action items (file upload) |

### Task Prioritization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/prioritize` | Extract & prioritize tasks (JSON body) |
| `POST` | `/prioritize/upload` | Extract & prioritize tasks (file upload) |

### Vector Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search` | Semantic search over indexed documents |
| `POST` | `/index` | Index a single document |
| `POST` | `/index/batch` | Index multiple documents |
| `POST` | `/index/upload` | Index an uploaded file |
| `DELETE` | `/index` | Clear the entire index |

### Full Analysis Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Summarize + extract + prioritize (JSON) |
| `POST` | `/analyze/upload` | Full pipeline (file upload) |

---

## Sample Usage (cURL)

### Summarize a Meeting

```bash
# From JSON
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice: We need to finalize the Q1 budget. Bob will prepare the spreadsheet by Friday. Carol should review the vendor contracts. TODO: send the updated proposal to the client by Wednesday."}'

# From file
curl -X POST http://localhost:8000/summarize/upload \
  -F "file=@samples/sprint_planning.txt"
```

### Extract Action Items

```bash
curl -X POST http://localhost:8000/action-items \
  -H "Content-Type: application/json" \
  -d '{"text": "Action item: Alice will update the docs. TODO: Fix the CI pipeline. Bob should deploy to staging by Wednesday."}'
```

### Run Full Analysis Pipeline

```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F "file=@samples/sprint_planning.txt"
```

**Example Response:**
```json
{
  "summary": "the sprint goals for Q1 are reviewed. Bob has finished the authentication module. the CI pipeline has been failing intermittently since last week.",
  "action_items": [
    {
      "text": "Carol: Action item: Carol will update the API documentation for the new auth endpoints by Friday.",
      "assignee": "Carol",
      "deadline": "Friday",
      "source_line": 7,
      "confidence": 0.85,
      "extraction_method": "rule"
    },
    {
      "text": "Dave: TODO: Dave needs to fix the CI pipeline.",
      "assignee": "Dave",
      "deadline": null,
      "source_line": 9,
      "confidence": 0.85,
      "extraction_method": "rule"
    }
  ],
  "prioritized_tasks": [
    {
      "text": "Bob: The database migration is a blocker — we need to resolve it ASAP.",
      "priority_score": 0.7521,
      "priority_label": "high",
      "assignee": "Bob",
      "deadline": null
    }
  ]
}
```

### Index & Search

```bash
# Index a transcript
curl -X POST http://localhost:8000/index/upload \
  -F "file=@samples/incident_postmortem.txt"

# Semantic search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database outage and disk space", "k": 3}'
```

---

## How Each Module Works — Key Learnings

### 1. Meeting Summarization (T5)

**File:** `backend/summarizer.py`

**Model:** [T5-small](https://huggingface.co/t5-small) (60M parameters)

**How T5 Summarization Works:**

T5 (Text-to-Text Transfer Transformer) treats *every* NLP task as converting one text string into another. For summarization, the input is prefixed with `"summarize: "` and the model generates a condensed version.

```
Input:  "summarize: Alice said we need to fix the CI pipeline..."
Output: "the CI pipeline needs to be fixed according to Alice"
```

**Key Learning — Chunked Summarization:**

T5-small has a **512-token context window**. Meeting transcripts are often 1,000–10,000+ tokens. Naively truncating loses critical information.

**Our solution: Hierarchical chunking**
1. Split the transcript into overlapping 450-token chunks (50-token overlap preserves context at boundaries)
2. Summarize each chunk independently
3. Concatenate chunk summaries and run a **second summarization pass** to produce a coherent final summary

```python
# Overlap prevents losing context at chunk boundaries
#   Chunk 1: tokens[0:450]
#   Chunk 2: tokens[400:850]    ← 50-token overlap
#   Chunk 3: tokens[800:1250]
```

**Key Learning — Model Caching with `@lru_cache`:**

Loading transformer weights takes 2–3 seconds. Using `@lru_cache` ensures the model loads exactly once and is reused across all requests:

```python
@lru_cache(maxsize=1)
def _load_summarization_pipeline():
    return pipeline("summarization", model="t5-small")
```

---

### 2. Action Item Extraction (BERT NER)

**File:** `backend/action_items.py`

**Models:** Regex patterns + [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)

**Two-Stage Extraction Strategy:**

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | Regex patterns | Detect action-oriented language ("TODO:", "will", "should", "please", "assigned to") |
| 2 | BERT NER | Identify **person names** (PER entities) to populate the assignee field |

**Key Learning — Why Hybrid (Rules + ML)?**

Pure ML extraction requires labeled training data we don't have. Pure regex misses nuanced language. The hybrid approach gives us:
- **High recall** from broad regex patterns
- **Structured enrichment** from BERT NER (assignee extraction)
- **No training data required** — all models are pre-trained

**Key Learning — BERT NER Token Classification:**

BERT NER is a *token classification* task. Each input token receives a label: `B-PER` (beginning of person), `I-PER` (inside person), `O` (outside), etc.

```
Input tokens:  ["Alice", "will", "fix", "the", "bug"]
NER labels:    ["B-PER", "O",    "O",   "O",   "O"]
```

The `aggregation_strategy="simple"` parameter merges multi-token entities automatically:
```
"John Smith" → B-PER + I-PER → {"entity_group": "PER", "word": "John Smith"}
```

---

### 3. Task Prioritization (BERT Embeddings)

**File:** `backend/prioritizer.py`

**Model:** [bert-base-uncased](https://huggingface.co/bert-base-uncased) (110M parameters)

**Priority Scoring Formula:**

```
priority = 0.40 × keyword_score + 0.45 × semantic_score + 0.15 × length_score
```

| Signal | Weight | How It Works |
|--------|--------|-------------|
| **Keyword score** | 40% | Weighted lookup of urgency keywords ("ASAP"→0.30, "critical"→0.28, "deadline"→0.18, etc.) |
| **Semantic score** | 45% | Cosine similarity between the task embedding and pre-defined "urgent reference" sentence embeddings |
| **Length score** | 15% | Slight bonus for more detailed items (more context = easier to act on) |

**Key Learning — Mean Pooling vs. [CLS] Token:**

BERT outputs a hidden state for each input token. To get a single sentence embedding, you have two options:

```python
# Option A: [CLS] token (simpler but less accurate)
embedding = outputs.last_hidden_state[:, 0, :]

# Option B: Mean pooling (better for similarity tasks) ← we use this
mask = attention_mask.unsqueeze(-1)
embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
```

Mean pooling averages *all* token embeddings (weighted by the attention mask to ignore padding). This captures information from the entire sentence, not just the [CLS] summary.

**Key Learning — `torch.no_grad()` for Inference:**

During inference (no backpropagation), wrapping computation in `torch.no_grad()` disables gradient tracking, reducing memory usage by ~50% and speeding up computation:

```python
with torch.no_grad():
    outputs = model(**inputs)  # no gradient graph built
```

**Key Learning — Cosine Similarity for Semantic Comparison:**

Two sentences with similar *meaning* will have embedding vectors pointing in similar directions. `torch.nn.functional.cosine_similarity` measures this:

```python
# Returns value between -1 (opposite) and 1 (identical meaning)
similarity = F.cosine_similarity(task_embedding, urgency_embedding, dim=1)
```

---

### 4. Vector Search (FAISS)

**File:** `backend/vector_search.py`

**Technology:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)

**How Semantic Search Works:**

```
1. Document Ingestion:
   "Meeting about CI pipeline"  →  BERT  →  [0.12, -0.34, 0.56, ...]  →  FAISS Index
   "Discussion on Q1 budget"    →  BERT  →  [0.78, 0.22, -0.11, ...]  →  FAISS Index

2. Query Time:
   "deployment issues"          →  BERT  →  [0.10, -0.30, 0.52, ...]
                                                      ↓
                                              FAISS kNN Search
                                                      ↓
                                   Result: "Meeting about CI pipeline" (score: 0.87)
```

**Key Learning — L2 Normalization + Inner Product = Cosine Similarity:**

FAISS `IndexFlatIP` computes inner (dot) product. By L2-normalizing all vectors first, inner product becomes mathematically equivalent to cosine similarity:

```python
faiss.normalize_L2(embeddings)  # now ||v|| = 1 for all vectors
# Inner product of normalized vectors = cosine similarity
```

This is faster than computing cosine similarity directly and lets us use FAISS's optimized inner product kernels.

**Key Learning — Metadata Sidecar Pattern:**

FAISS indices only store float vectors — no text or metadata. We use a **sidecar JSON file** to store document metadata alongside the FAISS index:

```
data/faiss_index/
├── index.faiss      # Binary FAISS index (just vectors)
└── metadata.json    # Document text, source, type for each vector
```

The array index in FAISS corresponds to the array index in the metadata list.

**Key Learning — Scaling Considerations:**

| Documents | Recommended FAISS Index | Search Time |
|-----------|------------------------|-------------|
| < 10,000 | `IndexFlatIP` (exact) | < 1ms |
| 10K–1M | `IndexIVFFlat` (approximate) | < 10ms |
| 1M–1B | `IndexHNSW` or `IndexIVFPQ` | < 50ms |

Our implementation uses `IndexFlatIP` (exact brute-force search), which is ideal for the typical meeting transcript corpus size.

---

## PyTorch Concepts Demonstrated

This project showcases several core PyTorch and deep learning concepts:

| Concept | Where Used | What It Does |
|---------|-----------|-------------|
| **`torch.no_grad()`** | Prioritizer, Vector Search | Disables gradient computation during inference → 50% less memory |
| **`model.eval()`** | Prioritizer, Vector Search | Switches off dropout/batch-norm training behavior |
| **Mean pooling** | Prioritizer, Vector Search | Averages token embeddings into sentence embeddings using attention masks |
| **Cosine similarity** | Prioritizer | Measures semantic similarity between task and urgency reference embeddings |
| **Token classification** | Action Items (NER) | BERT predicts a label (B-PER, I-PER, O, etc.) for each input token |
| **Encoder-decoder generation** | Summarizer (T5) | Encoder processes input, decoder auto-regressively generates summary tokens |
| **Pipeline abstraction** | Summarizer, Action Items | HuggingFace `pipeline()` wraps tokenization + inference + post-processing |
| **Tensor operations** | Throughout | Batched matrix operations, masking, normalization |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test class
pytest tests/test_app.py::TestSummarizer -v

# Run with output
pytest tests/ -v -s
```

The test suite covers:
- **Unit tests** for each module (summarizer, action_items, prioritizer, vector_search)
- **Integration tests** for all FastAPI endpoints
- **Edge cases** (empty input, empty index, string vs. dict input)

> **Note:** Tests download model weights on first run (~700 MB total). Subsequent runs use the HuggingFace cache.

---

## Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `VECTOR_INDEX_DIR` | `./data/faiss_index` | Where FAISS index is persisted |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `DEBUG` | `false` | Enable debug mode |

---

## Future Improvements

- [ ] **GPU acceleration** — Detect CUDA/MPS and move models to GPU for 10-50x speedup
- [ ] **Pinecone integration** — Replace FAISS with managed Pinecone for cloud-native vector search
- [ ] **Fine-tuned models** — Fine-tune T5 on meeting-specific summarization datasets (e.g., AMI Corpus)
- [ ] **Streaming responses** — Use FastAPI's `StreamingResponse` for real-time summary generation
- [ ] **WebSocket support** — Live transcription → real-time action item extraction
- [ ] **Authentication** — JWT-based auth for multi-tenant deployments
- [ ] **Frontend** — React/Next.js dashboard for uploading transcripts and viewing results
- [ ] **Speaker diarization** — Integrate Whisper + pyannote for audio → attributed transcript
- [ ] **Custom NER fine-tuning** — Train on domain-specific entities (project names, internal tools)

---

## License

MIT
