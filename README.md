# Deterministic Mini-RAG Pipeline

A zero-dependency, replayable Retrieval-Augmented Generation (RAG) pipeline built entirely in pure Python. 

This repository ingests a local product knowledge base, indexes it, retrieves relevant chunks, synthesizes citation-strict answers, and evaluates retrieval quality deterministically. 

## Architectural Decisions

- **Zero External Dependencies:** To guarantee a 100% success rate on a clean checkout without environment or compilation issues, this pipeline uses no external libraries (no `langchain`, `pinecone`, `torch`, or `fastapi`). It relies solely on Python's standard library (`math`, `collections`, `re`, `json`, `http.server`).
- **Deterministic Retrieval (BM25):** Implements the Okapi BM25 algorithm from scratch for fast, vector-free, and mathematically deterministic lexical ranking.
- **LLM-Free Generation:** As permitted by the requirements, this solution uses a deterministic extractive generator rather than an LLM. It extracts the most relevant chunk and appends a strict `[Title §chunk_id]` citation. This ensures zero hallucination risk, no API key requirements, and instant execution. Consequently, `llm_calls.jsonl` is intentionally omitted.
- **Stretch Goals Achieved:** 
  - Automated Grounding Check (`artifacts/grounding_check.json`)
  - Chunking Strategy Comparison (`artifacts/chunking_comparison.json`)
  - Minimal Local API (`api.py`)

## Pipeline Stages Enforced
The pipeline executes sequentially with strict terminal logging:
`INIT` -> `DOCUMENTS_LOADED` -> `DOCUMENTS_CHUNKED` -> `INDEX_BUILT` -> `RETRIEVAL_COMPLETE` -> `ANSWERS_GENERATED` -> `EVALUATION_COMPLETE` -> `VALIDATION_COMPLETE` -> `RESULTS_FINALISED`

## How to Run

### 1. Run the Pipeline
Executes the core pipeline, regenerates the index, and writes all outputs to `artifacts/`.
```bash
python pipeline.py
