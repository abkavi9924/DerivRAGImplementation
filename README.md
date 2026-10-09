# DerivRAGImplementation

This repository implements a deterministic mini-RAG pipeline for a small product knowledge base. It reads text files from the `kb/` directory, chunks them in code, retrieves the most relevant chunks for each question, generates grounded answers with citations, and evaluates retrieval quality with fixed rules.

## Design choices

- Document ingestion: reads each `kb/*.txt` file and parses `Title:` and `Section:` metadata.
- Chunking: deterministic sentence-based chunks are used by default; a fixed-size chunking strategy is also compared for analysis.
- Retrieval: local lexical ranking based on token overlap and TF-IDF-style weighting over chunk text.
- Answer generation: selects the best matching sentence from the retrieved chunk set and appends a citation in the format `[Doc Title §chunk_id]`.
- Validation: artifact checks ensure JSON validity, required outputs exist, answer labels use the controlled vocabulary, and citations match retrieved chunks.

This project does not use an external LLM. All logic runs locally and is fully reproducible from a clean checkout.

## Commands

```bash
python pipeline.py
python validate.py
```

The `pipeline.py` script regenerates the `artifacts/` outputs. The `validate.py` script checks that required artifacts exist and that the generated JSON is structurally consistent.
