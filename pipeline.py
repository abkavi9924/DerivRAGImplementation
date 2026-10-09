"""Replayable, deterministic mini-RAG pipeline without external dependencies."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# CONTROLLED VOCABULARIES & SCHEMAS
# =============================================================================

ALLOWED_ANSWER_LABELS = {
    "grounded_answer",
    "insufficient_context",
    "conflicting_context",
}

ALLOWED_RETRIEVAL_STATUSES = {
    "hit",
    "partial_hit",
    "miss",
}

CITATION_PATTERN = re.compile(r"^\[(?P<title>.+?)\s+§(?P<chunk_id>chunk_\d+)\]$")

STAGES = [
    "INIT",
    "DOCUMENTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]


def log_stage(stage_name: str) -> None:
    """Logs pipeline execution stages with strict formatting."""
    print(f"-> {stage_name}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Document:
    filepath: Path
    title: str
    section: str
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_title: str
    section: str
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# BM25 SEARCH ENGINE
# =============================================================================


class BM25Index:
    """Deterministic Okapi BM25 index implemented with standard library math."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size > 0 else 0.0
        self.doc_freqs: List[Counter[str]] = [Counter(doc) for doc in corpus_tokens]
        self.idf: Dict[str, float] = {}

        df: Counter[str] = Counter()
        for frequencies in self.doc_freqs:
            for word in frequencies:
                df[word] += 1

        for word, freq in df.items():
            # Standard smoothed IDF calculation
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores: List[float] = []
        for i, frequencies in enumerate(self.doc_freqs):
            score = 0.0
            doc_length = self.doc_len[i]
            for token in query_tokens:
                if token not in frequencies:
                    continue
                freq = frequencies[token]
                numerator = self.idf[token] * freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_length / self.avgdl))
                score += numerator / denominator
            scores.append(score)
        return scores


def tokenize(text: str) -> List[str]:
    """Tokenize alphanumeric words in lowercase."""
    return re.findall(r"\w+", text.lower())


# =============================================================================
# PIPELINE STAGES
# =============================================================================


def load_documents(kb_dir: Path) -> List[Document]:
    """Reads and parses title/section metadata from knowledge base files."""
    docs: List[Document] = []
    for filepath in sorted(kb_dir.glob("*.txt")):
        content = filepath.read_text(encoding="utf-8")

        title_match = re.search(r"^Title:\s*(.+)$", content, flags=re.MULTILINE)
        section_match = re.search(r"^Section:\s*(.+)$", content, flags=re.MULTILINE)

        title = title_match.group(1).strip() if title_match else filepath.stem
        section = section_match.group(1).strip() if section_match else "General"

        # Strip header lines cleanly
        body = re.sub(r"^Title:.*?\n", "", content, flags=re.MULTILINE)
        body = re.sub(r"^Section:.*?\n", "", body, flags=re.MULTILINE).strip()

        docs.append(Document(filepath=filepath, title=title, section=section, body=body))

    log_stage("DOCUMENTS_LOADED")
    return docs


def chunk_documents(docs: List[Document], strategy: str = "line") -> List[Chunk]:
    """Chunks documents using line boundaries or fixed-character windows."""
    chunks: List[Chunk] = []
    chunk_idx = 1

    for doc in docs:
        if strategy == "line":
            current_offset = 0
            for line in doc.body.splitlines():
                stripped_line = line.strip()
                if not stripped_line:
                    current_offset += len(line) + 1
                    continue

                start = current_offset
                end = current_offset + len(stripped_line)

                chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{chunk_idx}",
                        doc_title=doc.title,
                        section=doc.section,
                        text=stripped_line,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_idx += 1
                current_offset += len(line) + 1

        elif strategy == "fixed":
            window = 140
            overlap = 20
            start = 0
            text = doc.body
            while start < len(text):
                end = min(start + window, len(text))
                slice_text = text[start:end].strip()
                if len(slice_text) > 15:
                    chunks.append(
                        Chunk(
                            chunk_id=f"chunk_{chunk_idx}",
                            doc_title=doc.title,
                            section=doc.section,
                            text=slice_text,
                            start_char=start,
                            end_char=end,
                        )
                    )
                    chunk_idx += 1
                start += window - overlap

    return chunks


def build_index(chunks: List[Chunk]) -> BM25Index:
    """Tokenizes chunk text and builds BM25 index."""
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    index = BM25Index(tokenized_corpus)
    log_stage("INDEX_BUILT")
    return index


def retrieve(
    index: BM25Index, chunks: List[Chunk], queries: List[Dict[str, Any]], top_k_count: int = 3
) -> List[Dict[str, Any]]:
    """Retrieves top-k chunks per query using BM25 ranking."""
    retrievals: List[Dict[str, Any]] = []

    for q in queries:
        q_tokens = tokenize(q["question"])
        scores = index.get_scores(q_tokens)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

        top_k: List[Dict[str, Any]] = []
        for rank, (score, chunk) in enumerate(ranked[:top_k_count], start=1):
            top_k.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "doc_title": chunk.doc_title,
                    "score": round(float(score), 4),
                    "chunk_text": chunk.text,
                }
            )

        retrievals.append(
            {
                "query_id": q["query_id"],
                "question": q["question"],
                "top_k": top_k,
            }
        )

    return retrievals


def generate_answers(retrievals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synthesizes citation-grounded answers based strictly on top-ranked chunks."""
    answers: List[Dict[str, Any]] = []

    for r in retrievals:
        top_result = r["top_k"][0]
        score = top_result["score"]

        # Deterministic boundary: score < 0.5 implies insufficient context
        if score < 0.5:
            answers.append(
                {
                    "query_id": r["query_id"],
                    "answer_label": "insufficient_context",
                    "answer": "Context provided is insufficient to answer the question.",
                    "citations": [],
                    "used_chunk_ids": [],
                }
            )
        else:
            citation = f"[{top_result['doc_title']} §{top_result['chunk_id']}]"
            answers.append(
                {
                    "query_id": r["query_id"],
                    "answer_label": "grounded_answer",
                    "answer": f"{top_result['chunk_text']} {citation}",
                    "citations": [citation],
                    "used_chunk_ids": [top_result["chunk_id"]],
                }
            )

    return answers


def evaluate_retrieval(
    retrievals: List[Dict[str, Any]], queries: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluates top-3 retrieval outcomes against ground truth titles."""
    query_lookup = {q["query_id"]: q for q in queries}
    evaluations: List[Dict[str, Any]] = []
    hits = partial_hits = misses = 0

    for r in retrievals:
        q_id = r["query_id"]
        expected_titles = query_lookup.get(q_id, {}).get("expected_doc_titles", [])
        retrieved_titles = [c["doc_title"] for c in r["top_k"]]

        expected = expected_titles[0] if expected_titles else ""

        if retrieved_titles and retrieved_titles[0] == expected:
            status = "hit"
            explanation = "Expected title found at rank 1"
            hits += 1
            matched = True
        elif expected in retrieved_titles:
            rank = retrieved_titles.index(expected) + 1
            status = "partial_hit"
            explanation = f"Expected title found at rank {rank}"
            partial_hits += 1
            matched = True
        else:
            status = "miss"
            explanation = "Expected title not found in top 3"
            misses += 1
            matched = False

        evaluations.append(
            {
                "query_id": q_id,
                "expected_doc_titles": expected_titles,
                "retrieved_doc_titles_top3": retrieved_titles,
                "retrieval_status": status,
                "matched_expected_title": matched,
                "explanation": explanation,
            }
        )

    total = len(queries)
    top3_rate = round((hits + partial_hits) / total, 4) if total > 0 else 0.0
    summary = {
        "top3_hit_rate": top3_rate,
        "total_queries": total,
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses,
    }

    return evaluations, summary


def run_grounding_check(
    answers: List[Dict[str, Any]], retrievals: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Checks whether citations reference retrieved chunks and maintain lexical grounding."""
    results: List[Dict[str, Any]] = []
    retrieval_map = {r["query_id"]: {c["chunk_id"]: c for c in r["top_k"]} for r in retrievals}

    for ans in answers:
        q_id = ans["query_id"]
        top_chunks = retrieval_map.get(q_id, {})
        grounded = True
        reasons: List[str] = []

        for cid in ans["used_chunk_ids"]:
            if cid not in top_chunks:
                grounded = False
                reasons.append(f"Citation violation: {cid} was not retrieved in top-k")
                continue

            chunk_text = top_chunks[cid]["chunk_text"]
            # Verify lexical support in raw chunk
            clean_ans = re.sub(r"\[.+? §chunk_\d+\]", "", ans["answer"]).strip()
            if clean_ans not in chunk_text and chunk_text not in clean_ans:
                grounded = False
                reasons.append(f"Lexical mismatch between answer and chunk text: {cid}")

        results.append(
            {
                "query_id": q_id,
                "is_grounded": grounded,
                "checked_chunk_ids": ans["used_chunk_ids"],
                "reasons": reasons,
            }
        )

    return results


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================


def run_pipeline() -> None:
    # Stage: INIT
    log_stage("INIT")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    kb_dir = Path("kb")
    queries_file = Path("queries.json")

    if not queries_file.exists():
        raise FileNotFoundError(f"Missing {queries_file}")

    queries: List[Dict[str, Any]] = json.loads(queries_file.read_text(encoding="utf-8"))

    # Stage: DOCUMENTS_LOADED
    docs = load_documents(kb_dir)

    # Stage: DOCUMENTS_CHUNKED
    chunks = chunk_documents(docs, strategy="line")
    save_json(artifacts_dir / "chunks.json", [c.to_dict() for c in chunks])
    log_stage("DOCUMENTS_CHUNKED")

    # Stage: INDEX_BUILT
    index = build_index(chunks)

    # Stage: RETRIEVAL_COMPLETE
    retrievals = retrieve(index, chunks, queries, top_k_count=3)
    save_json(artifacts_dir / "retrieval.json", retrievals)
    log_stage("RETRIEVAL_COMPLETE")

    # Stage: ANSWERS_GENERATED
    answers = generate_answers(retrievals)
    save_json(artifacts_dir / "answers.json", answers)
    log_stage("ANSWERS_GENERATED")

    # Stage: EVALUATION_COMPLETE
    eval_records, summary = evaluate_retrieval(retrievals, queries)
    save_json(
        artifacts_dir / "eval.json",
        {"summary": summary, "records": eval_records},
    )
    log_stage("EVALUATION_COMPLETE")

    # Stretch 1: Grounding Check
    grounding = run_grounding_check(answers, retrievals)
    save_json(artifacts_dir / "grounding_check.json", grounding)

    # Stretch 2: Chunking Strategy Comparison
    fixed_chunks = chunk_documents(docs, strategy="fixed")
    fixed_index = build_index(fixed_chunks)
    fixed_retrievals = retrieve(fixed_index, fixed_chunks, queries, top_k_count=3)
    _, fixed_summary = evaluate_retrieval(fixed_retrievals, queries)

    comparison = {
        "strategy_line": summary,
        "strategy_fixed": fixed_summary,
        "tradeoff_analysis": (
            "Line-based chunking preserves document semantic boundaries, "
            "whereas fixed character windows split sentences across boundaries, "
            "degrading exact term alignment for BM25."
        ),
    }
    save_json(artifacts_dir / "chunking_comparison.json", comparison)

    # Stage: VALIDATION_COMPLETE
    from validate import run_validations

    run_validations()
    log_stage("VALIDATION_COMPLETE")

    # Stage: RESULTS_FINALISED
    log_stage("RESULTS_FINALISED")


if __name__ == "__main__":
    run_pipeline()