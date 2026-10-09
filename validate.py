"""Deterministic validation suite for mini-RAG outputs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ALLOWED_ANSWER_LABELS: Set[str] = {
    "grounded_answer",
    "insufficient_context",
    "conflicting_context",
}

ALLOWED_RETRIEVAL_STATUSES: Set[str] = {
    "hit",
    "partial_hit",
    "miss",
}

CITATION_REGEX = re.compile(r"^\[.+?\s+§chunk_\d+\]$")


def validate_file_exists(filepath: Path) -> None:
    if not filepath.is_file():
        raise AssertionError(f"Required artifact missing: {filepath}")


def load_json_file(filepath: Path) -> Any:
    validate_file_exists(filepath)
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON file: {filepath} ({exc})") from exc


def run_validations() -> None:
    artifacts_dir = Path("artifacts")
    queries_path = Path("queries.json")

    # 1. Verify existence of required core artifacts
    required_artifacts = [
        artifacts_dir / "chunks.json",
        artifacts_dir / "retrieval.json",
        artifacts_dir / "answers.json",
        artifacts_dir / "eval.json",
    ]
    for path in required_artifacts:
        validate_file_exists(path)

    # 2. Load artifacts
    queries: List[Dict[str, Any]] = load_json_file(queries_path)
    chunks: List[Dict[str, Any]] = load_json_file(artifacts_dir / "chunks.json")
    retrieval: List[Dict[str, Any]] = load_json_file(artifacts_dir / "retrieval.json")
    answers: List[Dict[str, Any]] = load_json_file(artifacts_dir / "answers.json")
    eval_data: Dict[str, Any] = load_json_file(artifacts_dir / "eval.json")

    query_ids = [q["query_id"] for q in queries]
    total_query_count = len(query_ids)

    # 3. Validate Chunks
    if not chunks:
        raise AssertionError("chunks.json is empty")
    for chunk in chunks:
        for key in ("chunk_id", "doc_title", "section", "text", "start_char", "end_char"):
            if key not in chunk:
                raise AssertionError(f"Chunk missing required key '{key}': {chunk}")
        if not isinstance(chunk["start_char"], int) or not isinstance(chunk["end_char"], int):
            raise AssertionError(f"Chunk character offsets must be integers: {chunk['chunk_id']}")

    # 4. Validate Retrieval
    retrieval_q_ids = [r["query_id"] for r in retrieval]
    if set(retrieval_q_ids) != set(query_ids):
        raise AssertionError("Not all queries are present in retrieval.json")

    retrieval_map: Dict[str, Set[str]] = {}
    for r in retrieval:
        top_k = r.get("top_k", [])
        if len(top_k) < 3:
            raise AssertionError(f"Query {r['query_id']} has fewer than 3 retrieved chunks")

        retrieval_map[r["query_id"]] = set()
        for item in top_k:
            for field in ("rank", "chunk_id", "doc_title", "score", "chunk_text"):
                if field not in item:
                    raise AssertionError(f"Item in top_k missing '{field}': {item}")
            if not isinstance(item["score"], (int, float)):
                raise AssertionError(f"Retrieval score must be numeric: {item['score']}")
            retrieval_map[r["query_id"]].add(item["chunk_id"])

    # 5. Validate Answers & Citations
    answer_q_ids = [a["query_id"] for a in answers]
    if set(answer_q_ids) != set(query_ids):
        raise AssertionError("Not all queries are present in answers.json")

    for ans in answers:
        label = ans.get("answer_label")
        if label not in ALLOWED_ANSWER_LABELS:
            raise AssertionError(f"Invalid answer label '{label}' in query {ans['query_id']}")

        citations: List[str] = ans.get("citations", [])
        used_chunks: List[str] = ans.get("used_chunk_ids", [])

        if label == "grounded_answer":
            if not citations:
                raise AssertionError(f"Grounded answer lacks citations: {ans['query_id']}")
            if not used_chunks:
                raise AssertionError(f"Grounded answer lacks used_chunk_ids: {ans['query_id']}")

        # Ensure citations match required format and belong to retrieved top_k
        valid_retrieved_chunks = retrieval_map.get(ans["query_id"], set())
        for cit in citations:
            if not CITATION_REGEX.match(cit):
                raise AssertionError(f"Citation does not match required format '[Title §chunk_id]': {cit}")

        for chunk_id in used_chunks:
            if chunk_id not in valid_retrieved_chunks:
                raise AssertionError(
                    f"Cited chunk {chunk_id} was NOT among retrieved top_k for query {ans['query_id']}"
                )

    # 6. Validate Evaluation and Summary
    if "summary" not in eval_data or "records" not in eval_data:
        raise AssertionError("eval.json must contain both 'summary' and 'records'")

    summary = eval_data["summary"]
    required_summary_fields = ("top3_hit_rate", "total_queries", "hits", "partial_hits", "misses")
    for field in required_summary_fields:
        if field not in summary:
            raise AssertionError(f"eval.json summary missing field: {field}")

    if summary["total_queries"] != total_query_count:
        raise AssertionError(
            f"Summary total_queries ({summary['total_queries']}) does not match queries count ({total_query_count})"
        )

    calculated_total = summary["hits"] + summary["partial_hits"] + summary["misses"]
    if calculated_total != summary["total_queries"]:
        raise AssertionError("Summary hits + partial_hits + misses does not equal total_queries")

    for record in eval_data["records"]:
        status = record.get("retrieval_status")
        if status not in ALLOWED_RETRIEVAL_STATUSES:
            raise AssertionError(f"Invalid retrieval status '{status}' in record {record}")
        for key in ("query_id", "expected_doc_titles", "retrieved_doc_titles_top3", "matched_expected_title", "explanation"):
            if key not in record:
                raise AssertionError(f"Record missing key '{key}': {record}")

    print("All validation assertions passed successfully.")


if __name__ == "__main__":
    try:
        run_validations()
        sys.exit(0)
    except AssertionError as err:
        print(f"Validation FAILED: {err}", file=sys.stderr)
        sys.exit(1)