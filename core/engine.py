import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config import CHROMA_COLLECTION_PREFIX, CHROMA_DB_PATH
from core.embedding import (
    build_embedding_split_debug,
    generate_embeddings,
    group_payloads_for_embedding,
    prepare_embedding_payloads,
)
from core.vector_db import ChromaVectorStore


@dataclass
class IndexingStats:
    source_path: str
    project_name: str
    passports_total: int
    payloads_total: int
    embedded_total: int
    upserted_total: int
    collection_name: str
    db_path: str


@dataclass
class SplitPreviewStats:
    source_path: str
    project_name: str
    passports_total: int
    payloads_total: int
    batches_total: int
    batches: List[Dict[str, Any]]


@dataclass
class PayloadPreviewStats:
    source_path: str
    project_name: str
    passports_total: int
    payloads_total: int
    batches_total: int
    batches: List[Dict[str, Any]]


def _derive_project_name(source_path: Path) -> str:
    stem = source_path.stem.lower().replace(" ", "_")
    return stem or "default_project"


def _build_collection_name(project_name: str) -> str:
    return f"{CHROMA_COLLECTION_PREFIX}_{project_name}"


def _load_passports(source_path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    passports = raw.get("passports", [])
    if not isinstance(passports, list):
        raise ValueError(f"Invalid passports format in {source_path}")
    return [item for item in passports if isinstance(item, dict)]


def index_passports_from_file(source_path: Path) -> IndexingStats:
    if not source_path.exists():
        raise FileNotFoundError(f"Batch file not found: {source_path}")

    passports = _load_passports(source_path=source_path)
    project_name = _derive_project_name(source_path=source_path)
    collection_name = _build_collection_name(project_name=project_name)
    vector_store = ChromaVectorStore(
        db_path=CHROMA_DB_PATH,
        collection_name=collection_name,
    )

    payloads = prepare_embedding_payloads(
        passports=passports,
        project_name=project_name,
    )
    embedded = generate_embeddings(payloads=payloads)

    upsert_rows = [
        (payload.doc_id, payload.text, payload.metadata, vector)
        for payload, vector in embedded
    ]
    upserted_total = vector_store.upsert_embeddings(rows=upsert_rows)

    return IndexingStats(
        source_path=str(source_path),
        project_name=project_name,
        passports_total=len(passports),
        payloads_total=len(payloads),
        embedded_total=len(embedded),
        upserted_total=upserted_total,
        collection_name=collection_name,
        db_path=str(CHROMA_DB_PATH),
    )


def preview_embedding_split_from_file(source_path: Path) -> SplitPreviewStats:
    if not source_path.exists():
        raise FileNotFoundError(f"Batch file not found: {source_path}")

    passports = _load_passports(source_path=source_path)
    project_name = _derive_project_name(source_path=source_path)
    payloads = prepare_embedding_payloads(
        passports=passports,
        project_name=project_name,
    )
    debug_rows = build_embedding_split_debug(payloads=payloads)
    batch_rows = [
        {
            "batch_index": row.batch_index,
            "items_count": row.items_count,
            "chars_count": row.chars_count,
            "paths": row.paths,
        }
        for row in debug_rows
    ]
    return SplitPreviewStats(
        source_path=str(source_path),
        project_name=project_name,
        passports_total=len(passports),
        payloads_total=len(payloads),
        batches_total=len(batch_rows),
        batches=batch_rows,
    )


def preview_embedding_payloads_from_file(source_path: Path) -> PayloadPreviewStats:
    if not source_path.exists():
        raise FileNotFoundError(f"Batch file not found: {source_path}")

    passports = _load_passports(source_path=source_path)
    project_name = _derive_project_name(source_path=source_path)
    payloads = prepare_embedding_payloads(
        passports=passports,
        project_name=project_name,
    )
    payload_batches = group_payloads_for_embedding(payloads=payloads)
    batch_rows: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(payload_batches, start=1):
        batch_rows.append(
            {
                "batch_index": batch_index,
                "items_count": len(batch),
                "chars_count": sum(len(item.text) for item in batch),
                "items": [
                    {
                        "doc_id": item.doc_id,
                        "path": str(item.metadata.get("path", "")),
                        "text": item.text,
                    }
                    for item in batch
                ],
            }
        )

    return PayloadPreviewStats(
        source_path=str(source_path),
        project_name=project_name,
        passports_total=len(passports),
        payloads_total=len(payloads),
        batches_total=len(batch_rows),
        batches=batch_rows,
    )


def index_passports(
    passports: Sequence[Dict[str, Any]],
    project_name: str,
) -> IndexingStats:
    collection_name = _build_collection_name(project_name=project_name)
    vector_store = ChromaVectorStore(
        db_path=CHROMA_DB_PATH,
        collection_name=collection_name,
    )

    payloads = prepare_embedding_payloads(
        passports=passports,
        project_name=project_name,
    )
    embedded = generate_embeddings(payloads=payloads)
    upsert_rows = [
        (payload.doc_id, payload.text, payload.metadata, vector)
        for payload, vector in embedded
    ]
    upserted_total = vector_store.upsert_embeddings(rows=upsert_rows)

    return IndexingStats(
        source_path="in_memory",
        project_name=project_name,
        passports_total=len(passports),
        payloads_total=len(payloads),
        embedded_total=len(embedded),
        upserted_total=upserted_total,
        collection_name=collection_name,
        db_path=str(CHROMA_DB_PATH),
    )
