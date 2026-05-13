import hashlib
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import (
    CHROMA_COLLECTION_PREFIX,
    CHROMA_DB_PATH,
    SUMMARY_MAX_WORKERS,
)
from core.embedding import (
    generate_embeddings,
    iter_payload_chunks,
    prepare_embedding_payloads,
)
from core.processor import create_smart_batches, enrich_code_batch
from core.vector_db import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class CachedPassport:
    passport: Dict[str, Any]
    content_hash: str


@dataclass
class SummaryStageStats:
    project_name: str
    passports_total: int
    cached_before_run: int
    generated_now: int
    failed_files: int
    rejected_passports: int


@dataclass
class EmbeddingStageStats:
    payloads_total: int
    already_indexed: int
    embedded_now: int
    upserted_now: int


def _passport_file_name(file_path: str) -> str:
    digest = hashlib.sha1(file_path.encode("utf-8")).hexdigest()
    return f"{digest}.json"


def load_cached_passports(passports_dir: Path) -> Dict[str, CachedPassport]:
    """Loads cached passports from the directory."""
    result: Dict[str, CachedPassport] = {}
    if not passports_dir.exists():
        return result

    for file_path in sorted(passports_dir.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"Skip broken passport file {file_path}: {error}")
            continue

        if not isinstance(payload, dict):
            continue
        source_path = payload.get("path")
        passport = payload.get("passport")
        if not isinstance(source_path, str) or not source_path.strip():
            continue
        if not isinstance(passport, dict):
            continue

        raw_hash = payload.get("content_hash", "")
        result[source_path] = CachedPassport(
            passport=passport,
            content_hash=raw_hash if isinstance(raw_hash, str) else "",
        )

    return result


def _save_passport(
    passports_dir: Path,
    passport: Dict[str, Any],
    content_hash: str = "",
) -> bool:
    source_path = passport.get("path")
    if not isinstance(source_path, str) or not source_path.strip():
        return False

    output_path = passports_dir / _passport_file_name(source_path)
    output_payload = {
        "path": source_path,
        "content_hash": content_hash,
        "passport": passport,
    }
    serialized = json.dumps(output_payload, ensure_ascii=False, indent=2)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=passports_dir,
            prefix=f"{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(serialized)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)
        os.replace(str(temp_path), str(output_path))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return True


def print_batch_preview(batches: List[List[Dict[str, str]]], limit: int) -> None:
    selected = batches if limit == 0 else batches[:limit]
    print(f"Created {len(batches)} total batches, selected {len(selected)} for run")
    for index, batch in enumerate(selected, 1):
        total_chars = sum(len(item.get("content", "")) for item in batch)
        print(f"Batch #{index}, files: {len(batch)}, chars: {total_chars}")
        for item in batch:
            print(f"  - {item['file_path']}")
        print("-" * 100)


def run_summary_stage(
    project_data: List[Dict[str, str]],
    passports_dir: Path,
    batch_limit: int,
    project_name: str,
    max_workers: int = SUMMARY_MAX_WORKERS,
) -> SummaryStageStats:
    """Generates summaries (Technical Passports) for project files through Gemini."""
    cached_passports = load_cached_passports(passports_dir=passports_dir)
    initial_cached_count = len(cached_passports)

    # Mapping file_path -> content_hash for comparison with cache and writing when saving.
    hash_by_path: Dict[str, str] = {
        item["file_path"]: item.get("content_hash", "")
        for item in project_data
    }

    # Backfill: for passports in old format (without content_hash) write the actual hash
    # to disk without calling AI, so that the comparison works correctly when running again.
    passports_to_backfill = [
        (path, entry)
        for path, entry in cached_passports.items()
        if not entry.content_hash and hash_by_path.get(path)
    ]
    if passports_to_backfill:
        print(f"[summary] backfilling content_hash for {len(passports_to_backfill)} old passports")
        for path, entry in passports_to_backfill:
            new_hash = hash_by_path[path]
            _save_passport(passports_dir, entry.passport, content_hash=new_hash)
            cached_passports[path] = CachedPassport(
                passport=entry.passport, content_hash=new_hash,
            )

    # Filtering: skip files that have a passport and content_hash matches.
    # If file has changed (hash doesn't match) — passport will be recreated.
    files_to_process: List[Dict[str, str]] = []
    for file_info in project_data:
        path = file_info.get("file_path", "")
        cached = cached_passports.get(path)
        if cached is None:
            files_to_process.append(file_info)
        elif cached.content_hash != file_info.get("content_hash", ""):
            files_to_process.append(file_info)

    generated_now = 0
    failed_files = 0
    rejected_passports = 0
    cached_valid = len(project_data) - len(files_to_process)
    print(
        f"[summary] project={project_name}, files_total={len(project_data)}, "
        f"cached_valid={cached_valid}, pending={len(files_to_process)}"
    )

    if not files_to_process:
        return SummaryStageStats(
            project_name=project_name,
            passports_total=len(cached_passports),
            cached_before_run=initial_cached_count,
            generated_now=0,
            failed_files=0,
            rejected_passports=0,
        )

    summary_batches = create_smart_batches(files_to_process)
    selected_batches = summary_batches if batch_limit == 0 else summary_batches[:batch_limit]
    print(
        f"[summary] batches_total={len(summary_batches)}, "
        f"selected={len(selected_batches)}, workers={max_workers}"
    )

    lock = threading.Lock()

    def _process_batch(
        batch_index: int, batch: List[Dict[str, str]],
    ) -> Tuple[int, int, int, int]:
        passports, rejected = enrich_code_batch(batch)
        saved_in_batch = 0

        for passport in passports:
            path = passport.get("path", "")
            content_hash = hash_by_path.get(path, "")
            if _save_passport(
                passports_dir=passports_dir,
                passport=passport,
                content_hash=content_hash,
            ):
                with lock:
                    cached_passports[path] = CachedPassport(
                        passport=passport, content_hash=content_hash,
                    )
                saved_in_batch += 1

        with lock:
            unresolved = [
                item["file_path"]
                for item in batch
                if item["file_path"] not in cached_passports
            ]

        return saved_in_batch, len(unresolved), batch_index, rejected

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_batch, idx, batch): idx
            for idx, batch in enumerate(selected_batches, start=1)
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                saved, unresolved_count, _, batch_rejected = future.result()
            except Exception as error:
                logger.exception("[summary] batch=%s failed: %s", batch_idx, error)
                saved = 0
                unresolved_count = len(selected_batches[batch_idx - 1])
                batch_rejected = 0

            generated_now += saved
            failed_files += unresolved_count
            rejected_passports += batch_rejected
            print(
                f"[summary] batch={batch_idx}/{len(selected_batches)}, "
                f"saved={saved}, unresolved={unresolved_count}, "
                f"rejected={batch_rejected}"
            )

    total_attempted = generated_now + failed_files + rejected_passports
    if total_attempted > 0:
        reject_pct = (rejected_passports / total_attempted) * 100
        print(
            f"[summary] quality report: generated={generated_now}, "
            f"rejected={rejected_passports} ({reject_pct:.1f}%), "
            f"failed={failed_files}, total_attempted={total_attempted}"
        )

    return SummaryStageStats(
        project_name=project_name,
        passports_total=len(cached_passports),
        cached_before_run=initial_cached_count,
        generated_now=generated_now,
        failed_files=failed_files,
        rejected_passports=rejected_passports,
    )


def run_embedding_stage(
    passports_dir: Path,
    project_name: str,
    embedding_chunk_size: int,
) -> EmbeddingStageStats:
    """Reads passports from disk, creates embeddings and writes to ChromaDB.

    Incremental logic: compare content_hash in payload.metadata with what is stored in Chroma.
    Reindex only new and changed records.
    """
    if embedding_chunk_size <= 0:
        raise ValueError("embedding_chunk_size must be > 0")

    cached_entries = load_cached_passports(passports_dir=passports_dir)
    passports = [entry.passport for entry in cached_entries.values()]

    # Mapping path -> content_hash for passing into metadata of each payload.
    hash_by_path: Dict[str, str] = {
        path: entry.content_hash
        for path, entry in cached_entries.items()
    }

    payloads = prepare_embedding_payloads(
        passports=passports,
        project_name=project_name,
        hash_by_path=hash_by_path,
    )

    collection_name = f"{CHROMA_COLLECTION_PREFIX}_{project_name}"
    vector_store = ChromaVectorStore(
        db_path=CHROMA_DB_PATH,
        collection_name=collection_name,
    )

    # Get existing records together with metadata for comparing content_hash.
    doc_ids = [payload.doc_id for payload in payloads]
    existing_meta = vector_store.get_ids_with_metadata(ids=doc_ids)

    pending_payloads = []
    already_indexed = 0
    for payload in payloads:
        stored = existing_meta.get(payload.doc_id)
        if stored is None:
            pending_payloads.append(payload)
        elif stored.get("content_hash", "") != payload.metadata.get("content_hash", ""):
            pending_payloads.append(payload)
        else:
            already_indexed += 1

    print(
        f"[embedding] payloads_total={len(payloads)}, unchanged={already_indexed}, "
        f"pending={len(pending_payloads)}"
    )

    embedded_now = 0
    upserted_now = 0
    payload_chunks = list(
        iter_payload_chunks(
            payloads=pending_payloads,
            chunk_size=embedding_chunk_size,
        )
    )
    for chunk_index, payload_chunk in enumerate(payload_chunks, start=1):
        embedded_chunk = generate_embeddings(payloads=payload_chunk)
        upsert_rows = [
            (payload.doc_id, payload.text, payload.metadata, vector)
            for payload, vector in embedded_chunk
        ]
        chunk_upserted = vector_store.upsert_embeddings(rows=upsert_rows)
        embedded_now += len(embedded_chunk)
        upserted_now += chunk_upserted
        print(
            f"[embedding] chunk={chunk_index}/{len(payload_chunks)}, "
            f"embedded={len(embedded_chunk)}, upserted={chunk_upserted}"
        )

    return EmbeddingStageStats(
        payloads_total=len(payloads),
        already_indexed=already_indexed,
        embedded_now=embedded_now,
        upserted_now=upserted_now,
    )
