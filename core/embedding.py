import json
import os
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from google import genai

from config import (
    EMBEDDING_REQUESTS_PER_MINUTE,
    EMBEDDING_MODEL,
    EMBEDDING_MAX_BATCH_SIZE,
    EMBEDDING_MAX_BATCH_CHARS,
    EMBEDDING_MAX_TEXT_CHARS,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_BASE_DELAY_SEC,
    GEMINI_API_KEY,
)
from core.rate_limiter import get_or_create_limiter

logger = logging.getLogger(__name__)
_gemini_client: Optional[Any] = None
_gemini_rate_limiter = get_or_create_limiter(
    name="gemini-embed",
    max_requests=EMBEDDING_REQUESTS_PER_MINUTE,
)


@dataclass(frozen=True)
class EmbeddingPayload:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EmbeddingBatchDebugInfo:
    batch_index: int
    items_count: int
    chars_count: int
    paths: List[str]


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Keep this marker to make truncation explicit inside embeddings input.
    return f"{text[:max_chars]} ...[truncated]"


def _build_doc_id(project_name: str, file_path: str) -> str:
    raw = f"{project_name}:{file_path}"
    hash_value = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"{project_name}:{hash_value}"


def _build_embedding_text(passport: Dict[str, Any], max_text_chars: int) -> str:
    path = _normalize_text(passport.get("path"))
    domain = _normalize_text(passport.get("domain"))
    summary = _normalize_text(passport.get("summary"))
    business_rules = passport.get("business_rules", [])
    interactions = passport.get("interactions", [])

    lines: List[str] = [f"path: {path}"]
    if domain:
        lines.append(f"domain: {domain}")
    if summary:
        lines.append(f"summary: {summary}")

    if isinstance(business_rules, list) and business_rules:
        lines.append("business_rules:")
        for rule in business_rules:
            rule_text = _normalize_text(rule)
            if rule_text:
                lines.append(f"- {rule_text}")

    if isinstance(interactions, list) and interactions:
        lines.append("interactions:")
        for item in interactions:
            item_text = _normalize_text(item)
            if item_text:
                lines.append(f"- {item_text}")

    # Intentionally skip full_content to keep embeddings compact and cheap.
    text = "\n".join(lines)
    return _truncate(text, max_chars=max_text_chars)


def prepare_embedding_payloads(
    passports: Sequence[Dict[str, Any]],
    project_name: str,
    max_text_chars: int = EMBEDDING_MAX_TEXT_CHARS,
    hash_by_path: Optional[Dict[str, str]] = None,
) -> List[EmbeddingPayload]:
    payloads: List[EmbeddingPayload] = []
    for passport in passports:
        path = _normalize_text(passport.get("path"))
        if not path:
            continue

        text = _build_embedding_text(passport=passport, max_text_chars=max_text_chars)
        if not text:
            continue

        metadata: Dict[str, Any] = {
            "project": project_name,
            "path": path,
            "domain": _normalize_text(passport.get("domain")),
            "summary": _truncate(_normalize_text(passport.get("summary")), max_chars=1000),
            "business_rules_count": len(passport.get("business_rules", []))
            if isinstance(passport.get("business_rules", []), list)
            else 0,
            "interactions_count": len(passport.get("interactions", []))
            if isinstance(passport.get("interactions", []), list)
            else 0,
            "passport_json": _truncate(
                _safe_json(
                    {
                        key: value
                        for key, value in passport.items()
                        if key != "full_content"
                    }
                ),
                max_chars=1500,
            ),
        }
        if hash_by_path is not None:
            metadata["content_hash"] = hash_by_path.get(path, "")

        payloads.append(
            EmbeddingPayload(
                doc_id=_build_doc_id(project_name=project_name, file_path=path),
                text=text,
                metadata=metadata,
            )
        )
    return payloads


def _group_payloads_for_embedding(
    payloads: Sequence[EmbeddingPayload],
    max_batch_size: int,
    max_batch_chars: int,
) -> List[List[EmbeddingPayload]]:
    batches: List[List[EmbeddingPayload]] = []
    current_batch: List[EmbeddingPayload] = []
    current_chars = 0

    for payload in payloads:
        text_size = len(payload.text)
        is_new_batch_needed = (
            len(current_batch) >= max_batch_size
            or (current_batch and current_chars + text_size > max_batch_chars)
        )
        if is_new_batch_needed:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(payload)
        current_chars += text_size

    if current_batch:
        batches.append(current_batch)

    return batches


def _extract_vectors(response: Any) -> List[List[float]]:
    embeddings_obj = getattr(response, "embeddings", None)
    if embeddings_obj is None and isinstance(response, dict):
        embeddings_obj = response.get("embeddings")
    if embeddings_obj is None:
        raise ValueError("Embedding response does not contain embeddings")

    vectors: List[List[float]] = []
    for item in embeddings_obj:
        values = getattr(item, "values", None)
        if values is None and isinstance(item, dict):
            values = item.get("values")
        if not isinstance(values, list):
            raise ValueError("Embedding response item does not contain vector values")
        vectors.append(values)
    return vectors


def _embed_with_retry(
    texts: Sequence[str],
    model: str,
    max_retries: int,
    base_delay_sec: float,
) -> List[List[float]]:
    client = _get_gemini_client()
    attempt = 0
    last_error: Optional[Exception] = None

    while attempt < max_retries:
        try:
            _gemini_rate_limiter.wait_for_slot()
            response = client.models.embed_content(
                model=model,
                contents=list(texts),
            )
            return _extract_vectors(response=response)
        except Exception as error:
            last_error = error
            attempt += 1
            if attempt >= max_retries:
                break
            sleep_for = base_delay_sec * (2 ** (attempt - 1))
            logger.warning(
                "Embedding attempt %s/%s failed: %s. Retrying in %.2f sec",
                attempt,
                max_retries,
                error,
                sleep_for,
            )
            time.sleep(sleep_for)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedding failed without explicit exception")


def generate_embeddings(
    payloads: Sequence[EmbeddingPayload],
    model: str = EMBEDDING_MODEL,
    max_batch_size: int = EMBEDDING_MAX_BATCH_SIZE,
    max_batch_chars: int = EMBEDDING_MAX_BATCH_CHARS,
    max_retries: int = EMBEDDING_MAX_RETRIES,
    base_delay_sec: float = EMBEDDING_RETRY_BASE_DELAY_SEC,
) -> List[Tuple[EmbeddingPayload, List[float]]]:
    if not payloads:
        return []

    batches = _group_payloads_for_embedding(
        payloads=payloads,
        max_batch_size=max_batch_size,
        max_batch_chars=max_batch_chars,
    )

    results: List[Tuple[EmbeddingPayload, List[float]]] = []
    for batch_index, batch in enumerate(batches, start=1):
        texts = [item.text for item in batch]
        vectors = _embed_with_retry(
            texts=texts,
            model=model,
            max_retries=max_retries,
            base_delay_sec=base_delay_sec,
        )
        if len(vectors) != len(batch):
            raise ValueError(
                f"Embedding response size mismatch in batch {batch_index}: "
                f"got {len(vectors)}, expected {len(batch)}"
            )
        for payload, vector in zip(batch, vectors):
            results.append((payload, vector))

    return results


def iter_payload_chunks(
    payloads: Sequence[EmbeddingPayload],
    chunk_size: int,
) -> Iterable[List[EmbeddingPayload]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    for start in range(0, len(payloads), chunk_size):
        yield list(payloads[start : start + chunk_size])


def group_payloads_for_embedding(
    payloads: Sequence[EmbeddingPayload],
    max_batch_size: int = EMBEDDING_MAX_BATCH_SIZE,
    max_batch_chars: int = EMBEDDING_MAX_BATCH_CHARS,
) -> List[List[EmbeddingPayload]]:
    # Expose batching logic for dry-run debugging before model calls.
    return _group_payloads_for_embedding(
        payloads=payloads,
        max_batch_size=max_batch_size,
        max_batch_chars=max_batch_chars,
    )


def build_embedding_split_debug(
    payloads: Sequence[EmbeddingPayload],
    max_batch_size: int = EMBEDDING_MAX_BATCH_SIZE,
    max_batch_chars: int = EMBEDDING_MAX_BATCH_CHARS,
) -> List[EmbeddingBatchDebugInfo]:
    # This function does not call the model API. It only shows how payloads
    # will be split by the same batching logic used during real embedding.
    batches = _group_payloads_for_embedding(
        payloads=payloads,
        max_batch_size=max_batch_size,
        max_batch_chars=max_batch_chars,
    )

    debug_rows: List[EmbeddingBatchDebugInfo] = []
    for batch_index, batch in enumerate(batches, start=1):
        debug_rows.append(
            EmbeddingBatchDebugInfo(
                batch_index=batch_index,
                items_count=len(batch),
                chars_count=sum(len(item.text) for item in batch),
                paths=[str(item.metadata.get("path", "")) for item in batch],
            )
        )
    return debug_rows
