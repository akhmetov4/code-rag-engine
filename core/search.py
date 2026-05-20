import logging
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from google import genai

from config import (
    CHROMA_COLLECTION_PREFIX,
    CHROMA_DB_PATH,
    DOMAIN_EXPANSION_MAX_DOMAINS,
    DOMAIN_EXPANSION_MAX_FILES,
    EMBEDDING_MODEL,
    EMBEDDING_REQUESTS_PER_MINUTE,
    GEMINI_API_KEY,
    SEARCH_DISTANCE_THRESHOLD,
)
from core.rate_limiter import get_or_create_limiter
from core.vector_db import ChromaVectorStore

logger = logging.getLogger(__name__)

_gemini_client: Optional[Any] = None
# Common rate limiter with core/embedding.py — one limit for all embedding requests.
_rate_limiter = get_or_create_limiter(
    name="gemini-embed",
    max_requests=EMBEDDING_REQUESTS_PER_MINUTE,
)

# ChromaVectorStore is expensive to open (PersistentClient init). Reuse per project.
_store_cache: Dict[str, ChromaVectorStore] = {}
_store_cache_lock = threading.Lock()


@dataclass
class SearchResult:
    rank: int
    path: str
    domain: str
    summary: str
    distance: float
    doc_id: str


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_store(project_name: str) -> ChromaVectorStore:
    cached = _store_cache.get(project_name)
    if cached is not None:
        return cached
    with _store_cache_lock:
        cached = _store_cache.get(project_name)
        if cached is not None:
            return cached
        collection_name = f"{CHROMA_COLLECTION_PREFIX}_{project_name}"
        store = ChromaVectorStore(
            db_path=CHROMA_DB_PATH,
            collection_name=collection_name,
        )
        _store_cache[project_name] = store
        return store


@lru_cache(maxsize=512)
def _embed_query_cached(query_text: str, model: str) -> Tuple[float, ...]:
    """Cached embedding for repeat queries. Same text+model → same vector deterministically."""
    client = _get_gemini_client()
    _rate_limiter.wait_for_slot()
    response = client.models.embed_content(
        model=model,
        contents=[query_text],
    )
    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise ValueError("Embedding API returned empty response")
    values = getattr(embeddings[0], "values", None)
    if not isinstance(values, list):
        raise ValueError("Embedding response does not contain vector values")
    return tuple(values)


def embed_query(query_text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Converts user query text into a vector through Gemini Embedding API."""
    return list(_embed_query_cached(query_text, model))


def search_codebase(
    query_text: str,
    project_name: str,
    top_k: int = 5,
) -> List[SearchResult]:
    """Semantic search: embed query → find nearest documents in ChromaDB."""
    return search_codebase_multi(
        query_text=query_text,
        project_names=[project_name],
        top_k=top_k,
    )


def search_codebase_multi(
    query_text: str,
    project_names: List[str],
    top_k: int = 5,
    distance_threshold: float = SEARCH_DISTANCE_THRESHOLD,
) -> List[SearchResult]:
    """Semantic search across multiple projects.

    Embedding query is done once, then the vector is reused for query by each collection.
    The results are merged and sorted by distance (L2 — less = closer).
    Results with distance > distance_threshold are discarded.
    Returns up to top_k best results.
    """
    query_vector = embed_query(query_text)
    all_results: List[SearchResult] = []

    for project_name in project_names:
        store = _get_store(project_name)
        raw = store.query(query_embedding=query_vector, top_k=top_k)

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc_id, _doc, meta, dist in zip(ids, documents, metadatas, distances):
            all_results.append(
                SearchResult(
                    rank=0,
                    path=meta.get("path", ""),
                    domain=meta.get("domain", ""),
                    summary=meta.get("summary", ""),
                    distance=dist,
                    doc_id=doc_id,
                )
            )

    all_results.sort(key=lambda r: r.distance)
    all_results = all_results[:top_k]

    # Фильтрация по порогу дистанции — убираем слабые совпадения.
    before_filter = len(all_results)
    all_results = [r for r in all_results if r.distance <= distance_threshold]
    filtered_out = before_filter - len(all_results)
    if filtered_out:
        logger.info(
            "[search] distance threshold=%.2f: kept %d, discarded %d weak results",
            distance_threshold, len(all_results), filtered_out,
        )

    for idx, result in enumerate(all_results, start=1):
        result.rank = idx

    return all_results


def expand_by_domains(
    seed_results: List[SearchResult],
    project_name: str,
) -> List[SearchResult]:
    """Extracts ALL files from domains found in seed results.

    Returns only files that are NOT in seed_results (expansion).
    Expanded results have distance = -1 (not ranked by query).
    """
    return expand_by_domains_multi(
        seed_results=seed_results,
        project_names=[project_name],
    )


def expand_by_domains_multi(
    seed_results: List[SearchResult],
    project_names: List[str],
    max_domains: int = DOMAIN_EXPANSION_MAX_DOMAINS,
    max_files: int = DOMAIN_EXPANSION_MAX_FILES,
) -> List[SearchResult]:
    """Domain expansion across multiple projects.

    Collects domains from seed results (up to max_domains, prioritised by frequency
    in seeds), then extracts up to max_files files that are not in seed_results.
    """
    # Приоритет доменов — по частоте появления в seed-результатах.
    domain_freq: Dict[str, int] = {}
    for r in seed_results:
        if r.domain:
            domain_freq[r.domain] = domain_freq.get(r.domain, 0) + 1
    domains = sorted(domain_freq, key=lambda d: domain_freq[d], reverse=True)

    if not domains:
        return []

    # Ограничиваем число доменов.
    if len(domains) > max_domains:
        skipped = domains[max_domains:]
        domains = domains[:max_domains]
        logger.info(
            "[expand] limited to %d domains, skipped: %s", max_domains, skipped,
        )

    where: Dict[str, Any] = (
        {"domain": domains[0]}
        if len(domains) == 1
        else {"domain": {"$in": domains}}
    )

    seed_ids = {r.doc_id for r in seed_results}
    expanded: List[SearchResult] = []

    for project_name in project_names:
        store = _get_store(project_name)
        raw = store.get_by_metadata(where=where)

        ids = raw.get("ids", [])
        metadatas = raw.get("metadatas", [])

        for doc_id, meta in zip(ids, metadatas):
            if doc_id in seed_ids:
                continue
            expanded.append(
                SearchResult(
                    rank=0,
                    path=meta.get("path", ""),
                    domain=meta.get("domain", ""),
                    summary=meta.get("summary", ""),
                    distance=-1.0,
                    doc_id=doc_id,
                )
            )

    expanded.sort(key=lambda r: r.path)

    # Ограничиваем число expansion-файлов.
    if len(expanded) > max_files:
        logger.info(
            "[expand] trimmed expansion from %d to %d files", len(expanded), max_files,
        )
        expanded = expanded[:max_files]

    for idx, result in enumerate(expanded, start=len(seed_results) + 1):
        result.rank = idx

    return expanded
