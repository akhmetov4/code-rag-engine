import json
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import chromadb
from chromadb.api.models.Collection import Collection


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _sanitize_metadata_value(value)
        for key, value in metadata.items()
    }


class ChromaVectorStore:
    def __init__(self, db_path: Path, collection_name: str) -> None:
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @property
    def collection(self) -> Collection:
        return self._collection

    def upsert_embeddings(
        self,
        rows: Sequence[Tuple[str, str, Dict[str, Any], List[float]]],
    ) -> int:
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        documents = [row[1] for row in rows]
        metadatas = [_sanitize_metadata(row[2]) for row in rows]
        embeddings = [row[3] for row in rows]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(rows)

    def _iter_id_chunks(
        self, ids: Sequence[str], chunk_size: int,
    ) -> Iterable[List[str]]:
        iterator = iter(ids)
        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break
            yield chunk

    def get_existing_ids(
        self,
        ids: Sequence[str],
        chunk_size: int = 500,
    ) -> Set[str]:
        if not ids:
            return set()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        existing: Set[str] = set()
        for chunk in self._iter_id_chunks(ids, chunk_size):
            response = self._collection.get(ids=chunk)
            found_ids = response.get("ids", []) if isinstance(response, dict) else []
            for item in found_ids:
                if isinstance(item, str):
                    existing.add(item)
        return existing

    def get_ids_with_metadata(
        self,
        ids: Sequence[str],
        chunk_size: int = 500,
    ) -> Dict[str, Dict[str, Any]]:
        """Returns {doc_id: metadata} for records found in the collection."""
        if not ids:
            return {}
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        result: Dict[str, Dict[str, Any]] = {}
        for chunk in self._iter_id_chunks(ids, chunk_size):
            response = self._collection.get(ids=chunk, include=["metadatas"])
            found_ids = response.get("ids", []) if isinstance(response, dict) else []
            found_metas = response.get("metadatas", []) if isinstance(response, dict) else []
            for doc_id, meta in zip(found_ids, found_metas):
                if isinstance(doc_id, str):
                    result[doc_id] = meta if isinstance(meta, dict) else {}
        return result

    def get_by_metadata(
        self,
        where: Dict[str, Any],
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Returns documents by metadata filter (without semantic ranking)."""
        return self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
