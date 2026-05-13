import argparse
import json
from pathlib import Path
from typing import Dict

import chromadb

from config import CHROMA_DB_PATH
from core.engine import (
    preview_embedding_payloads_from_file,
    preview_embedding_split_from_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug tools for embedding batching and payload previews."
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="debug_output",
        help="Directory containing generated batch_*.json files.",
    )
    parser.add_argument(
        "--index-batch-file",
        type=str,
        default="",
        help="Path to a single batch JSON file with passports.",
    )
    parser.add_argument(
        "--embedding-split",
        action="store_true",
        help="Show embedding split stats for a single batch file.",
    )
    parser.add_argument(
        "--embedding-split-all",
        action="store_true",
        help="Show embedding split stats for all batch_*.json files.",
    )
    parser.add_argument(
        "--embedding-split-output",
        type=str,
        default="",
        help="Optional output file path for split-all JSON report.",
    )
    parser.add_argument(
        "--embedding-payloads",
        action="store_true",
        help="Show exact texts that will be sent to embedding API (single file).",
    )
    parser.add_argument(
        "--embedding-payloads-all",
        action="store_true",
        help="Show exact texts for all batch_*.json files in --debug-dir.",
    )
    parser.add_argument(
        "--chroma-stats",
        action="store_true",
        help="Show Chroma DB collections and document counts.",
    )
    parser.add_argument(
        "--chroma-collection",
        type=str,
        default="",
        help="Optional collection name for sample rows preview.",
    )
    parser.add_argument(
        "--chroma-limit",
        type=int,
        default=3,
        help="How many sample rows to show for --chroma-collection.",
    )
    return parser


def _print_chroma_stats(collection_name: str, limit: int) -> None:
    if limit <= 0:
        raise ValueError("--chroma-limit must be > 0")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collections = client.list_collections()
    print(f"Chroma DB path: {CHROMA_DB_PATH}")
    print(f"Collections total: {len(collections)}")
    for item in collections:
        collection = client.get_collection(item.name)
        print(f"- {item.name}: count={collection.count()}")

    if not collection_name:
        return

    print("-" * 100)
    print(f"Sample rows from collection: {collection_name}")
    collection = client.get_collection(collection_name)
    rows = collection.get(limit=limit, include=["documents", "metadatas"])
    ids = rows.get("ids", [])
    documents = rows.get("documents", [])
    metadatas = rows.get("metadatas", [])
    print(f"Rows returned: {len(ids)}")
    for index, row_id in enumerate(ids, start=1):
        print(f"[{index}] id={row_id}")
        if index - 1 < len(metadatas):
            print(f"  metadata={metadatas[index - 1]}")
        if index - 1 < len(documents):
            print(f"  document={documents[index - 1]}")
        print("  " + "-" * 80)


def _print_embedding_payload_preview(index_batch_file: Path) -> None:
    preview = preview_embedding_payloads_from_file(index_batch_file)
    print("Embedding payload preview (no model call)")
    print(f"  Source file: {preview.source_path}")
    print(f"  Project: {preview.project_name}")
    print(f"  Passports: {preview.passports_total}")
    print(f"  Payloads: {preview.payloads_total}")
    print(f"  Batches: {preview.batches_total}")
    for batch in preview.batches:
        print(
            f"  Batch #{batch['batch_index']}: "
            f"items={batch['items_count']}, chars={batch['chars_count']}"
        )
        items = batch["items"]
        if isinstance(items, list):
            for item in items:
                print(f"    path: {item['path']}")
                print(f"    doc_id: {item['doc_id']}")
                print("    text_to_embed_start")
                print(item["text"])
                print("    text_to_embed_end")
                print("    " + "-" * 60)


def _run_embedding_split_all(debug_dir: Path, output_path_str: str) -> None:
    if not debug_dir.exists():
        raise FileNotFoundError(f"Debug directory not found: {debug_dir}")

    batch_files = sorted(debug_dir.glob("batch_*.json"))
    if not batch_files:
        raise ValueError(f"No batch_*.json files found in: {debug_dir}")

    aggregated: Dict[str, object] = {
        "debug_dir": str(debug_dir),
        "files_total": len(batch_files),
        "files": [],
    }

    total_passports = 0
    total_payloads = 0
    total_batches = 0

    print(f"Embedding split preview for {len(batch_files)} files (no model call)")
    for batch_file in batch_files:
        preview = preview_embedding_split_from_file(batch_file)
        total_passports += preview.passports_total
        total_payloads += preview.payloads_total
        total_batches += preview.batches_total

        print(
            f"- {batch_file.name}: passports={preview.passports_total}, "
            f"payloads={preview.payloads_total}, batches={preview.batches_total}"
        )
        for row in preview.batches:
            print(
                f"    batch#{row['batch_index']}: "
                f"items={row['items_count']}, chars={row['chars_count']}"
            )

        files_list = aggregated["files"]
        if isinstance(files_list, list):
            files_list.append(
                {
                    "source_file": preview.source_path,
                    "project_name": preview.project_name,
                    "passports_total": preview.passports_total,
                    "payloads_total": preview.payloads_total,
                    "batches_total": preview.batches_total,
                    "batches": preview.batches,
                }
            )

    print(
        "Totals: "
        f"passports={total_passports}, payloads={total_payloads}, batches={total_batches}"
    )
    aggregated["totals"] = {
        "passports_total": total_passports,
        "payloads_total": total_payloads,
        "batches_total": total_batches,
    }

    if output_path_str:
        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(aggregated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved split report: {output_path}")


def main() -> None:
    args = _build_parser().parse_args()

    if args.chroma_stats:
        _print_chroma_stats(
            collection_name=args.chroma_collection,
            limit=args.chroma_limit,
        )
        return

    if args.embedding_payloads_all:
        debug_dir = Path(args.debug_dir)
        if not debug_dir.exists():
            raise FileNotFoundError(f"Debug directory not found: {debug_dir}")
        batch_files = sorted(debug_dir.glob("batch_*.json"))
        if not batch_files:
            raise ValueError(f"No batch_*.json files found in: {debug_dir}")
        for batch_file in batch_files:
            _print_embedding_payload_preview(index_batch_file=batch_file)
            print("=" * 120)
        return

    if args.embedding_payloads:
        if not args.index_batch_file:
            raise ValueError("--embedding-payloads requires --index-batch-file")
        _print_embedding_payload_preview(index_batch_file=Path(args.index_batch_file))
        return

    if args.embedding_split_all:
        _run_embedding_split_all(
            debug_dir=Path(args.debug_dir),
            output_path_str=args.embedding_split_output,
        )
        return

    if args.embedding_split:
        if not args.index_batch_file:
            raise ValueError("--embedding-split requires --index-batch-file")
        preview = preview_embedding_split_from_file(Path(args.index_batch_file))
        print("Embedding split preview (no model call)")
        print(f"  Source file: {preview.source_path}")
        print(f"  Project: {preview.project_name}")
        print(f"  Passports: {preview.passports_total}")
        print(f"  Payloads: {preview.payloads_total}")
        print(f"  Batches: {preview.batches_total}")
        for row in preview.batches:
            print(
                f"  Batch #{row['batch_index']}: "
                f"items={row['items_count']}, chars={row['chars_count']}"
            )
            for path in row["paths"]:
                print(f"    - {path}")
        return

    raise ValueError(
        "Choose one debug mode: --embedding-split, --embedding-split-all, "
        "--embedding-payloads, --embedding-payloads-all, or --chroma-stats."
    )


if __name__ == "__main__":
    main()
