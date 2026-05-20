import argparse
import logging
from pathlib import Path

from config import (
    CHROMA_DB_PATH,
    PROJECT_ROOTS,
    SUMMARY_MAX_WORKERS,
)
from core.pipeline import (
    print_batch_preview,
    run_embedding_stage,
    run_summary_stage,
)
from core.processor import create_smart_batches
from core.reader import prepare_project_data
from core.generator import (
    build_context_data,
    generate_analysis,
    save_report,
    translate_to_english,
)
from core.pipeline import load_cached_passports
from core.search import (
    expand_by_domains_multi,
    search_codebase_multi,
)

logger = logging.getLogger(__name__)


def _derive_project_name(root_path: str) -> str:
    stem = Path(root_path).stem.lower().replace(" ", "_")
    return stem or "default_project"


def _add_project_args(
    parser: argparse.ArgumentParser,
    *,
    default: int | None = 0,
) -> None:
    """Common arguments for commands that work with a specific project."""
    help_text = "Project index from PROJECT_ROOTS."
    if default is None:
        help_text += " If not specified, search all projects."
    parser.add_argument(
        "--project-index",
        type=int,
        default=default,
        help=help_text,
    )


def _add_pipeline_args(
    parser: argparse.ArgumentParser,
    *,
    project_default: int | None = 0,
) -> None:
    """Common arguments for summary/embed/process."""
    _add_project_args(parser, default=project_default)
    parser.add_argument(
        "--state-dir",
        type=str,
        default="pipeline_state",
        help="Base folder for intermediate passports.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Code RAG Engine: summary → embedding → search."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- summary ---
    sp_summary = subparsers.add_parser(
        "summary", help="Generate technical passports (summaries) for project files.",
    )
    _add_pipeline_args(sp_summary)
    sp_summary.add_argument(
        "--batch-limit", type=int, default=0,
        help="How many batches to process (0 = all).",
    )
    sp_summary.add_argument(
        "--workers", type=int, default=SUMMARY_MAX_WORKERS,
        help="Number of parallel workers.",
    )
    sp_summary.add_argument(
        "--dry-run", action="store_true",
        help="Only show batch plan, without processing.",
    )

    # --- embed ---
    sp_embed = subparsers.add_parser(
        "embed", help="Embedding summaries in ChromaDB.",
    )
    _add_pipeline_args(sp_embed)
    sp_embed.add_argument(
        "--chunk-size", type=int, default=120,
        help="Size of payload chunks for incremental embedding/upsert.",
    )

    # --- process ---
    sp_process = subparsers.add_parser(
        "process", help="Summary + Embedding (full pipeline).",
    )
    _add_pipeline_args(sp_process, project_default=None)
    sp_process.add_argument(
        "--batch-limit", type=int, default=0,
        help="How many summary batches to process (0 = all).",
    )
    sp_process.add_argument(
        "--workers", type=int, default=SUMMARY_MAX_WORKERS,
        help="Number of parallel workers for summary.",
    )
    sp_process.add_argument(
        "--chunk-size", type=int, default=120,
        help="Size of payload chunks for embedding.",
    )

    # --- search ---
    sp_search = subparsers.add_parser(
        "search", help="Search the codebase.",
    )
    _add_project_args(sp_search, default=None)
    sp_search.add_argument(
        "query", type=str,
        help="Text query for search.",
    )
    sp_search.add_argument(
        "--top-k", type=int, default=5,
        help="Number of seed results for semantic search.",
    )
    sp_search.add_argument(
        "--no-expand", action="store_true",
        help="Disable domain expansion (enabled by default).",
    )

    # --- ask ---
    sp_ask = subparsers.add_parser(
        "ask", help="Full RAG: search → generate documentation → file.",
    )
    _add_pipeline_args(sp_ask, project_default=None)
    sp_ask.add_argument(
        "query", type=str,
        help="Question in any language.",
    )
    sp_ask.add_argument(
        "--top-k", type=int, default=5,
        help="Number of seed results for semantic search.",
    )
    sp_ask.add_argument(
        "--no-expand", action="store_true",
        help="Disable domain expansion.",
    )
    sp_ask.add_argument(
        "--output-dir", type=str, default="output",
        help="Folder for saving reports.",
    )

    return parser


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def _resolve_project(args: argparse.Namespace) -> tuple[str, str]:
    """Returns (root_path, project_name) by --project-index."""
    idx = args.project_index
    if idx < 0 or idx >= len(PROJECT_ROOTS):
        raise ValueError(
            f"--project-index must be in range [0, {len(PROJECT_ROOTS) - 1}]"
        )
    root_path = str(PROJECT_ROOTS[idx])
    project_name = _derive_project_name(root_path)
    return root_path, project_name


def _resolve_project_names(args: argparse.Namespace) -> list[str]:
    """If --project-index is specified, one project, otherwise all from PROJECT_ROOTS."""
    if args.project_index is not None:
        _, name = _resolve_project(args)
        return [name]
    return [_derive_project_name(str(root)) for root in PROJECT_ROOTS]


def _resolve_passports_dir(args: argparse.Namespace, project_name: str) -> Path:
    return Path(args.state_dir) / project_name / "passports"


# ---------------------------------------------------------------------------
#  Commands
# ---------------------------------------------------------------------------

def _cmd_summary(args: argparse.Namespace) -> None:
    root_path, project_name = _resolve_project(args)
    project_data = prepare_project_data(root_path)
    all_batches = create_smart_batches(project_data)
    print_batch_preview(batches=all_batches, limit=args.batch_limit)

    if args.dry_run:
        print("Dry run: processing skipped.")
        return

    passports_dir = _resolve_passports_dir(args, project_name)
    passports_dir.mkdir(parents=True, exist_ok=True)

    stats = run_summary_stage(
        project_data=project_data,
        passports_dir=passports_dir,
        batch_limit=args.batch_limit,
        project_name=project_name,
        max_workers=args.workers,
    )
    print("Summary stage completed")
    print(f"  Project: {stats.project_name}")
    print(f"  Cached before run: {stats.cached_before_run}")
    print(f"  Generated now: {stats.generated_now}")
    print(f"  Passports total: {stats.passports_total}")
    print(f"  Rejected (invalid): {stats.rejected_passports}")
    print(f"  Unresolved files: {stats.failed_files}")
    print(f"  Passports folder: {passports_dir}")


def _cmd_embed(args: argparse.Namespace) -> None:
    _, project_name = _resolve_project(args)
    passports_dir = _resolve_passports_dir(args, project_name)

    if not passports_dir.exists():
        raise FileNotFoundError(
            f"Passports directory not found: {passports_dir}. "
            "Run 'summary' command first."
        )

    stats = run_embedding_stage(
        passports_dir=passports_dir,
        project_name=project_name,
        embedding_chunk_size=args.chunk_size,
    )
    print("Embedding stage completed")
    print(f"  Payloads total: {stats.payloads_total}")
    print(f"  Already indexed: {stats.already_indexed}")
    print(f"  Embedded now: {stats.embedded_now}")
    print(f"  Upserted now: {stats.upserted_now}")
    print(f"  Chroma DB path: {CHROMA_DB_PATH}")


def _process_one_project(args: argparse.Namespace) -> None:
    root_path, project_name = _resolve_project(args)
    print(f"\n{'='*80}")
    print(f"  PROCESSING: {project_name}  ({root_path})")
    print(f"{'='*80}")

    project_data = prepare_project_data(root_path)
    all_batches = create_smart_batches(project_data)
    print_batch_preview(batches=all_batches, limit=args.batch_limit)

    passports_dir = _resolve_passports_dir(args, project_name)
    passports_dir.mkdir(parents=True, exist_ok=True)

    summary_stats = run_summary_stage(
        project_data=project_data,
        passports_dir=passports_dir,
        batch_limit=args.batch_limit,
        project_name=project_name,
        max_workers=args.workers,
    )
    print("Summary stage completed")
    print(f"  Project: {summary_stats.project_name}")
    print(f"  Cached before run: {summary_stats.cached_before_run}")
    print(f"  Generated now: {summary_stats.generated_now}")
    print(f"  Passports total: {summary_stats.passports_total}")
    print(f"  Rejected (invalid): {summary_stats.rejected_passports}")
    print(f"  Unresolved files: {summary_stats.failed_files}")

    embedding_stats = run_embedding_stage(
        passports_dir=passports_dir,
        project_name=project_name,
        embedding_chunk_size=args.chunk_size,
    )
    print("Embedding stage completed")
    print(f"  Payloads total: {embedding_stats.payloads_total}")
    print(f"  Already indexed: {embedding_stats.already_indexed}")
    print(f"  Embedded now: {embedding_stats.embedded_now}")
    print(f"  Upserted now: {embedding_stats.upserted_now}")
    print(f"  Chroma DB path: {CHROMA_DB_PATH}")


def _cmd_process(args: argparse.Namespace) -> None:
    if args.project_index is None:
        if not PROJECT_ROOTS:
            print("[process] PROJECT_ROOTS is empty. Set it in .env.")
            return
        indices = list(range(len(PROJECT_ROOTS)))
        print(
            f"[process] No --project-index given. "
            f"Running for ALL {len(indices)} projects in PROJECT_ROOTS."
        )
    else:
        indices = [args.project_index]

    failures: list[tuple[int, str]] = []
    for idx in indices:
        args.project_index = idx
        try:
            _process_one_project(args)
        except Exception as exc:
            logger.exception("[process] failed for project index %d", idx)
            print(f"[process] FAILED for project index {idx}: {exc}")
            print("[process] Continuing with next project...")
            failures.append((idx, str(exc)))

    print(f"\n{'='*80}")
    print(f"  PROCESS SUMMARY")
    print(f"{'='*80}")
    print(f"  Projects attempted: {len(indices)}")
    print(f"  Succeeded: {len(indices) - len(failures)}")
    if failures:
        print(f"  Failed: {len(failures)}")
        for idx, err in failures:
            print(f"    - index {idx}: {err}")


def _cmd_search(args: argparse.Namespace) -> None:
    project_names = _resolve_project_names(args)
    use_expand = not args.no_expand
    print(
        f"[search] query=\"{args.query}\", projects={project_names}, "
        f"top_k={args.top_k}, domain_expand={use_expand}"
    )

    seed_results = search_codebase_multi(
        query_text=args.query,
        project_names=project_names,
        top_k=args.top_k,
    )

    if not seed_results:
        print("[search] Nothing found (all results filtered by distance threshold).")
        return

    domains_found = sorted({r.domain for r in seed_results if r.domain})
    print(
        f"[search] Seed results after filtering: {len(seed_results)}/{args.top_k}, "
        f"domains: {domains_found}"
    )

    print(f"\n{'='*80}")
    print(f"  SEED RESULTS (semantic search, top {args.top_k})")
    print(f"{'='*80}")
    for result in seed_results:
        print(f"\n  #{result.rank} (distance: {result.distance:.4f})")
        print(f"    Path:    {result.path}")
        print(f"    Domain:  {result.domain}")
        print(f"    Summary: {result.summary}")

    if not use_expand:
        return

    expanded = expand_by_domains_multi(
        seed_results=seed_results,
        project_names=project_names,
    )

    if not expanded:
        print(f"\n[search] Domain expansion: no additional files.")
        return

    total = len(seed_results) + len(expanded)
    print(f"\n{'='*80}")
    print(f"  DOMAIN EXPANSION (+{len(expanded)} files, {total} total)")
    print(f"{'='*80}")
    for result in expanded:
        print(f"\n  #{result.rank}")
        print(f"    Path:    {result.path}")
        print(f"    Domain:  {result.domain}")
        print(f"    Summary: {result.summary}")


def _cmd_ask(args: argparse.Namespace) -> None:
    project_names = _resolve_project_names(args)

    # Load passports (with full_content) from all selected projects.
    passports_by_path: dict[str, object] = {}
    for pname in project_names:
        pdir = _resolve_passports_dir(args, pname)
        if not pdir.exists():
            logger.warning(
                f"Passports dir not found for '{pname}': {pdir}, пропускаем."
            )
            continue
        cached = load_cached_passports(pdir)
        passports_by_path.update(
            {path: entry.passport for path, entry in cached.items()}
        )

    if not passports_by_path:
        raise FileNotFoundError(
            "No passports found for any project. "
            "Run 'process' or 'summary' command first."
        )

    # 1. Translate query to English for better semantic search.
    print(f"[ask] Original query: {args.query}")
    english_query = translate_to_english(args.query)
    print(f"[ask] English query:  {english_query}")
    print(f"[ask] Projects: {project_names}")

    # 2. Semantic search across all collections.
    seed_results = search_codebase_multi(
        query_text=english_query,
        project_names=project_names,
        top_k=args.top_k,
    )
    if not seed_results:
        print("[ask] Ничего не найдено (все результаты отфильтрованы по distance threshold).")
        return

    domains = sorted({r.domain for r in seed_results if r.domain})
    print(
        f"[ask] Seed results after filtering: {len(seed_results)}/{args.top_k}, "
        f"domains: {domains}"
    )

    # 3. Domain expansion — get files from found domains (with limits).
    all_results = list(seed_results)
    if not args.no_expand:
        expanded = expand_by_domains_multi(seed_results, project_names)
        all_results.extend(expanded)
        if expanded:
            print(f"[ask] Domain expansion: +{len(expanded)} files, {len(all_results)} total")
        else:
            print("[ask] Domain expansion: no additional files.")

    matched = sum(1 for r in all_results if r.path in passports_by_path)
    print(f"[ask] Files with source code: {matched}/{len(all_results)}")

    # 4. Build context with budget limits and generate.
    context_data = build_context_data(all_results, passports_by_path)
    print(f"[ask] Context: {len(context_data):,} chars in final prompt")
    print(f"[ask] Generating analysis...")

    analysis = generate_analysis(
        context_data=context_data,
        user_query=args.query,
    )

    # 5. Save report.
    output_path = save_report(
        content=analysis,
        query=args.query,
        output_dir=Path(args.output_dir),
    )
    print(f"[ask] Report saved: {output_path}")

    preview_lines = analysis.strip().splitlines()[:15]
    print(f"\n{'='*80}")
    print("  PREVIEW")
    print(f"{'='*80}")
    for line in preview_lines:
        print(f"  {line}")
    if len(analysis.strip().splitlines()) > 15:
        print(f"  ... (more {len(analysis.strip().splitlines()) - 15} lines)")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "summary": _cmd_summary,
        "embed": _cmd_embed,
        "process": _cmd_process,
        "search": _cmd_search,
        "ask": _cmd_ask,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return

    handler(args)


if __name__ == "__main__":
    main()
