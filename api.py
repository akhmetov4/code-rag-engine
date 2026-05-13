import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from config import (
    API_HOST,
    API_PORT,
    PROJECT_ROOT,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
    SUMMARY_MAX_WORKERS,
)
from core.generator import build_context_data, generate_analysis, translate_to_english
from core.pipeline import load_cached_passports, run_embedding_stage, run_summary_stage
from core.processor import create_smart_batches
from core.reader import prepare_project_data
from core.search import expand_by_domains_multi, search_codebase_multi

logger = logging.getLogger(__name__)

app = FastAPI(title="Code RAG Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# job_id -> { status: "running"|"done"|"error", logs: [...], error: str|None }
_jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env"
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _derive_project_name(path: str) -> str:
    stem = Path(path).stem.lower().replace(" ", "_")
    return stem or "default_project"


def _passports_dir(project_name: str) -> Path:
    return PROJECT_ROOT / "pipeline_state" / project_name / "passports"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_project_status(
    project_id: str,
    status: str,
    index_error: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {"status": status, "updated_at": _now_iso()}
    if index_error is not None:
        payload["index_error"] = index_error[:500]
    else:
        payload["index_error"] = None
    try:
        _supabase().table("projects").update(payload).eq("id", project_id).execute()
    except Exception:
        logger.exception("Failed to update project status in Supabase")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    path: str
    workers: int = SUMMARY_MAX_WORKERS
    chunk_size: int = 120


class AskRequest(BaseModel):
    path: str
    query: str
    top_k: int = 5
    expand: bool = True


class GlobalAskRequest(BaseModel):
    user_id: str
    query: str
    rules: Optional[str] = None
    top_k: int = 5
    expand: bool = True


class SearchRequest(BaseModel):
    path: str
    query: str
    top_k: int = 5
    expand: bool = True


# ---------------------------------------------------------------------------
# Background process job
# ---------------------------------------------------------------------------

def _run_process(
    job_id: str,
    project_id: str,
    project_path: str,
    workers: int,
    chunk_size: int,
) -> None:
    job = _jobs[job_id]

    def log(msg: str) -> None:
        logger.info(msg)
        job["logs"].append(msg)

    try:
        project_name = _derive_project_name(project_path)
        log(f"Starting process: project='{project_name}', path={project_path}")

        project_data = prepare_project_data(project_path)
        log(f"Files discovered: {len(project_data)}")

        passports_dir = _passports_dir(project_name)
        passports_dir.mkdir(parents=True, exist_ok=True)

        summary_stats = run_summary_stage(
            project_data=project_data,
            passports_dir=passports_dir,
            batch_limit=0,
            project_name=project_name,
            max_workers=workers,
        )
        log(
            f"Summary done: generated={summary_stats.generated_now}, "
            f"total={summary_stats.passports_total}, "
            f"failed={summary_stats.failed_files}"
        )

        embedding_stats = run_embedding_stage(
            passports_dir=passports_dir,
            project_name=project_name,
            embedding_chunk_size=chunk_size,
        )
        log(
            f"Embedding done: embedded={embedding_stats.embedded_now}, "
            f"upserted={embedding_stats.upserted_now}, "
            f"unchanged={embedding_stats.already_indexed}"
        )

        _update_project_status(project_id, "indexed")
        job["status"] = "done"
        log("Process completed successfully")

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("Process failed: project_id=%s", project_id)
        job["status"] = "error"
        job["error"] = error_msg
        _update_project_status(project_id, "error", index_error=error_msg)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/projects/{project_id}/process")
def process_project(
    project_id: str,
    req: ProcessRequest,
) -> Dict[str, str]:
    """Kick off the full summary + embedding pipeline for a project.

    Returns immediately with a job_id. Poll GET /jobs/{job_id} for status.
    Supabase project status is updated: not_indexed → indexing → indexed|error.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "logs": [], "error": None}

    _update_project_status(project_id, "indexing")

    thread = threading.Thread(
        target=_run_process,
        args=(job_id, project_id, req.path, req.workers, req.chunk_size),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    """Poll indexing job status and logs."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/projects/{project_id}/search")
def search_project(project_id: str, req: SearchRequest) -> Dict[str, Any]:
    """Semantic search over an indexed project."""
    project_name = _derive_project_name(req.path)

    seed_results = search_codebase_multi(
        query_text=req.query,
        project_names=[project_name],
        top_k=req.top_k,
    )

    expanded = []
    if req.expand and seed_results:
        expanded = expand_by_domains_multi(
            seed_results=seed_results,
            project_names=[project_name],
        )

    def _to_dict(r: Any) -> Dict[str, Any]:
        return {
            "rank": r.rank,
            "path": r.path,
            "domain": r.domain,
            "summary": r.summary,
            "distance": r.distance,
        }

    return {
        "seed_results": [_to_dict(r) for r in seed_results],
        "expanded": [_to_dict(r) for r in expanded],
    }


@app.post("/projects/{project_id}/ask")
def ask_project(project_id: str, req: AskRequest) -> Dict[str, Any]:
    """Full RAG: translate query → search → domain expand → generate answer."""
    project_name = _derive_project_name(req.path)
    passports_dir = _passports_dir(project_name)

    if not passports_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Project is not indexed yet. Run process first.",
        )

    cached = load_cached_passports(passports_dir)
    if not cached:
        raise HTTPException(
            status_code=400,
            detail="No passports found. Run process first.",
        )

    passports_by_path = {path: entry.passport for path, entry in cached.items()}

    english_query = translate_to_english(req.query)
    logger.info("[ask] original=%r  english=%r", req.query, english_query)

    seed_results = search_codebase_multi(
        query_text=english_query,
        project_names=[project_name],
        top_k=req.top_k,
    )

    if not seed_results:
        return {"answer": "No relevant files found for this query.", "sources": []}

    all_results = list(seed_results)
    if req.expand:
        expanded = expand_by_domains_multi(seed_results, [project_name])
        all_results.extend(expanded)

    context_data = build_context_data(all_results, passports_by_path)
    answer = generate_analysis(context_data=context_data, user_query=req.query)

    sources = [
        {"path": r.path, "domain": r.domain, "distance": r.distance}
        for r in seed_results
    ]

    return {"answer": answer, "sources": sources}


@app.post("/ask")
def ask_global(req: GlobalAskRequest) -> Dict[str, Any]:
    """Full RAG across ALL indexed projects for a user.

    Fetches indexed projects from Supabase, searches all of them, generates
    an answer, persists the conversation, and returns it.
    """
    # 1. Fetch all indexed projects for this user
    try:
        rows = (
            _supabase()
            .table("projects")
            .select("id, name, path")
            .eq("user_id", req.user_id)
            .eq("status", "indexed")
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Supabase error: {exc}") from exc

    projects = rows.data or []
    if not projects:
        raise HTTPException(
            status_code=400,
            detail="No indexed projects found. Index at least one project first.",
        )

    # 2. Collect passports from all projects
    project_names: List[str] = []
    passports_by_path: Dict[str, Any] = {}

    for proj in projects:
        pname = _derive_project_name(proj["path"])
        pdir = _passports_dir(pname)
        if not pdir.exists():
            logger.warning("Passports dir missing for project '%s', skipping", pname)
            continue
        cached = load_cached_passports(pdir)
        if not cached:
            continue
        project_names.append(pname)
        passports_by_path.update({path: entry.passport for path, entry in cached.items()})

    if not project_names:
        raise HTTPException(
            status_code=400,
            detail="Indexed projects found in Supabase but no passports on disk. Re-run indexing.",
        )

    # 3. Build effective query (incorporate rules if provided)
    if req.rules and req.rules.strip():
        effective_query = (
            f"**Role & instructions from user:**\n{req.rules.strip()}\n\n"
            f"**Question:**\n{req.query}"
        )
    else:
        effective_query = req.query

    # 4. Translate → search → expand → generate
    english_query = translate_to_english(req.query)
    logger.info("[ask_global] original=%r  english=%r  projects=%s", req.query, english_query, project_names)

    seed_results = search_codebase_multi(
        query_text=english_query,
        project_names=project_names,
        top_k=req.top_k,
    )

    if not seed_results:
        raise HTTPException(
            status_code=404,
            detail="No relevant files found for this query across indexed projects.",
        )

    all_results = list(seed_results)
    if req.expand:
        expanded = expand_by_domains_multi(seed_results, project_names)
        all_results.extend(expanded)

    context_data = build_context_data(all_results, passports_by_path)
    answer = generate_analysis(context_data=context_data, user_query=effective_query)

    sources = [
        {"path": r.path, "domain": r.domain, "distance": r.distance}
        for r in seed_results
    ]

    # 5. Persist conversation to Supabase
    conversation_id = str(uuid.uuid4())
    try:
        _supabase().table("conversations").insert({
            "id": conversation_id,
            "user_id": req.user_id,
            "query": req.query,
            "rules": req.rules or None,
            "answer": answer,
            "sources": sources,
            "project_names": project_names,
        }).execute()
    except Exception:
        logger.exception("Failed to save conversation to Supabase")
        # Don't fail the request — answer is already generated

    return {
        "id": conversation_id,
        "query": req.query,
        "answer": answer,
        "sources": sources,
        "project_names": project_names,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    uvicorn.run(app, host=API_HOST, port=API_PORT)
