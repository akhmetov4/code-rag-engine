import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai

from config import (
    CONTEXT_MAX_CHARS,
    CONTEXT_MAX_FILES,
    GEMINI_API_KEY,
    GENERATIVE_MODEL,
    GENERATIVE_MODEL_FALLBACKS,
    GENERATIVE_REQUESTS_PER_MINUTE,
    PROJECT_ROOT,
)
from core.rate_limiter import get_or_create_limiter
from core.search import SearchResult

logger = logging.getLogger(__name__)

_gemini_client: Optional[Any] = None
# Common rate limiter with core/processor.py — one limit for all generative requests.
_rate_limiter = get_or_create_limiter(
    name="gemini-generate",
    max_requests=GENERATIVE_REQUESTS_PER_MINUTE,
)

ANALYSIS_PROMPT_PATH = PROJECT_ROOT / "prompts" / "analysis_prompt.txt"


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _generate_text(prompt: str) -> str:
    """Calls Gemini to generate a text (markdown) response."""
    client = _get_gemini_client()
    model_candidates = [GENERATIVE_MODEL] + [
        m for m in GENERATIVE_MODEL_FALLBACKS if m != GENERATIVE_MODEL
    ]
    last_error: Optional[Exception] = None

    for model_name in model_candidates:
        try:
            _rate_limiter.wait_for_slot()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not text:
                raise ValueError("Gemini returned empty response")
            return text
        except Exception as error:
            last_error = error
            error_text = str(error)
            is_model_unavailable = (
                "404" in error_text
                and "NOT_FOUND" in error_text
                and "model" in error_text.lower()
            )
            if is_model_unavailable:
                logger.warning("Model %s unavailable, trying next", model_name)
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("No available generative model")


def translate_to_english(text: str) -> str:
    """Translates user query to English for better semantic search."""
    prompt = (
        "Translate the following text to English. "
        "Return ONLY the translation, no explanations.\n\n"
        f"{text}"
    )
    return _generate_text(prompt).strip()


def _build_file_section(
    result: SearchResult,
    passport: Dict[str, Any],
) -> str:
    """Формирует текстовую секцию контекста для одного файла."""
    lines = [f"=== FILE: {result.path} ==="]
    lines.append(f"Domain: {result.domain}")
    lines.append(f"Summary: {result.summary}")

    business_rules = passport.get("business_rules", [])
    if isinstance(business_rules, list) and business_rules:
        lines.append("Business Rules:")
        for rule in business_rules:
            lines.append(f"  - {rule}")

    interactions = passport.get("interactions", [])
    if isinstance(interactions, list) and interactions:
        lines.append("Interactions:")
        for item in interactions:
            lines.append(f"  - {item}")

    full_content = passport.get("full_content")
    if full_content:
        lines.append(f"\nSource Code:\n{full_content}")
    else:
        lines.append("\n[Source code not available for this file]")

    return "\n".join(lines)


def build_context_data(
    results: List[SearchResult],
    passports_by_path: Dict[str, Dict[str, Any]],
    max_files: int = CONTEXT_MAX_FILES,
    max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """Collects context from search results with budget limits.

    Seed-результаты (distance >= 0) идут первыми, затем expansion (distance == -1).
    Сборка останавливается, когда достигнут лимит файлов или символов.
    """
    sections: List[str] = []
    total_chars = 0
    files_included = 0
    files_skipped_budget = 0
    separator = "\n\n---\n\n"

    for result in results:
        if files_included >= max_files:
            files_skipped_budget += len(results) - files_included - files_skipped_budget
            break

        passport = passports_by_path.get(result.path, {})
        section = _build_file_section(result, passport)

        section_chars = len(section) + len(separator)
        if total_chars + section_chars > max_chars and sections:
            files_skipped_budget += 1
            continue

        sections.append(section)
        total_chars += section_chars
        files_included += 1

    if files_skipped_budget:
        logger.info(
            "[context] budget: included %d files (%d chars), skipped %d (max_files=%d, max_chars=%d)",
            files_included, total_chars, files_skipped_budget, max_files, max_chars,
        )
    else:
        logger.info(
            "[context] included all %d files (%d chars)", files_included, total_chars,
        )

    return separator.join(sections)


def generate_analysis(
    context_data: str,
    user_query: str,
    prompt_template_path: Path = ANALYSIS_PROMPT_PATH,
) -> str:
    """Generates an analysis of a feature based on the context and user query."""
    template = prompt_template_path.read_text(encoding="utf-8")
    prompt = template.format(
        context_data=context_data,
        user_query=user_query,
    )
    return _generate_text(prompt)


def _slugify(text: str, max_len: int = 50) -> str:
    slug = text[:max_len].strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "_", slug)
    return slug.strip("_") or "report"


def save_report(
    content: str,
    query: str,
    output_dir: Path,
) -> Path:
    """Saves the generated report to a markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(query)
    filename = f"{timestamp}_{slug}.md"
    output_path = output_dir / filename
    output_path.write_text(content, encoding="utf-8")
    return output_path
