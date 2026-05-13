import os
import re
import json
import logging
from json_repair import repair_json
from typing import Any, List, Dict, Optional, Tuple
from collections import defaultdict
from config import (
    GEMINI_API_KEY,
    GENERATIVE_REQUESTS_PER_MINUTE,
    GENERATIVE_MODEL,
    GENERATIVE_MODEL_FALLBACKS,
    PASSPORT_SUMMARY_MIN_CHARS,
    PROJECT_ROOT,
)
from google import genai
from core.rate_limiter import get_or_create_limiter

logger = logging.getLogger(__name__)
gemini_client: Optional[Any] = None
_gemini_rate_limiter = get_or_create_limiter(
    name="gemini-generate",
    max_requests=GENERATIVE_REQUESTS_PER_MINUTE,
)


def _get_gemini_client() -> Any:
    # Lazy initialization of the Gemini client.
    global gemini_client
    if gemini_client is not None:
        return gemini_client

    # Get the actual value from the environment (if the variable is set later),
    # or use the value from config.py if not set.
    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    gemini_client = genai.Client(api_key=api_key)
    return gemini_client


def _generate_model_response(prompt: str) -> Any:
    client = _get_gemini_client()
    # Set the priority to the model from the config, then try the fallbacks without duplicates.
    model_candidates = [GENERATIVE_MODEL] + [
        model_name for model_name in GENERATIVE_MODEL_FALLBACKS if model_name != GENERATIVE_MODEL
    ]
    last_error: Optional[Exception] = None

    for model_name in model_candidates:
        try:
            _gemini_rate_limiter.wait_for_slot()
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        except Exception as error:
            last_error = error
            error_text = str(error)
            is_model_unavailable = (
                "404" in error_text
                and "NOT_FOUND" in error_text
                and "model" in error_text.lower()
            )
            if is_model_unavailable:
                logger.warning(
                    "Model %s is unavailable, trying next fallback model", model_name
                )
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("No available generative model candidates")



def _folder_depth(folder_path: str) -> int:
    if not folder_path:
        return 0
    return folder_path.count(os.sep) + 1


def _split_folder_into_batches(
    files: List[Dict[str, str]],
    max_batch_size: int,
    max_chars: Optional[int],
) -> List[List[Dict[str, str]]]:
    # Stable order is needed for reproducible results between runs.
    sorted_files = sorted(files, key=lambda item: item["file_path"])
    folder_batches: List[List[Dict[str, str]]] = []
    current_batch: List[Dict[str, str]] = []
    current_chars = 0

    for file_info in sorted_files:
        file_chars = len(file_info.get("content", ""))
        exceeds_file_limit = max_chars is not None and file_chars > max_chars
        exceeds_batch_size = len(current_batch) >= max_batch_size
        exceeds_chars_limit = (
            max_chars is not None
            and current_batch
            and current_chars + file_chars > max_chars
        )

        if current_batch and (exceeds_batch_size or exceeds_chars_limit):
            folder_batches.append(current_batch)
            current_batch = []
            current_chars = 0

        # Very large file goes into a separate batch to not break the limits.
        if exceeds_file_limit:
            folder_batches.append([file_info])
            continue

        current_batch.append(file_info)
        current_chars += file_chars

    if current_batch:
        folder_batches.append(current_batch)

    return folder_batches


def create_smart_batches(
    files_data: List[Dict[str, str]],
    max_batch_size: int = 12,
    min_batch_size: int = 3,
    max_chars: Optional[int] = 120_000,
) -> List[List[Dict[str, str]]]:
    folder_groups = defaultdict(list)
    for file in files_data:
        folder = os.path.dirname(file["file_path"])
        folder_groups[folder].append(file)

    # Raise small groups up the directory structure (bottom-up),
    # to reduce the number of batches from a single file.
    folders = sorted(folder_groups.keys(), key=_folder_depth, reverse=True)
    for folder in folders:
        if folder not in folder_groups:
            continue
        if len(folder_groups[folder]) >= min_batch_size or folder == "":
            continue

        parent_folder = os.path.dirname(folder)
        folder_groups[parent_folder].extend(folder_groups[folder])
        del folder_groups[folder]

    batches: List[List[Dict[str, str]]] = []
    for folder in sorted(folder_groups.keys()):
        files = folder_groups[folder]
        batches.extend(
            _split_folder_into_batches(
                files=files,
                max_batch_size=max_batch_size,
                max_chars=max_chars,
            )
        )

    return batches

def clean_ai_json(text: str) -> str:
    """Очищает ответ LLM от markdown-обёртки и чинит невалидный JSON.
    Сначала пробуем стандартный json.loads; если падает — repair_json."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        logger.warning("[processor] JSON от LLM невалиден, пробуем repair_json")
        repaired = repair_json(text, return_objects=False)
        return repaired


_REQUIRED_PASSPORT_FIELDS: Dict[str, type] = {
    "path": str,
    "domain": str,
    "summary": str,
    "business_rules": list,
    "interactions": list,
}

# Паттерны заглушек, которые модель иногда возвращает вместо настоящего summary.
_PLACEHOLDER_PREFIXES = (
    "n/a", "todo", "tbd", "not available", "no summary",
    "no description", "undefined", "none",
)


def _validate_single_passport(
    item: Any,
    index: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Проверяет один passport-объект. Возвращает (passport, None) если валиден,
    или (None, причина_отказа) если невалиден."""
    if not isinstance(item, dict):
        return None, f"item[{index}]: not a dict (got {type(item).__name__})"

    for field_name, expected_type in _REQUIRED_PASSPORT_FIELDS.items():
        value = item.get(field_name)
        if value is None:
            return None, f"item[{index}]: missing required field '{field_name}'"
        if not isinstance(value, expected_type):
            return None, (
                f"item[{index}]: field '{field_name}' has type "
                f"{type(value).__name__}, expected {expected_type.__name__}"
            )

    path = item["path"].strip()
    if not path:
        return None, f"item[{index}]: 'path' is empty string"

    domain = item["domain"].strip()
    if not domain:
        return None, f"item[{index}] ({path}): 'domain' is empty"

    summary = item["summary"].strip()
    if not summary:
        return None, f"item[{index}] ({path}): 'summary' is empty"

    if len(summary) < PASSPORT_SUMMARY_MIN_CHARS:
        return None, (
            f"item[{index}] ({path}): summary too short "
            f"({len(summary)} chars < {PASSPORT_SUMMARY_MIN_CHARS})"
        )

    summary_lower = summary.lower().strip(" .")
    for placeholder in _PLACEHOLDER_PREFIXES:
        if summary_lower == placeholder:
            return None, f"item[{index}] ({path}): summary is a placeholder ('{summary}')"

    return item, None


def _validate_passports(raw_data: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Валидирует список passport'ов от модели.
    Возвращает (список_валидных, список_причин_отказа)."""
    if not isinstance(raw_data, list):
        raise ValueError("Gemini response must be a JSON array")

    validated: List[Dict[str, Any]] = []
    rejections: List[str] = []

    for index, item in enumerate(raw_data):
        passport, reason = _validate_single_passport(item, index)
        if passport is not None:
            validated.append(passport)
        elif reason is not None:
            rejections.append(reason)

    return validated, rejections

def enrich_code_batch(
    batch: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Генерирует passport'ы для батча файлов через Gemini.
    Возвращает (валидные_passport'ы, количество_отброшенных)."""
    if not batch:
        return [], 0

    context_text = ""
    for file in batch:
        context_text += f"\n--- START_FILE: {file['file_path']} ---\n{file['content']}\n--- END_FILE: {file['file_path']} ---\n"

    prompt_template = (PROJECT_ROOT / "prompts" / "summary_prompt.txt").read_text()
    prompt = prompt_template.format(context_text=context_text)
    try:
        response = _generate_model_response(prompt)
        response_text = getattr(response, "text", None)
        if not response_text:
            raise ValueError("Gemini returned empty text response")

        parsed_data = json.loads(clean_ai_json(response_text))
        json_data, rejections = _validate_passports(parsed_data)

        if rejections:
            logger.warning(
                "[processor] validation: %d valid, %d rejected out of %d raw items",
                len(json_data), len(rejections), len(parsed_data),
            )
            for reason in rejections:
                logger.warning("[processor] rejected: %s", reason)

        content_by_path = {file["file_path"]: file["content"] for file in batch}
        for passport in json_data:
            passport["full_content"] = content_by_path.get(passport["path"])

        return json_data, len(rejections)
    except Exception as e:
        logger.exception("Error enriching code batch: %s", e)
        return [], 0
