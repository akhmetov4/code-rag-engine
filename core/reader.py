import hashlib
import os
from fnmatch import fnmatch
from config import IGNORE_DIRS, ALLOWED_EXTENSIONS, IGNORE_FILES, IGNORE_PATTERNS
from typing import Optional

def get_all_files(root_dir: str) -> list[str]:
    valid_files = []
    root_dir = os.path.abspath(root_dir)
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        files[:] = [
            f for f in files
            if f not in IGNORE_FILES and not any(fnmatch(f, pattern) for pattern in IGNORE_PATTERNS)
        ]
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ALLOWED_EXTENSIONS:
                full_path = os.path.join(root, file)
                valid_files.append(full_path)
    return valid_files

def read_file(file_path: str) -> Optional[str]:
    """Reads file completely, preserving all content including comments and docstring's —
    they carry valuable context for analysis through Gemini."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def prepare_project_data(root_dir: str) -> list[dict[str, str]]:
    raw_file_list = get_all_files(root_dir)
    processed_data = []

    print(f"Found {len(raw_file_list)} files in {root_dir}")

    for full_path in raw_file_list:
        relative_path = os.path.relpath(full_path, root_dir)
        content = read_file(full_path)
        if content is None:
            continue
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        processed_data.append({
            "file_path": relative_path,
            "content": content,
            "content_hash": content_hash,
        })

    return processed_data
