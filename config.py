import os
from pathlib import Path
from dotenv import load_dotenv  

PROJECT_ROOT = Path(__file__).resolve().parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

IGNORE_DIRS = {
    "node_modules",
    "dist",
    "build",
    ".storybook",
    ".git",
    ".vscode",
    ".cursor",
    ".venv",
    ".env",
    ".helm",
    "install",
    "docker",
    "migrations",
    "config",
}

IGNORE_FILES = {
    ".gitignore",
    ".git",
    "package-lock.json",
    "package.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "pnpm-workspace.yaml",
    "composer.lock",
    "composer.json",
    "composer.phar",
    "swagger.json",
    ".env",
    ".env.local",
    ".env.example",
}

IGNORE_PATTERNS = {
    "*.spec.js",
    "*.spec.ts",
    "*.css",
}

ALLOWED_EXTENSIONS = {
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",

    ".php",
    ".py",
}

# Comma-separated absolute (or ~) paths to the codebases to index, set in .env:
#   PROJECT_ROOTS=/path/to/repo-a,/path/to/repo-b
PROJECT_ROOTS = [
    Path(p.strip()).expanduser()
    for p in os.getenv("PROJECT_ROOTS", "").split(",")
    if p.strip()
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Supabase (used by the API server to read project info and write status back)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# API server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8765"))
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATIVE_MODEL = "gemini-2.5-flash"

GENERATIVE_REQUESTS_PER_MINUTE = 100
EMBEDDING_REQUESTS_PER_MINUTE = 2000
GENERATIVE_MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]

# Embedding pipeline settings
EMBEDDING_MAX_BATCH_SIZE = 24
EMBEDDING_MAX_BATCH_CHARS = 24_000
EMBEDDING_MAX_TEXT_CHARS = 4_000
EMBEDDING_MAX_RETRIES = 4
EMBEDDING_RETRY_BASE_DELAY_SEC = 1.5

# Parallelism settings for summary stage. How many batches to process at once.
SUMMARY_MAX_WORKERS = 5

# Vector database settings
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma"
CHROMA_COLLECTION_PREFIX = "code_passports"

# --- Passport validation settings ---
# Минимальная длина summary, чтобы считать passport валидным.
PASSPORT_SUMMARY_MIN_CHARS = 20

# --- Retrieval quality settings ---
# Максимальная L2-дистанция для seed-результатов. Результаты дальше порога отбрасываются.
SEARCH_DISTANCE_THRESHOLD = 1.5

# Ограничения на domain expansion.
DOMAIN_EXPANSION_MAX_DOMAINS = 3
DOMAIN_EXPANSION_MAX_FILES = 15

# Context budget — лимиты для финального prompt, чтобы не перегружать LLM шумом.
CONTEXT_MAX_FILES = 20
CONTEXT_MAX_CHARS = 200_000