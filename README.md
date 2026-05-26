# Code RAG Engine

Indexes codebases via per-file "technical passports" (generated with Gemini) and runs semantic search over them in ChromaDB. Entry point: `main.py`.

## Installation

```bash
cd /path/to/code-rag-engine
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key
# Comma-separated list of project paths to index and search over.
# Indices 0, 1, 2 … are used by the --project-index flag.
PROJECT_ROOTS=/path/to/repo-a,/path/to/repo-b
```

Default vector DB location: `data/chroma/`. Intermediate passports: `pipeline_state/<project_name>/passports/`.

---

## General help

```bash
python main.py --help
python main.py <command> --help
```

---

## Commands and examples

### `summary` — generate per-file passports (summaries)

Without writing anything to disk, just print the batch plan:

```bash
python main.py summary --dry-run
```

Full run for the first project in `PROJECT_ROOTS` (default index `0`):

```bash
python main.py summary
```

Limit the number of batches and set the worker count:

```bash
python main.py summary --batch-limit 3 --workers 2
```

A different project and a custom state directory:

```bash
python main.py summary --project-index 1 --state-dir my_state
```

### `embed` — load passports into ChromaDB

A passports directory must already exist (produced by `summary` or `process`). Example:

```bash
python main.py embed
python main.py embed --project-index 1 --chunk-size 120 --state-dir pipeline_state
```

### `process` — full pipeline: `summary` + `embed`

```bash
python main.py process
python main.py process --batch-limit 5 --workers 3 --chunk-size 120 --project-index 0
```

### `search` — semantic search over already-indexed collections

Across all projects in `PROJECT_ROOTS` (no `--project-index` argument):

```bash
python main.py search "where is payment processed"
```

A single project only:

```bash
python main.py search "authentication middleware" --project-index 0
```

More candidates on the first step and no domain expansion:

```bash
python main.py search "night delivery" --top-k 10 --no-expand
```

### `ask` — search + context assembly + report to file (full RAG)

Requires saved passports (from `summary` or `process`). By default all projects are considered; pass an index to scope to one.

```bash
python main.py ask "How does night delivery work?"
python main.py ask "What happens on buy button click?" --project-index 0 --top-k 8
python main.py ask "describe the orders API" --no-expand --output-dir reports
```

The report is saved to the specified folder (default `output/`) as a markdown file.

---

## Typical workflow

1. Configure `PROJECT_ROOTS` and `.env`.
2. Run indexing once: `python main.py process` (or step by step with `summary`, then `embed`).
3. Search: `python main.py search "..."`.
4. Deep answer with a saved file: `python main.py ask "..."`.

After changing code in a target repo, re-run `summary`/`process` to refresh passports and embeddings.
