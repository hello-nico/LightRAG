# Repository Guidelines

## Project Structure & Module Organization

- `lightrag/` — core library (retrieval, graph/vector storage, utils). API under `lightrag/api/` with FastAPI routes and `webui/` assets; console entrypoints: `lightrag-server`, `lightrag-gunicorn`.
- `cli.py` — CLI app (query modes + PDF extract). Uses core instance manager.
- `tests/` — script-style tests: `test_graph_storage.py`, `test_lightrag_ollama_chat.py`.
- `docs/` — docs and notes (see `docs/cli_improve.md`).
- Runtime data defaults: `inputs/` (ingest) and `rag_storage/` (generated storage).

## Build, Test, and Development Commands

- Setup: `python -m venv .venv && source .venv/bin/activate`
- Install (dev + API): `pip install -e .[api]` (or `uv pip install -e .[api]`).
- Lint/format: `pre-commit install && pre-commit run -a` or `ruff format . && ruff check . --fix`.
- Run server (dev): `python -m lightrag.api.lightrag_server` (reads `.env`) or `lightrag-server`.
- Run server (prod): `lightrag-gunicorn --workers 2`.
- CLI query: `python -m cli -q local "什么是RAG"`.
- CLI extract (offline-ready): `python -m cli extract --input-dir rag_pdfs --output-dir ./rag_storage --mode dry-run`.
- Tests:
  - Graph storage (interactive): `python tests/test_graph_storage.py` (ensure `.env` storages configured).
  - API compatibility: start server, then `python tests/test_lightrag_ollama_chat.py --tests non_stream stream`.

## Coding Style & Naming Conventions

- Python 3.10+, 4-space indentation, type hints preferred.
- Names: modules/functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Use `ruff` for style/formatting; keep functions small, log via `lightrag.utils.logger`.

## Testing Guidelines

- Place tests under `tests/` as `test_<area>.py`. For async pytest tests, use `pytest.mark.asyncio` if applicable.
- No strict coverage gate; include tests for new behavior and edge/failure paths.
- Reuse data in `examples/`/`datasets/` where possible; avoid committing large binaries.

## Commit & Pull Request Guidelines

- Commits: imperative, concise, scoped. Examples: "Add Neo4J batch edge ops", "Fix CoT render fallback".
- PRs: clear description, reproduction steps, env vars touched, linked issues; screenshots/GIFs for Web UI; note config migrations.
- CI hygiene: pass `pre-commit`; verify server starts locally without regressions.

## Security & Configuration Tips

- Copy `env.example` to `.env`; never commit secrets. Key vars: `LLM_BINDING`, `OPENAI_API_KEY`, `WORKING_DIR`, `LIGHTRAG_*_STORAGE`.
- Optional Qwen embedding via `QWEN_API_KEY`, `QWEN_EMBEDDING_MODEL`, `QWEN_EMBEDDING_HOST` (only used when set).
- Default storages are file-based; DB backends require their envs set before running tests or the API.
