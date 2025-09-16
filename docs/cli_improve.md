# CLI Improvements Summary

## Overview

This iteration refactors the CLI to use the unified core instance system and adds a robust PDF processing command. The design focuses on clear UX, offline-friendliness, and safe configuration usage.

## Key Changes

- Core-backed queries: `query` command now retrieves a LightRAG instance via `lightrag.core` to ensure consistent configuration and lifecycle management.
- New `extract` command for PDFs:
  - Modes: `--mode dry-run|insert|full` (default `dry-run`).
    - dry-run: extract text/metadata only; no network or API keys required.
    - insert: extract + insert into storages; no extra reasoning.
    - full: full processing; requires configured LLM/Embedding backends.
  - Concurrency: `--max-concurrent` controls async processing.
  - Working dir override: `--output-dir` overrides `WORKING_DIR`.
- Optional Qwen embedding: auto-enabled when `QWEN_*` env vars are present; remains disabled otherwise (no hard dependency, lazy loaded).
- Security & DX: avoids printing secrets; removes unsafe defaults; cleans unused imports.

## Usage

- Query (stream on/off):
  - `python -m cli -q local "什么是RAG"`
  - `python -m cli -q local --no-stream "什么是RAG"`
- Extract PDFs:
  - `python -m cli extract --input-dir rag_pdfs --output-dir ./rag_storage --max-concurrent 2 --mode dry-run`
  - `python -m cli extract --input-dir rag_pdfs --mode insert`
  - With Qwen: set `QWEN_API_KEY`, `QWEN_EMBEDDING_MODEL`, `QWEN_EMBEDDING_HOST` then run `--mode full`.

## Testing Performed

- Local regression: `modes`, `version`, stream/non-stream query paths.
- PDF pipeline:
  - dry-run: processes N PDFs; prints success/failure stats; exit code 0 on full success.
  - insert: extracts + inserts with `ids=file_stem`, `file_paths=path`.
  - full (optional): with Qwen env configured, produces vector/status artifacts in working dir.
- Quality: `pre-commit run -a` (ruff format/check, misc hooks) passes.

## Notes

- The CLI now defers to `lightrag.core` for configuration; CLI options override `.env` where provided.
- Qwen is strictly optional; when not configured, the default embedding path is used.
