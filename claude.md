# Project: LightRAG

## Overview

LightRAG is a simple and fast Retrieval-Augmented Generation (RAG) system that combines knowledge graph construction with vector search for enhanced document understanding and querying. It extracts entities and relationships from documents to build a comprehensive knowledge graph, enabling sophisticated query capabilities beyond traditional RAG systems.

## Architecture

- **Core RAG Engine**: Combines vector search with knowledge graph traversal
- **Multi-Storage Support**: JSON, PostgreSQL, Redis, Neo4j, MongoDB, Milvus, Qdrant, Faiss
- **LLM Integration**: Supports OpenAI, Hugging Face, Ollama, and custom models
- **Async Processing**: Full asynchronous support for high-performance document processing
- **Web UI**: Built-in FastAPI server with visualization capabilities

## Technology Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI (Web UI), AsyncIO (Core)
- **Dependencies**: aiohttp, pydantic, networkx, numpy, pandas, nano-vectordb
- **Build System**: Setuptools with pyproject.toml
- **Storage**: Multiple database backends with unified interface

## Directory Structure

```
LightRAG/
├── lightrag/           # Core RAG engine → ./lightrag/claude.md
├── examples/           # Usage examples → ./examples/claude.md
├── tests/              # Test suite → ./tests/claude.md
├── reproduce/          # Reproducibility scripts → ./reproduce/claude.md
├── lightrag_webui/     # Web interface → ./lightrag_webui/claude.md
├── lightrag/api/       # REST API server → ./lightrag/api/claude.md
└── docs/               # Documentation → ./docs/claude.md
```

## Key Entry Points

- **Main Application**: `lightrag/lightrag.py:LightRAG`
- **API Endpoints**: `lightrag/api/routers/*.py`
- **LLM Integrations**: `lightrag/llm/*.py`
- **Storage Implementations**: `lightrag/kg/*_impl.py`
- **Knowledge Graph**: `lightrag/kg/shared_storage.py`

## Development Workflow

1. **Installation**: `pip install -e .` for development, `pip install lightrag-hku` for production
2. **Testing**: Run `pytest tests/` for unit tests
3. **API Server**: `lightrag-server` command launches the web interface
4. **Documentation**: See README.md for comprehensive usage guides

## Navigation Index

- **Core RAG Engine**: lightrag/lightrag.py
- **Entity Extraction**: lightrag/operate.py
- **Knowledge Graph Storage**: lightrag/kg/
- **LLM Integrations**: lightrag/llm/
- **API Routes**: lightrag/api/routers/
- **Web UI**: lightrag_webui/
- **Examples**: examples/
- **Testing**: tests/

## Key Patterns & Conventions

- **Async First**: All core operations are asynchronous
- **Dependency Injection**: LLM and embedding functions are injected at initialization
- **Multi-Storage**: Unified interface across different database backends
- **Modular Design**: Separate components for storage, processing, and querying
- **Extensible**: Easy to add new LLM providers and storage backends

## Performance Considerations

- **Entity Extraction**: CPU-intensive, benefits from parallel processing
- **Vector Search**: Memory-intensive, requires appropriate hardware
- **Knowledge Graph**: Scale considerations for large document sets
- **LLM Calls**: Major bottleneck, requires efficient batching and caching

## Document Indexing Mechanism

### 1. Multi-Storage Architecture

LightRAG uses a sophisticated 4-layer storage system:

- **KV Storage** (`BaseKVStorage`): Stores documents, text chunks, and LLM response cache
- **Vector Storage** (`BaseVectorStorage`): Stores embeddings for entities, relations, and chunks
- **Graph Storage** (`BaseGraphStorage`): Stores entity-relationship knowledge graph
- **Doc Status Storage** (`DocStatusStorage`): Tracks document processing state

### 2. Indexing Pipeline

1. **Document Ingestion**: `LightRAG.ainsert()` method accepts documents
2. **Chunking**: `chunking_by_token_size()` splits documents into manageable chunks
3. **Entity Extraction**: `extract_entities()` uses LLM to extract entities and relationships
4. **Knowledge Graph Construction**: Builds semantic relationships between entities
5. **Vector Embedding**: Generates embeddings for semantic search
6. **Multi-Storage Persistence**: Saves data to appropriate storage backends

### 3. Core Indexing Components

- **`LightRAG.ainsert()`**: Main entry point for document indexing
- **`apipeline_enqueue_documents()`**: Validates and enqueues documents for processing
- **`apipeline_process_enqueue_documents()`**: Processes queued documents through the pipeline
- **`extract_entities()`**: Core LLM-driven entity and relationship extraction
- **Storage Implementations**: Multiple backends (JSON, PostgreSQL, Neo4j, Milvus, etc.)

### 4. Key Technical Features

- **Async Processing**: Full asynchronous pipeline for high throughput
- **LLM Integration**: Uses LLMs for intelligent entity extraction
- **Smart Caching**: LLM response caching to reduce API calls
- **Incremental Processing**: Supports batch and incremental document addition
- **Namespace Isolation**: Data isolation between different LightRAG instances

### 5. Storage Implementation Details

Each storage type has multiple implementations:

- **KV Storage**: JSON, PostgreSQL, Redis, MongoDB
- **Vector Storage**: NanoVectorDB, PostgreSQL, Milvus, FAISS, Qdrant
- **Graph Storage**: NetworkX, Neo4j, PostgreSQL+AGE, Memgraph
- **Doc Status**: JSON, PostgreSQL, MongoDB

The system automatically handles storage initialization, data migration, and consistency maintenance across all storage layers.
