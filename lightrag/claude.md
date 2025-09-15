# LightRAG Core Module Documentation

## Module Purpose and Role

LightRAG is a sophisticated Retrieval-Augmented Generation (RAG) framework that combines knowledge graph construction, vector search, and large language models to create intelligent document processing and query systems. The core module provides:

- **Document Ingestion & Processing**: Automatic extraction of entities and relationships from documents
- **Knowledge Graph Construction**: Building semantic relationships between extracted entities
- **Multi-Modal Storage**: Support for multiple storage backends (KV, Vector, Graph)
- **Intelligent Querying**: Hybrid retrieval combining vector search and knowledge graph traversal
- **LLM Integration**: Unified interface for multiple LLM providers

## File Structure Overview

```
lightrag/
├── __init__.py              # Module initialization and main exports
├── base.py                  # Abstract base classes and core interfaces
├── lightrag.py             # Main LightRAG class and orchestration logic
├── namespace.py            # Namespace definitions for storage isolation
├── constants.py            # Configuration constants and defaults
├── types.py               # Pydantic models and type definitions
├── utils.py               # Utility functions and helper classes
├── utils_graph.py         # Graph-specific utilities
├── operate.py             # Core operational logic (chunking, extraction, querying)
├── prompt.py              # LLM prompt templates and configurations
├── rerank.py              # Result reranking functionality
├── exceptions.py          # Custom exception classes
│
├── api/                   # REST API implementation
│   ├── __init__.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── document_routes.py
│   │   ├── query_routes.py
│   │   ├── graph_routes.py
│   │   └── ollama_api.py
│   └── webui/             # Web interface assets
│
├── kg/                    # Knowledge Graph storage implementations
│   ├── __init__.py        # Storage registry and implementation mapping
│   ├── shared_storage.py  # Shared storage utilities
│   ├── json_kv_impl.py    # JSON-based KV storage
│   ├── json_doc_status_impl.py
│   ├── nano_vector_db_impl.py
│   ├── milvus_impl.py     # Milvus vector database
│   ├── faiss_impl.py      # FAISS vector index
│   ├── qdrant_impl.py     # Qdrant vector database
│   ├── redis_impl.py      # Redis storage
│   ├── postgres_impl.py   # PostgreSQL storage
│   ├── mongo_impl.py      # MongoDB storage
│   ├── neo4j_impl.py      # Neo4j graph database
│   ├── memgraph_impl.py   # Memgraph database
│   ├── age_impl.py        # Apache AGE (PostgreSQL extension)
│   └── networkx_impl.py   # NetworkX in-memory graph
│
├── llm/                   # LLM provider integrations
│   ├── __init__.py
│   ├── binding_options.py # Configuration options framework
│   ├── openai.py          # OpenAI API integration
│   ├── azure_openai.py    # Azure OpenAI integration
│   ├── anthropic.py       # Anthropic Claude integration
│   ├── ollama.py          # Ollama local models
│   ├── hf.py              # Hugging Face models
│   ├── lmdeploy.py        # LMDeploy inference engine
│   ├── lollms.py          # Local LLM server
│   ├── jina.py            # Jina AI models
│   ├── siliconcloud.py    # Silicon Cloud models
│   ├── zhipu.py           # Zhipu AI models
│   ├── nvidia_openai.py   # NVIDIA NIM integration
│   └── bedrock.py         # AWS Bedrock integration
│
└── tools/                 # Development and operational tools
    ├── __init__.py
    ├── check_initialization.py
    └── lightrag_visualizer/  # Knowledge graph visualization
```

## Core Components and Relationships

### 1. Storage Architecture

LightRAG uses a multi-storage architecture with four distinct storage types:

- **KV Storage**: Key-value storage for documents, chunks, and cache
- **Vector Storage**: Dense vector embeddings for semantic search
- **Graph Storage**: Knowledge graph for entity relationships
- **Doc Status Storage**: Document processing state tracking

### 2. Main Classes

#### LightRAG (lightrag.py)
The central orchestrator class that manages:
- Document ingestion and processing pipeline
- Storage initialization and management
- Query execution and result composition
- Configuration and state management

#### Base Storage Interfaces (base.py)
Abstract base classes defining the storage contract:
- `BaseKVStorage`: Key-value operations
- `BaseVectorStorage`: Vector search and embedding storage
- `BaseGraphStorage`: Graph operations and traversal
- `DocStatusStorage`: Document state tracking

### 3. Processing Pipeline

The document processing workflow:
1. **Chunking**: Split documents into manageable text chunks
2. **Entity Extraction**: Identify entities using LLM extraction
3. **Relationship Extraction**: Discover relationships between entities
4. **Knowledge Graph Construction**: Build semantic graph structure
5. **Vector Embedding**: Generate embeddings for semantic search
6. **Storage Persistence**: Save to appropriate storage backends

## Architectural Patterns

### 1. Plugin Architecture
LightRAG uses a plugin-based architecture for storage implementations:
- Storage providers are dynamically loaded based on configuration
- Each storage type has multiple implementations
- Environment-based configuration with fallback defaults

### 2. Namespace Isolation
All storage operations use namespaces to isolate:
- Different document collections
- Processing stages
- Cache layers
- Entity types

### 3. Async-First Design
- All operations are designed for asynchronous execution
- Batch processing with configurable parallelism
- Timeout management for LLM operations

### 4. Configuration Management
- Centralized constants with sensible defaults
- Environment variable support with type conversion
- Command-line argument integration
- Dynamic configuration validation

## Key Interfaces and Extension Points

### Storage Provider Interface
To add a new storage implementation:
1. Implement the appropriate base interface
2. Register in `kg/__init__.py` STORAGES mapping
3. Define environment requirements
4. Implement required methods

### LLM Provider Interface
To add a new LLM provider:
1. Create binding options class inheriting from `BindingOptions`
2. Implement provider-specific integration
3. Register in the LLM binding system

### Custom Processing Hooks
Extension points available:
- Custom chunking strategies
- Entity extraction post-processing
- Relationship validation rules
- Query result filtering

## Performance Considerations

### 1. Storage Backend Selection
- **Memory-constrained**: Use JSON/NetworkX for development
- **Production scale**: Use Milvus/PostgreSQL/Neo4j
- **High throughput**: Redis/MongoDB for KV operations
- **Graph complexity**: Neo4j/Memgraph for complex relationships

### 2. Batch Processing Configuration
- `DEFAULT_MAX_ASYNC`: Controls parallel processing (default: 4)
- `DEFAULT_MAX_PARALLEL_INSERT`: Batch insert operations (default: 2)
- `DEFAULT_EMBEDDING_BATCH_NUM`: Embedding computation batch size (default: 10)

### 3. Token Management
- `DEFAULT_MAX_TOTAL_TOKENS`: Context window management (default: 30000)
- `DEFAULT_MAX_ENTITY_TOKENS`: Entity description limits (default: 6000)
- `DEFAULT_MAX_RELATION_TOKENS`: Relationship context limits (default: 8000)

### 4. Query Optimization
- `DEFAULT_TOP_K`: Initial vector search results (default: 40)
- `DEFAULT_CHUNK_TOP_K`: Chunk filtering (default: 20)
- `DEFAULT_COSINE_THRESHOLD`: Similarity threshold (default: 0.2)
- `DEFAULT_RELATED_CHUNK_NUMBER`: Graph-based expansion (default: 5)

## Best Practices

### 1. Configuration Management
- Use environment variables for sensitive credentials
- Leverage `.env` files for local development
- Validate storage implementation compatibility

### 2. Document Processing
- Pre-process documents to remove noise
- Use appropriate chunking strategy for content type
- Monitor entity extraction quality

### 3. Query Optimization
- Choose appropriate query mode based on use case
- Use hybrid retrieval for complex queries
- Implement caching for frequent queries

### 4. Monitoring and Maintenance
- Monitor storage connection health
- Track document processing status
- Regular knowledge graph validation
- Embedding model performance monitoring

### 5. Scaling Considerations
- Distribute storage across dedicated servers
- Use connection pooling for database connections
- Implement load balancing for API endpoints
- Monitor memory usage for in-memory components

## Integration Patterns

### 1. API Integration
- RESTful endpoints for document management
- GraphQL-like query interface
- Ollama-compatible API for tool compatibility

### 2. Custom Application Integration
- Direct Python API for embedded usage
- Async/await support for web applications
- Event-driven processing pipelines

### 3. Data Export/Import
- Knowledge graph export formats
- Document collection backup/restore
- Cross-storage migration tools

## Development Guidelines

### 1. Adding New Features
- Follow existing patterns for consistency
- Use abstract base classes for interfaces
- Implement comprehensive error handling
- Include appropriate logging

### 2. Testing Strategy
- Unit tests for individual components
- Integration tests for storage implementations
- Performance testing for scaling validation
- End-to-end testing for complete workflows

### 3. Documentation Standards
- Type annotations for all public interfaces
- Docstrings following Google format
- Example usage in documentation
- Configuration examples

This architecture provides a flexible, scalable foundation for building intelligent document processing and query systems with strong separation of concerns and extensive customization capabilities.