# LightRAG API Documentation

## Overview

LightRAG provides a comprehensive REST API built on FastAPI for Retrieval-Augmented Generation with knowledge graph capabilities. This documentation covers the API architecture, endpoints, authentication, and integration patterns.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [API Endpoints](#api-endpoints)
- [Authentication & Security](#authentication--security)
- [Web UI Integration](#web-ui-integration)
- [Configuration & Deployment](#configuration--deployment)
- [Integration Patterns](#integration-patterns)
- [Development Guide](#development-guide)

## Architecture Overview

### Core Components

```python
# Main API server structure
lightrag/api/
├── __init__.py          # Module initialization
├── lightrag_server.py   # Main FastAPI server
├── config.py            # Configuration management
├── auth.py              # Authentication handlers
├── utils_api.py         # Utility functions
└── routers/             # Modular API routers
    ├── document_routes.py
    ├── query_routes.py
    ├── graph_routes.py
    └── ollama_api.py
```

### FastAPI Framework

The API is built on FastAPI with:
- Automatic OpenAPI/Swagger documentation
- Pydantic models for request/response validation
- Async/await support for high concurrency
- Dependency injection for authentication
- Streaming responses for real-time processing

### Storage Backends

LightRAG supports multiple storage backends:
- **Vector Storage**: NanoVectorDB, PostgreSQL, MongoDB, Redis, ChromaDB
- **Graph Storage**: NetworkX, Neo4j, RedisGraph, ArangoDB
- **KV Storage**: JSON, Redis, MongoDB, PostgreSQL
- **Document Status**: JSON, Redis, MongoDB

## API Endpoints

### Query Endpoints

#### POST /api/v1/query
Perform RAG queries with multiple modes:

```python
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The query text")
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(default="mix")
    only_need_context: Optional[bool] = Field(default=None)
    only_need_prompt: Optional[bool] = Field(default=None)
    response_type: Optional[str] = Field(default="Multiple Paragraphs")
    stream: Optional[bool] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    chunk_top_k: Optional[int] = Field(default=None)
    max_entity_tokens: Optional[int] = Field(default=None)
    max_relation_tokens: Optional[int] = Field(default=None)
    max_total_tokens: Optional[int] = Field(default=None)
    hl_keywords: Optional[List[str]] = Field(default=None)
    ll_keywords: Optional[List[str]] = Field(default=None)
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None)
    history_turns: Optional[int] = Field(default=None)
    model_func: Optional[Callable] = Field(default=None)
    user_prompt: Optional[str] = Field(default=None)
    enable_rerank: Optional[bool] = Field(default=None)
```

**Query Modes**:
- `local`: Context-dependent information retrieval
- `global`: Global knowledge utilization
- `hybrid`: Combined local and global retrieval
- `naive`: Basic search without advanced techniques
- `mix`: Knowledge graph + vector retrieval (default)
- `bypass`: Direct LLM without knowledge retrieval

#### GET /api/v1/query/stream
Streaming query endpoint for real-time responses.

### Document Management Endpoints

#### POST /api/v1/documents
Upload and process documents:

```python
class DocumentUploadRequest(BaseModel):
    content: str = Field(..., description="Document content")
    id: Optional[str] = Field(None, description="Custom document ID")
    file_path: Optional[str] = Field(None, description="File path for citation")
    track_id: Optional[str] = Field(None, description="Tracking ID for monitoring")
```

#### GET /api/v1/documents
List documents with pagination and filtering:

```python
class DocumentListRequest(BaseModel):
    status: Optional[DocStatus] = Field(None, description="Filter by status")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=10, le=200, description="Items per page")
    sort_field: str = Field("updated_at", description="Field to sort by")
    sort_direction: str = Field("desc", description="Sort direction")
```

#### DELETE /api/v1/documents/{doc_id}
Delete document and all associated data.

#### GET /api/v1/documents/status
Get document processing status counts.

### Knowledge Graph Endpoints

#### GET /api/v1/graph
Retrieve knowledge graph for a specific entity:

```python
class GraphRequest(BaseModel):
    node_label: str = Field(..., description="Entity label to query")
    max_depth: int = Field(3, ge=1, le=10, description="Graph traversal depth")
    max_nodes: int = Field(1000, ge=10, le=5000, description="Maximum nodes to return")
```

#### Entity Management
- `POST /api/v1/graph/entities`: Create new entity
- `GET /api/v1/graph/entities/{entity_name}`: Get entity info
- `PUT /api/v1/graph/entities/{entity_name}`: Edit entity
- `DELETE /api/v1/graph/entities/{entity_name}`: Delete entity

#### Relation Management
- `POST /api/v1/graph/relations`: Create new relation
- `GET /api/v1/graph/relations`: Get relation info
- `PUT /api/v1/graph/relations`: Edit relation
- `DELETE /api/v1/graph/relations`: Delete relation

#### POST /api/v1/graph/entities/merge
Merge multiple entities into one:

```python
class EntityMergeRequest(BaseModel):
    source_entities: List[str] = Field(..., description="Entities to merge")
    target_entity: str = Field(..., description="Target entity name")
    merge_strategy: Optional[Dict[str, str]] = Field(None, description="Merge strategy")
    target_entity_data: Optional[Dict[str, Any]] = Field(None, description="Target data")
```

### Ollama-Compatible Endpoints

LightRAG provides full Ollama API compatibility:

- `POST /api/chat`: Ollama chat completion
- `POST /api/generate`: Ollama text generation
- `GET /api/tags`: List available models
- `GET /api/version`: Get server version

```python
class OllamaChatRequest(BaseModel):
    model: str = Field(..., description="Model name")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    stream: bool = Field(False, description="Stream response")
    options: Optional[Dict[str, Any]] = Field(None, description="Model options")
```

## Authentication & Security

### Authentication Methods

#### JWT Authentication
```python
class AuthHandler:
    def create_token(self, username: str, role: str = "user") -> str:
        """Create JWT token with role-based expiration"""
        payload = {
            "sub": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(
                hours=24 if role == "admin" else 1
            ),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
```

#### API Key Authentication
```python
class APIKeyAuth:
    def __init__(self, api_keys: List[str] = None):
        self.valid_keys = set(api_keys or [])
    
    def validate_key(self, api_key: str) -> bool:
        return api_key in self.valid_keys
```

#### IP Whitelisting
```python
class IPWhitelist:
    def __init__(self, allowed_ips: List[str] = None):
        self.allowed_ips = set(allowed_ips or ["127.0.0.1", "::1"])
    
    def validate_ip(self, client_host: str) -> bool:
        return client_host in self.allowed_ips
```

### Security Configuration

Environment variables for security:
```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256

# API Key Configuration
API_KEYS=key1,key2,key3

# IP Whitelist
ALLOWED_IPS=192.168.1.0/24,10.0.0.1

# CORS Settings
CORS_ORIGINS=https://example.com,http://localhost:3000
```

### Role-Based Access Control

- **Admin**: Full access to all endpoints
- **User**: Read/write access to queries and documents
- **Read-only**: Query access only

## Web UI Integration

### Built-in Web Interface

LightRAG includes a comprehensive web UI with:
- Document management and upload
- Real-time query interface
- Knowledge graph visualization
- Processing status monitoring
- System configuration

### API Integration Points

#### Frontend Configuration
```javascript
// LightRAG JavaScript client
const lightrag = new LightRAGClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key',
  // Auto-retry and error handling
  retry: 3,
  timeout: 30000,
});
```

#### Real-time Updates
WebSocket endpoints for real-time notifications:
- Document processing status
- Query completion events
- System health metrics

### Custom UI Development

Example React component for query interface:
```jsx
function QueryInterface() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [streaming, setStreaming] = useState(false);

  const handleQuery = async () => {
    setStreaming(true);
    const stream = await lightrag.queryStream(query);
    
    for await (const chunk of stream) {
      setResponse(prev => prev + chunk);
    }
    setStreaming(false);
  };

  return (
    <div>
      <textarea value={query} onChange={(e) => setQuery(e.target.value)} />
      <button onClick={handleQuery} disabled={streaming}>
        {streaming ? 'Processing...' : 'Query'}
      </button>
      <div>{response}</div>
    </div>
  );
}
```

## Configuration & Deployment

### Configuration Sources

LightRAG supports multiple configuration sources:

1. **Environment Variables** (highest priority)
2. **.env File** (project directory)
3. **config.ini** (legacy support)
4. **Command Line Arguments**
5. **Default Values**

### Key Configuration Options

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
LOG_LEVEL=INFO

# Storage Configuration
KV_STORAGE=JsonKVStorage
VECTOR_STORAGE=NanoVectorDBStorage
GRAPH_STORAGE=NetworkXStorage
DOC_STATUS_STORAGE=JsonDocStatusStorage

# LLM Configuration
LLM_MODEL_NAME=gpt-4o-mini
LLM_MAX_ASYNC=8
LLM_TIMEOUT=120

# Embedding Configuration
EMBEDDING_BATCH_NUM=10
EMBEDDING_FUNC_MAX_ASYNC=8
EMBEDDING_TIMEOUT=60

# Processing Configuration
MAX_PARALLEL_INSERT=4
MAX_GRAPH_NODES=1000
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
```

### Deployment Options

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "lightrag.api.lightrag_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Systemd Service
```ini
[Unit]
Description=LightRAG API Server
After=network.target

[Service]
User=lightrag
Group=lightrag
WorkingDirectory=/opt/lightrag
Environment=PYTHONPATH=/opt/lightrag
ExecStart=/usr/local/bin/uvicorn lightrag.api.lightrag_server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: lightrag
        image: lightrag:latest
        ports:
        - containerPort: 8000
        env:
        - name: WORKERS
          value: "2"
        - name: LOG_LEVEL
          value: "INFO"
```

### Monitoring & Health Checks

#### Health Endpoints
- `GET /health`: Basic health check
- `GET /health/detailed`: Detailed system health
- `GET /metrics`: Prometheus metrics

#### Logging Configuration
```python
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'lightrag.log',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
})
```

## Integration Patterns

### Ollama Compatibility

LightRAG fully implements the Ollama API specification:

```python
# Complete Ollama compatibility
class OllamaAPI:
    async def chat(self, request: OllamaChatRequest):
        """Ollama-compatible chat endpoint"""
        # Implementation matches Ollama's response format
        return {
            "model": request.model,
            "created_at": datetime.utcnow().isoformat(),
            "message": {"role": "assistant", "content": response},
            "done": True
        }
```

### LLM Backend Integration

Supported LLM providers:
- OpenAI (GPT models)
- Ollama (local models)
- Azure OpenAI
- AWS Bedrock
- Custom model endpoints

### Storage Backend Integration

#### PostgreSQL Example
```python
from lightrag.kg.postgres_kv_impl import PostgresKVStorage
from lightrag.kg.postgres_vector_impl import PostgresVectorStorage

# Configure PostgreSQL storage
rag = LightRAG(
    kv_storage="PostgresKVStorage",
    vector_storage="PostgresVectorStorage",
    graph_storage="NetworkXStorage",  # Graph in memory, vectors in PostgreSQL
    doc_status_storage="PostgresDocStatusStorage"
)
```

#### MongoDB Example
```python
from lightrag.kg.mongodb_kv_impl import MongoDBKVStorage
from lightrag.kg.mongodb_vector_impl import MongoDBVectorStorage

# Configure MongoDB storage
rag = LightRAG(
    kv_storage="MongoDBKVStorage",
    vector_storage="MongoDBVectorStorage",
    graph_storage="NetworkXStorage",
    doc_status_storage="MongoDBDocStatusStorage"
)
```

### Custom Embedding Functions

```python
def custom_embedding_function(texts: List[str]) -> List[List[float]]:
    """Custom embedding function implementation"""
    # Your embedding logic here
    return embeddings

# Use custom embedding function
rag = LightRAG(embedding_func=custom_embedding_function)
```

### Webhook Integration

```python
# Webhook configuration for external integrations
WEBHOOK_URLS = os.getenv("WEBHOOK_URLS", "").split(",")

async def send_webhook(event_type: str, data: dict):
    """Send webhook notifications for important events"""
    for url in WEBHOOK_URLS:
        if url:
            async with httpx.AsyncClient() as client:
                await client.post(url, json={
                    "event": event_type,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                })
```

## Development Guide

### Extending the API

#### Adding New Endpoints

1. Create a new router module:
```python
# routers/custom_routes.py
from fastapi import APIRouter, Depends
from .auth import get_current_user

router = APIRouter(prefix="/api/v1/custom")

@router.get("/endpoint")
async def custom_endpoint(user: dict = Depends(get_current_user)):
    return {"message": "Custom endpoint", "user": user}
```

2. Register the router in the main server:
```python
# lightrag_server.py
from .routers.custom_routes import router as custom_router

app.include_router(custom_router)
```

#### Custom Authentication

```python
# auth/custom_auth.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def custom_api_key_auth(api_key: str = Security(api_key_header)):
    if not validate_custom_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"user_id": "custom_user"}
```

### Testing the API

#### Unit Tests
```python
# test_api.py
from fastapi.testclient import TestClient
from lightrag.api.lightrag_server import app

client = TestClient(app)

def test_query_endpoint():
    response = client.post("/api/v1/query", json={
        "query": "test query",
        "mode": "mix"
    })
    assert response.status_code == 200
    assert "response" in response.json()
```

#### Integration Tests
```python
# test_integration.py
import pytest
from lightrag import LightRAG

@pytest.mark.asyncio
async def test_document_processing():
    rag = LightRAG()
    await rag.initialize_storages()
    
    track_id = await rag.ainsert("Test document content")
    
    # Wait for processing to complete
    await asyncio.sleep(2)
    
    status = await rag.get_processing_status()
    assert status["processed"] > 0
```

### Performance Optimization

#### Caching Strategies
```python
# Enable LLM response caching
rag = LightRAG(enable_llm_cache=True)

# Enable embedding cache
rag = LightRAG(embedding_cache_config={
    "enabled": True,
    "similarity_threshold": 0.95,
    "use_llm_check": False
})
```

#### Batch Processing
```python
# Batch document insertion
documents = ["doc1 content", "doc2 content", "doc3 content"]
track_id = await rag.ainsert(documents)

# Batch query processing
queries = ["query1", "query2", "query3"]
results = await asyncio.gather(*[
    rag.aquery(query) for query in queries
])
```

### Error Handling

#### Custom Exception Handlers
```python
# exception_handlers.py
from fastapi import Request
from fastapi.responses import JSONResponse

async def custom_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "request_id": request.state.request_id
        }
    )

# Register handler
app.add_exception_handler(Exception, custom_exception_handler)
```

#### Validation Error Responses
```python
# Custom validation error format
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
            "body": exc.body
        }
    )
```

## Conclusion

The LightRAG API provides a comprehensive, production-ready interface for RAG systems with extensive customization options, multiple authentication methods, and support for various storage backends. Its Ollama compatibility makes it easy to integrate with existing AI chat interfaces, while the modular architecture allows for easy extension and customization.

For more detailed information, refer to the auto-generated OpenAPI documentation at `/docs` when the server is running.