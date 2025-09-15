# LightRAG Examples Guide

This guide provides a comprehensive overview of the LightRAG examples directory, helping users understand and utilize the various demonstration patterns and use cases.

## Overview

The examples directory contains demonstration scripts showcasing LightRAG's capabilities across different LLM providers, storage backends, and advanced features. These examples serve as practical starting points for implementing RAG systems.

## Example Types and Purposes

### 1. Core LLM Provider Examples

**OpenAI Integration**

- `lightrag_openai_demo.py`: Basic OpenAI integration with GPT-4o-mini
- Demonstrates all query modes (naive, local, global, hybrid)
- Includes logging configuration and cleanup procedures

**Azure OpenAI Integration**

- `lightrag_azure_openai_demo.py`: Azure-specific OpenAI configuration
- Shows custom LLM and embedding function implementations
- Supports separate deployments for chat and embedding models

**Gemini Integration**

- `lightrag_gemini_demo.py`: Google Gemini integration
- Uses SentenceTransformer for local embeddings
- Demonstrates prompt construction for Gemini API

**Ollama Integration**

- `lightrag_ollama_demo.py`: Local Ollama model deployment
- Supports streaming responses
- Configurable model names and embedding dimensions

### 2. Specialized Provider Examples

**Hugging Face Models**

- `lightrag_hf_demo.py`: Local Hugging Face model integration
- Uses transformers library for model loading
- Supports custom tokenizers and embedding models

**Amazon Bedrock**

- `lightrag_bedrock_demo.py`: AWS Bedrock integration
- Demonstrates Anthropic Claude models
- Includes boto3 configuration and logging setup

**OpenAI-Compatible APIs**

- `lightrag_openai_compatible_demo.py`: Generic OpenAI-compatible endpoints
- Supports DeepSeek, TogetherAI, and other providers
- Configurable base URLs and API keys

### 3. Storage Backend Examples

**MongoDB Graph Storage**

- `lightrag_openai_mongodb_graph_demo.py`: MongoDB integration for graph storage
- Demonstrates external database configuration
- Environment variable-based configuration

**Custom Knowledge Graph Insertion**

- `insert_custom_kg.py`: Manual knowledge graph creation
- Shows structured entity and relationship definition
- Useful for domain-specific knowledge injection

### 4. Advanced Feature Examples

**Reranking Integration**

- `rerank_example.py`: Cohere reranking functionality
- Demonstrates per-query rerank control
- Shows direct rerank API usage

**Multimodal Processing (RAGAnything)**

- `raganything_example.py`: Multimodal document processing
- Supports images, tables, and equations
- Integration with MinerU parser

**Direct Modal Processors**

- `modalprocessors_example.py`: Individual modal processor usage
- Image, table, and equation processing examples
- Standalone processor initialization

### 5. Visualization Examples

**HTML Graph Visualization**

- `graph_visual_with_html.py`: Interactive graph visualization
- Uses Pyvis and NetworkX libraries
- Generates interactive HTML visualizations

**Neo4j Integration**

- `graph_visual_with_neo4j.py`: Neo4j graph database export
- Converts GraphML to Neo4j format
- Batch processing for large graphs

## Key Demonstration Patterns

### Basic RAG Setup Pattern

```python
# 1. Import required modules
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# 2. Configure working directory
WORKING_DIR = "./data"

# 3. Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
)

# 4. Insert documents
rag.insert("Your document text here")

# 5. Query with different modes
result = rag.query("Your question", param=QueryParam(mode="hybrid"))
```

### Custom LLM Function Pattern

```python
async def custom_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    # Your custom LLM implementation
    return response

async def custom_embed_func(texts):
    # Your custom embedding implementation
    return embeddings
```

### Environment Configuration Pattern

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

# Access environment variables
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")
base_url = os.getenv("BASE_URL")
```

## Integration Examples

### LLM Provider Integration Matrix

| Provider | Example File | Key Features |
|----------|-------------|-------------|
| OpenAI | `lightrag_openai_demo.py` | Standard API, all query modes |
| Azure OpenAI | `lightrag_azure_openai_demo.py` | Azure endpoints, separate deployments |
| Google Gemini | `lightrag_gemini_demo.py` | Gemini API, local embeddings |
| Ollama | `lightrag_ollama_demo.py` | Local models, streaming support |
| Hugging Face | `lightrag_hf_demo.py` | Local models, transformers library |
| Amazon Bedrock | `lightrag_bedrock_demo.py` | AWS service, Claude models |
| OpenAI-compatible | `lightrag_openai_compatible_demo.py` | Generic endpoints, multiple providers |

### Storage Backend Configuration

**File-based Storage (Default)**

- JSON files for chunks, entities, relationships
- GraphML for knowledge graph
- Simple and portable

**MongoDB Storage**

- Configure with `graph_storage="MongoGraphStorage"`
- Set MongoDB connection string via environment variables
- Suitable for production deployments

**PostgreSQL Storage**

- Configure with appropriate storage class
- Requires database setup and connection string

## Advanced Usage Patterns

### Streaming Responses

```python
# Enable streaming in query parameters
param = QueryParam(mode="hybrid", stream=True)

# Handle streaming response
async for chunk in rag.aquery("question", param=param):
    print(chunk, end="", flush=True)
```

### Multimodal Processing

```python
from raganything import RAGAnything, RAGAnythingConfig

# Configure multimodal processing
config = RAGAnythingConfig(
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True
)

# Process multimodal content
await rag.process_document_complete(file_path, output_dir)
```

### Custom Knowledge Graph

```python
custom_kg = {
    "entities": [
        {
            "entity_name": "CustomEntity",
            "entity_type": "Type",
            "description": "Description",
            "source_id": "Source"
        }
    ],
    "relationships": [
        {
            "src_id": "Entity1",
            "tgt_id": "Entity2",
            "description": "Relationship description",
            "keywords": "key, words",
            "weight": 1.0
        }
    ]
}

rag.insert_custom_kg(custom_kg)
```

## Running and Modifying Examples

### Prerequisites

1. Python 3.10+
2. Required dependencies: `pip install lightrag-hku`
3. Environment variables for API keys
4. Optional: `.env` file for configuration

### Basic Execution

```bash
# Set environment variables
export OPENAI_API_KEY="your-key"

# Run example
python examples/lightrag_openai_demo.py
```

### Customization Steps

1. **Copy the example** to a new file
2. **Modify configuration** (API keys, model names)
3. **Adjust working directory** for your data
4. **Customize LLM/embedding functions** if needed
5. **Add your documents** for processing

### Common Modifications

**Changing LLM Provider**

```python
# From OpenAI to Ollama
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=ollama_embed
    )
)
```

**Adding Custom Processing**

```python
# Add pre-processing hook
async def custom_preprocess(text):
    # Your custom processing logic
    return processed_text

# Use in insertion
processed_text = await custom_preprocess(raw_text)
rag.insert(processed_text)
```

## Best Practices

### 1. Environment Management

- Use `.env` files for sensitive configuration
- Set appropriate logging levels
- Configure working directories properly

### 2. Error Handling

- Implement proper exception handling
- Validate API responses
- Handle rate limiting and retries

### 3. Performance Optimization

- Use appropriate batch sizes
- Cache embeddings where possible
- Monitor token usage and costs

### 4. Monitoring and Logging

- Configure comprehensive logging
- Monitor API usage and costs
- Track processing performance

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure environment variables are set
2. **Import Errors**: Check package installation and dependencies
3. **Memory Issues**: Reduce batch sizes or use smaller models
4. **Network Errors**: Verify API endpoints and connectivity

### Debug Mode

Enable verbose debugging:

```python
from lightrag.utils import set_verbose_debug
set_verbose_debug(True)
```

## Next Steps

1. **Start with** `lightrag_openai_demo.py` for basic functionality
2. **Explore provider-specific** examples for your preferred LLM
3. **Experiment with** advanced features like reranking and multimodal
4. **Customize** for your specific use case and data
5. **Deploy** with appropriate storage backends for production

For more detailed information, refer to the main LightRAG documentation and individual example files.
