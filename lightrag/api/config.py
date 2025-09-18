"""
Configs for the LightRAG API.
Enhanced with core configuration system integration.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from lightrag.utils import get_env_value
from lightrag.llm.binding_options import (
    OllamaEmbeddingOptions,
    OllamaLLMOptions,
    OpenAILLMOptions,
)
from lightrag.base import OllamaServerInfos
import sys

from lightrag.constants import (
    DEFAULT_WOKERS,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_MAX_ASYNC,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    DEFAULT_RERANK_BINDING,
    DEFAULT_ENTITY_TYPES,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


ollama_server_infos = OllamaServerInfos()


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "NetworkXStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


def get_default_host(binding_type: str) -> str:
    default_hosts = {
        "ollama": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
        "lollms": os.getenv("LLM_BINDING_HOST", "http://localhost:9600"),
        "azure_openai": os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        "openai": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        "qwen": os.getenv("QWEN_EMBEDDING_HOST", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"),
    }
    return default_hosts.get(
        binding_type, os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    )  # fallback to ollama if unknown


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Args:
        is_uvicorn_mode: Whether running under uvicorn mode

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="LightRAG API Server")

    # Server configuration
    parser.add_argument(
        "--host",
        default=get_env_value("HOST", "0.0.0.0"),
        help="Server host (default: from env or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_value("PORT", 9621, int),
        help="Server port (default: from env or 9621)",
    )

    # Directory configuration
    parser.add_argument(
        "--working-dir",
        default=get_env_value("WORKING_DIR", "./rag_storage"),
        help="Working directory for RAG storage (default: from env or ./rag_storage)",
    )
    parser.add_argument(
        "--input-dir",
        default=get_env_value("INPUT_DIR", "./inputs"),
        help="Directory containing input documents (default: from env or ./inputs)",
    )

    parser.add_argument(
        "--timeout",
        default=get_env_value("TIMEOUT", DEFAULT_TIMEOUT, int, special_none=True),
        type=int,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout",
    )

    # RAG configuration
    parser.add_argument(
        "--max-async",
        type=int,
        default=get_env_value("MAX_ASYNC", DEFAULT_MAX_ASYNC, int),
        help=f"Maximum async operations (default: from env or {DEFAULT_MAX_ASYNC})",
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=get_env_value("SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS, int),
        help=f"Maximum token size for entity/relation summary(default: from env or {DEFAULT_SUMMARY_MAX_TOKENS})",
    )
    parser.add_argument(
        "--summary-context-size",
        type=int,
        default=get_env_value(
            "SUMMARY_CONTEXT_SIZE", DEFAULT_SUMMARY_CONTEXT_SIZE, int
        ),
        help=f"LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_CONTEXT_SIZE})",
    )
    parser.add_argument(
        "--summary-length-recommended",
        type=int,
        default=get_env_value(
            "SUMMARY_LENGTH_RECOMMENDED", DEFAULT_SUMMARY_LENGTH_RECOMMENDED, int
        ),
        help=f"LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_LENGTH_RECOMMENDED})",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        default=get_env_value("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from env or INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=get_env_value("VERBOSE", False, bool),
        help="Enable verbose debug output(only valid for DEBUG log-level)",
    )

    parser.add_argument(
        "--key",
        type=str,
        default=get_env_value("LIGHTRAG_API_KEY", None),
        help="API key for authentication. This protects lightrag server against unauthorized access",
    )

    # Optional https parameters
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=get_env_value("SSL", False, bool),
        help="Enable HTTPS (default: from env or False)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=get_env_value("SSL_CERTFILE", None),
        help="Path to SSL certificate file (required if --ssl is enabled)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=get_env_value("SSL_KEYFILE", None),
        help="Path to SSL private key file (required if --ssl is enabled)",
    )

    # Ollama model configuration
    parser.add_argument(
        "--simulated-model-name",
        type=str,
        default=get_env_value("OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME),
        help="Name for the simulated Ollama model (default: from env or lightrag)",
    )

    parser.add_argument(
        "--simulated-model-tag",
        type=str,
        default=get_env_value("OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG),
        help="Tag for the simulated Ollama model (default: from env or latest)",
    )

    # Namespace
    parser.add_argument(
        "--workspace",
        type=str,
        default=get_env_value("WORKSPACE", ""),
        help="Default workspace for all storage",
    )

    parser.add_argument(
        "--auto-scan-at-startup",
        action="store_true",
        default=False,
        help="Enable automatic scanning when the program starts",
    )

    # Server workers configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=get_env_value("WORKERS", DEFAULT_WOKERS, int),
        help="Number of worker processes (default: from env or 1)",
    )

    # LLM and embedding bindings
    parser.add_argument(
        "--llm-binding",
        type=str,
        default=get_env_value("LLM_BINDING", "ollama"),
        choices=[
            "lollms",
            "ollama",
            "openai",
            "openai-ollama",
            "azure_openai",
            "aws_bedrock",
        ],
        help="LLM binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--embedding-binding",
        type=str,
        default=get_env_value("EMBEDDING_BINDING", "ollama"),
        choices=["lollms", "ollama", "openai", "azure_openai", "aws_bedrock", "jina", "qwen"],
        help="Embedding binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--rerank-binding",
        type=str,
        default=get_env_value("RERANK_BINDING", DEFAULT_RERANK_BINDING),
        choices=["null", "cohere", "jina", "aliyun"],
        help=f"Rerank binding type (default: from env or {DEFAULT_RERANK_BINDING})",
    )

    # Conditionally add binding options defined in binding_options module
    # This will add command line arguments for all binding options (e.g., --ollama-embedding-num_ctx)
    # and corresponding environment variables (e.g., OLLAMA_EMBEDDING_NUM_CTX)
    if "--llm-binding" in sys.argv:
        try:
            idx = sys.argv.index("--llm-binding")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "ollama":
                OllamaLLMOptions.add_args(parser)
        except IndexError:
            pass
    elif os.environ.get("LLM_BINDING") == "ollama":
        OllamaLLMOptions.add_args(parser)

    if "--embedding-binding" in sys.argv:
        try:
            idx = sys.argv.index("--embedding-binding")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "ollama":
                OllamaEmbeddingOptions.add_args(parser)
        except IndexError:
            pass
    elif os.environ.get("EMBEDDING_BINDING") == "ollama":
        OllamaEmbeddingOptions.add_args(parser)

    # Add OpenAI LLM options when llm-binding is openai or azure_openai
    if "--llm-binding" in sys.argv:
        try:
            idx = sys.argv.index("--llm-binding")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in [
                "openai",
                "azure_openai",
            ]:
                OpenAILLMOptions.add_args(parser)
        except IndexError:
            pass
    elif os.environ.get("LLM_BINDING") in ["openai", "azure_openai"]:
        OpenAILLMOptions.add_args(parser)

    args = parser.parse_args()

    # convert relative path to absolute path
    args.working_dir = os.path.abspath(args.working_dir)
    args.input_dir = os.path.abspath(args.input_dir)

    # Inject storage configuration from environment variables
    args.kv_storage = get_env_value(
        "LIGHTRAG_KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE
    )
    args.doc_status_storage = get_env_value(
        "LIGHTRAG_DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE
    )
    args.graph_storage = get_env_value(
        "LIGHTRAG_GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE
    )
    args.vector_storage = get_env_value(
        "LIGHTRAG_VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE
    )

    # Get MAX_PARALLEL_INSERT from environment
    args.max_parallel_insert = get_env_value("MAX_PARALLEL_INSERT", 2, int)

    # Get MAX_GRAPH_NODES from environment
    args.max_graph_nodes = get_env_value("MAX_GRAPH_NODES", 1000, int)

    # Handle openai-ollama special case
    if args.llm_binding == "openai-ollama":
        args.llm_binding = "openai"
        args.embedding_binding = "ollama"

    # Ollama ctx_num
    args.ollama_num_ctx = get_env_value("OLLAMA_NUM_CTX", 32768, int)

    args.llm_binding_host = get_env_value(
        "LLM_BINDING_HOST", get_default_host(args.llm_binding)
    )
    args.embedding_binding_host = get_env_value(
        "EMBEDDING_BINDING_HOST", get_default_host(args.embedding_binding)
    )
    args.llm_binding_api_key = get_env_value("LLM_BINDING_API_KEY", None)
    args.embedding_binding_api_key = get_env_value("EMBEDDING_BINDING_API_KEY", "")

    # Inject model configuration
    args.llm_model = get_env_value("LLM_MODEL", "mistral-nemo:latest")
    args.embedding_model = get_env_value("EMBEDDING_MODEL", "bge-m3:latest")
    args.embedding_dim = get_env_value("EMBEDDING_DIM", 1024, int)

    # Inject chunk configuration
    args.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
    args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)

    # Inject LLM cache configuration
    args.enable_llm_cache_for_extract = get_env_value(
        "ENABLE_LLM_CACHE_FOR_EXTRACT", True, bool
    )
    args.enable_llm_cache = get_env_value("ENABLE_LLM_CACHE", True, bool)

    # Select Document loading tool (DOCLING, DEFAULT)
    args.document_loading_engine = get_env_value("DOCUMENT_LOADING_ENGINE", "DEFAULT")

    # Add environment variables that were previously read directly
    args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    args.summary_language = get_env_value("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE)
    args.entity_types = get_env_value("ENTITY_TYPES", DEFAULT_ENTITY_TYPES, list)
    args.whitelist_paths = get_env_value("WHITELIST_PATHS", "/health,/api/*")

    # For JWT Auth
    args.auth_accounts = get_env_value("AUTH_ACCOUNTS", "")
    args.token_secret = get_env_value("TOKEN_SECRET", "lightrag-jwt-default-secret")
    args.token_expire_hours = get_env_value("TOKEN_EXPIRE_HOURS", 48, int)
    args.guest_token_expire_hours = get_env_value("GUEST_TOKEN_EXPIRE_HOURS", 24, int)
    args.jwt_algorithm = get_env_value("JWT_ALGORITHM", "HS256")

    # Rerank model configuration
    args.rerank_model = get_env_value("RERANK_MODEL", None)
    args.rerank_binding_host = get_env_value("RERANK_BINDING_HOST", None)
    args.rerank_binding_api_key = get_env_value("RERANK_BINDING_API_KEY", None)
    # Note: rerank_binding is already set by argparse, no need to override from env

    # Min rerank score configuration
    args.min_rerank_score = get_env_value(
        "MIN_RERANK_SCORE", DEFAULT_MIN_RERANK_SCORE, float
    )

    # Query configuration
    args.history_turns = get_env_value("HISTORY_TURNS", DEFAULT_HISTORY_TURNS, int)
    args.top_k = get_env_value("TOP_K", DEFAULT_TOP_K, int)
    args.chunk_top_k = get_env_value("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K, int)
    args.max_entity_tokens = get_env_value(
        "MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS, int
    )
    args.max_relation_tokens = get_env_value(
        "MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS, int
    )
    args.max_total_tokens = get_env_value(
        "MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS, int
    )
    args.cosine_threshold = get_env_value(
        "COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, float
    )
    args.related_chunk_number = get_env_value(
        "RELATED_CHUNK_NUMBER", DEFAULT_RELATED_CHUNK_NUMBER, int
    )

    # Add missing environment variables for health endpoint
    args.force_llm_summary_on_merge = get_env_value(
        "FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
    )
    args.embedding_func_max_async = get_env_value(
        "EMBEDDING_FUNC_MAX_ASYNC", DEFAULT_EMBEDDING_FUNC_MAX_ASYNC, int
    )
    args.embedding_batch_num = get_env_value(
        "EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM, int
    )

    ollama_server_infos.LIGHTRAG_NAME = args.simulated_model_name
    ollama_server_infos.LIGHTRAG_TAG = args.simulated_model_tag

    return args


def update_uvicorn_mode_config():
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    if global_args.workers > 1:
        original_workers = global_args.workers
        global_args.workers = 1
        # Log warning directly here
        logging.warning(
            f">> Forcing workers=1 in uvicorn mode(Ignoring workers={original_workers})"
        )


global_args = parse_args()


# Core Configuration Integration
# These functions provide integration with the new core configuration system
# while maintaining backward compatibility

def get_api_core_config():
    """
    Get core configuration that's compatible with API usage.
    This integrates the core config system with the existing API config.

    Returns:
        dict: Core configuration that can be used with the instance manager
    """
    try:
        from lightrag.core.config import load_core_config
        core_config = load_core_config()

        # Override with API-specific values from global_args
        api_config = {
            "instance_name": get_env_value("WORKSPACE", "default"),
            "auto_init": False,  # API handles initialization separately
            "working_dir": global_args.working_dir,
            "input_dir": global_args.input_dir,
            "workspace": global_args.workspace,
            "entity_types": global_args.entity_types,
            "summary_language": global_args.summary_language,
            "prompts_json_path": get_env_value("PROMPTS_JSON_PATH", None),

            # Storage configuration
            "kv_storage": global_args.kv_storage,
            "vector_storage": global_args.vector_storage,
            "graph_storage": global_args.graph_storage,
            "doc_status_storage": global_args.doc_status_storage,

            # Custom configuration from environment
            "custom_config": {}
        }

        # Add custom LIGHTRAG_ prefixed environment variables
        env_prefix = "LIGHTRAG_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix) and key not in [
                "LIGHTRAG_INSTANCE_NAME",
                "LIGHTRAG_AUTO_INIT",
                "LIGHTRAG_KV_STORAGE",
                "LIGHTRAG_VECTOR_STORAGE",
                "LIGHTRAG_GRAPH_STORAGE",
                "LIGHTRAG_DOC_STATUS_STORAGE"
            ]:
                config_key = key[len(env_prefix):].lower()
                try:
                    import json
                    api_config["custom_config"][config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    api_config["custom_config"][config_key] = value

        return api_config

    except ImportError:
        # If core config is not available, return basic config
        logging.warning("Core configuration system not available, using basic config")
        return {
            "instance_name": "api_server",
            "auto_init": False,
            "working_dir": global_args.working_dir,
            "input_dir": global_args.input_dir,
            "workspace": global_args.workspace,
            "entity_types": global_args.entity_types,
            "summary_language": global_args.summary_language,
            "prompts_json_path": None,
            "custom_config": {}
        }


def get_api_instance_config():
    """
    Get LightRAG configuration for API server instance creation.
    This provides the configuration needed to create a LightRAG instance
    that's compatible with the API server's requirements.

    Returns:
        dict: Configuration for LightRAG instance creation
    """
    core_config = get_api_core_config()

    # Convert to LightRAG constructor format
    instance_config = {
        "working_dir": core_config["working_dir"],
        "input_dir": core_config["input_dir"],
        "workspace": core_config["workspace"],
        "entity_types": core_config["entity_types"],
        "summary_language": core_config["summary_language"],

        # Storage configuration
        "kv_storage": core_config["kv_storage"],
        "vector_storage": core_config["vector_storage"],
        "graph_storage": core_config["graph_storage"],
        "doc_status_storage": core_config["doc_status_storage"],

        # Additional parameters from global_args
        "chunk_token_size": int(global_args.chunk_size),
        "chunk_overlap_token_size": int(global_args.chunk_overlap_size),
        "max_async": global_args.max_async,
        "summary_max_tokens": global_args.summary_max_tokens,
        "summary_context_size": global_args.summary_context_size,
        "enable_llm_cache_for_entity_extract": global_args.enable_llm_cache_for_extract,
        "enable_llm_cache": global_args.enable_llm_cache,
        "max_parallel_insert": global_args.max_parallel_insert,
        "max_graph_nodes": global_args.max_graph_nodes,

        # Addon parameters
        "addon_params": {
            "language": core_config["summary_language"],
            "entity_types": core_config["entity_types"],
        }
    }

    # Add custom configuration
    instance_config["addon_params"].update(core_config["custom_config"])

    return instance_config


def register_api_instance(rag_instance):
    """
    Register the API server's LightRAG instance with the global instance manager.
    This allows external components to access the same instance.

    Args:
        rag_instance: The LightRAG instance created by the API server
    """
    try:
        from lightrag.core import set_instance

        core_config = get_api_core_config()
        import asyncio

        # Create async function to register instance
        async def register_instance():
            await set_instance(
                name=core_config["instance_name"],
                instance=rag_instance
            )
            logging.info(f"Registered API instance as '{core_config['instance_name']}'")

        # Run in current event loop or create new one
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the registration
            asyncio.create_task(register_instance())
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(register_instance())

    except ImportError:
        logging.warning("Core instance manager not available, skipping API instance registration")
    except Exception as e:
        logging.error(f"Failed to register API instance: {e}")


def get_api_prompts_config():
    """
    Get prompts configuration for the API server.
    Returns the prompts JSON path if configured, or None.

    Returns:
        str or None: Path to prompts JSON file, or None if not configured
    """
    try:
        core_config = get_api_core_config()
        return core_config.get("prompts_json_path")
    except Exception as e:
        logging.warning(f"Failed to get prompts config: {e}")
        return None
