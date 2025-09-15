import sys
import os

if sys.version_info < (3, 9):
    pass
else:
    pass
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("aiohttp"):
    pm.install("aiohttp")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

import numpy as np
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    logger,
)
from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def qwen_embedding(
    texts: list[str],
    model: str = "text-embedding-v3",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
    api_key: str = None,
    dimensions: int = 1536,
    **kwargs
) -> np.ndarray:
    """Generate embeddings for a list of texts using Qwen's API.

    Args:
        texts: List of texts to embed.
        model: The embedding model to use (default: text-embedding-v3).
        base_url: The base URL for the Qwen API.
        api_key: Qwen API key. If None, uses the QWEN_API_KEY environment variable.
        dimensions: The embedding dimensions (default: 1536 for text-embedding-v3).
        **kwargs: Additional parameters to pass to the API.

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        ValueError: If API key is not provided or invalid.
        Exception: If API call fails.
    """
    if api_key:
        os.environ["QWEN_API_KEY"] = api_key

    if "QWEN_API_KEY" not in os.environ:
        raise ValueError("QWEN_API_KEY environment variable is required")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['QWEN_API_KEY']}",
    }

    # Convert single text to list if needed
    if isinstance(texts, str):
        texts = [texts]

    # Prepare payload
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
        "dimensions": dimensions,
    }

    # Add additional kwargs to payload
    payload.update(kwargs)

    logger.debug(
        f"Qwen embedding request: {len(texts)} texts, model: {model}, dimensions: {dimensions}"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    
                    # Check if the error response is HTML
                    content_type = response.headers.get("content-type", "").lower()
                    is_html_error = (
                        error_text.strip().startswith("<!DOCTYPE html>")
                        or "text/html" in content_type
                    )

                    if is_html_error:
                        # Provide clean error messages for HTML error pages
                        if response.status == 502:
                            clean_error = "Bad Gateway (502) - Qwen service temporarily unavailable. Please try again in a few minutes."
                        elif response.status == 503:
                            clean_error = "Service Unavailable (503) - Qwen service is temporarily overloaded. Please try again later."
                        elif response.status == 504:
                            clean_error = "Gateway Timeout (504) - Qwen service request timed out. Please try again."
                        else:
                            clean_error = f"HTTP {response.status} - Qwen service error. Please try again later."
                    else:
                        clean_error = error_text

                    logger.error(f"Qwen API error {response.status}: {clean_error}")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Qwen API error: {clean_error}",
                    )

                response_json = await response.json()
                data_list = response_json.get("data", [])

                if not data_list:
                    logger.error("Qwen API returned empty data list")
                    raise ValueError("Qwen API returned empty data list")

                if len(data_list) != len(texts):
                    logger.error(
                        f"Qwen API returned {len(data_list)} embeddings for {len(texts)} texts"
                    )
                    raise ValueError(
                        f"Qwen API returned {len(data_list)} embeddings for {len(texts)} texts"
                    )

                # Extract embeddings from response
                embeddings = []
                for item in data_list:
                    embedding = item.get("embedding")
                    if embedding:
                        embeddings.append(embedding)

                if len(embeddings) != len(texts):
                    logger.error(
                        f"Only {len(embeddings)} valid embeddings found for {len(texts)} texts"
                    )
                    raise ValueError(
                        f"Only {len(embeddings)} valid embeddings found for {len(texts)} texts"
                    )

                embeddings_array = np.array(embeddings)
                logger.debug(f"Qwen embeddings generated: shape {embeddings_array.shape}")

                return embeddings_array

    except Exception as e:
        logger.error(f"Qwen embedding error: {e}")
        raise


@wrap_embedding_func_with_attrs(embedding_dim=1536)
async def qwen_embedding_func(
    texts: list[str],
    model: str = "text-embedding-v3",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
    api_key: str = None,
    dimensions: int = 1536,
    **kwargs
) -> np.ndarray:
    """Qwen embedding function with standard LightRAG interface.
    
    This function is decorated with wrap_embedding_func_with_attrs to ensure
    compatibility with LightRAG's embedding system.
    """
    return await qwen_embedding(
        texts=texts,
        model=model,
        base_url=base_url,
        api_key=api_key,
        dimensions=dimensions,
        **kwargs
    )