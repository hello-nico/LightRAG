"""
LightRAG DeerFlow 集成实现

此模块实现了 DeerFlow 适配器，利用 LightRAG 实例完成检索、
相似度计算、资源列举等功能。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lightrag.base import QueryParam
from lightrag.core import get_instance_names, get_lightrag_instance
from lightrag.kg.shared_storage import initialize_pipeline_status

from .models import (
    RetrievalResult,
    RetrievalRequest
)
from .context_parser import ContextParser

logger = logging.getLogger(__name__)


class ResourceRequest(BaseModel):
    """ResourceRequest is a class that represents a resource request."""
    query: str = Field(..., description="The query string")


class ResourceResponse(BaseModel):
    """ResourceResponse is a class that represents a resource response."""
    resources: List[Dict[str, Any]] = Field(..., description="The resources")


class DeerFlowChunk:
    """DeerFlow 标准的 Chunk 类"""

    def __init__(self, content: str, similarity: float):
        self.content = content
        self.similarity = similarity


class DeerFlowDocument:
    """DeerFlow 标准的 Document 类"""

    def __init__(
        self,
        id: str,
        url: str | None = None,
        title: str | None = None,
        chunks: List[DeerFlowChunk] = None,
    ):
        self.id = id
        self.url = url
        self.title = title
        self.chunks = chunks or []


class DeerFlowResource(BaseModel):
    """DeerFlow 标准的 Resource 类"""

    uri: str = Field(..., description="The URI of the resource")
    title: str = Field(..., description="The title of the resource")
    description: Optional[str] = Field("", description="The description of the resource")


class DeerFlowRetriever:
    """DeerFlow 标准的检索器接口"""

    def __init__(self, similarity_threshold: float = 0.5, default_mode: str = "mix", max_results: int = 20):
        
        self.similarity_threshold = similarity_threshold
        self.default_mode = default_mode
        self.max_results = max_results
        self.rag_instance = None
    
    async def list_resources(self, query: str | None = None) -> List[DeerFlowResource]:
        """
        列出可用资源LightRAG实例

        Args:
            query: 查询字符串

        Returns:
            List[DeerFlowResource]: 资源列表
        """
        self.rag_instance = await get_lightrag_instance(auto_init=True)
        instance_names = await get_instance_names()
        return [DeerFlowResource(uri=f"lightrag://{instance_name}", title=instance_name) for instance_name in instance_names]
        
    async def retrieve(self, instance_name: str, request: RetrievalRequest) -> RetrievalResult:
        """
        执行检索
        """
        try:
            rag_instance = await get_lightrag_instance(instance_name)
            await rag_instance.initialize_storages()
            await initialize_pipeline_status()
            top_k = request.max_results
            param_kwargs = {}
            param_kwargs.setdefault("mode", self.default_mode)
            param_kwargs.setdefault("top_k", top_k * 2)
            param_kwargs.setdefault("chunk_top_k", top_k)
            param_kwargs["only_need_context"] = True

            query_param = QueryParam(**param_kwargs)

            start_time = time.perf_counter()
            raw_response = await rag_instance.aquery(request.query, param=query_param)

            if hasattr(raw_response, "__aiter__"):
                parts: List[str] = []
                async for chunk in raw_response:
                    parts.append(chunk)
                context_text = "".join(parts)
            else:
                context_text = raw_response or ""
            
            parsed_context = ContextParser().parse(context_text)

            limited_chunks = [
                chunk.model_copy(update={"chunk_index": idx})
                for idx, chunk in enumerate(parsed_context.chunks[:top_k])
            ]

            retrieval_time = time.perf_counter() - start_time

            metadata: Dict[str, Any] = {
                "instance": instance_name,
                "mode": query_param.mode,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "retrieved_chunks": len(limited_chunks),
            }

            return RetrievalResult(
                query=request.query,
                chunks=limited_chunks,
                entities=parsed_context.entities,
                relationships=parsed_context.relationships,
                context=context_text,
                metadata=metadata,
                total_results=len(limited_chunks),
                retrieval_time=retrieval_time,
            )

        except Exception as exc:
            logger.error(f"Retrieval failed: {exc}")
            raise Exception(f"Failed to retrieve results: {exc}") from exc

    async def query_relevant_documents(
        self, query: str, resources: List[DeerFlowResource] = None
    ) -> List[DeerFlowDocument]:
        """
        DeerFlow 标准接口：文档查询

        Args:
            query: 查询字符串
            resources: 可选的资源列表，用于过滤

        Returns:
            List[DeerFlowDocument]: DeerFlow 文档列表
        """
        logger.info(f"Querying relevant documents: {query}")
        if resources is None:
            resources = await self.list_resources()
        try:
            retrieval_request = RetrievalRequest(
                query=query,
                max_results=self.max_results,
                min_score=self.similarity_threshold
            )

            retrieval_result = await self.retrieve(resources[0].uri.split("://")[1], retrieval_request)

            logger.info(f"Found {len(retrieval_result.chunks)} relevant documents")
            return retrieval_result.chunks

        except Exception as e:
            logger.error(f"Query relevant documents failed: {e}")
            raise Exception(f"Failed to query relevant documents: {e}")
