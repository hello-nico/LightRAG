"""
LightRAG DeerFlow 集成实现

此模块实现了 DeerFlow 适配器，利用 LightRAG 实例完成检索、
相似度计算、资源列举等功能。
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lightrag.base import QueryParam
from lightrag.core import get_instance_names, get_lightrag_instance
from lightrag.kg.shared_storage import initialize_pipeline_status

from .models import (
    RetrievalResult,
    RetrievalRequest,
    Chunk,
    Entity,
    Relationship
)

from lightrag.utils import logger


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
        extra: dict | None = None,
    ):
        self.id = id
        self.url = url
        self.title = title
        self.chunks = chunks or []
        self.extra = extra or {}

    def add_chunk(self, content: str, similarity: float | None = None) -> None:
        """追加一个转换后的 chunk"""
        self.chunks.append(DeerFlowChunk(content=content, similarity=similarity or 0.0))


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

    def _convert_to_chunk(self, chunk_data: dict, index: int) -> Chunk:
        """将结构化块数据转换为Chunk对象"""
        return Chunk(
            id=str(chunk_data.get("id", f"chunk_{index}")),
            doc_id=chunk_data.get("file_path", ""),
            content=chunk_data.get("content", ""),
            chunk_index=chunk_data.get("chunk_index", index),
            score=chunk_data.get("score"),
            similarity=chunk_data.get("similarity"),
        )

    @staticmethod
    def _build_document_identifier(chunk: Chunk) -> str:
        return chunk.doc_id or chunk.id

    @staticmethod
    def _build_document_title(doc_identifier: str) -> str:
        path = Path(doc_identifier)
        title = path.stem if path.stem else doc_identifier
        return title or doc_identifier

    @staticmethod
    def _build_document_url(doc_identifier: str, chunk: Chunk, instance_name: str | None) -> str | None:
        raw_path = chunk.doc_id or ""
        if raw_path and "://" in raw_path:
            return raw_path
        if raw_path:
            normalized = raw_path.replace("\\", "/")
        else:
            normalized = doc_identifier.replace("\\", "/")
        if instance_name:
            normalized_path = normalized.lstrip("/")
            if normalized_path:
                return f"lightrag://{instance_name}/{normalized_path}"
            return f"lightrag://{instance_name}"
        return normalized or None

    @staticmethod
    def _chunk_similarity(chunk: Chunk) -> float:
        if chunk.similarity is not None:
            return float(chunk.similarity)
        if chunk.score is not None:
            return float(chunk.score)
        return 0.0

    def _convert_to_entity(self, entity_data: dict) -> Entity:
        """将结构化实体数据转换为Entity对象"""
        return Entity(
            id=str(entity_data.get("id", "")),
            entity=entity_data.get("entity", ""),
            type=entity_data.get("type", ""),
            description=entity_data.get("description"),
        )

    def _convert_to_relationship(self, rel_data: dict) -> Relationship:
        """将结构化关系数据转换为Relationship对象"""
        return Relationship(
            id=str(rel_data.get("id", "")),
            source_entity_id=str(rel_data.get("source_entity_id", "")),
            target_entity_id=str(rel_data.get("target_entity_id", "")),
            description=rel_data.get("description"),
        )
    
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
            param_kwargs["return_structured"] = True  # 使用结构化数据返回

            query_param = QueryParam(**param_kwargs)

            start_time = time.perf_counter()
            raw_response = await rag_instance.aquery(request.query, param=query_param)

            # 结构化数据格式
            limited_chunks = [
                self._convert_to_chunk(chunk_data, idx)
                for idx, chunk_data in enumerate(raw_response["chunks"][:top_k])
            ]
            entities = [
                self._convert_to_entity(entity_data)
                for entity_data in raw_response["entities"]
            ]
            relationships = [
                self._convert_to_relationship(rel_data)
                for rel_data in raw_response["relationships"]
            ]
            context_for_response = raw_response

            retrieval_time = time.perf_counter() - start_time

            metadata: Dict[str, Any] = {
                "instance": instance_name,
                "mode": query_param.mode,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "retrieved_chunks": len(limited_chunks),
                "structured_data": isinstance(context_for_response, dict),
            }

            return RetrievalResult(
                query=request.query,
                chunks=limited_chunks,
                entities=entities,
                relationships=relationships,
                context=context_for_response,
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
        if not resources:
            logger.warning("No resources available for query")
            return []
        try:
            retrieval_request = RetrievalRequest(
                query=query,
                max_results=self.max_results,
            )

            target_resource = resources[0]
            instance_name = target_resource.uri.split("://")[1]
            retrieval_result = await self.retrieve(instance_name, retrieval_request)

            documents = self._group_chunks_to_documents(
                retrieval_result.chunks, instance_name
            )
            limited_documents = documents[: self.max_results]
            logger.info(
                "Found %s documents with %s total chunks",
                len(limited_documents),
                sum(len(doc.chunks) for doc in limited_documents),
            )
            return limited_documents

        except Exception as e:
            logger.error(f"Query relevant documents failed: {e}")
            raise Exception(f"Failed to query relevant documents: {e}")

    def _group_chunks_to_documents(
        self, chunks: List[Chunk], instance_name: str | None
    ) -> List[DeerFlowDocument]:
        documents: List[DeerFlowDocument] = []
        documents_map: Dict[str, DeerFlowDocument] = {}

        for chunk in chunks:
            doc_identifier = self._build_document_identifier(chunk)
            if not doc_identifier:
                continue

            document = documents_map.get(doc_identifier)
            if document is None:
                url = self._build_document_url(doc_identifier, chunk, instance_name)
                title = self._build_document_title(doc_identifier)
                source_file = chunk.doc_id or ""
                extra = {
                    "source": {
                        "doc_id": doc_identifier,
                        "file_path": source_file,
                        "instance": instance_name,
                        "chunk_ids": [],
                    }
                }
                document = DeerFlowDocument(
                    id=doc_identifier,
                    url=url,
                    title=title,
                    extra=extra,
                )
                documents_map[doc_identifier] = document
                documents.append(document)

            document.add_chunk(
                content=chunk.content,
                similarity=self._chunk_similarity(chunk),
            )
            chunk_id_list = document.extra.setdefault("source", {}).setdefault(
                "chunk_ids", []
            )
            if chunk.id not in chunk_id_list:
                chunk_id_list.append(chunk.id)

        return documents
