"""
LightRAG 转换工具

此模块提供从解析结果到统一模型的转换工具，负责补全缺失的
chunk_id、doc_id、metadata 等信息。
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import uuid
import hashlib

from .models import Chunk, Document, Entity, Relationship, RetrievalResult
from .context_parser import ParsedContext

logger = logging.getLogger(__name__)


@dataclass
class ConversionContext:
    """转换上下文，包含 LightRAG 实例和缓存信息"""
    rag_instance: Any = None
    doc_cache: Dict[str, Document] = None
    chunk_cache: Dict[str, Chunk] = None

    def __post_init__(self):
        if self.doc_cache is None:
            self.doc_cache = {}
        if self.chunk_cache is None:
            self.chunk_cache = {}


class ConversionService:
    """转换服务"""

    def __init__(self, rag_instance: Any = None):
        self.rag_instance = rag_instance
        self.doc_cache: Dict[str, Document] = {}
        self.chunk_cache: Dict[str, Chunk] = {}

    def set_rag_instance(self, rag_instance: Any):
        """设置 LightRAG 实例"""
        self.rag_instance = rag_instance
        self._clear_cache()

    def _clear_cache(self):
        """清空缓存"""
        self.doc_cache.clear()
        self.chunk_cache.clear()

    async def convert_to_retrieval_result(
        self,
        parsed_context: ParsedContext,
        query: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        将解析后的上下文转换为检索结果

        Args:
            parsed_context: 解析后的上下文
            query: 原始查询
            doc_id: 可选的文档ID
            metadata: 额外的元数据

        Returns:
            RetrievalResult: 转换后的检索结果
        """
        result = RetrievalResult(
            query=query,
            chunks=[],
            entities=[],
            relationships=[],
            context="",  # 可选择性地存储原始上下文
            metadata=metadata or {}
        )

        # 转换并补全 chunks
        for chunk in parsed_context.chunks:
            converted_chunk = await self._convert_chunk(chunk, doc_id)
            result.chunks.append(converted_chunk)

        # 转换 entities
        for entity in parsed_context.entities:
            converted_entity = self._convert_entity(entity)
            result.entities.append(converted_entity)

        # 转换 relationships
        for relationship in parsed_context.relationships:
            converted_relationship = self._convert_relationship(relationship)
            result.relationships.append(converted_relationship)

        # 更新统计信息
        result.total_results = len(result.chunks)
        result.metadata.update(parsed_context.metadata)

        return result

    async def _convert_chunk(self, chunk: Chunk, doc_id: Optional[str] = None) -> Chunk:
        """转换并补全 chunk 信息"""
        # 补全 doc_id
        if not chunk.doc_id and doc_id:
            chunk.doc_id = doc_id
        elif not chunk.doc_id:
            chunk.doc_id = await self._infer_doc_id_from_content(chunk.content)

        # 补全 chunk_id（如果需要）
        if not chunk.id or chunk.id.startswith("chunk_"):
            chunk.id = self._generate_chunk_id(chunk.content, chunk.doc_id, chunk.chunk_index)

        # 补充元数据
        chunk.metadata.update(await self._enrich_chunk_metadata(chunk))

        # 尝试获取相似度分数
        if chunk.similarity is None and self.rag_instance:
            chunk.similarity = await self._calculate_similarity(chunk, chunk.doc_id)

        return chunk

    def _convert_entity(self, entity: Entity) -> Entity:
        """转换并补全 entity 信息"""
        # 补全 entity ID
        if not entity.id or entity.id.startswith("entity_"):
            entity.id = self._generate_entity_id(entity.name, entity.type)

        # 补充元数据
        entity.metadata.update(self._enrich_entity_metadata(entity))

        return entity

    def _convert_relationship(self, relationship: Relationship) -> Relationship:
        """转换并补全 relationship 信息"""
        # 补全 relationship ID
        if not relationship.id or relationship.id.startswith("rel_"):
            relationship.id = self._generate_relationship_id(
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.relation_type
            )

        # 补充元数据
        relationship.metadata.update(self._enrich_relationship_metadata(relationship))

        return relationship

    async def _infer_doc_id_from_content(self, content: str) -> str:
        """从内容推断文档ID"""
        if not self.rag_instance:
            return f"doc_{uuid.uuid4().hex[:8]}"

        # 尝试从 LightRAG 实例获取文档信息
        try:
            if hasattr(self.rag_instance, 'doc_status'):
                try:
                    # 尝试异步调用
                    docs_result = await self.rag_instance.doc_status.get_docs_paginated_async(
                        page=1, page_size=1
                    )
                except AttributeError:
                    # 回退到同步方法
                    logger.warning("doc_status.get_docs_paginated_async not available, falling back to sync method")
                    docs_result = self.rag_instance.doc_status.get_docs_paginated(
                        page=1, page_size=1
                    )

                if docs_result and docs_result[0]:
                    return docs_result[0][0]  # 返回第一个文档ID
        except Exception as e:
            logger.warning(f"推断文档ID时发生错误: {e}")

        return f"doc_{uuid.uuid4().hex[:8]}"

    def _generate_chunk_id(self, content: str, doc_id: str, index: int) -> str:
        """生成唯一的 chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{index}_{content_hash}"

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """生成唯一的 entity ID"""
        name_hash = hashlib.md5(f"{name}:{entity_type}".encode()).hexdigest()[:8]
        return f"entity_{entity_type}_{name_hash}"

    def _generate_relationship_id(self, source_id: str, target_id: str, rel_type: str) -> str:
        """生成唯一的 relationship ID"""
        rel_hash = hashlib.md5(f"{source_id}:{target_id}:{rel_type}".encode()).hexdigest()[:8]
        return f"rel_{rel_type}_{rel_hash}"

    async def _enrich_chunk_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """补充 chunk 元数据"""
        enriched = {}

        if self.rag_instance:
            try:
                # 获取文档信息
                if chunk.doc_id and hasattr(self.rag_instance, 'doc_status'):
                    try:
                        # 尝试异步调用
                        doc_status = await self.rag_instance.doc_status.get_by_id_async(chunk.doc_id)
                    except AttributeError:
                        # 回退到同步方法
                        logger.warning("doc_status.get_by_id_async not available, falling back to sync method")
                        doc_status = self.rag_instance.doc_status.get_by_id(chunk.doc_id)

                    if doc_status:
                        enriched["doc_title"] = doc_status.get("title", "")
                        enriched["doc_status"] = doc_status.get("status", "unknown")

                # 获取文本块信息
                if hasattr(self.rag_instance, 'text_chunks'):
                    # 注意：这里需要根据实际的 text_chunks API 调整
                    # 当前简化处理，避免复杂的异步调用
                    pass

            except Exception as e:
                logger.warning(f"补充 chunk 元数据时发生错误: {e}")

        # 添加通用元数据
        enriched.update({
            "content_length": len(chunk.content),
            "word_count": len(chunk.content.split()),
            "has_entities": bool(chunk.metadata.get("entities", [])),
            "has_relationships": bool(chunk.metadata.get("relationships", []))
        })

        return enriched

    def _enrich_entity_metadata(self, entity: Entity) -> Dict[str, Any]:
        """补充 entity 元数据"""
        enriched = {}

        # 添加通用元数据
        enriched.update({
            "name_length": len(entity.name),
            "type_length": len(entity.type),
            "has_description": bool(entity.description)
        })

        return enriched

    def _enrich_relationship_metadata(self, relationship: Relationship) -> Dict[str, Any]:
        """补充 relationship 元数据"""
        enriched = {}

        # 添加通用元数据
        enriched.update({
            "relation_type_length": len(relationship.relation_type),
            "has_description": bool(relationship.description),
            "same_entity_type": relationship.source_entity_id.split("_")[1] == relationship.target_entity_id.split("_")[1]
        })

        return enriched

    async def _calculate_similarity(self, chunk: Chunk, doc_id: str) -> Optional[float]:
        """计算 chunk 的相似度分数"""
        if not self.rag_instance or not hasattr(self.rag_instance, 'chunks_vdb'):
            return None

        try:
            # 使用 chunks_vdb 计算相似度
            # 使用异步 API 获取嵌入向量
            try:
                embeddings = await self.rag_instance.embedding_func([chunk.content])
            except TypeError:
                # 如果嵌入函数不支持异步，回退到同步调用
                logger.warning("Embedding function is not async, falling back to sync call")
                embeddings = self.rag_instance.embedding_func([chunk.content])

            query_embedding = embeddings[0]

            # 搜索相似的 chunks
            try:
                search_results = await self.rag_instance.chunks_vdb.query(
                    chunk.content,
                    top_k=1
                )
            except AttributeError:
                # 如果不支持异步查询，回退到同步方法
                logger.warning("chunks_vdb.query is not async, falling back to sync call")
                search_results = self.rag_instance.chunks_vdb.query(
                    chunk.content,
                    top_k=1
                )

            if search_results:
                return float(search_results[0].get("score", 0.0))

        except Exception as e:
            logger.warning(f"计算相似度时发生错误: {e}")

        return None

    async def batch_convert(
        self,
        parsed_contexts: List[ParsedContext],
        queries: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[RetrievalResult]:
        """
        批量转换解析后的上下文

        Args:
            parsed_contexts: 解析后的上下文列表
            queries: 查询列表
            doc_ids: 可选的文档ID列表
            metadata_list: 额外的元数据列表

        Returns:
            List[RetrievalResult]: 转换后的检索结果列表
        """
        if len(parsed_contexts) != len(queries):
            raise ValueError("parsed_contexts 和 queries 的长度必须相同")

        if doc_ids and len(doc_ids) != len(parsed_contexts):
            raise ValueError("doc_ids 的长度必须与 parsed_contexts 相同")

        if metadata_list and len(metadata_list) != len(parsed_contexts):
            raise ValueError("metadata_list 的长度必须与 parsed_contexts 相同")

        results = []
        for i, (parsed_context, query) in enumerate(zip(parsed_contexts, queries)):
            doc_id = doc_ids[i] if doc_ids else None
            metadata = metadata_list[i] if metadata_list else None

            result = await self.convert_to_retrieval_result(
                parsed_context, query, doc_id, metadata
            )
            results.append(result)

        return results

    async def get_document_info(self, doc_id: str) -> Optional[Document]:
        """获取文档信息"""
        if not self.rag_instance or not doc_id:
            return None

        # 检查缓存
        if doc_id in self.doc_cache:
            return self.doc_cache[doc_id]

        try:
            if hasattr(self.rag_instance, 'doc_status'):
                try:
                    # 尝试异步调用
                    doc_status = await self.rag_instance.doc_status.get_by_id_async(doc_id)
                except AttributeError:
                    # 回退到同步方法
                    logger.warning("doc_status.get_by_id_async not available, falling back to sync method")
                    doc_status = self.rag_instance.doc_status.get_by_id(doc_id)

                if doc_status:
                    doc = Document(
                        id=doc_id,
                        title=doc_status.get("title", f"Document {doc_id}"),
                        description=doc_status.get("description", ""),
                        chunk_count=doc_status.get("chunks_count", 0),
                        metadata=doc_status
                    )
                    self.doc_cache[doc_id] = doc
                    return doc

        except Exception as e:
            logger.warning(f"获取文档信息时发生错误: {e}")

        return None