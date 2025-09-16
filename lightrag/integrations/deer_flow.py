"""
LightRAG DeerFlow 集成实现

此模块实现了 DeerFlow 适配器，利用 LightRAG 实例完成检索、
相似度计算、资源列举等功能。
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from .base import BaseIntegration, IntegrationError, ResourceNotFoundError
from .models import (
    Resource,
    Document,
    Chunk,
    RetrievalResult,
    RetrievalRequest,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    ListResourcesRequest,
    ListResourcesResponse,
    ResourceType,
)
from .context_parser import ContextParser, ParsedContext
from .converters import ConversionService

logger = logging.getLogger(__name__)


class DeerFlowIntegration(BaseIntegration):
    """DeerFlow 集成实现"""

    version = "1.0.0"
    description = "LightRAG DeerFlow integration for retrieval operations"
    capabilities = ["retrieve", "batch_retrieve", "list_resources", "similarity_calculation"]

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 DeerFlow 集成

        Args:
            config: 集成配置，必须包含 rag_instance
        """
        super().__init__(config)
        self.rag_instance: Optional[Any] = None
        self.context_parser = ContextParser()
        self.conversion_service = ConversionService()
        self.similarity_threshold = self.get_config_value("similarity_threshold", 0.5)
        self.default_mode = self.get_config_value("default_mode", "mix")
        self.max_results = self.get_config_value("max_results", 10)

    async def initialize(self) -> None:
        """初始化集成实例"""
        if self.is_initialized:
            return

        logger.info("Initializing DeerFlow integration")

        # 验证配置
        errors = await self.validate_config()
        if errors:
            raise IntegrationError(f"Configuration validation failed: {errors}")

        # 获取 LightRAG 实例
        self.rag_instance = self.get_config_value("rag_instance")
        if not self.rag_instance:
            raise IntegrationError("rag_instance is required in configuration")

        # 设置转换服务的 LightRAG 实例
        self.conversion_service.set_rag_instance(self.rag_instance)

        self.is_initialized = True
        logger.info("DeerFlow integration initialized successfully")

    async def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []

        if "rag_instance" not in self.config:
            errors.append("rag_instance is required")

        if "similarity_threshold" in self.config:
            threshold = self.config["similarity_threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                errors.append("similarity_threshold must be a number between 0 and 1")

        if "default_mode" in self.config:
            mode = self.config["default_mode"]
            valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
            if mode not in valid_modes:
                errors.append(f"default_mode must be one of {valid_modes}")

        if "max_results" in self.config:
            max_results = self.config["max_results"]
            if not isinstance(max_results, int) or max_results <= 0:
                errors.append("max_results must be a positive integer")

        return errors

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        执行单次检索

        Args:
            request: 检索请求

        Returns:
            RetrievalResult: 检索结果
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        logger.info(f"Executing retrieval query: {request.query}")

        try:
            # 设置查询参数
            from lightrag.base import QueryParam

            query_param = QueryParam(
                mode=self.default_mode,
                only_need_context=True,  # 只返回上下文，不生成回答
                top_k=min(request.max_results, self.max_results),
                chunk_top_k=min(request.max_results, self.max_results),
            )

            # 应用过滤条件
            if request.min_score is not None:
                query_param.cosine_better_than_threshold = request.min_score

            # 执行查询
            context = await self.rag_instance.aquery(request.query, param=query_param)

            # 解析上下文
            parsed_context = self.context_parser.parse(context)

            # 转换为检索结果
            result = await self.conversion_service.convert_to_retrieval_result(
                parsed_context,
                request.query,
                metadata=request.filters if request.filters else {}
            )

            # 计算检索时间
            result.retrieval_time = time.time() - start_time

            # 应用分数过滤
            if request.min_score is not None:
                result.chunks = [
                    chunk for chunk in result.chunks
                    if (chunk.score or 0) >= request.min_score and
                       (chunk.similarity or 0) >= request.min_score
                ]
                result.total_results = len(result.chunks)

            logger.info(f"Retrieval completed: {result.total_results} chunks, {len(result.entities)} entities, {len(result.relationships)} relationships")
            return result

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise IntegrationError(f"Retrieval operation failed: {e}")

    async def batch_retrieve(self, request: BatchRetrievalRequest) -> BatchRetrievalResponse:
        """
        执行批量检索

        Args:
            request: 批量检索请求

        Returns:
            BatchRetrievalResponse: 批量检索结果
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"Executing batch retrieval for {len(request.queries)} queries")

        try:
            # 并发执行多个检索
            tasks = []
            for query in request.queries:
                retrieval_request = RetrievalRequest(
                    query=query,
                    max_results=request.max_results_per_query,
                    min_score=request.min_score,
                    include_metadata=request.include_metadata,
                    filters=request.filters
                )
                tasks.append(self.retrieve(retrieval_request))

            # 等待所有检索完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            successful_results = []
            errors = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Query {i} failed: {result}")
                    errors.append({
                        "query_index": i,
                        "query": request.queries[i],
                        "error": str(result)
                    })
                else:
                    successful_results.append(result)

            response = BatchRetrievalResponse(
                results=successful_results,
                metadata={
                    "total_queries": len(request.queries),
                    "successful_queries": len(successful_results),
                    "failed_queries": len(errors),
                    "errors": errors
                }
            )

            logger.info(f"Batch retrieval completed: {len(successful_results)} successful, {len(errors)} failed")
            return response

        except Exception as e:
            logger.error(f"Batch retrieval failed: {e}")
            raise IntegrationError(f"Batch retrieval operation failed: {e}")

    async def list_resources(self, request: ListResourcesRequest) -> ListResourcesResponse:
        """
        列出可用资源

        Args:
            request: 列出资源请求

        Returns:
            ListResourcesResponse: 资源列表
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info(f"Listing resources (type: {request.resource_type})")

        try:
            resources = []

            # 获取文档资源
            if request.resource_type is None or request.resource_type == ResourceType.DOCUMENT:
                if hasattr(self.rag_instance, 'doc_status'):
                    try:
                        # 使用异步 API 获取文档列表
                        docs_result = await self.rag_instance.doc_status.get_docs_paginated_async(
                            page=1, page_size=1000  # 获取较大数量用于过滤
                        )
                        if docs_result and docs_result[0]:
                            for doc_id, doc_info in docs_result[0]:
                                doc = Document(
                                    id=doc_id,
                                    title=doc_info.get("title", f"Document {doc_id}"),
                                    description=doc_info.get("description", ""),
                                    chunk_count=doc_info.get("chunks_count", 0),
                                    file_path=doc_info.get("file_path", ""),
                                    metadata=doc_info
                                )
                                resources.append(doc)
                    except AttributeError:
                        # 回退到同步方法
                        logger.warning("doc_status.get_docs_paginated_async not available, falling back to sync method")
                        docs_result = self.rag_instance.doc_status.get_docs_paginated(
                            page=1, page_size=1000
                        )
                        if docs_result and docs_result[0]:
                            for doc_id, doc_info in docs_result[0]:
                                doc = Document(
                                    id=doc_id,
                                    title=doc_info.get("title", f"Document {doc_id}"),
                                    description=doc_info.get("description", ""),
                                    chunk_count=doc_info.get("chunks_count", 0),
                                    file_path=doc_info.get("file_path", ""),
                                    metadata=doc_info
                                )
                                resources.append(doc)

            # 获取文本块资源（简化处理，避免复杂的异步调用）
            # 由于 text_chunks 的异步调用较为复杂，这里暂时跳过
            # 可以在后续版本中完善

            # 应用过滤条件
            if request.filters:
                filtered_resources = []
                for resource in resources:
                    match = True
                    for key, value in request.filters.items():
                        if key in resource.metadata and resource.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_resources.append(resource)
                resources = filtered_resources

            # 应用分页
            total_count = len(resources)
            start_offset = request.offset
            end_offset = start_offset + request.limit
            paginated_resources = resources[start_offset:end_offset]

            response = ListResourcesResponse(
                resources=paginated_resources,
                total_count=total_count,
                has_more=end_offset < total_count
            )

            logger.info(f"Listed {len(paginated_resources)} resources (total: {total_count})")
            return response

        except Exception as e:
            logger.error(f"Failed to list resources: {e}")
            raise IntegrationError(f"Failed to list resources: {e}")

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """
        获取指定资源

        Args:
            resource_id: 资源ID

        Returns:
            Optional[Resource]: 资源对象，如果不存在则返回None
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # 尝试从文档状态中查找
            if hasattr(self.rag_instance, 'doc_status'):
                try:
                    doc_info = await self.rag_instance.doc_status.get_by_id_async(resource_id)
                    if doc_info:
                        return Document(
                            id=resource_id,
                            title=doc_info.get("title", f"Document {resource_id}"),
                            description=doc_info.get("description", ""),
                            chunk_count=doc_info.get("chunks_count", 0),
                            file_path=doc_info.get("file_path", ""),
                            metadata=doc_info
                        )
                except AttributeError:
                    # 回退到同步方法
                    logger.warning("doc_status.get_by_id_async not available, falling back to sync method")
                    doc_info = self.rag_instance.doc_status.get_by_id(resource_id)
                    if doc_info:
                        return Document(
                            id=resource_id,
                            title=doc_info.get("title", f"Document {resource_id}"),
                            description=doc_info.get("description", ""),
                            chunk_count=doc_info.get("chunks_count", 0),
                            file_path=doc_info.get("file_path", ""),
                            metadata=doc_info
                        )

            # 文本块查找暂时跳过，因为需要更复杂的异步调用
            # 可以在后续版本中完善

            logger.warning(f"Resource not found: {resource_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get resource {resource_id}: {e}")
            raise IntegrationError(f"Failed to get resource {resource_id}: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "message": "Integration is not initialized"
            }

        try:
            # 检查 LightRAG 实例是否可用
            if not self.rag_instance:
                return {
                    "status": "error",
                    "message": "LightRAG instance is not available"
                }

            # 检查必要的方法是否存在
            required_methods = ['aquery', 'doc_status']
            missing_methods = []
            for method in required_methods:
                if not hasattr(self.rag_instance, method):
                    missing_methods.append(method)

            if missing_methods:
                return {
                    "status": "error",
                    "message": f"Missing required methods: {missing_methods}"
                }

            # 执行简单的测试查询
            test_result = await self.retrieve(RetrievalRequest(query="test"))

            return {
                "status": "healthy",
                "message": "Integration is functioning properly",
                "details": {
                    "rag_instance_available": True,
                    "required_methods_present": len(missing_methods) == 0,
                    "test_query_success": True,
                    "test_result_count": test_result.total_results,
                    "config": {
                        "similarity_threshold": self.similarity_threshold,
                        "default_mode": self.default_mode,
                        "max_results": self.max_results
                    }
                }
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "message": f"Health check failed: {e}"
            }

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            float: 相似度分数（0-1）
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # 使用 LightRAG 的嵌入函数计算相似度
            if hasattr(self.rag_instance, 'embedding_func'):
                embedding_func = self.rag_instance.embedding_func
                try:
                    # 尝试异步调用
                    embeddings = await embedding_func([text1, text2])
                except TypeError:
                    # 如果嵌入函数不支持异步，回退到同步调用
                    logger.warning("Embedding function is not async, falling back to sync call")
                    embeddings = embedding_func([text1, text2])

                # 计算余弦相似度
                import numpy as np
                vec1 = np.array(embeddings[0])
                vec2 = np.array(embeddings[1])

                cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                return float(cosine_similarity)
            else:
                logger.warning("Embedding function not available, using simple text similarity")
                # 简单的文本相似度计算
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0.0

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            raise IntegrationError(f"Similarity calculation failed: {e}")