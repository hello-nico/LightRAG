"""
LightRAG 检索集成 API 路由

此模块提供了统一的检索接口，用于与外部系统集成。
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.integrations import (
    get_global_integration_manager,
    RetrievalRequest,
    RetrievalResult,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    ListResourcesRequest,
    ListResourcesResponse,
    Resource,
    ResourceType,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["retrieval"])

# 请求/响应模型
class RetrievalQueryRequest(BaseModel):
    """检索查询请求"""
    query: str = Field(..., min_length=1, description="检索查询文本")
    max_results: int = Field(10, ge=1, le=100, description="最大结果数")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="最小分数阈值")
    include_metadata: bool = Field(True, description="是否包含元数据")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    integration_name: Optional[str] = Field(None, description="集成名称，留空使用默认集成")

class RetrievalQueryResponse(BaseModel):
    """检索查询响应"""
    success: bool = Field(..., description="是否成功")
    result: Optional[RetrievalResult] = Field(None, description="检索结果")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

class BatchRetrievalQueryRequest(BaseModel):
    """批量检索查询请求"""
    queries: List[str] = Field(..., min_items=1, max_items=100, description="查询列表")
    max_results_per_query: int = Field(10, ge=1, le=100, description="每个查询的最大结果数")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="最小分数阈值")
    include_metadata: bool = Field(True, description="是否包含元数据")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    integration_name: Optional[str] = Field(None, description="集成名称，留空使用默认集成")

class BatchRetrievalQueryResponse(BaseModel):
    """批量检索查询响应"""
    success: bool = Field(..., description="是否成功")
    result: Optional[BatchRetrievalResponse] = Field(None, description="批量检索结果")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

class ListResourcesQueryRequest(BaseModel):
    """列出资源请求"""
    resource_type: Optional[ResourceType] = Field(None, description="资源类型过滤")
    limit: int = Field(100, ge=1, le=1000, description="限制数量")
    offset: int = Field(0, ge=0, description="偏移量")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    integration_name: Optional[str] = Field(None, description="集成名称，留空使用默认集成")

class ListResourcesQueryResponse(BaseModel):
    """列出资源响应"""
    success: bool = Field(..., description="是否成功")
    result: Optional[ListResourcesResponse] = Field(None, description="资源列表")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

class GetResourceRequest(BaseModel):
    """获取资源请求"""
    resource_id: str = Field(..., description="资源ID")
    integration_name: Optional[str] = Field(None, description="集成名称，留空使用默认集成")

class GetResourceResponse(BaseModel):
    """获取资源响应"""
    success: bool = Field(..., description="是否成功")
    result: Optional[Resource] = Field(None, description="资源对象")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    success: bool = Field(..., description="是否成功")
    result: Dict[str, Any] = Field(..., description="健康检查结果")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

class SimilarityRequest(BaseModel):
    """相似度计算请求"""
    text1: str = Field(..., min_length=1, description="第一个文本")
    text2: str = Field(..., min_length=1, description="第二个文本")
    integration_name: Optional[str] = Field(None, description="集成名称，留空使用默认集成")

class SimilarityResponse(BaseModel):
    """相似度计算响应"""
    success: bool = Field(..., description="是否成功")
    result: Optional[float] = Field(None, description="相似度分数（0-1）")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

def create_retrieval_routes(rag, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    # 工具函数
    async def get_integration_manager():
        """获取并初始化集成管理器"""
        try:
            manager = get_global_integration_manager()
            if not manager.is_initialized:
                await manager.initialize()
            return manager
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get integration manager: {e}")

    async def handle_integration_error(func_name: str, error: Exception) -> str:
        """处理集成错误"""
        logger.error(f"Error in {func_name}: {error}")
        return f"Integration error in {func_name}: {error}"

    @router.post("/retrieve", response_model=RetrievalQueryResponse, dependencies=[Depends(combined_auth)])
    async def retrieve(
        request: RetrievalQueryRequest,
        manager=Depends(get_integration_manager)
    ):
        """
        执行单次检索
        """
        start_time = time.time()

        try:
            # 构建检索请求
            retrieval_request = RetrievalRequest(
                query=request.query,
                max_results=request.max_results,
                min_score=request.min_score,
                include_metadata=request.include_metadata,
                filters=request.filters
            )

            # 执行检索
            result = await manager.retrieve(retrieval_request, request.integration_name)

            return RetrievalQueryResponse(
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = await handle_integration_error("retrieve", e)
            return RetrievalQueryResponse(
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time
            )

    @router.post("/batch", response_model=BatchRetrievalQueryResponse, dependencies=[Depends(combined_auth)])
    async def batch_retrieve(
        request: BatchRetrievalQueryRequest,
        manager=Depends(get_integration_manager)
    ):
        """
        执行批量检索
        """
        start_time = time.time()

        try:
            # 构建批量检索请求
            batch_request = BatchRetrievalRequest(
                queries=request.queries,
                max_results_per_query=request.max_results_per_query,
                min_score=request.min_score,
                include_metadata=request.include_metadata,
                filters=request.filters
            )

            # 执行批量检索
            result = await manager.batch_retrieve(batch_request, request.integration_name)

            return BatchRetrievalQueryResponse(
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = await handle_integration_error("batch_retrieve", e)
            return BatchRetrievalQueryResponse(
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time
            )

    @router.post("/resources", response_model=ListResourcesQueryResponse, dependencies=[Depends(combined_auth)])
    async def list_resources(
        request: ListResourcesQueryRequest,
        manager=Depends(get_integration_manager)
    ):
        """
        列出可用资源
        """
        start_time = time.time()

        try:
            # 构建列出资源请求
            list_request = ListResourcesRequest(
                resource_type=request.resource_type,
                limit=request.limit,
                offset=request.offset,
                filters=request.filters
            )

            # 执行列出资源
            result = await manager.list_resources(list_request, request.integration_name)

            return ListResourcesQueryResponse(
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = await handle_integration_error("list_resources", e)
            return ListResourcesQueryResponse(
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time
            )

    @router.get("/resources/{resource_id}", response_model=GetResourceResponse, dependencies=[Depends(combined_auth)])
    async def get_resource(
        resource_id: str,
        integration_name: Optional[str] = Query(None, description="集成名称，留空使用默认集成"),
        manager=Depends(get_integration_manager)
    ):
        """
        获取指定资源
        """
        start_time = time.time()

        try:
            # 执行获取资源
            result = await manager.get_resource(resource_id, integration_name)

            return GetResourceResponse(
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = await handle_integration_error("get_resource", e)
            return GetResourceResponse(
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time
            )

    @router.get("/health", response_model=HealthCheckResponse, dependencies=[Depends(combined_auth)])
    async def health_check(
        integration_name: Optional[str] = Query(None, description="集成名称，留空检查所有集成"),
        manager=Depends(get_integration_manager)
    ):
        """
        健康检查
        """
        start_time = time.time()

        try:
            # 执行健康检查
            result = await manager.health_check(integration_name)

            return HealthCheckResponse(
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = await handle_integration_error("health_check", e)
            return HealthCheckResponse(
                success=False,
                result={},
                error=error_msg,
                execution_time=time.time() - start_time
            )

    @router.post("/similarity", response_model=SimilarityResponse, dependencies=[Depends(combined_auth)])
    async def calculate_similarity(
        request: SimilarityRequest,
        manager=Depends(get_integration_manager)
    ):
        """
        计算文本相似度
        """
        start_time = time.time()

        try:
            # 获取集成实例
            integration = await manager.get_integration(request.integration_name)

            # 计算相似度
            if hasattr(integration, 'calculate_similarity'):
                result = await integration.calculate_similarity(request.text1, request.text2)
            else:
                raise HTTPException(status_code=400, detail="Integration does not support similarity calculation")

            return SimilarityResponse(
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = await handle_integration_error("calculate_similarity", e)
            return SimilarityResponse(
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time
            )

    @router.get("/integrations", dependencies=[Depends(combined_auth)])
    async def list_integrations(
        manager=Depends(get_integration_manager)
    ):
        """
        列出所有可用的集成
        """
        try:
            integrations = manager.list_integrations()
            integration_info = manager.get_integration_info()

            return {
                "success": True,
                "integrations": integrations,
                "integration_info": integration_info
            }

        except Exception as e:
            logger.error(f"Error listing integrations: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list integrations: {e}")

    @router.get("/config", dependencies=[Depends(combined_auth)])
    async def get_config(
        manager=Depends(get_integration_manager)
    ):
        """
        获取集成配置
        """
        try:
            # 这里需要实现获取配置的逻辑
            # 暂时返回基本信息
            return {
                "success": True,
                "config": {
                    "default_integration": manager.default_integration,
                    "available_integrations": manager.list_integrations()
                }
            }

        except Exception as e:
            logger.error(f"Error getting config: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get config: {e}")

    return router, combined_auth