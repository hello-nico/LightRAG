"""
LightRAG 检索集成 API 路由

此模块提供了基于 DeerFlowRetriever 的检索接口。
"""
import resource
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.integrations import DeerFlowRetriever, RetrievalRequest, DeerFlowRetrievalResult

router = APIRouter(tags=["retrieval"])

# 请求/响应模型
class RetrievalQueryRequest(BaseModel):
    """检索查询请求"""
    query: str = Field(..., min_length=1, description="检索查询文本")
    max_results: int = Field(10, ge=1, le=100, description="最大结果数")
    resources: List[str] = Field(default_factory=list, description="资源列表")

class RetrievalQueryResponse(BaseModel):
    """检索查询响应"""
    success: bool = Field(..., description="是否成功")
    result: Optional[DeerFlowRetrievalResult] = Field(None, description="检索结果")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

class ListResourcesResponse(BaseModel):
    """列出资源响应"""
    success: bool = Field(..., description="是否成功")
    resources: List[Any] = Field(..., description="资源列表")
    error: Optional[str] = Field(None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")

def create_retrieval_routes(api_key: Optional[str] = None):
    """创建检索路由"""
    combined_auth = get_combined_auth_dependency(api_key)
    retriever = DeerFlowRetriever()

    @router.get("/resources", response_model=ListResourcesResponse, dependencies=[Depends(combined_auth)])
    async def list_resources():
        """列出可用资源"""
        start_time = time.time()
        try:
            resources = await retriever.list_resources()
            return ListResourcesResponse(
                success=True,
                resources=resources,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ListResourcesResponse(
                success=False,
                resources=[],
                error=str(e),
                execution_time=time.time() - start_time
            )

    @router.post("/retrieve", response_model=RetrievalQueryResponse, dependencies=[Depends(combined_auth)])
    async def retrieve(request: RetrievalQueryRequest):
        """执行检索"""
        start_time = time.time()
        try:
            # 构建检索请求
            retrieval_request = RetrievalRequest(
                query=request.query,
                max_results=request.max_results
            )
            # todo 当前只能返回一个资源实例列表
            resource_instances = []
            for r in request.resources:
                print(r)
                all_resources = await retriever.list_resources()
                for resource in all_resources:
                    if resource.uri == r:
                        resource_instances.append(resource)
                        break
                    
            if not resource_instances:
                return RetrievalQueryResponse(
                    success=False,
                    error="No available resources",
                    execution_time=time.time() - start_time
                )
                
            instance_name = resource_instances[0].uri.split("://")[1]
            result = await retriever.retrieve(instance_name, retrieval_request)

            return RetrievalQueryResponse(
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return RetrievalQueryResponse(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    return router