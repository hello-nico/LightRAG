"""
LightRAG 检索集成的数据模型定义

此模块定义了所有与检索相关的数据结构，用于标准化接口。
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ResourceType(str, Enum):
    """资源类型枚举"""
    DOCUMENT = "document"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TEXT_CHUNK = "text_chunk"


class Resource(BaseModel):
    """基础资源模型"""
    id: str = Field(..., description="资源唯一标识")
    type: ResourceType = Field(..., description="资源类型")
    title: Optional[str] = Field(None, description="资源标题")
    description: Optional[str] = Field(None, description="资源描述")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="资源元数据")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")

    class Config:
        use_enum_values = True


class Document(Resource):
    """文档模型"""
    type: ResourceType = Field(ResourceType.DOCUMENT, literal=True)
    content: Optional[str] = Field(None, description="文档内容")
    chunk_count: int = Field(0, description="包含的块数量")
    file_path: Optional[str] = Field(None, description="文件路径")
    file_size: Optional[int] = Field(None, description="文件大小（字节）")
    mime_type: Optional[str] = Field(None, description="MIME类型")


class Chunk(BaseModel):
    """文本块模型"""
    id: str = Field(..., description="块唯一标识")
    doc_id: str = Field(..., description="所属文档ID")
    content: str = Field(..., description="块内容")
    chunk_index: int = Field(..., description="块索引")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="块元数据")
    score: Optional[float] = Field(None, description="相关度分数")
    similarity: Optional[float] = Field(None, description="相似度分数")
    created_at: Optional[datetime] = Field(None, description="创建时间")


class Entity(BaseModel):
    """实体模型"""
    id: str = Field(..., description="实体唯一标识")
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")
    description: Optional[str] = Field(None, description="实体描述")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="实体元数据")


class Relationship(BaseModel):
    """关系模型"""
    id: str = Field(..., description="关系唯一标识")
    source_entity_id: str = Field(..., description="源实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    relation_type: str = Field(..., description="关系类型")
    description: Optional[str] = Field(None, description="关系描述")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="关系元数据")


class RetrievalResult(BaseModel):
    """检索结果模型"""
    query: str = Field(..., description="检索查询")
    chunks: List[Chunk] = Field(default_factory=list, description="检索到的文本块")
    entities: List[Entity] = Field(default_factory=list, description="检索到的实体")
    relationships: List[Relationship] = Field(default_factory=list, description="检索到的关系")
    context: Optional[str] = Field(None, description="原始上下文")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="检索元数据")
    total_results: int = Field(0, description="总结果数")
    retrieval_time: Optional[float] = Field(None, description="检索耗时（秒）")


class RetrievalRequest(BaseModel):
    """检索请求模型"""
    query: str = Field(..., description="检索查询")
    max_results: int = Field(10, description="最大结果数")
    min_score: Optional[float] = Field(None, description="最小分数阈值")
    include_metadata: bool = Field(True, description="是否包含元数据")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")


class BatchRetrievalRequest(BaseModel):
    """批量检索请求模型"""
    queries: List[str] = Field(..., description="检索查询列表")
    max_results_per_query: int = Field(10, description="每个查询的最大结果数")
    min_score: Optional[float] = Field(None, description="最小分数阈值")
    include_metadata: bool = Field(True, description="是否包含元数据")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")


class ListResourcesRequest(BaseModel):
    """列出资源请求模型"""
    resource_type: Optional[ResourceType] = Field(None, description="资源类型过滤")
    limit: int = Field(100, description="限制数量")
    offset: int = Field(0, description="偏移量")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")


class ListResourcesResponse(BaseModel):
    """列出资源响应模型"""
    resources: List[Resource] = Field(default_factory=list, description="资源列表")
    total_count: int = Field(0, description="总资源数")
    has_more: bool = Field(False, description="是否有更多结果")


class BatchRetrievalResponse(BaseModel):
    """批量检索响应模型"""
    results: List[RetrievalResult] = Field(default_factory=list, description="批量检索结果")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="批量检索元数据")