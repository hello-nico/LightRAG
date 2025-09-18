"""
LightRAG 检索集成的数据模型定义

此模块定义了所有与检索相关的数据结构，用于标准化接口。
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field



class Document(BaseModel):
    """文档模型"""
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
    score: Optional[float] = Field(None, description="相关度分数")
    similarity: Optional[float] = Field(None, description="相似度分数")


class Entity(BaseModel):
    """实体模型"""
    id: str = Field(..., description="实体唯一标识")
    entity: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")
    description: Optional[str] = Field(None, description="实体描述")


class Relationship(BaseModel):
    """关系模型"""
    id: str = Field(..., description="关系唯一标识")
    source_entity_id: str = Field(..., description="源实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    description: Optional[str] = Field(None, description="关系描述")


class RetrievalResult(BaseModel):
    """检索结果模型"""
    query: str = Field(..., description="检索查询")
    chunks: List[Chunk] = Field(default_factory=list, description="检索到的文本块")
    entities: List[Entity] = Field(default_factory=list, description="检索到的实体")
    relationships: List[Relationship] = Field(default_factory=list, description="检索到的关系")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="检索元数据")
    total_results: int = Field(0, description="总结果数")
    retrieval_time: Optional[float] = Field(None, description="检索耗时（秒）")


class RetrievalRequest(BaseModel):
    """检索请求模型"""
    query: str = Field(..., description="检索查询")
    max_results: int = Field(10, description="最大结果数")
    min_score: Optional[float] = Field(None, description="最小分数阈值")