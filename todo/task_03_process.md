```python
import abc

from pydantic import BaseModel, Field

class Chunk:
    content: str
    similarity: float

    def __init__(self, content: str, similarity: float):
        self.content = content
        self.similarity = similarity

class Document:
    """
    Document is a class that represents a document.
    """

    id: str
    url: str | None = None
    title: str | None = None
    chunks: list[Chunk] = []

    def __init__(
        self,
        id: str,
        url: str | None = None,
        title: str | None = None,
        chunks: list[Chunk] = [],
    ):
        self.id = id
        self.url = url
        self.title = title
        self.chunks = chunks

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "content": "\n\n".join([chunk.content for chunk in self.chunks]),
        }
        if self.url:
            d["url"] = self.url
        if self.title:
            d["title"] = self.title
        return d

class Resource(BaseModel):
    """
    Resource is a class that represents a resource.
    """

    uri: str = Field(..., description="The URI of the resource")
    title: str = Field(..., description="The title of the resource")
    description: str | None = Field("", description="The description of the resource")

class Retriever(abc.ABC):
    """
    Define a RAG provider, which can be used to query documents and resources.
    """

    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        """
        List resources from the rag provider.
        """
        pass

    @abc.abstractmethod
    def query_relevant_documents(
        self, query: str, resources: list[Resource] = []
    ) -> list[Document]:
        """
        Query relevant documents from the resources.
        """
        pass
```

这是deer-flow的检索器接口，我们需要调整我们当前集成的检索器，将lightrag本身的资源类型（Document、Chunk、relationship、entity）转换为deer-flow的资源类型（Resource）。
我们当前已经实现了提供完整的集成管理器，以及一个简单的deer-flow集成。

## 现状分析

### 当前集成架构

1. **集成管理器** (`manager.py`): 提供统一的集成管理接口
2. **DeerFlow集成** (`deer_flow.py`): 基于现有集成框架的实现
3. **数据模型** (`models.py`): 定义了完整的检索数据结构
4. **上下文解析器** (`context_parser.py`): 解析LightRAG的检索结果
5. **转换器** (`converters.py`): 提供数据转换和补全功能

### 发现的问题

1. **接口不匹配**: deer-flow要求实现`list_resources(query)`和`query_relevant_documents(query, resources)`方法
2. **数据模型差异**: deer-flow的Resource需要`uri`字段，Document需要包含Chunk列表
3. **缺失功能**: DeerFlowIntegration中的list_resources方法未完全实现

### 接口对比

**DeerFlow标准接口**:

```python
class Retriever(abc.ABC):
    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]: pass

    @abc.abstractmethod
    def query_relevant_documents(self, query: str, resources: list[Resource] = []) -> list[Document]: pass
```

**当前LightRAG集成接口**:

```python
class BaseIntegration(abc.ABC):
    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult: pass

    @abstractmethod
    async def list_resources(self, request: ListResourcesRequest) -> ListResourcesResponse: pass
```

## 解决方案

### 基于现有架构的修改计划

#### 1. 修改 `models.py`

- 在现有数据模型基础上添加deer-flow兼容字段
- 添加DeerFlowResource、DeerFlowDocument、DeerFlowChunk类
- 保持现有模型的向后兼容性

#### 2. 扩展 `converters.py`

- 添加 `convert_to_deerflow_resources()` 方法
- 添加 `convert_to_deerflow_documents()` 方法
- 实现URI生成策略（`lightrag://doc/{doc_id}` 格式）
- 添加实体/关系到Resource的转换逻辑

#### 3. 完善 `deer_flow.py`

- 实现 `list_resources()` 方法的具体逻辑
- 从LightRAG文档状态获取可用资源列表
- 修改检索结果输出格式，符合deer-flow标准
- 补充 `get_resource()` 方法的完整实现

#### 4. 修改 `base.py`

- 添加deer-flow Retriever接口抽象类
- 保持现有BaseIntegration接口不变

#### 5. 更新 `__init__.py`

- 注册deer-flow接口适配器
- 提供统一的接口访问入口

### 核心转换逻辑

#### URI生成策略

- 文档资源: `lightrag://doc/{doc_id}`
- 实体资源: `lightrag://entity/{entity_id}`
- 关系资源: `lightrag://relationship/{rel_id}`
- 文本块资源: `lightrag://chunk/{chunk_id}`

#### 数据映射

- LightRAG Document → DeerFlow Document
- LightRAG Chunk → DeerFlow Chunk (similarity字段映射)
- LightRAG Entity/Relationship → DeerFlow Resource

#### 资源列表实现

- 从LightRAG的doc_status获取文档列表
- 将文档转换为Resource格式
- 支持按查询过滤资源

### 实施步骤

1. **数据模型扩展**: 在models.py中添加deer-flow兼容类
2. **转换器增强**: 在converters.py中添加转换方法
3. **集成完善**: 完善deer_flow.py的缺失方法
4. **接口适配**: 添加deer-flow标准接口支持
5. **测试验证**: 确保接口兼容性和功能正确性

### 关键技术点

- 复用现有的转换服务架构
- 最小化代码改动，确保向后兼容
- 基于现有的LightRAG实例获取资源信息
- 统一的URI生成策略

这个方案在现有架构基础上进行最小化修改，实现deer-flow标准接口的完整支持。
