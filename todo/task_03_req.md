# LightRAG 本地检索器实现计划

## 概述

基于当前LightRAG项目，创建一个本地检索器，能够直接返回检索到的文档、关系和实体信息，并将这些信息有机整合后提供给外部服务使用。

## 核心设计思路

### 1. 利用现有的only_need_context参数

- LightRAG已有`only_need_context`参数，设置为True时直接返回检索上下文而非生成回答
- 上下文格式：包含Entities(KG)、Relationships(KG)、Document Chunks(DC)三个部分

### 2. 创建专门的检索器接口

- 新建`LocalRetriever`类，封装LightRAG的检索功能
- 提供统一的数据格式输出，便于外部服务使用
- 支持同步和异步调用

### 3. 数据格式规范化

- 将原始上下文解析为结构化数据
- 提供文档、关系、实体的分离和组合视图
- 添加元数据（检索时间、参数设置等）

## 实现步骤

### 1. 创建本地检索器核心模块 (`lightrag/local_retriever.py`)

```python
class LocalRetriever:
    - __init__: 初始化LightRAG实例
    - retrieve: 同步检索方法
    - aretrieve: 异步检索方法
    - _parse_context: 解析上下文为结构化数据
    - _format_output: 格式化输出结果
```

### 2. 创建数据模型 (`lightrag/retriever_models.py`)

```python
class RetrievalResult:
    - query: 查询文本
    - entities: 实体列表
    - relationships: 关系列表
    - documents: 文档块列表
    - metadata: 元数据

class Entity, Relationship, DocumentChunk: 对应的数据结构
```

### 3. 创建API路由 (`lightrag/api/routers/retriever_routes.py`)

- POST `/retriever/retrieve` - 基础检索
- POST `/retriever/aretrieve` - 异步检索
- POST `/retriever/batch` - 批量检索

### 4. 创建示例服务脚本

- 演示如何使用检索器
- 提供不同输出格式的示例
- 集成到现有FastAPI应用

## 关键特性

### 1. 多模式支持

- 支持local/global/hybrid/mix等所有检索模式
- 可配置返回结果数量和token限制
- 支持重排序和过滤

### 2. 灵活输出格式

- JSON格式：便于程序处理
- 原始格式：与LightRAG兼容
- 自定义格式：根据需求调整

### 3. 性能优化

- 复用现有embedding和缓存机制
- 支持批量处理
- 异步操作提升吞吐量

## 文件结构

```
lightrag/
├── local_retriever.py      # 核心检索器
├── retriever_models.py     # 数据模型
├── api/routers/
│   └── retriever_routes.py # API路由
└── examples/
    └── retriever_demo.py   # 使用示例
```

## 预期效果

- 外部服务可以通过简单API获取结构化的检索结果
- 保持与LightRAG现有功能的完全兼容性
- 提供更灵活的数据访问方式

## 技术细节分析

### 现有架构分析

1. **查询入口**: `LightRAG.aquery()` / `LightRAG.query()`
2. **检索模式**:
   - `kg_query()` - 知识图谱检索（local/global/hybrid/mix）
   - `naive_query()` - 向量检索（naive）
   - `bypass` - 直接LLM
3. **上下文构建**: `_build_query_context()` 函数
4. **数据格式**: 返回格式化的文本，包含Entities、Relationships、Document Chunks

### 关键发现

1. **only_need_context参数**: 当设置为True时，直接返回检索上下文
2. **上下文结构**:

   ```
   -----Entities(KG)-----
   ```json
   [...]
   ```

   -----Relationships(KG)-----

   ```json
   [...]
   ```

   -----Document Chunks(DC)-----

   ```json
   [...]
   ```

   ```
3. **数据源**:
   - Entities: 来自实体向量数据库和知识图谱
   - Relationships: 来自关系向量数据库和知识图谱
   - Document Chunks: 来自文档向量数据库和文本块存储

### 实现策略

1. **解析现有输出**: 解析LightRAG的文本输出为结构化数据
2. **保持兼容**: 不修改现有逻辑，只添加新的接口层
3. **统一封装**: 提供简单易用的API接口

## 下一步行动

- [ ] 创建local_retriever.py核心模块
- [ ] 定义retriever_models.py数据模型
- [ ] 实现API路由
- [ ] 编写使用示例
- [ ] 测试和优化性能

---

# DeerFlow项目私有域知识检索集成方案分析

## 概述

本文档详细分析了DeerFlow项目中私有域知识检索的集成方案。通过对项目代码的全面分析，阐述了该项目的RAG（检索增强生成）架构、技术实现和特色功能。

## 项目结构分析

### 目录结构

```
src/
├── rag/                    # RAG核心模块
│   ├── __init__.py        # 模块导出
│   ├── retriever.py       # 抽象检索器接口
│   ├── builder.py         # 检索器工厂
│   ├── ragflow.py         # RAGFlow提供商实现
│   ├── vikingdb_knowledge_base.py  # VikingDB知识库实现
│   └── milvus.py          # Milvus向量数据库实现
├── tools/                 # 工具集成
│   └── retriever.py       # LangChain检索工具
└── server/                # 服务器API
    └── rag_request.py     # RAG相关API请求处理
```

### 核心架构设计

DeerFlow项目采用了**多层次的RAG架构**，通过抽象层设计实现了多个检索后端的统一接入。核心设计特点：

1. **抽象层设计**：`Retriever`抽象类定义了统一的检索接口
2. **工厂模式**：`build_retriever()`函数根据环境配置动态创建检索器实例
3. **插件化架构**：支持通过实现`Retriever`接口添加新的检索后端
4. **配置驱动**：所有功能通过环境变量配置，支持不同环境的快速切换

## RAG实现分析

### 核心接口设计

```python
class Retriever(abc.ABC):
    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        """列出可用的知识库资源"""
        pass

    @abc.abstractmethod
    def query_relevant_documents(
        self, query: str, resources: list[Resource] = []
    ) -> list[Document]:
        """查询相关文档"""
        pass
```

### 数据模型设计

1. **Resource**：表示知识库资源，包含URI、标题和描述
2. **Document**：表示检索到的文档，包含ID、URL、标题和内容块列表
3. **Chunk**：表示文档内容块，包含具体内容和相似度分数

### 工厂模式实现

```python
def build_retriever() -> Retriever | None:
    if SELECTED_RAG_PROVIDER == RAGProvider.RAGFLOW.value:
        return RAGFlowProvider()
    elif SELECTED_RAG_PROVIDER == RAGProvider.VIKINGDB_KNOWLEDGE_BASE.value:
        return VikingDBKnowledgeBaseProvider()
    elif SELECTED_RAG_PROVIDER == RAGProvider.MILVUS.value:
        return MilvusProvider()
    return None
```

## 向量存储分析

### Milvus向量数据库集成

项目实现了完整的Milvus集成方案，支持本地和云端部署：

#### 1. 双模式支持

- **Milvus Lite**：本地文件数据库（如`./milvus_demo.db`）
- **远程Milvus**：连接到Milvus服务器（如`http://localhost:19530`）

#### 2. 嵌入模型集成

- **OpenAI Embeddings**：支持OpenAI的文本嵌入模型
- **Dashscope Embeddings**：支持阿里云Dashscope嵌入服务
- **统一接口**：通过封装提供统一的嵌入接口

#### 3. 向量存储架构

```python
class CollectionSchema:
    - id: VARCHAR (主键)
    - embedding: FLOAT_VECTOR (向量字段)
    - content: VARCHAR (文档内容)
    - title: VARCHAR (文档标题)
    - url: VARCHAR (文档URL)
    - 动态字段支持 (metadata)
```

#### 4. 索引配置

- **索引类型**：IVF_FLAT
- **度量类型**：内积(IP)
- **参数配置**：nlist=1024, nprobe=10

### 检索流程

1. **查询向量化**：将用户查询转换为向量表示
2. **向量搜索**：在向量空间中执行相似度搜索
3. **结果过滤**：根据资源列表过滤搜索结果
4. **结果聚合**：将同一文档的不同块重新组合
5. **相似度计算**：返回每个匹配块的相似度分数

## 文档处理流程

### 文档摄入流程

Milvus提供商实现了完整的文档自动摄入功能：

#### 1. 文档发现

- 自动扫描`examples/`目录下的markdown文件
- 支持配置自定义示例目录(`MILVUS_EXAMPLES_DIR`)
- 支持启用/禁用自动加载(`MILVUS_AUTO_LOAD_EXAMPLES`)

#### 2. 文档处理

```python
def _load_example_files(self):
    # 1. 发现markdown文件
    md_files = list(examples_path.glob("*.md"))

    # 2. 检查是否已存在
    existing_docs = self._get_existing_document_ids()

    # 3. 处理每个文件
    for md_file in md_files:
        if doc_id in existing_docs:
            continue  # 跳过已加载的文件

        # 4. 读取和解析内容
        content = md_file.read_text(encoding="utf-8")
        title = self._extract_title_from_markdown(content, md_file.name)

        # 5. 内容分块
        chunks = self._split_content(content)

        # 6. 向量化和存储
        for i, chunk in enumerate(chunks):
            # 生成嵌入向量并存储
```

#### 3. 内容分块策略

- **分块大小**：默认4000字符
- **分块方式**：按段落分割（以`\n\n`为分隔符）
- **智能处理**：保持段落完整性，避免在段落中间分割

#### 4. 去重机制

- **ID生成**：基于文件名、大小和修改时间生成唯一ID
- **重复检查**：存储前检查文档是否已存在
- **版本管理**：支持强制重新加载(`force_reload=True`)

### 元数据管理

每个文档块都包含丰富的元数据：

```python
metadata = {
    "source": "examples",        # 数据源
    "file": "document.md",       # 文件名
    "title": "文档标题",         # 文档标题
    "url": "milvus://collection/document.md",  # 文档URL
    "doc_id": "unique_doc_id",   # 文档唯一标识
}
```

## 集成方案总结

### 支持的检索后端

#### 1. RAGFlow提供商

- **定位**：企业级RAG服务平台
- **API集成**：通过REST API调用
- **特性**：
  - 支持跨语言检索(`RAGFLOW_CROSS_LANGUAGES`)
  - 数据集级别过滤
  - 分页控制(`RAGFLOW_PAGE_SIZE`)
- **配置要求**：`RAGFLOW_API_URL`, `RAGFLOW_API_KEY`

##### RAGFlow API接口详细分析

###### 核心组件

**RAGFlowProvider类** (`src/rag/ragflow.py:13-137`)

- 实现了`Retriever`抽象基类
- 通过HTTP API与RAGFlow服务通信
- 支持文档检索和资源列表功能

###### 环境配置

**必需的环境变量**：

- `RAGFLOW_API_URL`: RAGFlow服务地址 (如: `http://localhost:9388`)
- `RAGFLOW_API_KEY`: API认证密钥
- `RAG_PROVIDER`: 设置为`ragflow`启用该提供商

**可选配置**：

- `RAGFLOW_PAGE_SIZE`: 每页返回结果数量 (默认10)
- `RAGFLOW_CROSS_LANGUAGES`: 跨语言搜索支持 (如: `English,Chinese,Spanish`)

###### API接口分析

###### A. 文档检索接口

- **URL**: `{API_URL}/api/v1/retrieval`
- **方法**: POST
- **认证**: Bearer Token
- **请求头**:

  ```json
  {
    "Authorization": "Bearer {API_KEY}",
    "Content-Type": "application/json"
  }
  ```

**请求体** (`src/rag/ragflow.py:60-68`):

```json
{
  "question": "用户查询问题",
  "dataset_ids": ["数据集ID列表"],
  "document_ids": ["文档ID列表"],
  "page_size": 10,
  "cross_languages": ["English", "Chinese"]
}
```

**响应格式** (`src/rag/ragflow.py:77-87`):

```json
{
  "data": {
    "doc_aggs": [
      {
        "doc_id": "文档ID",
        "doc_name": "文档名称"
      }
    ],
    "chunks": [
      {
        "document_id": "所属文档ID",
        "content": "片段内容",
        "similarity": 0.95
      }
    ]
  }
}
```

###### B. 数据集列表接口

- **URL**: `{API_URL}/api/v1/datasets`
- **方法**: GET
- **认证**: Bearer Token
- **查询参数**: `name` (可选，用于过滤数据集名称)

**响应格式** (`src/rag/ragflow.py:118-127`):

```json
{
  "data": [
    {
      "id": "数据集ID",
      "name": "数据集名称",
      "description": "数据集描述"
    }
  ]
}
```

###### 数据模型

**核心数据结构** (`src/rag/retriever.py`):

- **Chunk**: 文档片段
  - `content`: 文本内容
  - `similarity`: 相似度分数

- **Document**: 文档对象
  - `id`: 文档ID
  - `title`: 文档标题
  - `chunks`: 包含的片段列表

- **Resource**: 资源对象
  - `uri`: 资源标识符 (格式: `rag://dataset/{dataset_id}#{document_id}`)
  - `title`: 资源标题
  - `description`: 资源描述

###### 使用方式

**初始化配置**:

```python
provider = RAGFlowProvider()
```

**查询相关文档**:

```python
documents = provider.query_relevant_documents(
    query="用户问题",
    resources=[Resource(uri="rag://dataset/123", title="数据集")]
)
```

**列出可用资源**:

```python
resources = provider.list_resources(query="搜索关键词")
```

#### 2. VikingDB知识库

- **定位**：字节跳动内部知识库系统
- **安全认证**：HMAC-SHA256签名机制
- **高级功能**：
  - 查询预处理（指令增强、查询重写）
  - 后处理（重排序、块扩散、附件链接）
  - 密集检索和稀疏检索的权重控制
- **配置要求**：API密钥对、Region设置

#### 3. Milvus向量数据库

- **定位**：开源向量数据库
- **部署模式**：本地Lite模式 + 远程服务器模式
- **特色功能**：
  - 自动示例文档加载
  - 灵活的嵌入模型选择
  - 动态字段支持
  - 完整的CRUD操作
- **配置灵活性**：支持多种部署和配置选项

### 工具集成与使用

#### 1. LangChain工具集成

```python
class RetrieverTool(BaseTool):
    name: str = "local_search_tool"
    description: str = "Useful for retrieving information from the file with `rag://` uri prefix"

    def _run(self, keywords: str) -> list[Document]:
        documents = self.retriever.query_relevant_documents(keywords, self.resources)
        return [doc.to_dict() for doc in documents]
```

#### 2. API端点

- `GET /api/rag/resources` - 列出可用知识库
- `GET /api/rag/config` - 获取RAG配置信息
- 支持知识库的动态发现和配置

#### 3. 优先级设计

- **本地检索优先**：本地检索工具优先于网络搜索
- **资源过滤**：支持在特定知识库中检索
- **上下文集成**：与LangGraph工作流无缝集成

### 环境配置

#### 关键配置项

```bash
# RAG提供商选择
RAG_PROVIDER=ragflow|vikingdb_knowledge_base|milvus

# Milvus配置
MILVUS_URI=./milvus_demo.db
MILVUS_COLLECTION=documents
MILVUS_EMBEDDING_PROVIDER=openai
MILVUS_EMBEDDING_MODEL=text-embedding-ada-002
MILVUS_EMBEDDING_DIM=1536
MILVUS_TOP_K=10
MILVUS_AUTO_LOAD_EXAMPLES=true
MILVUS_EXAMPLES_DIR=examples

# RAGFlow配置
RAGFLOW_API_URL=https://api.ragflow.com
RAGFLOW_API_KEY=your_api_key
RAGFLOW_PAGE_SIZE=10
RAGFLOW_CROSS_LANGUAGES=zh,en

# VikingDB配置
VIKINGDB_KNOWLEDGE_BASE_API_URL=your_api_url
VIKINGDB_KNOWLEDGE_BASE_API_AK=your_access_key
VIKINGDB_KNOWLEDGE_BASE_API_SK=your_secret_key
```

### 特色功能

#### 1. 多级过滤能力

- **数据集级别**：限制在特定数据集中搜索
- **文档级别**：在特定文档内检索
- **元数据过滤**：基于文档属性进行过滤

#### 2. 智能示例管理

- **自动加载**：启动时自动加载示例文档
- **去重处理**：避免重复加载相同文档
- **版本管理**：支持文档的强制重新加载
- **状态查询**：查询已加载的示例文档列表

#### 3. 灵活的部署模式

- **完全本地化**：Milvus Lite支持完全本地部署
- **云端部署**：支持连接远程向量数据库
- **混合模式**：同时使用本地和云端资源

#### 4. 企业级特性

- **安全性**：支持API密钥认证和签名机制
- **可扩展性**：插件化架构便于添加新的检索后端
- **可维护性**：统一的接口设计便于维护和测试
- **监控友好**：完整的日志记录和错误处理

### 扩展性设计

#### 1. 插件化架构

通过实现`Retriever`接口可以轻松添加新的检索后端：

```python
class CustomRetriever(Retriever):
    def list_resources(self, query: str | None = None) -> list[Resource]:
        # 实现资源列表
        pass

    def query_relevant_documents(self, query: str, resources: list[Resource] = []) -> list[Document]:
        # 实现文档检索
        pass
```

##### 实现新的RAG Provider详细指南

###### 架构设计建议

**基于现有抽象基类**

最简单的方式是继承`Retriever`抽象基类 (`src/rag/retriever.py:62-82`)：

```python
class Retriever(abc.ABC):
    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        pass

    @abc.abstractmethod
    def query_relevant_documents(
        self, query: str, resources: list[Resource] = []
    ) -> list[Document]:
        pass
```

###### 配置管理扩展

需要在以下文件中注册新的Provider：

**src/config/tools.py** - 添加新的枚举值：

```python
class RAGProvider(enum.Enum):
    RAGFLOW = "ragflow"
    VIKINGDB_KNOWLEDGE_BASE = "vikingdb_knowledge_base"
    MILVUS = "milvus"
    YOUR_NEW_PROVIDER = "your_provider"  # 新增
```

**.env.example** - 添加环境变量模板：

```bash
# RAG_PROVIDER=your_provider
# YOUR_PROVIDER_API_URL="https://your-api.com"
# YOUR_PROVIDER_API_KEY="your-key"
# YOUR_PROVIDER_PAGE_SIZE=10
```

###### 实现模式选择

**推荐方式：完全参考RAGFlowProvider**

优点：

- 代码结构清晰，易于维护
- 统一的错误处理和认证机制
- 与现有系统完美集成

**简化方式：基于HTTP的通用Provider**

如果你的RAG服务有简单的REST API，可以创建一个通用的HTTP Provider：

```python
class GenericHTTPProvider(Retriever):
    def __init__(self):
        self.api_url = os.getenv("GENERIC_RAG_API_URL")
        self.api_key = os.getenv("GENERIC_RAG_API_KEY")

    def query_relevant_documents(self, query: str, resources: list[Resource] = []):
        # 通用HTTP请求逻辑
        pass
```

###### 工厂模式集成

deer-flow使用了工厂模式来选择Provider，你只需要：

1. 实现你的Provider类
2. 在配置中注册
3. 系统会自动根据`RAG_PROVIDER`环境变量选择

###### 最佳实践建议

**保持一致性**：

- 使用相同的请求/响应格式
- 统一的错误处理机制
- 相同的认证方式

**灵活性考虑**：

- 支持可选配置参数
- 提供合理的默认值
- 支持分页和过滤

**测试建议**：

- 参考`tests/unit/rag/test_ragflow.py`编写测试
- 测试不同场景（成功、失败、边界情况）

#### 2. 配置驱动扩展

- 所有功能通过环境变量配置
- 支持不同环境的快速切换
- 便于容器化部署和云原生应用

## 总结

DeerFlow项目的私有域知识检索集成方案体现了**企业级RAG系统的最佳实践**，特别是在以下几个方面表现出色：

1. **架构设计**：抽象层设计实现了多个检索后端的统一接入
2. **技术实现**：支持向量数据库、企业RAG平台和内部知识库
3. **易用性**：自动化文档处理、智能示例管理和灵活的配置
4. **可扩展性**：插件化架构便于添加新的检索后端
5. **企业级特性**：安全性、可维护性和监控友好性

该方案为构建企业级智能问答系统提供了完整的技术参考，特别是在处理私有域知识检索方面具有很强的实用价值。
