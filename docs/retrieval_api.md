# /api/v1/retrieve 接口使用说明

本说明文档面向需要通过 `LightRAG` 服务检索知识内容的整合方。接口基于 `FastAPI`，并由 `DeerFlowRetriever` 提供具体检索能力。当前主要暴露两个端点：

- `GET /api/v1/resources` — 获取可供检索的 LightRAG 实例列表
- `POST /api/v1/retrieve` — 按指定模式执行检索

## 1. 鉴权

部署时可启用两类鉴权方式：

- **API Key**：通过 HTTP 头 `X-API-Key: <your-key>` 传入
- **OAuth2 Password**：在 `Authorization` 头携带 `Bearer <token>`

若服务配置了任一鉴权方式，调用方至少需满足其中之一。路径命中白名单时可跳过鉴权，具体由服务端配置决定。

## 2. 资源列表

### 请求

```
GET /api/v1/retrieve/resources
```

### 响应

```json
{
  "success": true,
  "resources": [
    {
      "uri": "rag://default",
      "title": "default",
      "description": "lightrag workspace"
    }
  ],
  "execution_time": 0.012
}
```

调用方需从 `resources` 数组中选择目标实例的 `uri`，在后续检索请求里使用。

## 3. 检索请求

### 请求路径

```
POST /api/v1/retrieve
Content-Type: application/json
```

### 请求体字段

| 字段 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `query` | `string` | 是 | 需要检索的自然语言问题或关键词，最小长度 1 |
| `max_results` | `int` | 否 | 期望返回的最大检索条数，范围 `1-100`，默认 `10` |
| `resources` | `string[]` | **是** | 目标资源 `uri` 列表，需至少包含一个 `rag://<instance>` 值 |
| `local_search` | `bool` | 否 | 设置为 `true` 时执行实体优先的本地检索，并返回结构化检索上下文 |
| `background_search` | `bool` | 否 | 设置为 `true` 时执行全局背景检索，返回回答与检索详情 |

> **说明**：当前实现要求 `local_search` 与 `background_search` 二选一；若二者均为 `false`，会返回错误 `Invalid search type`。

### 示例请求（本地检索）

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key>" \
  -d '{
        "query": "什么是 RAG",
        "max_results": 5,
        "resources": ["rag://default"],
        "local_search": true
      }'
```

## 4. 响应格式

顶层响应统一为：

```json
{
  "success": true,
  "result": { ... },
  "error": null,
  "execution_time": 0.183
}
```

- `success`：是否成功
- `result`：成功时包含具体检索结果；失败时为空
- `error`：失败时的错误描述
- `execution_time`：接口执行耗时，单位秒

### 4.1 本地检索 (`local_search=true`)

当调用方请求本地检索时，`result` 为 `DeerFlowRetrievalResult` 对象：

```json
{
  "query": "什么是 RAG",
  "chunks": [
    {
      "id": "chunk_0",
      "doc_id": "docs/rag.md",
      "content": "Retrieval-Augmented Generation (RAG) 是一种...",
      "chunk_index": 0,
      "score": 0.92,
      "similarity": 0.87
    }
  ],
  "metadata": {
    "query": "什么是 RAG",
    "mode": "local",
    "top_k": 5,
    "chunk_top_k": 5,
    "retrieved_chunks": 3
  },
  "total_results": 3
}
```

字段含义：

- `chunks`：命中的文本块列表（已按相似度去重、合并），携带原始文件路径及分数
- `metadata`：检索参数与命中统计，便于调用方审计
- `total_results`：此次返回的 chunk 数量

### 4.2 背景检索 (`background_search=true`)

背景检索会携带答案与结构化检索来源：

```json
{
  "background": "检索增强生成（RAG）是一种...",
  "entities": [
    {
      "id": 1,
      "entity": "Retrieval-Augmented Generation",
      "type": "concept",
      "description": "..."
    }
  ],
  "relationships": [
    {
      "id": 1,
      "entity1": "LLM",
      "entity2": "RAG",
      "description": "LLM 在 RAG 流程中的角色"
    }
  ],
  "metadata": {
    "total_entities": 5,
    "total_relationships": 3,
    "total_chunks": 3,
    "mode": "global",
    "query": "什么是 RAG"
  }
}
```

- `background`：聚合后的最终回答
- `entities` / `relationships`：检索到的知识图谱实体与关系（结构已按 `kg_query` 参数裁剪）
- `metadata.total_chunks`：与回答拼接的文本块数量，便于调用方做事实核查

## 5. 失败示例

```json
{
  "success": false,
  "result": null,
  "error": "No available resources",
  "execution_time": 0.005
}
```

常见错误说明：

- `No available resources`：`resources` 列表为空，或填写的 `uri` 不存在
- `Invalid search type`：`local_search` 与 `background_search` 均为 `false`
- 鉴权失败：返回 `401/403`，需检查 `X-API-Key` 或 `Authorization` 头

## 6. 接入建议

1. **资源缓存**：`/resources` 结果可做短期缓存，减少重复调用
2. **参数幂等**：同一查询+资源组合可复用返回值，LightRAG 内部已启用缓存，重复请求成本较低
3. **错误重试**：针对 `5xx` 或网络异常建议实现指数退避策略
4. **日志审计**：可将 `metadata` 与 `execution_time` 记录至调用链，辅助分析检索质量

如需进一步扩展（例如自定义检索模式、组合查询），可参考 `docs/query.md` 与 `docs/local_query_analysis.md` 中对 LightRAG 查询模式的介绍。
