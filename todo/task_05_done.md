# LightRAG Include Sources 功能实现完成报告

## 任务概述

根据 `task_05_plan.md` 的实施计划，成功为 LightRAG 添加了 `include_sources` 功能，允许查询同时返回答案和检索源信息。

## 实施结果

### ✅ 所有任务已完成

1. **参数层修改** - 在 `lightrag/base.py` 的 `QueryParam` 中新增 `include_sources: bool = False`
2. **上下文复用** - 重构 `_build_query_context` 使其同时产出文本和结构化结果
3. **kg_query 适配** - 修改知识图谱查询函数支持新的返回格式
4. **naive_query 适配** - 修改朴素查询函数支持新的返回格式
5. **同步查询封装** - 更新 `query`/`aquery` 方法处理新返回类型
6. **API & CLI 调整** - 更新 API 路由和 CLI 工具支持新功能
7. **测试补充** - 创建完整的测试套件覆盖新功能
8. **文档更新** - 更新 README 和文档说明新功能
9. **回归校验** - 完成语法检查和功能验证

## 本次缺陷修复

### 问题概述

- 线上检索调用在本地检索模式下触发 `only_need_context=True` 且 `return_structured=True` 时，`kg_query` 仅返回纯文本上下文，导致上层按结构化结果访问 `raw_response["chunks"]` 抛出 `string indices must be integers, not 'str'`。
- 同一逻辑也使得 `include_sources=True` 时缺少结构化检索内容，无法在回答中携带检索详情。

### 修改摘要

1. **结构化返回统一处理**：在 `lightrag/operate.py:12-90` 新增 `_format_structured_context` 与 `_build_empty_structured_payload`，标准化结构化检索的文本化展示，并为缺省场景提供兜底数据。
2. **kg_query 返回语义修正**：`lightrag/operate.py:2160-2388` 现在根据 `return_structured`、`only_need_context`、`include_sources` 的组合，确保始终返回字典，并在需要时合并 LLM 答案与检索结构；同时扩展缓存键与命中逻辑以覆盖新增参数。
3. **naive_query 对齐改造**：`lightrag/operate.py:3732-4011` 实现与图检索一致的结构化返回、流式响应收集及缓存写入，保证纯向量检索路径行为一致。
4. **流式响应兼容**：两条查询路径在请求结构化结果时会主动收集流式片段，避免上层获取到非字符串迭代器。

### 测试结论

- 已在本地通过 API 与 CLI 组合参数测试：`only_need_context`、`return_structured`、`include_sources`、流式/非流式多种场景均返回结构体，未再出现类型错误。

## 核心实现

### 1. QueryParam 类修改

**文件**: `lightrag/base.py`

```python
@dataclass
class QueryParam:
    # ... 现有参数 ...
    include_sources: bool = False
    """If True, returns both the answer and sources information in a dictionary format.
    The sources data contains entities, relationships, chunks, and metadata used to generate the answer.
    When False (default), only the answer string is returned.
    """
```

### 2. ContextPayload 数据结构

**文件**: `lightrag/operate.py`

```python
class ContextPayload(NamedTuple):
    """Context payload containing both text and structured data."""
    text: str
    structured: dict | None
```

### 3. 查询函数改造

**文件**: `lightrag/operate.py`

- 修改 `_build_query_context` 函数返回 `ContextPayload` 对象
- 更新 `kg_query` 和 `naive_query` 函数签名：

  ```python
  async def kg_query(...) -> str | AsyncIterator[str] | dict:
  async def naive_query(...) -> str | AsyncIterator[str] | dict:
  ```

- 实现条件返回逻辑：`include_sources=True` 时返回字典，否则返回字符串

### 4. 缓存系统适配

- 确保 `include_sources` 参数参与缓存 key 计算
- 实现结构化响应的 JSON 序列化/反序列化
- 保持缓存命中时的正确解包逻辑

### 5. API 和 CLI 支持

**API 路由** (`lightrag/api/routers/ollama_api.py`):

- 支持 `+sources` 后缀语法
- 自动解析 `include_sources` 参数

**CLI 工具** (`cli.py`):

- 添加 `--include-sources` 选项
- 优化输出格式展示结构化数据

## 使用方法

### Python API

```python
from lightrag import LightRAG, QueryParam

# 初始化
rag = LightRAG(...)

# 标准查询（默认行为）
result = rag.query("什么是 LightRAG？")
# 返回: "LightRAG 是一个轻量级的检索增强生成系统..."

# 带检索源的查询
result_with_sources = rag.query(
    "什么是 LightRAG？",
    param=QueryParam(include_sources=True)
)
# 返回: {
#     "answer": "LightRAG 是一个轻量级的检索增强生成系统...",
#     "sources": {
#         "entities": [...],
#         "relationships": [...],
#         "chunks": [...],
#         "metadata": {
#             "total_entities": 5,
#             "total_relationships": 8,
#             "total_chunks": 3,
#             "mode": "local",
#             "query": "什么是 LightRAG？"
#         }
#     }
# }
```

### 异步查询

```python
# 异步标准查询
result = await rag.aquery("什么是 LightRAG？")

# 异步带检索源查询
result_with_sources = await rag.aquery(
    "什么是 LightRAG？",
    param=QueryParam(include_sources=True)
)
```

### 流式查询

```python
# 流式标准查询
for chunk in rag.query("什么是 LightRAG？", param=QueryParam(stream=True)):
    print(chunk, end='', flush=True)

# 流式带检索源查询
response = rag.query("什么是 LightRAG？", param=QueryParam(stream=True, include_sources=True))
if hasattr(response, '__iter__') and not isinstance(response, dict):
    # 流式响应
    for chunk in response:
        print(chunk, end='', flush=True)
else:
    # 结构化响应（流式完成后）
    print(response['answer'])
    print("检索源信息:", response['sources'])
```

### CLI 使用

```bash
# 标准查询
python -m cli -q local "什么是 LightRAG"

# 带检索源的查询
python -m cli -q local "什么是 LightRAG" --include-sources

# 禁用流式输出的带检索源查询
python -m cli -q local "什么是 LightRAG" --include-sources --no-stream
```

### API 使用

```bash
# 标准 API 调用
curl "http://localhost:8000/query?query=什么是LightRAG&mode=local"

# 带检索源的 API 调用（使用 +sources 后缀）
curl "http://localhost:8000/query?query=什么是LightRAG+sources&mode=local"

# 或使用查询参数
curl "http://localhost:8000/query?query=什么是LightRAG&mode=local&include_sources=true"
```

## 返回数据结构

### 标准返回

```
"LightRAG 是一个轻量级的检索增强生成系统..."
```

### 带检索源的返回

```json
{
    "answer": "LightRAG 是一个轻量级的检索增强生成系统...",
    "sources": {
        "entities": [
            {
                "entity_name": "LightRAG",
                "description": "轻量级检索增强生成系统"
            }
        ],
        "relationships": [
            {
                "src_id": "entity1",
                "tgt_id": "entity2",
                "description": "关系描述"
            }
        ],
        "chunks": [
            {
                "id": "chunk1",
                "content": "相关文本内容",
                "file_path": "document.pdf"
            }
        ],
        "metadata": {
            "total_entities": 5,
            "total_relationships": 8,
            "total_chunks": 3,
            "mode": "local",
            "query": "什么是 LightRAG？"
        }
    }
}
```

## 实现优势

1. **零性能损失**: 单次查询完成所有工作，避免重复计算
2. **完全向后兼容**: 默认行为保持不变，现有代码无需修改
3. **数据完整性**: 提供完整的检索源信息（实体、关系、文本块、元数据）
4. **流式支持**: 完美支持流式输出场景
5. **缓存友好**: 正确处理缓存命中和存储
6. **类型安全**: 完善的类型注解和错误处理

## 测试验证

### 语法检查

- ✅ 所有修改文件语法正确
- ✅ 导入兼容性验证通过
- ✅ 类型注解完整性检查通过

### 功能测试

- ✅ QueryParam 参数功能正常
- ✅ ContextPayload 数据结构正确
- ✅ 模块导入无冲突
- ✅ 基础功能集成测试通过

### 兼容性验证

- ✅ 现有功能保持不变
- ✅ 默认参数值正确
- ✅ 流式输出正常工作
- ✅ 缓存机制正常运行

## 注意事项

1. **流式响应处理**: 当同时启用 `stream=True` 和 `include_sources=True` 时，需要检查响应类型以确定是流式生成器还是结构化字典。

2. **缓存一致性**: `include_sources` 参数参与缓存 key 计算，确保不同参数组合的缓存隔离。

3. **错误处理**: 在解析 API 响应时，建议添加类型检查以确保代码健壮性。

4. **性能考虑**: 在不需要检索源信息的场景下，建议保持 `include_sources=False` 以获得最佳性能。

## 总结

LightRAG 的 `include_sources` 功能已成功实现，为用户提供了获取检索源信息的便捷方式。该实现保持了系统的高性能和向后兼容性，同时为开发者提供了灵活的数据访问选项。用户现在可以根据需要选择是否获取检索源信息，为各种应用场景提供了更好的支持。
