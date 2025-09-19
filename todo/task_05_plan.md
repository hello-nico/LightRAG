## 可执行性评审

- ✅ QueryParam 已使用 `@dataclass` 定义，新增布尔开关不会破坏构造逻辑；`kg_query` / `naive_query` 也集中在单一文件，便于一次修改。
- ⚠️ `kg_query` 当前直接把 `_build_query_context` 的字符串结果塞入 prompt，若改为结构化数据，需要同时保留字符串形态以兼容 LLM prompt；必须避免重复构建上下文导致性能回退。
- ⚠️ `QueryParam.stream=True` 时会返回 `AsyncIterator[str]`，若包装成 dict 需要额外处理（例如包装生成器或返回二元组）；否则上层期望的流式响应将被破坏。
- ⚠️ LLM 缓存使用 JSON 序列化字符串；返回字典时要统一序列化/反序列化，且 `args_hash` 需包含新参数以避免缓存串味。
- ⚠️ 现有 API/CLI/测试默认接收字符串，需要逐一排查调用点（API 路由、CLI 输出、docs 示例），防止因返回类型改变导致崩溃。

## TODO 清单（按执行顺序）

1. **参数层**：在 `lightrag/base.py` 的 `QueryParam` 中新增 `include_sources: bool = False`，并补充 docstring；确认默认值保持兼容。
2. **上下文复用**：重构 `_build_query_context` 使其同时产出文本 & 结构化两套结果（例如返回 `ContextPayload(text: str, structured: dict | None)`）；保证只构建一次上下文并向下游提供两种视图。
3. **kg_query 适配**：
   - 使用步骤2的返回结构来生成 prompt（继续用文本版）。
   - 当 `include_sources` 为真时，携带结构化结果并在最终返回值封装 `{ "answer": xxx, "sources": {...}}`。
   - 处理 `stream=True`：考虑返回自定义对象（如 `QueryStreamResult(answer_stream, sources)`）或生成器包装器，确保调用方可迭代且同时访问 `sources`。
   - 更新失败分支和 `cache` 命中逻辑，保证布尔参数参与缓存 key，并在缓存中存储/取出统一的 JSON 字符串。
4. **naive_query 适配**：复制与步骤3同样的改造（上下文产出、返回格式、流式、缓存）。
5. **同步查询封装**：在 `lightrag/lightrag.py` 中的 `query`/`aquery` 处理新返回类型，确保同步接口继续返回字符串或带源信息的结构体；为流式场景提供统一访问方式。
6. **API & CLI 调整**：排查 `lightrag/api` 路由与 `cli.py`，根据 `include_sources` 判定输出格式（JSON 响应、控制台展示源信息）。
7. **测试补充**：新增/调整测试覆盖（至少增加一个包含 `include_sources=True` 的单元或集成测试，验证普通/流式/缓存命中场景）。
8. **文档更新**：更新 README / docs / 示例代码，说明如何开启 source 输出、返回结构示例，以及流式模式的使用方法。
9. **回归校验**：运行既定校验脚本（`ruff format`, `ruff check`, 相关测试）并记录结果；若新增测试需在 `tests/` 下编写。

## 实施思路摘要

- 先改造上下文构建以避免重复开销，再逐层向外扩散改动，减少回滚成本。
- 对流式返回可采用「包装生成器 + 附加属性」或自定义数据类的方案，提前在 API/CLI 约定访问方式。
- 缓存写读采用 JSON 字符串统一格式，命中后需解包恢复 `{answer, sources}` 结构，保持 include_sources=False 场景的透明性。
- 对外接口统一通过文档和示例告知返回格式可能是字符串或结构体，提醒调用方适配。

# LightRAG Query方法增强计划：同时返回答案和检索源信息

## 分析背景

通过对LightRAG项目的深入分析，发现现有的query方法已经具备完整的检索能力，但默认只返回问答结果。用户需要能够同时获取检索源信息和问答结果，以便了解答案的来源和依据。

## 核心发现

### 1. 数据流分析

- **主入口点**: `lightrag.py:2035-2056` - `query()` 和 `aquery()` 方法
- **检索处理**: `operate.py:2100` - `kg_query()` 函数处理知识图谱查询
- **上下文构建**: `operate.py:2465` - `_build_query_context()` 函数构建检索上下文
- **LLM调用**: `operate.py:2213-2218` - 将检索上下文注入prompt并调用LLM

### 2. 检索源信息结构

在 `_build_query_context` 函数中，已经存在完整的检索源信息：

```python
# 结构化数据格式（当 return_structured=True 时）
{
    "entities": [...],           # 实体列表
    "relationships": [...],      # 关系列表
    "chunks": [...],            # 文本块列表
    "metadata": {
        "total_entities": int,
        "total_relationships": int,
        "total_chunks": int,
        "mode": str,
        "query": str
    }
}
```

### 3. 关键数据流

```
用户查询 → kg_query/naive_query → _build_query_context → LLM调用 → 返回答案
                                    ↑
                               检索源信息已在此处生成
```

## 实现方案

### 方案选择：单次查询 + 条件返回

**优势**：

- 零性能损失：避免重复计算
- 向后兼容：默认行为不变
- 数据完整：利用现有检索数据
- 代码简洁：最小化修改

### 具体修改计划

#### 1. 修改 QueryParam 类

**文件**: `lightrag/base.py:162` 行后

**添加参数**:

```python
include_sources: bool = False  # 控制是否返回检索源信息
```

#### 2. 修改 kg_query 函数返回逻辑

**文件**: `lightrag/operate.py:2213` 行附近

**修改内容**:

```python
# 在 LLM 调用前保存检索源信息
sources_data = None
if query_param.include_sources:
    # 利用 _build_query_context 返回的结构化数据
    sources_data = {
        "entities": entities_context,
        "relationships": relations_context,
        "chunks": text_units_context,
        "metadata": {
            "total_entities": len(entities_context),
            "total_relationships": len(relations_context),
            "total_chunks": len(text_units_context),
            "mode": query_param.mode,
            "query": query
        }
    }

# LLM 调用获取答案
response = await use_model_func(...)

# 根据参数决定返回格式
if query_param.include_sources and sources_data:
    return {
        "answer": response,
        "sources": sources_data
    }
else:
    return response
```

#### 3. 修改 naive_query 函数

**文件**: `lightrag/operate.py` - naive_query 函数

采用相同的模式修改返回逻辑，确保一致性。

#### 4. 修改函数返回类型注解

**文件**: `lightrag/lightrag.py:2061` 行

**修改内容**:

```python
-> str | AsyncIterator[str] | dict
```

## 使用示例

### 标准查询（保持不变）

```python
result = rag.query("What is LightRAG?")
# 返回: "LightRAG is a lightweight retrieval-augmented generation system..."
```

### 带检索源的查询

```python
result_with_sources = rag.query(
    "What is LightRAG?",
    param=QueryParam(include_sources=True)
)
# 返回: {
#     "answer": "LightRAG is a lightweight retrieval-augmented generation system...",
#     "sources": {
#         "entities": [...],
#         "relationships": [...],
#         "chunks": [...],
#         "metadata": {...}
#     }
# }
```

## 实现优势

1. **性能最优**: 一次查询完成所有工作，无重复计算
2. **向后兼容**: 现有代码无需修改，默认行为保持不变
3. **数据完整**: 提供完整的检索源信息，包括实体、关系和文本块
4. **扩展性强**: 为后续功能增强奠定基础
5. **代码简洁**: 最小化修改，降低引入错误的风险

## 注意事项

1. 需要同时修改 `kg_query` 和 `naive_query` 函数确保一致性
2. 需要考虑流式输出的处理逻辑
3. 需要更新相关文档和示例代码
4. 需要确保类型注解的准确性

## 实施步骤

1. **修改 QueryParam 类** - 添加新参数
2. **修改 kg_query 函数** - 实现条件返回逻辑
3. **修改 naive_query 函数** - 确保一致性
4. **更新类型注解** - 修正返回类型
5. **测试验证** - 确保功能正常且向后兼容
6. **文档更新** - 更新使用说明和示例

这个方案完美实现了用户需求：在保持现有功能完整性的同时，提供检索源信息的返回能力。
