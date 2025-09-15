# LightRAG Global Query 过程详细分析

## 概述

LightRAG 的 global query 是一种基于关系检索的查询模式，专注于获取与查询相关的实体间关系和全局知识结构。与 local query 的实体检索不同，global query 首先检索关系，然后通过关系推导相关实体，从而提供更全面的概念性理解。本文档将详细分析 global query 的完整执行流程，包括每个步骤的具体操作和实现机制。

## Global Query 完整流程

### 第一步：查询初始化和缓存检查

**函数**: `kg_query()` (lightrag/operate.py:2100)

**操作步骤**:

1. **参数验证**: 检查查询是否为空，如果为空则返回失败响应
2. **模型函数选择**:
   - 如果 `query_param.model_func` 存在，使用指定的模型函数
   - 否则使用全局配置中的 LLM 模型函数，并设置高优先级（5）
3. **缓存处理**:
   - 计算查询参数的哈希值 (`compute_args_hash`)
   - 检查缓存中是否存在相同的查询结果
   - 如果命中缓存，直接返回缓存结果

**关键代码**:

```python
# Handle cache
args_hash = compute_args_hash(
    query_param.mode,
    query,
    query_param.response_type,
    query_param.top_k,
    query_param.chunk_top_k,
    query_param.max_entity_tokens,
    query_param.max_relation_tokens,
    query_param.max_total_tokens,
    query_param.hl_keywords or [],
    query_param.ll_keywords or [],
    query_param.user_prompt or "",
    query_param.enable_rerank,
)
cached_response = await handle_cache(
    hashing_kv, args_hash, query, query_param.mode, cache_type="query"
)
if cached_response is not None:
    return cached_response
```

### 第二步：关键词提取

**函数**: `get_keywords_from_query()` (lightrag/operate.py:2260)

**操作步骤**:

1. **预定义关键词检查**: 检查查询参数中是否已提供关键词
2. **LLM 关键词提取**: 如果没有预定义关键词，调用 LLM 提取关键词
3. **关键词分类**: 将关键词分为高层关键词（hl_keywords）和低层关键词（ll_keywords）
4. **空关键词处理**: 如果关键词为空，进行相应的警告和处理

**关键代码**:

```python
hl_keywords, ll_keywords = await get_keywords_from_query(
    query, query_param, global_config, hashing_kv
)

# Handle empty keywords
if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
    logger.warning("high_level_keywords is empty")
if hl_keywords == [] and ll_keywords == []:
    if len(query) < 50:
        logger.warning("Both high and low level keywords are empty, fallback to naive mode")
        query_param.mode = "naive"
```

### 第三步：关系检索（Global 模式核心）

**函数**: `_get_edge_data()` (lightrag/operate.py:3072)

**操作步骤**:

1. **向量相似度搜索**: 在关系向量数据库中搜索与高层关键词相似的关系
2. **关系数据获取**: 批量获取关系的详细属性和度数
3. **实体推导**: 从检索到的关系中推导相关实体
4. **数据格式化**: 整合关系和实体数据，添加排序信息

**详细过程**:

```python
# 1. 向量相似度搜索
results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

# 2. 提取关系对
edge_pairs = [(r["src_id"], r["tgt_id"]) for r in results]

# 3. 批量获取关系数据和度数
edge_data_dict, edge_degrees_dict = await asyncio.gather(
    knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
    knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
)

# 4. 格式化关系数据
all_edges_data = [
    {
        "src_tgt": pair,
        "rank": edge_degrees_dict.get(pair, 0),
        **edge_props,
    }
    for pair, edge_props in zip(all_edges, edge_data_dict.values())
    if edge_props is not None
]

# 5. 推导相关实体
use_entities = await _find_most_related_entities_from_relationships(
    edge_datas, query_param, knowledge_graph_inst
)
```

**返回结果结构**:

- `edge_datas`: 格式化的关系数据列表，每个关系包含 `src_tgt`、`rank`、`weight` 等属性
- `use_entities`: 从关系推导出的实体数据列表

### 第四步：上下文构建

**函数**: `_build_query_context()` (lightrag/operate.py:2465)

**操作步骤**:

1. **模式判断**: 确认查询模式为 "global" 且存在高层关键词
2. **调用关系检索**: 获取关系和实体数据
3. **数据合并**: 使用轮询算法合并全局和本地数据（如果有）
4. **Token 控制**: 对关系和实体数据进行 token 限制处理
5. **文本块检索**: 基于关系检索相关的文本块

**关键代码**:

```python
# Handle global mode
elif query_param.mode == "global" and len(hl_keywords) > 0:
    global_relations, global_entities = await _get_edge_data(
        hl_keywords,
        knowledge_graph_inst,
        relationships_vdb,
        query_param,
    )

# Token control and truncation
relations_context = truncate_list_by_token_size(
    relations_context,
    key=lambda x: json.dumps(x, ensure_ascii=False),
    max_token_size=max_relation_tokens,
    tokenizer=tokenizer,
)
```

### 第五步：文本块检索和处理

**函数**: `_find_related_text_unit_from_relations()` (lightrag/operate.py:3276)

**操作步骤**:

1. **关系块收集**: 为每个关系收集相关的文本块
2. **策略选择**: 根据配置选择文本块选择策略（WEIGHT 或 VECTOR）
3. **权重计算**: 计算每个文本块的权重
4. **去重处理**: 对文本块进行去重处理
5. **Token 限制**: 根据可用的 token 预算限制文本块数量

**支持的选择策略**:

- **WEIGHT**: 基于出现次数的线性梯度权重轮询
- **VECTOR**: 基于嵌入向量的余弦相似度选择

**关键代码**:

```python
# Collect text chunks from relationships
for edge in edge_datas:
    src_entity = edge.get("src_id", "")
    tgt_entity = edge.get("tgt_id", "")
    if src_entity and tgt_entity:
        chunk_ids = set()
        # Get chunks from source entity
        if src_entity in entity_source_id_map:
            chunk_ids.update(split_string_by_multi_markers(
                entity_source_id_map[src_entity], [GRAPH_FIELD_SEP]
            ))
        # Get chunks from target entity
        if tgt_entity in entity_source_id_map:
            chunk_ids.update(split_string_by_multi_markers(
                entity_source_id_map[tgt_entity], [GRAPH_FIELD_SEP]
            ))
```

### 第六步：最终上下文格式化

**操作步骤**:

1. **Token 预算计算**: 计算可用于文本块的 token 数量
2. **动态分块**: 根据可用 token 数量动态调整文本块
3. **上下文组装**: 将关系、实体和文本块组装成最终上下文
4. **格式化输出**: 按照指定格式输出上下文信息

**Token 计算过程**:

```python
# 计算 KG 上下文 token 数量
kg_context_tokens = len(tokenizer.encode(kg_context))

# 计算系统提示词开销
sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

# 计算可用于文本块的 token 数量
available_chunk_tokens = max_total_tokens - (kg_context_tokens + sys_prompt_overhead + buffer_tokens)
```

### 第七步：LLM 生成响应

**操作步骤**:

1. **提示词构建**: 构建包含上下文的系统提示词
2. **LLM 调用**: 使用 LLM 生成最终响应
3. **响应清理**: 清理响应中的冗余信息
4. **缓存保存**: 将结果保存到缓存中

**关键代码**:

```python
# 构建系统提示词
sys_prompt = sys_prompt_temp.format(
    context_data=context,
    response_type=query_param.response_type,
    history=history_context,
    user_prompt=user_prompt,
)

# 调用 LLM
response = await use_model_func(
    query,
    system_prompt=sys_prompt,
    stream=query_param.stream,
    enable_cot=True,
)
```

### 第八步：结果返回和缓存

**操作步骤**:

1. **响应清理**: 移除响应中的系统提示词和查询文本
2. **缓存存储**: 将结果存储到 LLM 缓存中
3. **结果返回**: 返回最终生成的响应

## 关键技术细节

### 1. relationships_vdb.query 工作机制

`relationships_vdb.query()` 方法返回的关系数据结构：

```python
results = [
    {
        "src_id": "entity1",
        "tgt_id": "entity2",
        "content": "entity1 + entity2 + description",
        "description": "relationship description",
        "source_id": "doc1,doc2",
        "weight": 1.0,
        "distance": 0.85,
        "created_at": "timestamp"
    }
    # ... 更多关系
]
```

### 2. 关系权重和边属性处理

**权重计算**:

- 默认权重为 1.0
- 如果关系中缺少 `weight` 属性，系统会发出警告并使用默认值
- 最终排序基于 `rank` 和 `weight` 的组合：`key=lambda x: (x["rank"], x["weight"])`

**边属性处理**:

```python
if "weight" not in edge_props:
    logger.warning(f"Edge {pair} missing 'weight' attribute, using default value 1.0")
    edge_props["weight"] = 1.0
```

### 3. 轮询合并算法

在混合模式下，global query 使用轮询算法合并不同来源的数据：

```python
# Round-robin merge chunks from different sources
merged_chunks = []
seen_chunk_ids = set()
max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))

for i in range(max_len):
    if i < len(vector_chunks) and vector_chunks[i]["chunk_id"] not in seen_chunk_ids:
        merged_chunks.append(vector_chunks[i])
        seen_chunk_ids.add(vector_chunks[i]["chunk_id"])
    if i < len(entity_chunks) and entity_chunks[i]["chunk_id"] not in seen_chunk_ids:
        merged_chunks.append(entity_chunks[i])
        seen_chunk_ids.add(entity_chunks[i]["chunk_id"])
    if i < len(relation_chunks) and relation_chunks[i]["chunk_id"] not in seen_chunk_ids:
        merged_chunks.append(relation_chunks[i])
        seen_chunk_ids.add(relation_chunks[i]["chunk_id"])
```

### 4. Token 控制机制

Global 模式的 Token 分配策略：

- **关系上下文**: `max_relation_tokens`（默认 8000）
- **实体上下文**: `max_entity_tokens`（默认 6000）
- **总 Token 预算**: `max_total_tokens`（默认 30000）
- **安全缓冲**: 预留 100 个 token

### 5. 空高层关键词的处理逻辑

当 `hl_keywords` 为空时：

```python
if query_param.mode == "global" and len(hl_keywords) > 0:
    # 执行 global 查询
    global_relations, global_entities = await _get_edge_data(...)
else:
    # 跳过 global 查询
    logger.warning("high_level_keywords is empty")
```

## 性能优化机制

### 1. 缓存策略

- **查询缓存**: 基于查询参数哈希的缓存
- **关键词缓存**: 关键词提取结果缓存
- **LLM 响应缓存**: LLM 生成结果缓存

### 2. 批处理操作

```python
# 并发执行多个批处理操作
edge_data_dict, edge_degrees_dict = await asyncio.gather(
    knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
    knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
)
```

### 3. 去重机制

- **关系去重**: 基于排序后的实体对去重
- **实体去重**: 基于实体名称的去重
- **文本块去重**: 基于块 ID 的去重

### 4. 并发处理

- **批量操作**: 关系数据获取和度数计算并发执行
- **异步 I/O**: 所有数据库操作都使用异步 I/O
- **批处理**: 边缘数据获取使用批处理操作

## 与 Local 模式的对比

### 检索策略差异

| 特性 | Local 模式 | Global 模式 |
|------|------------|-------------|
| **检索起点** | 实体向量检索 | 关系向量检索 |
| **核心逻辑** | 实体 → 关系 | 关系 → 实体 |
| **关键词依赖** | 低层关键词 | 高层关键词 |
| **适用查询** | 具体技术细节 | 概念性理解 |

### 上下文构建差异

**Local 模式**:

```python
# 先获取实体，再找关系
local_entities, local_relations = await _get_node_data(
    ll_keywords, knowledge_graph_inst, entities_vdb, query_param
)
```

**Global 模式**:

```python
# 先获取关系，再推导实体
global_relations, global_entities = await _get_edge_data(
    hl_keywords, knowledge_graph_inst, relationships_vdb, query_param
)
```

### Token 分配差异

- **Local 模式**: 优先分配 token 给实体上下文
- **Global 模式**: 优先分配 token 给关系上下文
- **Hybrid 模式**: 动态调整 token 分配比例

## 数据流向图

```
查询输入 → 缓存检查 → 关键词提取 → 关系检索 → 实体推导 → 上下文构建 → 文本块检索 → Token控制 → LLM生成 → 结果返回
     ↓            ↓            ↓           ↓           ↓           ↓           ↓          ↓         ↓
  缓存命中     高/低层词     向量搜索    关系数据    实体数据    格式化上下文  相关文本块   分块处理    最终响应
```

## 性能特点

### 优势

1. **概念理解**: 基于关系检索，适合概念性查询
2. **全面性**: 提供实体间的关系网络信息
3. **Token 效率**: 智能的 token 控制机制
4. **缓存优化**: 多层缓存提高响应速度

### 适用场景

- 概念性和理论性问题
- 需要理解实体间关系的查询
- 宏观层面的知识理解
- 抽象概念的探索

## 参数调优建议

### 关键参数

- `top_k`: 控制检索的关系数量（默认 60）
- `max_relation_tokens`: 关系上下文 token 限制（默认 8000）
- `max_entity_tokens`: 实体上下文 token 限制（默认 6000）
- `max_total_tokens`: 总 token 预算（默认 30000）

### 调优策略

1. **简单查询**: 减少 `top_k` 和 token 限制
2. **复杂查询**: 增加 `top_k` 和 token 限制
3. **性能优先**: 启用缓存和重排序
4. **准确性优先**: 使用更高的 token 限制

## 常见问题解决方案

### 1. 高层关键词为空

**问题**: `hl_keywords` 为空导致 global 查询被跳过
**解决方案**:

- 检查查询内容是否过于具体
- 调整关键词提取提示词
- 考虑使用 hybrid 模式

### 2. 关系检索结果质量差

**问题**: 检索到的关系与查询不相关
**解决方案**:

- 调整关系向量数据库的嵌入模型
- 优化关系描述的质量
- 增加 `top_k` 参数值

### 3. Token 超限

**问题**: 上下文超过模型 token 限制
**解决方案**:

- 调整 `max_relation_tokens` 和 `max_entity_tokens` 比例
- 启用文本块重排序
- 优化关系和实体的选择策略

## 总结

LightRAG 的 global query 过程是一个精心设计的检索增强生成流程，通过关系检索、实体推导、文本块检索和智能的 token 控制机制，为用户提供了高质量、概念性强的查询响应。整个过程充分利用了异步处理、缓存优化和并发计算等技术，确保了系统的性能和可扩展性。

global 模式特别适合需要理解概念间关系和宏观知识结构的查询场景，是 LightRAG 六种查询模式中最具概念性和全局视野的一种。与 local 模式的互补使用可以为用户提供从具体细节到宏观概念的完整知识图谱查询体验。
