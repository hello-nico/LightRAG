# LightRAG Local Query 过程详细分析

## 概述

LightRAG 的 local query 是一种基于实体检索的查询模式，专注于获取与查询相关的实体及其关系的详细信息。本文档将详细分析 local query 的完整执行流程，包括每个步骤的具体操作和实现机制。

## Local Query 完整流程

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
if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
    logger.warning("low_level_keywords is empty")
```

### 第三步：实体检索（Local 模式核心）

**函数**: `_get_node_data()` (lightrag/operate.py:3007)

**操作步骤**:

1. **向量相似度搜索**: 在实体向量数据库中搜索与低层关键词相似的实体
2. **实体数据获取**: 批量获取实体的详细信息和度数
3. **关系发现**: 从检索到的实体中找到最相关的关系
4. **数据格式化**: 整合实体和关系数据，添加排序信息

**详细过程**:

```python
# 1. 向量相似度搜索
results = await entities_vdb.query(query, top_k=query_param.top_k)

# 2. 提取实体ID
node_ids = [r["entity_name"] for r in results]

# 3. 批量获取实体数据和度数
nodes_dict, degrees_dict = await asyncio.gather(
    knowledge_graph_inst.get_nodes_batch(node_ids),
    knowledge_graph_inst.node_degrees_batch(node_ids),
)

# 4. 格式化实体数据
node_datas = [
    {
        **n,
        "entity_name": k["entity_name"],
        "rank": d,
        "created_at": k.get("created_at"),
    }
    for k, n, d in zip(results, node_datas, node_degrees)
    if n is not None
]

# 5. 查找相关关系
use_relations = await _find_most_related_edges_from_entities(
    node_datas, query_param, knowledge_graph_inst
)
```

### 第四步：上下文构建

**函数**: `_build_query_context()` (lightrag/operate.py:2465)

**操作步骤**:

1. **模式判断**: 确认查询模式为 "local" 且存在低层关键词
2. **调用实体检索**: 获取实体和关系数据
3. **数据合并**: 使用轮询算法合并本地和全局数据（如果有）
4. **Token 控制**: 对实体和关系数据进行 token 限制处理
5. **文本块检索**: 基于实体检索相关的文本块

**关键代码**:

```python
# Handle local mode
if query_param.mode == "local" and len(ll_keywords) > 0:
    local_entities, local_relations = await _get_node_data(
        ll_keywords, knowledge_graph_inst, entities_vdb, query_param
    )

# Token control and truncation
entities_context = truncate_list_by_token_size(
    entities_context,
    key=lambda x: json.dumps(x, ensure_ascii=False),
    max_token_size=max_entity_tokens,
    tokenizer=tokenizer,
)
```

### 第五步：文本块检索和处理

**函数**: `_find_related_text_unit_from_entities()` (lightrag/operate.py:3121)

**操作步骤**:

1. **实体块收集**: 为每个实体收集相关的文本块
2. **策略选择**: 根据配置选择文本块选择策略（WEIGHT 或 VECTOR）
3. **权重计算**: 计算每个文本块的权重
4. **去重处理**: 对文本块进行去重处理
5. **Token 限制**: 根据可用的 token 预算限制文本块数量

**支持的选择策略**:

- **WEIGHT**: 基于出现次数的线性梯度权重轮询
- **VECTOR**: 基于嵌入向量的余弦相似度选择

### 第六步：最终上下文格式化

**操作步骤**:

1. **Token 预算计算**: 计算可用于文本块的 token 数量
2. **动态分块**: 根据可用 token 数量动态调整文本块
3. **上下文组装**: 将实体、关系和文本块组装成最终上下文
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

## 关键优化机制

### 1. 缓存机制

- **查询缓存**: 基于查询参数哈希的缓存
- **关键词缓存**: 关键词提取结果缓存
- **LLM 响应缓存**: LLM 生成结果缓存

### 2. Token 控制系统

- **分层限制**: 实体、关系、文本块的分层 token 限制
- **动态调整**: 根据实际使用情况动态调整各部分 token 预算
- **安全缓冲**: 预留 100 个 token 作为安全缓冲

### 3. 并发处理

- **批量操作**: 实体数据获取和度数计算并发执行
- **异步 I/O**: 所有数据库操作都使用异步 I/O
- **批处理**: 边缘数据获取使用批处理操作

### 4. 去重机制

- **实体去重**: 基于实体名称的去重
- **关系去重**: 基于实体对的去重
- **文本块去重**: 基于块 ID 的去重

## 数据流向图

```
查询输入 → 缓存检查 → 关键词提取 → 实体检索 → 关系发现 → 上下文构建 → 文本块检索 → Token控制 → LLM生成 → 结果返回
     ↓            ↓            ↓           ↓           ↓           ↓           ↓          ↓         ↓
  缓存命中     高/低层词     向量搜索    实体数据    关系数据    格式化上下文  相关文本块   分块处理    最终响应
```

## 性能特点

### 优势

1. **精准检索**: 基于实体向量检索，准确性高
2. **详细信息**: 提供实体的详细描述和关系
3. **Token 效率**: 智能的 token 控制机制
4. **缓存优化**: 多层缓存提高响应速度

### 适用场景

- 技术实现细节查询
- 特定概念的深入理解
- 需要具体案例和应用场景的问题
- 基于实体的专业知识查询

## 参数调优建议

### 关键参数

- `top_k`: 控制检索的实体数量（默认 60）
- `max_entity_tokens`: 实体上下文 token 限制（默认 6000）
- `max_relation_tokens`: 关系上下文 token 限制（默认 8000）
- `max_total_tokens`: 总 token 预算（默认 30000）

### 调优策略

1. **简单查询**: 减少 `top_k` 和 token 限制
2. **复杂查询**: 增加 `top_k` 和 token 限制
3. **性能优先**: 启用缓存和重排序
4. **准确性优先**: 使用更高的 token 限制

## 总结

LightRAG 的 local query 过程是一个精心设计的检索增强生成流程，通过实体检索、关系发现、文本块检索和智能的 token 控制机制，为用户提供了高质量、信息丰富的查询响应。整个过程充分利用了异步处理、缓存优化和并发计算等技术，确保了系统的性能和可扩展性。

local 模式特别适合需要详细技术信息和深入理解特定概念的查询场景，是 LightRAG 六种查询模式中最具专业性和技术深度的一种。
