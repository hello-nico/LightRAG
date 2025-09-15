# LightRAG Hybrid Query 过程详细分析

## 概述

LightRAG 的 hybrid query 是一种结合了 local 和 global 查询优势的混合模式，通过同时利用实体检索和关系检索来提供全面的知识图谱查询能力。hybrid 模式既关注具体的技术细节，又理解概念间的关系网络，为用户提供从微观到宏观的完整知识视图。本文档将详细分析 hybrid query 的完整执行流程，包括每个步骤的具体操作和实现机制。

## Hybrid Query 完整流程

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
if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
    logger.warning("high_level_keywords is empty")
if hl_keywords == [] and ll_keywords == []:
    if len(query) < 50:
        logger.warning("Both high and low level keywords are empty, fallback to naive mode")
        query_param.mode = "naive"
```

### 第三步：混合数据检索（Hybrid 模式核心）

**函数**: `_build_query_context()` (lightrag/operate.py:2533)

**操作步骤**:

1. **双路径并行检索**:
   - Local 路径：使用低层关键词检索实体和关系
   - Global 路径：使用高层关键词检索关系和实体
2. **异步并发执行**: 两个检索路径并行执行，提高效率
3. **数据去重合并**: 使用轮询算法合并两个路径的结果

**详细过程**:

```python
else:  # hybrid or mix mode
    if len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )
    if len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )
```

**关键特性**:
- **选择性执行**: 只有当对应的关键词不为空时才执行对应路径
- **并行处理**: Local 和 Global 路径可以并行执行
- **独立性**: 两个路径的检索结果独立，后续进行合并

### 第四步：Round-Robin 数据合并

**函数**: `_build_query_context()` (lightrag/operate.py:2570)

**操作步骤**:

1. **实体合并**: 使用轮询算法合并 local 和 global 实体
2. **关系合并**: 使用轮询算法合并 local 和 global 关系
3. **去重处理**: 基于实体名称和关系对的去重
4. **公平性保证**: 确保两个来源的数据都有公平的表现机会

**详细实现**:

```python
# Use round-robin merge to combine local and global data fairly
final_entities = []
seen_entities = set()
# Round-robin merge entities
max_len = max(len(local_entities), len(global_entities))
for i in range(max_len):
    # First from local
    if i < len(local_entities):
        entity = local_entities[i]
        entity_name = entity.get("entity_name")
        if entity_name and entity_name not in seen_entities:
            final_entities.append(entity)
            seen_entities.add(entity_name)
    # Then from global
    if i < len(global_entities):
        entity = global_entities[i]
        entity_name = entity.get("entity_name")
        if entity_name and entity_name not in seen_entities:
            final_entities.append(entity)
            seen_entities.add(entity_name)
```

**关系合并逻辑**:
```python
# Round-robin merge relations
final_relations = []
seen_relations = set()
max_len = max(len(local_relations), len(global_relations))
for i in range(max_len):
    # First from local
    if i < len(local_relations):
        relation = local_relations[i]
        relation_key = (relation.get("src_id"), relation.get("tgt_id"))
        if relation_key not in seen_relations:
            final_relations.append(relation)
            seen_relations.add(relation_key)
    # Then from global
    if i < len(global_relations):
        relation = global_relations[i]
        relation_key = (relation.get("src_id"), relation.get("tgt_id"))
        if relation_key not in seen_relations:
            final_relations.append(relation)
            seen_relations.add(relation_key)
```

### 第五步：文本块检索和处理

**操作步骤**:

1. **多源文本块检索**:
   - 从实体相关的文本块（local 路径）
   - 从关系相关的文本块（global 路径）
2. **策略选择**: 根据配置选择文本块选择策略（WEIGHT 或 VECTOR）
3. **权重计算**: 计算每个文本块的权重和出现频率
4. **去重处理**: 对文本块进行去重处理

**关键代码**:

```python
# Find deduplicated chunks from entities
if final_entities:
    entity_chunks = await _find_related_text_unit_from_entities(
        final_entities,
        query_param,
        text_chunks_db,
        knowledge_graph_inst,
        query,
        chunks_vdb,
        chunk_tracking=chunk_tracking,
        query_embedding=query_embedding,
    )

# Find deduplicated chunks from relations
if final_relations:
    relation_chunks = await _find_related_text_unit_from_relations(
        final_relations,
        query_param,
        text_chunks_db,
        entity_chunks,  # Pass entity chunks for deduplication
        query,
        chunks_vdb,
        chunk_tracking=chunk_tracking,
        query_embedding=query_embedding,
    )
```

### 第六步：最终上下文格式化

**操作步骤**:

1. **Token 预算计算**: 计算可用于文本块的 token 数量
2. **三轮轮询合并**: 合并向量块、实体块和关系块
3. **上下文组装**: 将实体、关系和文本块组装成最终上下文
4. **格式化输出**: 按照指定格式输出上下文信息

**合并算法**:

```python
# Round-robin merge chunks from different sources with deduplication by chunk_id
merged_chunks = []
seen_chunk_ids = set()
max_len = max(len(entity_chunks), len(relation_chunks))
origin_len = len(entity_chunks) + len(relation_chunks)

for i in range(max_len):
    if i < len(entity_chunks) and entity_chunks[i]["chunk_id"] not in seen_chunk_ids:
        merged_chunks.append(entity_chunks[i])
        seen_chunk_ids.add(entity_chunks[i]["chunk_id"])
    if i < len(relation_chunks) and relation_chunks[i]["chunk_id"] not in seen_chunk_ids:
        merged_chunks.append(relation_chunks[i])
        seen_chunk_ids.add(relation_chunks[i]["chunk_id"])
```

### 第七步：LLM 生成响应

**操作步骤**:

1. **提示词构建**: 构建包含混合上下文的系统提示词
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

## 关键技术细节

### 1. Round-Robin 合并算法

**设计原理**:
- **公平性**: 确保 local 和 global 数据源都有平等的表现机会
- **顺序保证**: 优先考虑 local 数据，然后是 global 数据
- **去重机制**: 基于唯一标识符进行去重

**算法复杂度**:
- 时间复杂度: O(n)，其中 n 是两个数据源的最大长度
- 空间复杂度: O(n)，用于存储去重集合

**实现细节**:

```python
# 实体合并的键值构建
entity_key = entity.get("entity_name")

# 关系合并的键值构建
relation_key = (relation.get("src_id"), relation.get("tgt_id"))

# 文本块合并的键值构建
chunk_key = chunk.get("chunk_id")
```

### 2. Token 控制策略

**动态 Token 分配**:

```python
# 计算 KG 上下文 token 数量
kg_context_tokens = len(tokenizer.encode(kg_context))

# 计算系统提示词开销
sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

# 计算可用于文本块的 token 数量
available_chunk_tokens = max_total_tokens - (kg_context_tokens + sys_prompt_overhead + buffer_tokens)
```

**分层 Token 限制**:
- **实体上下文**: `max_entity_tokens`（默认 6000）
- **关系上下文**: `max_relation_tokens`（默认 8000）
- **总 Token 预算**: `max_total_tokens`（默认 30000）
- **安全缓冲**: 预留 100 个 token

### 3. 文本块选择策略

**WEIGHT 策略**:
- 基于出现次数的线性梯度权重轮询
- 出现频率越高的块，权重越大
- 适用于有明显重要性的场景

**VECTOR 策略**:
- 基于嵌入向量的余弦相似度选择
- 考虑语义相关性
- 需要预计算查询嵌入向量

**配置选择**:

```python
kg_chunk_pick_method = text_chunks_db.global_config.get(
    "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
)
```

### 4. 并发处理机制

**异步并发执行**:

```python
# Local 路径的并发操作
nodes_dict, degrees_dict = await asyncio.gather(
    knowledge_graph_inst.get_nodes_batch(node_ids),
    knowledge_graph_inst.node_degrees_batch(node_ids),
)

# Global 路径的并发操作
edge_data_dict, edge_degrees_dict = await asyncio.gather(
    knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
    knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
)
```

### 5. 错误处理和回退机制

**关键词缺失处理**:

```python
if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
    logger.warning("low_level_keywords is empty")
if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
    logger.warning("high_level_keywords is empty")
```

**双关键词为空的回退**:

```python
if hl_keywords == [] and ll_keywords == []:
    if len(query) < 50:
        logger.warning("Both high and low level keywords are empty, fallback to naive mode")
        query_param.mode = "naive"
```

## 性能优化机制

### 1. 缓存策略

- **查询缓存**: 基于查询参数哈希的缓存
- **关键词缓存**: 关键词提取结果缓存
- **LLM 响应缓存**: LLM 生成结果缓存

### 2. 批处理操作

```python
# 批量获取节点数据
nodes_dict = await knowledge_graph_inst.get_nodes_batch(node_ids)

# 批量获取节点度数
degrees_dict = await knowledge_graph_inst.node_degrees_batch(node_ids)

# 批量获取边数据
edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

# 批量获取边度数
edge_degrees_dict = await knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples)
```

### 3. 智能去重

- **实体去重**: 基于实体名称
- **关系去重**: 基于排序后的实体对
- **文本块去重**: 基于 chunk_id

### 4. 预计算优化

```python
# 预计算查询嵌入向量
if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
    embedding_func_config = text_chunks_db.embedding_func
    if embedding_func_config and embedding_func_config.func:
        try:
            query_embedding = await embedding_func_config.func([query])
            query_embedding = query_embedding[0]  # Extract first embedding
            logger.debug("Pre-computed query embedding for all vector operations")
        except Exception as e:
            logger.warning(f"Failed to pre-compute query embedding: {e}")
            query_embedding = None
```

## 与其他模式的对比

### Hybrid vs Local

| 特性 | Local 模式 | Hybrid 模式 |
|------|------------|-------------|
| **数据源** | 仅实体检索 | 实体 + 关系检索 |
| **关键词依赖** | 仅低层关键词 | 低层 + 高层关键词 |
| **上下文广度** | 窄（技术细节） | 宽（技术 + 概念） |
| **检索复杂度** | 低 | 中等 |
| **适用场景** | 具体技术问题 | 综合性技术问题 |

### Hybrid vs Global

| 特性 | Global 模式 | Hybrid 模式 |
|------|------------|-------------|
| **数据源** | 仅关系检索 | 关系 + 实体检索 |
| **关键词依赖** | 仅高层关键词 | 高层 + 低层关键词 |
| **上下文深度** | 浅（概念关系） | 深（概念 + 细节） |
| **检索复杂度** | 中等 | 中等 |
| **适用场景** | 概念性理解 | 深度概念分析 |

### Hybrid vs Mix

| 特性 | Hybrid 模式 | Mix 模式 |
|------|------------|---------|
| **数据源数量** | 2个 (Entity + Relationship) | 3个 (Vector + Entity + Relationship) |
| **向量检索** | 无 | 有 |
| **上下文完整性** | 良好 | 优秀 |
| **性能开销** | 中等 | 较高 |
| **适用场景** | 结构化知识查询 | 复杂综合查询 |

## 数据流向图

```
查询输入 → 缓存检查 → 关键词提取 → 并行检索 → 数据合并 → 文本块检索 → Token控制 → LLM生成 → 结果返回
     ↓            ↓            ↓           ↓           ↓           ↓          ↓         ↓
  缓存命中     高/低层词   Local/Global  Round-Robin  多源文本块   分块处理    最终响应
                          实体/关系      去重合并      去重合并
```

## 适用场景

### 1. 技术架构分析

**场景**: 分析系统架构、技术栈选择、设计模式等
**优势**: 既能理解具体技术实现，又能把握整体架构关系

### 2. 问题诊断和解决

**场景**: 复杂技术问题的诊断和解决方案设计
**优势**: 结合具体技术细节和概念性理解，提供全面的解决方案

### 3. 知识图谱探索

**场景**: 探索知识图谱中的概念关系和实体属性
**优势**: 双重视角的检索提供更完整的知识网络

### 4. 跨领域知识整合

**场景**: 需要整合多个领域知识的综合性查询
**优势**: 能够同时处理领域内的技术细节和跨领域的概念关系

## 参数调优建议

### 关键参数

- `top_k`: 控制检索的实体和关系数量（默认 60）
- `max_entity_tokens`: 实体上下文 token 限制（默认 6000）
- `max_relation_tokens`: 关系上下文 token 限制（默认 8000）
- `max_total_tokens`: 总 token 预算（默认 30000）
- `kg_chunk_pick_method`: 文本块选择策略（WEIGHT/VECTOR）

### 调优策略

1. **平衡检索**:
   - 调整 `top_k` 以平衡检索广度和精度
   - 根据查询复杂度动态调整 token 分配

2. **关键词质量**:
   - 优化关键词提取提示词
   - 考虑使用预定义关键词提高准确性

3. **性能优化**:
   - 启用缓存机制提高响应速度
   - 使用批处理操作提高并发效率

## 常见问题解决方案

### 1. 单一关键词为空

**问题**: 只有低层或高层关键词，导致检索不完整
**解决方案**:
- 检查查询内容的表述方式
- 优化关键词提取提示词
- 考虑使用 mix 模式补充向量检索

### 2. 数据源不平衡

**问题**: Local 和 Global 检索结果数量差异很大
**解决方案**:
- 调整各自的 `top_k` 参数
- 优化关键词的提取质量
- 考虑使用不同的检索策略

### 3. Token 超限

**问题**: 混合上下文超过模型 token 限制
**解决方案**:
- 调整 `max_entity_tokens` 和 `max_relation_tokens` 比例
- 启用文本块重排序
- 优化实体和关系的选择策略

### 4. 重复内容过多

**问题**: Local 和 Global 检索结果中有大量重复
**解决方案**:
- 加强去重机制
- 优化检索策略
- 调整关键词的特异性

## 总结

LightRAG 的 hybrid query 过程是一个精心设计的双路径检索增强生成流程，通过结合 local 和 global 模式的优势，为用户提供了全面、平衡的知识图谱查询体验。整个过程充分利用了异步处理、缓存优化、并发计算和智能合并等技术，确保了系统的性能和可扩展性。

hybrid 模式特别适合需要同时理解技术细节和概念关系的综合性查询场景，是 LightRAG 六种查询模式中最具平衡性和实用性的一种。通过合理配置参数和优化检索策略，hybrid 模式能够在性能和质量之间找到最佳平衡点。