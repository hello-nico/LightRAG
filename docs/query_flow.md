# LightRAG 查询执行流程

本文档汇总 `LightRAG.aquery` 触发的完整检索与生成链路，涵盖 6 种查询模式、`kg_query`/`naive_query` 的核心逻辑、上下文构建方式以及缓存策略，便于后续扩展或排障时快速定位关键步骤。

## 1. aquery 总览

调用入口为 `LightRAG.aquery(query, param, system_prompt)`，根据 `param.mode` 分流：

| mode | 描述 | 实际调用 |
| --- | --- | --- |
| `local` | 优先实体上下文，围绕低层关键词检索知识图谱 | `kg_query` |
| `global` | 聚焦关系、全局结构 | `kg_query` |
| `hybrid` | 先本地后全局，交替合并实体/关系 | `kg_query` |
| `mix` | KG + 向量联合检索，兼顾文本相似度 | `kg_query`（附加 `_get_vector_context`） |
| `naive` | 纯向量检索，无图谱参与 | `naive_query` |
| `bypass` | 直接走 LLM（可覆盖自定义 `model_func`） | 绕过检索，直接调用模型 |

执行结束后会触发 `_query_done()`，用于通知缓存存储器任务已完成。

> `QueryParam.stream=True` 时允许流式输出；若同时请求结构化结果（`include_sources` / `return_structured`），内部会收集流式片段并最终返回字典，避免上层拿到未消费的异步迭代器。

## 2. kg_query 流程

适用于 `local/global/hybrid/mix` 模式，聚焦知识图谱相关检索。

1. **缓存键生成**：通过 `compute_args_hash` 聚合 `mode`、`query`、`top_k`、`chunk_top_k`、`max_*_tokens`、高级/低级关键词、`user_prompt`、`enable_rerank` 以及 `include_sources`、`return_structured`；命中缓存时优先解析 JSON 并直接返回。
2. **关键词抽取**：若 `QueryParam` 未显式指定，调用 `get_keywords_from_query` 生成高/低层关键词，并处理空关键词、过短查询等边界。
3. **上下文构建 `_build_query_context`**：
   - `local`：调用 `_get_node_data` 获取实体与关联边。
   - `global`：调用 `_get_edge_data` 获取关系及其两端实体。
   - `hybrid/mix`：两者兼顾，并采用轮询合并策略消除重复；`mix` 额外调用 `_get_vector_context` 进行向量粗排。
   - 完成后依据实体/关系生成结构化列表、去除无用元信息、应用统一 token 限额，并按顺序检索关联文本块：
     - `_find_related_text_unit_from_entities`：根据实体 `source_id` 中的 chunk ID 统计频次，依赖权重或向量相似度选取文本。
     - `_find_related_text_unit_from_relations`：面向关系的补充检索，保持 chunk 唯一。
     - 轮询合并向量/实体/关系来源的 chunk，记录来源标记（C/E/R）。
     - `process_chunks_unified` 根据 token 预算再次裁剪，并构建最终 chunk 列表。
4. **结果封装**：
   - `_build_query_context` 返回 `ContextPayload(text, structured)` 或结构化 `dict`，其中 `structured` 包含 `entities`、`relationships`、`chunks`、`metadata`。
   - `kg_query` 若请求 `return_structured`，将 `structured_payload` 作为主返回；若请求 `include_sources`，在生成答案后把 `answer` 与 `context` 写回同一字典。
   - `only_need_context=True` 时跳过 LLM 调用，直接返回结构化/格式化上下文。
5. **生成阶段**：在需要模型回答时，基于 `PROMPTS['rag_response']` 拼装系统提示，执行模型调用，并在必要时剥离提示文本。
6. **缓存写入**：
   - 结构化结果会被序列化为 JSON 字符串保存，普通字符串直接写入。
   - 缓存数据中记录了查询参数快照，便于后续调试。

## 3. naive_query 流程

面向 `naive` 模式，逻辑与 `kg_query` 类似，但仅依赖向量存储：

1. **缓存**：同样根据 `mode`、`query`、token 限制、`include_sources`/`return_structured` 生成哈希。
2. **向量检索**：调用 `_get_vector_context` 获取候选 chunk，并通过 `process_chunks_unified` 根据 token 预算裁剪。
3. **结构化封装**：构造仅包含 `chunks` 的 `structured_payload`，供 `return_structured` / `include_sources` 场景使用。
4. **生成与缓存**：流程与 `kg_query` 保持一致，输出可为纯答案或 `{"answer": ..., **sources}` 的字典。

## 4. 文档块抽取要点

- **实体驱动**：实体记录的 `source_id` 会拆分出关联 chunk ID，通过出现频次确定权重，并保持首次命中的顺序。
- **关系驱动**：在实体提供的 chunk 基础上补充关系相关块，优先保留未在实体中出现的内容。
- **向量补强**：`mix` 模式中 vector 检索先行，保证纯相似度的结果至少出现一次。
- **Token 管控**：依赖统一 token 控制系统，根据 `max_total_tokens` 预留系统提示与安全 buffer 后，对实体、关系、chunk 分别做精简，确保最终提示不会超过模型限制。
- **来源标记**：检索日志中通过 `chunks: E1/1 R1/1 C1/1` 等形式指示 chunk 来源及排名，便于排障。

## 5. 缓存机制摘要

- 所有查询均尝试从 `llm_response_cache` 命中；默认仅当 `enable_llm_cache` 开启时写入。
- 对于包含结构化信息的响应，统一以 JSON 字符串形式缓存；命中时会自动反序列化。
- 缓存键采用 `{mode}:{cache_type}:{args_hash}`，其中 `args_hash` 取决于 `QueryParam` 的核心字段。
- 流式响应不会写入缓存，以避免存储未完整的生成结果。

## 6. 设计提示

- 扩展新模式时：优先考虑是否可复用 `kg_query`/`naive_query` 结构，确保 `args_hash` 覆盖新增参数。
- 修改上下文构建逻辑时：同步关注 `ContextPayload` 与 `structured_payload`，避免破坏 `return_structured`/`include_sources` 的返回契约。
- 若需接入第三方服务：建议阅读 `docs/retrieval_api.md`，了解 `/api/v1/retrieve` 如何组合上述逻辑。
