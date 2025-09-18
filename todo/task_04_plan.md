# task_04 设计与实施方案（更新版）

## 背景与目标

- deer-flow 需要一个遵循其 Document 统一格式的 local search 接口，以便在深度研究链路中优先利用本地知识。
- LightRAG 当前通过 `mix` 模式综合实体、关系、向量三路检索，我们需在不破坏原有优势的前提下，返回 deer-flow 友好的结构。
- 初期迭代聚焦“真实文档块聚合”，后续可按效果将实体/关系包装为虚拟文档或补充更多结构化上下文。

## 核心设计

### 1. 检索模式策略

- 默认维持 `QueryParam.mode = "mix"`：
  - 保障实体/关系与向量三路融合，避免因图谱尚未完善导致的召回缺失。
  - 即便我们对外只返回文档块，内部仍利用实体/关系筛选 chunk，发挥 LightRAG 深度检索优势。
- 预留轻量配置（后续可扩展）允许切换 `local` / `global` / `hybrid`，以适配特定资源形态或性能需求。

### 2. 文档聚合与元信息补齐

- 在 `DeerFlowRetriever.query_relevant_documents` 内：
  - 使用 `RetrievalResult.chunks` 按 `chunk.doc_id`（缺失则回退 `chunk.id`）聚合。
  - 构建 `DeerFlowDocument`：
    - `id`: `doc_id` 或回退值。
    - `chunks`: 将原始 chunk 转成 `DeerFlowChunk(content, similarity)`，`similarity` 优先取 `chunk.similarity`，否则用 `chunk.score`。
    - `url`: 优先使用 `chunk.doc_id` / `file_path` 中的外链；若缺失则拼接 `lightrag://{instance}/{doc_identifier}`。
    - `title`: 通过 `Path(file_path).stem` 生成；无文件名时回退为 doc 标识。
  - 控制返回数量：遵循 `RetrievalRequest.max_results`，确保 deer-flow 工具展示数量稳定。

### 3. 结构化信息拓展（extra 字段）

- 为 `DeerFlowDocument` 新增可选 `extra: dict`：
  - 初始迭代先携带基础来源信息：`{"source": {"doc_id": ..., "file_path": ...}}`。
  - 预留 `entities` / `relationships` 列表，用于后续注入结构化摘要或“虚拟文档”实现。
  - 若未来将实体/关系转成虚拟 Document，可复用该字段标记类型（如 `extra["type"] = "entity"`）。

### 4. 提示词与前端配合

- deer-flow researcher prompt 需补充说明：Document 可能包含 `extra` 元信息；若存在 `entities` / `relationships`，应在回答中引用。
- 前端 `RetrieverToolCall` 保持从 Document 渲染 title/url/content，后续如引入虚拟文档需区分展示风格（例如标注“实体摘要”）。

### 5. 验证与扩展

- 编写集成测试或模拟调用，确保：
  - 文档聚合逻辑正确，URL/标题回退可靠。
  - `extra` 字段输出符合预期，未影响 deer-flow 现有解析。
- 手动演练：使用真实 LightRAG 实例调用 deer-flow local search，观察前端展示和 Agent 行为。
- 后续迭代思路：
  1. 将实体/关系转成虚拟 Document（带标签），提升结构化信息利用率。
  2. 增强元数据检索链路（例如存储导入时记录真实 URL、作者等），进一步完善可追溯性。

## TODO 列表

- [x] 调整 `lightrag/integrations/deer_flow.py`：为 `DeerFlowDocument` 增加 `extra` 字段及构造辅助函数。
- [x] 实现 `query_relevant_documents` 聚合逻辑：按文档分组、组装 chunk 列表、生成 URL/标题回退，并返回 `DeerFlowDocument` 列表。
- [x] 将基础来源信息写入 `extra`，为后续结构化扩展做好占位（暂不注入实体/关系）。
- [ ] 更新 deer-flow 相关 prompt（若在本仓库维护）或提供补丁说明，指引 Agent 识别 `extra`。
- [ ] 增补测试：模拟 `RetrievalResult` 输入，校验 Document 聚合输出、`extra` 字段，以及无文档时的回退行为。
- [ ] 准备手动验证步骤：运行 local search，确认前端标题/URL 展示正常，记录后续加入虚拟文档的观察点。

## 风险与注意事项

- LightRAG chunk 元数据可能因后端差异出现缺口；需确保回退逻辑覆盖空值场景。
- deer-flow 前端或 Agent 若未立即适配 `extra`，需评估是否临时忽略该字段或提供降级方案。
- 实体/关系虚拟文档方案在后续迭代时要谨慎控制数量，避免污染真实文档列表。
