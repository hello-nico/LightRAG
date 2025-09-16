# LightRAG 检索集成架构计划

## 概述

LightRAG 需要面向 DeerFlow 以及未来的第三方系统暴露统一的检索接口。我们将通过新增一个“检索集成”模块，在不破坏现有核心逻辑的前提下，实现接口标准化、上下文解析、数据补全与路由集成。

## 全局架构设计

### 1. 模块划分

- `lightrag/integrations/`：新增目录，用于管理所有对外检索集成。
  - `models.py`：统一定义 `Resource`、`Document`、`Chunk`、`RetrievalResult` 等 Pydantic 数据模型。
  - `context_parser.py`：解析 `rag.aquery(..., only_need_context=True)` 返回的 Markdown，拆分出实体、关系、文本块等结构化数据。
  - `converters.py`：提供从解析结果到统一模型的转换工具，负责补全缺失的 `chunk_id`、`doc_id`、`metadata` 等信息。
  - `base.py`：声明 `BaseIntegration` 抽象类，约定 `retrieve`、`batch_retrieve`、`list_resources` 等能力。
  - `deer_flow.py`：实现 DeerFlow 适配器，利用 `LightRAG` 实例完成检索、相似度计算、资源列举。
  - `manager.py`：按名称或配置返回具体适配器实例，便于未来扩展更多集成。

### 2. API 路由

- 新增 `lightrag/api/routers/retrieval_routes.py`，暴露统一的 FastAPI 路由（如 `/retrieval/retrieve`、`/retrieval/batch`、`/retrieval/resources`）。
- 路由层处理鉴权、参数校验、错误转换；业务逻辑委托给 `IntegrationManager`。
- 在 `lightrag/api/lightrag_server.py` 里追加一次 `app.include_router(create_retrieval_routes(...))`，保持最小侵入。

### 3. 数据补全策略

- 解析上下文后，利用 `LightRAG.doc_status`、`LightRAG.text_chunks` 获取文档标题、`doc_id`、`chunk` 元信息。
- 若缺失相似度，可通过 `rag.chunks_vdb` 对候选 chunk 进行相似度估算，或在 DeerFlow 适配器内提供可配置的估算策略。
- 返回值统一遵循 DeerFlow 要求，同时保留原有实体、关系结构，确保 LightRAG 自身能力不受影响。

### 4. 扩展性与配置

- 通过配置或环境变量选择默认 provider（默认为 `deer_flow`）。
- IntegrationManager 支持按名称选择适配器，后续新增集成只需实现 `BaseIntegration` 并在注册表中声明即可。

## TODO 列表

### 阶段一：基础设施

- [ ] 在 `lightrag/integrations/` 建立目录与 `__init__.py`，定义公共数据模型 (`models.py`)。
- [ ] 实现 `context_parser.py`，完成 Markdown → 结构化数据的解析能力，并覆盖异常格式处理。
- [ ] 编写 `converters.py`，将解析结果补全为文档/块结构，并对接 `LightRAG` 现有存储以填充元数据。
- [ ] 定义 `BaseIntegration` 抽象类（`base.py`）和 `IntegrationManager`（`manager.py`）。

### 阶段二：DeerFlow 适配

- [ ] 在 `deer_flow.py` 中实现 `DeerFlowIntegration`，封装单次检索、批量检索、资源列表能力。
- [ ] 追加 DeerFlow 相关配置（默认 provider、相似度策略）并给出合理的默认值。
- [ ] 为 DeerFlow 适配逻辑编写核心单元测试（解析、转换、接口返回）。

### 阶段三：API 集成

- [ ] 新增 `lightrag/api/routers/retrieval_routes.py`，定义请求/响应模型与路由，实现与 `IntegrationManager` 的调用闭环。
- [ ] 在 `lightrag/api/lightrag_server.py` 中包含新路由，确保与现有鉴权/日志机制兼容。
- [ ] 为路由层补充 FastAPI 级别的集成测试或示例调用脚本。

### 阶段四：文档与验收

- [ ] 更新 README 或 API 文档，说明新的 `/retrieval` 接口、配置方法及 DeerFlow 对接方式。
- [ ] 提供最小复现示例（例如 `examples/retrieval_usage.py`），演示如何调用统一检索接口。
- [ ] 跑通既有验证脚本（lint、tests）并确保回归通过。

## 验证与后续演进建议

- 优先对 `context_parser` 编写稳健的单元测试，确保解析容错。
- 预留 `IntegrationManager` 的扩展点，未来可按需新增多租户、缓存等能力。
- 若 DeerFlow 对相似度或排序有更严苛要求，可在适配器层引入可插拔策略，而不影响统一接口设计。
