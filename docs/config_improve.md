# LightRAG 全局配置系统改进方案

## 问题分析

经过分析，我发现了当前配置系统的几个关键问题：

1. **API 服务器**：在 `lightrag/api/lightrag_server.py` 中使用 `lightrag/api/config.py` 的配置系统，基于 `.env` 文件和命令行参数
2. **独立应用**：在 `examples/rag_pdf_processor.py` 中独立创建 LightRAG 实例，配置不统一
3. **配置分散**：实体类型、提示词等关键配置分散在不同地方

## 解决方案

### 1. 创建统一配置模块 (`lightrag/core/config.py`)

- 重构现有 `api/config.py` 为核心配置系统
- 支持从 `.env` 文件读取所有配置（包括实体类型、提示词）
- 提供配置验证和默认值处理

### 2. 创建全局实例管理器 (`lightrag/core/instance_manager.py`)

- 实现单例模式的 LightRAG 实例管理
- 统一的实例化接口，支持 CLI 和 API 使用
- 提供实例获取、销毁、重新配置等方法

### 3. 扩展 .env 配置支持

在现有配置基础上添加：

```
# 实体类型配置
ENTITY_TYPES=["person","organization","location","event","concept"]

# 自定义提示词配置（JSON格式）
PROMPTS_JSON_PATH="./prompts/custom_prompts.json"

# 全局实例配置
LIGHTRAG_INSTANCE_NAME="default"
LIGHTRAG_AUTO_INIT=true
```

### 4. 重构现有代码

- 更新 `api/config.py` 使用新的核心配置系统
- 重构 `examples/rag_pdf_processor.py` 使用全局实例管理器
- 确保向后兼容性

### 5. 提供统一的使用接口

```python
# CLI 使用
from lightrag.core import get_lightrag_instance
rag = await get_lightrag_instance()

# API 使用
from lightrag.core import get_lightrag_instance
rag = await get_lightrag_instance()
```

## 优势

- **统一配置**：所有 LightRAG 实例使用相同的配置源
- **灵活定制**：通过 .env 可以修改实体类型和提示词
- **简单使用**：提供统一的接口获取实例
- **向后兼容**：不破坏现有代码

## 实施步骤

1. 创建 `lightrag/core/` 目录结构
2. 实现 `lightrag/core/config.py` 统一配置系统
3. 实现 `lightrag/core/instance_manager.py` 实例管理器
4. 更新 `lightrag/api/config.py` 依赖新的核心配置
5. 重构 `examples/rag_pdf_processor.py` 使用全局实例
6. 添加相关文档和示例

## 技术细节

### 配置系统设计

- 继承现有的 `parse_args()` 逻辑
- 添加新的配置项支持（实体类型、提示词路径）
- 提供配置验证和类型转换
- 保持环境变量和命令行参数的兼容性

### 实例管理器设计

- 使用单例模式确保全局唯一实例
- 支持延迟初始化和按需创建
- 提供实例生命周期管理
- 支持多实例场景（通过不同的实例名称）

### 向后兼容性

- 现有 API 服务器代码无需修改
- 现有示例代码可以继续工作
- 新功能作为可选增强

---

# 可行性评估与落地建议（补充）

以下内容基于当前仓库代码实际情况的全面阅读与对照分析，补充可行性评估、分阶段落地计划与兼容性策略，帮助在不破坏现有功能的前提下，稳妥推进统一配置与实例管理。

## 结论概览

- 可行性：高。核心方向正确，能解决“配置分散、实例不统一”的问题。
- 路线：建议渐进式重构，先引入核心配置与实例管理，再逐步迁移 API 与示例，避免一次性大改引入风险。
- 兼容性：通过“薄适配层 + 可选增强”的方式，保持现有 `api/config.py` 与 `lightrag_server.py` 对外行为不变。

## 现状对照（关键点）

- API 侧配置较完善：`lightrag/api/config.py` 已支持 `.env + argparse`，并能从环境读取大量参数（含 `ENTITY_TYPES` JSON 列表），默认值来源集中在 `lightrag/constants.py`，解析函数统一用 `lightrag/utils.get_env_value()`。
- 服务端构建实例逻辑复杂：`lightrag/api/lightrag_server.py` 内部直接构造 `LightRAG`，包含 LLM/Embedding/Rerank/超时/选项缓存等服务端特性（例如 `LLMConfigCache`、OpenAI/Azure/Ollama 的优化参数拼装、Rerank provider 动态绑定等）。
- 示例侧独立：`examples/rag_pdf_processor.py` 自建 `LightRAG`，并从 `datasets/rag_domain_prompts.json` 注入 PROMPTS；与 API 的配置来源与注入时机不一致。

结论：要实现“统一配置 + 统一实例管理”，需要让“配置解析/验证”可复用，同时尊重服务端已有的“构建流程（含优化与鉴权）”。

## 方案可行性细化评估

1) 统一配置模块（`lightrag/core/config.py`）

- 可行。建议作为“核心配置装载器 + 数据类/字典”，专注于从 `.env`（和可选 CLI）读取公共配置、提供类型校验与默认值。
- API 侧保留 `argparse`，但其内部可复用核心解析/校验逻辑；`api/config.py` 继续导出 `global_args` 与 `update_uvicorn_mode_config()`，对外行为不变。

2) 全局实例管理（`lightrag/core/instance_manager.py`）

- 可行。建议作为“每进程”实例管理器（多进程下为每 worker 进程一个单例集合），提供：
  - `async get_lightrag_instance(name='default', builder=None, auto_init=False)`
  - `set_instance(name, rag)` / `destroy_instance(name)`
  - 内部用异步锁防止并发创建；可选的延迟创建与自动初始化。
- API 端：保持由 FastAPI lifespan 管理 `initialize_storages/finalize_storages`；避免与 InstanceManager 的自动初始化冲突。推荐 API 端“创建后注册”，示例/CLI “通过管理器获取”。

3) PROMPTS 与实体类型配置

- 可行。现有示例通过 `PROMPTS.update()` 和 `datasets/rag_domain_prompts.json` 注入，API 端从 `.env` 读取 `ENTITY_TYPES`。
- 建议引入一个统一的 `PROMPTS_JSON_PATH`（或保留多路径键，但默认使用单文件）。当路径存在时：
  - 加载 JSON 并在实例创建后注入 `PROMPTS`；
  - 若 JSON 中含 `entity_types` 字段，则优先覆盖 `ENTITY_TYPES`，保持“一处定义”。

4) 统一使用接口

- CLI/示例：直接使用 `from lightrag.core import get_lightrag_instance`。
- API：短期不建议完全替换为 `get_lightrag_instance()`，因为服务端构建逻辑较多。建议先在服务端实例创建完成后执行 `set_instance(name, rag)`，对外暴露统一获取路径；后续再将“构建逻辑”提炼为可复用 builder。

## 分阶段落地方案

Phase 1（低风险，优先推进）

- 新增 `lightrag/core/config.py`
  - 提供 `load_core_config()`：从 `.env` 读取核心配置（含 `ENTITY_TYPES`、`PROMPTS_JSON_PATH`、`LIGHTRAG_INSTANCE_NAME`、`LIGHTRAG_AUTO_INIT` 等），并做类型校验与默认值填充。
  - 约定配置优先级：命令行 > 环境变量 > `.env` > 默认值（与 API 现状一致）。
- 新增 `lightrag/core/instance_manager.py`
  - 实现“每进程单例字典 + 异步锁 + 生命周期钩子”。
  - 提供 `get_lightrag_instance`、`set_instance`、`destroy_instance` 等接口。
- 在 `env.example` 增补：`PROMPTS_JSON_PATH`、`LIGHTRAG_INSTANCE_NAME`、`LIGHTRAG_AUTO_INIT` 示例与说明。

Phase 2（示例与 CLI 收敛）

- 调整 `examples/rag_pdf_processor.py`
  - 默认通过 `get_lightrag_instance()` 获取实例；如需 Qwen embedding，可通过 `EMBEDDING_BINDING=qwen` 与现有 `llm.qwen` 能力对接，或在 builder 中按绑定类型选择。
  - 若配置了 `PROMPTS_JSON_PATH`，在实例创建后一次性注入 PROMPTS（并仅注入一次）。
- CLI 示例改造为使用统一的 `get_lightrag_instance()`，减少独立配置点。

Phase 3（API 内部复用与统一构建）

- `lightrag/api/config.py` 内部复用 `core/config` 的解析/校验逻辑：
  - 外部接口保持不变（继续导出 `global_args` 等）；
  - 增加必要的兼容层，确保现有运行参数与 `.env` 效果不变。
- `lightrag/api/lightrag_server.py`：
  - 保持现有“构建 LightRAG 的优化流程”；
  - 构建成功后调用 `set_instance(name, rag)`，将实例纳入管理器统一对外暴露。
- 后续可将服务端的构建流程提炼为“可复用 builder”，供 CLI/示例共享，逐步实现“实例构建逻辑统一”。

## 风险与注意点

- 多进程/多 worker：InstanceManager 是“每进程单例”，FastAPI 在 Gunicorn 多 worker 模式下会在每个进程里持有各自实例。这是合理而预期的；需要在文档中明确说明。
- 生命周期管理：API 端应继续用 lifespan 控制 `initialize_storages/finalize_storages`；InstanceManager 的 `auto_init` 建议默认仅在 CLI/脚本启用，避免重复初始化。
- PROMPTS 注入：确保注入发生在实例创建后且仅一次；当文件不存在或解析失败时打印警告并跳过，不影响主流程。
- 配置覆盖顺序：严格遵循“命令行 > 环境 > .env > 默认”，确保与现有 `api/config.py` 一致。
- 复用与耦合：服务端构建中包含优化与兼容逻辑（OpenAI/Azure/Ollama 参数预处理、rerank 绑定、超时参数等），短期内不建议迁移至 core，以免打破行为一致性；宜分离出“可复用 builder”再逐步替换。

## 设计与接口建议

- `core/config.py`
  - `load_core_config()`：返回结构化配置（可用 `dataclass` 或 `dict`），内含通用键：
    - `ENTITY_TYPES`（list）、`PROMPTS_JSON_PATH`（str，可选）
    - `LIGHTRAG_INSTANCE_NAME`（str，默认 `default`）、`LIGHTRAG_AUTO_INIT`（bool）
    - 其他可选通用键：`WORKSPACE`、`SUMMARY_LANGUAGE` 等
  - 校验与默认值沿用 `lightrag/constants.py` 与 `utils.get_env_value()` 的逻辑。

- `core/instance_manager.py`
  - `async get_lightrag_instance(name='default', builder=None, auto_init=False)`：
    - 若已存在，直接返回；否则在锁内调用 `builder()` 生成实例，必要时执行初始化；
    - `builder` 由调用方提供（示例/CLI 可用简化 builder，API 端暂保持自建）。
  - `set_instance(name, rag)`：将外部已构建实例注册到管理器（API 场景）。
  - `destroy_instance(name)`：销毁并从管理器移除（谨慎使用，避免影响正在使用的请求）。

- PROMPTS 注入策略
  - 优先使用 `PROMPTS_JSON_PATH` 指定的文件；当存在 `entity_types` 字段时，覆盖 `ENTITY_TYPES`；
  - 注入前校验 JSON 结构，失败则写日志并跳过。

## 向后兼容策略

- `api/config.py`：外部接口不变，内部逐步复用 `core/config` 的解析/校验；`global_args`、`update_uvicorn_mode_config()` 保留。
- `lightrag_server.py`：继续按现有方式构建 `LightRAG`，仅在构建后调用 `set_instance()`；对外使用上允许通过 `get_lightrag_instance()` 取得同一实例。
- 示例与 CLI：以新增接口为优先路径，旧用法短期保留，文档中标注推荐迁移。

## 变更清单（建议）

- 新增：`lightrag/core/config.py`、`lightrag/core/instance_manager.py`
- 修改：`env.example` 补充示例键（`PROMPTS_JSON_PATH`、`LIGHTRAG_INSTANCE_NAME`、`LIGHTRAG_AUTO_INIT`）
- 文档：本文件补充说明 + 示例/CLI README 更新
- 可选：在 `lightrag_server.py` 成功创建实例后注册至管理器（不改变现有构建逻辑）

## 验证清单

- Uvicorn 单进程：启动 API，检查 `/health`、文档路由、跨域设置、Rerank 绑定是否与原行为一致。
- Gunicorn 多 worker：确认每个 worker 正常启动、存储初始化/清理正常，关停时无资源泄漏。
- CLI/示例：`initialize_rag()` 迁移到 `get_lightrag_instance()` 后，处理/查询路径不变，PROMPTS 能按 `PROMPTS_JSON_PATH` 注入。
- `.env`/命令行覆盖：验证优先级与 `api/config.py` 现状一致。

## 粗略工期与里程碑

- Phase 1：1–2 天（新增 core 配置与实例管理、补文档与示例配置）
- Phase 2：2–3 天（示例与 CLI 收敛到统一接口）
- Phase 3：3+ 天（API 内部复用与统一 builder，保守推进）

## 成功标准

- 统一配置源：示例/CLI 与 API 读取到一致的关键参数（实体类型、语言、工作空间等）。
- 统一实例通道：对外可通过 `get_lightrag_instance()` 获取命名实例；API 端注册实例后可被外部复用。
- 完全兼容：不破坏现有 API 行为；`.env` 与命令行参数语义一致；服务端优化（LLM/Embedding/Rerank/超时/日志）全部保留。
