# Task Review: LightRAG 全局配置系统改进（完成度评审）

## 结论概览

- 总体完成度：高。核心目标（统一配置源、全局实例管理、API/示例集成）已达成，向后兼容策略合理。
- 建议状态：先修复下述问题后可标记为“完成”。问题均为小修，改动面小、风险低。

## 阻断性问题（需修复）

1) CLI 调用签名不匹配（会导致运行时错误）

- 位置：`cli.py`
- 现象：`execute_query` 和 `execute_streaming_query` 调用 `processor.query(query_text, param=param)`，而新 `RAGPDFProcessor.query(self, query, mode='...')` 签名仅接受 `mode`，不接受 `param`。
- 影响：CLI 查询时报 `TypeError`。
- 修复建议：改为 `await processor.query(query_text, mode=mode)`；流式回退分支也同步修改。

2) 默认构建器导入路径错误（易 ImportError）

- 位置：`lightrag/core/instance_manager.py`，默认 builder
- 现象：`from lightrag.llm import gpt_4o_mini_complete`，但 `lightrag/llm/__init__.py` 未导出该符号；实际定义在 `lightrag/llm/openai.py`。
- 影响：调用默认构建器（如 `initialize_lightrag_with_config` 或 demo）时报导入错误。
- 修复建议：改为 `from lightrag.llm.openai import gpt_4o_mini_complete`；移除未使用的 `openai_complete_if_cache` 导入。

3) 默认 embedding 函数缺少 `embedding_dim`（插入/向量库会出错）

- 位置：`lightrag/core/instance_manager.py`，默认 builder 内部的 `default_embedding_func`
- 现象：返回 `np.ndarray`，但未通过 `wrap_embedding_func_with_attrs(embedding_dim=...)` 或 `EmbeddingFunc` 包装，`NanoVectorDBStorage` 等依赖 `embedding_func.embedding_dim`。
- 影响：执行插入或向量库初始化时 `AttributeError`。
- 修复建议：使用 `from lightrag.utils import wrap_embedding_func_with_attrs` 并装饰默认函数，例如：
  - `@wrap_embedding_func_with_attrs(embedding_dim=1024)` 装饰 `default_embedding_func`；或使用 `EmbeddingFunc(embedding_dim=..., func=...)` 封装。

4) PROMPTS 注入未同步覆盖实例的实体类型（配置未落地到实例）

- 位置：`lightrag/core/instance_manager.py`，`inject_prompts_from_config()`
- 现象：更新了 `PROMPTS` 与 `config.entity_types`，但未更新 `instance.addon_params['entity_types']`。
- 影响：实例仍使用创建时的实体类型，注入的实体类型不生效。
- 修复建议：在注入成功后，若 `merged_prompts` 含 `entity_types`，同步执行：
  - `instance.addon_params['entity_types'] = merged_prompts['entity_types']`
  - 若 `language` 也在 JSON 中，可一并覆盖 `instance.addon_params['language']`（可选）。

5) 示例导入未导出符号（会 ImportError）

- 位置：`examples/rag_pdf_processor.py`
- 现象：
  - `from lightrag.core import get_global_manager`（未在 `lightrag/core/__init__.py` 导出）
  - `from lightrag.core import load_core_config, get_default_config`（`get_default_config` 未导出）
- 影响：运行示例时报导入错误。
- 修复建议（二选一）：
  - A. 在 `lightrag/core/__init__.py` 中将 `get_global_manager` 与 `get_default_config` 添加到 `__all__`；
  - B. 修改示例为从子模块导入：
    - `from lightrag.core.instance_manager import get_global_manager`
    - `from lightrag.core.config import get_default_config`

6) 示例提示词 JSON 键名与系统不一致（注入后不被使用）

- 位置：`examples/prompts/rag_prompts_example.json`
- 现象：使用了 `entity_extraction` / `relationship_extraction` / `entity_summarization` 键；系统实际使用 `lightrag/prompt.py` 中定义的键，例如：
  - `entity_extraction_system_prompt`
  - `entity_extraction_user_prompt`
  - `entity_continue_extraction_user_prompt`
  - `summarize_entity_descriptions`
- 影响：注入的提示词不会被实际流程消费。
- 修复建议：
  - A. 将示例 JSON 的键名改为上述实际键名；或
  - B. 在 `inject_prompts_from_config()` 中加入键名映射逻辑（把示例键名自动映射到系统键）。

7) 示例中硬编码 Qwen 地址与密钥（不利于部署与安全）

- 位置：`examples/rag_pdf_processor.py` 的自定义 builder
- 现象：硬编码 `base_url` 与 `api_key`。
- 影响：不便于环境切换且存在安全风险。
- 修复建议：改为从环境变量读取（如 `QWEN_EMBEDDING_HOST`、`QWEN_EMBEDDING_API_KEY`、`QWEN_EMBEDDING_MODEL`、`EMBEDDING_DIM`），或通过 `EMBEDDING_BINDING=qwen` 协议化集成。

## 次要问题与文档一致性

- 文档键名一致性：`docs/config_improve.md` 最初示例包含多个提示词路径键（如 `ENTITY_EXTRACTION_PROMPT_PATH` 等），但最终实现采用单一 `PROMPTS_JSON_PATH`。建议在文档“方案描述”处也统一为 `PROMPTS_JSON_PATH`，避免认知分歧（补充章节已说明，但前文示例建议同步更新）。
- API 辅助函数一致性：`get_api_instance_config()` 返回了 `input_dir` 字段，但 `LightRAG` 构造函数并不接收该参数。当前看函数未被用于实例化，属于无害；可保留作信息聚合，或在注释中注明该字段只用于链路信息而非构造参数。

## 建议修复摘要（可操作清单）

- 调整 CLI：将 `processor.query(query_text, param=param)` 改为 `processor.query(query_text, mode=mode)`（两个函数处）。
- 调整默认 builder：
  - 更改导入为 `from lightrag.llm.openai import gpt_4o_mini_complete`；
  - 使用 `@wrap_embedding_func_with_attrs(embedding_dim=1024)` 装饰默认 embedding 函数，或 `EmbeddingFunc` 包装。
- 同步实例实体类型/语言：在 `inject_prompts_from_config()` 注入成功后，更新 `instance.addon_params` 中的 `entity_types`（和可选的 `language`）。
- 导出或修正示例导入：在 `lightrag/core/__init__.py` 导出 `get_global_manager` 与 `get_default_config`，或更改示例为从子模块导入。
- 修正提示词 JSON 键名：优先修改示例 JSON 文件为系统实际键名；如希望保留简化键，则在注入函数内添加键名映射。
- 移除示例中的硬编码敏感信息：改为读取环境变量或 `.env`。
- 文档一致性：将 `docs/config_improve.md` 前文的多路径键统一为 `PROMPTS_JSON_PATH`。

## 快速验证清单（修复后建议执行）

- CLI：`python -m cli -q local "what is RAG"` 能正常输出结果，无 TypeError。
- API：`lightrag-server` 启动后 `/health` 正常，日志显示实例已注册、PROMPTS 注入成功或被跳过。
- PROMPTS 注入：设置 `PROMPTS_JSON_PATH` 指向包含正确键名的 JSON，验证实体提取/提示词生效（抽取输出符合定制）。
- 多实例：设置不同 `LIGHTRAG_INSTANCE_NAME` 获取多个实例，并通过管理器列出实例名确认注册成功。

## 认可点

- 设计与落地遵循“薄适配 + 渐进式重构”，API 行为保持不变；
- 核心模块职责清晰，接口命名与职责边界良好；
- 文档与实现相互呼应，具备可维护性与扩展空间。
