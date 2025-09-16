# task_02_review — CLI 改造代码评审

本文对当前 `cli.py` 实现进行审查，列出阻断性问题、改进建议与可验证的修复建议，便于快速迭代。

## 阻断性问题（必须修复）

- 异步实例获取错误使用
  - `get_lightrag_instance` 为异步函数，`get_cli_instance()` 未 `await`，返回协程对象；调用处随后访问 `.aquery/.ainsert` 会抛错。
  - `get_cli_instance` 传参位置错误：`return get_lightrag_instance(config)` 把 `config` 传给了 `name` 形参。
  - 修复建议：
    - 将 `get_cli_instance` 改为异步，并正确传参：`await get_lightrag_instance(name="cli", config=cfg, auto_init=True)`；
    - 所有调用改为 `await get_cli_instance(...)`。

- `load_core_config` 用法错误
  - 该函数不接受 `working_dir=` 参数，当前调用会报 `TypeError`。
  - 修复建议：若需覆盖工作目录，使用 `load_core_config(custom_defaults={"working_dir": working_dir})`。

- Qwen Embedding 未真正生效
  - 仅设置环境变量（如 `EMBEDDING_MODEL`）不足以影响 `core.instance_manager` 默认构建器；当前默认构建器仍使用随机 embedding。
  - 顶层 `from lightrag.llm.qwen import qwen_embedding_func` 会触发依赖安装尝试，不利于离线模式。
  - 修复建议：
    - 懒加载 Qwen：在需要时（检测到 `QWEN_*` 环境变量）再导入；
    - 构造自定义 `builder`（创建 `embedding_func`，必要时用 `wrap_embedding_func_with_attrs` 设置 `embedding_dim`），并传入 `get_lightrag_instance(..., builder=builder)`。

## 重要问题（尽快修复）

- `ainsert` 未附带文档标识
  - 建议传入 `ids=pdf_path.stem` 与 `file_paths=str(pdf_path)` 便于追踪与引用。
- 输出敏感信息
  - `models()` 打印了 `LLM_BINDING_API_KEY`；`Args` 里也保留了不安全默认值。建议永不打印密钥，或用掩码显示；删除硬编码默认 Key。
- 未使用的导入与代码
  - `tqdm`、`concurrent.futures`、`ExtractedContent` 等未使用，影响代码整洁和 `ruff` 检查。
- 环境变量对 LLM/Embedding 的真实影响不明确
  - `LLM_MODEL` 等仅设置了环境变量，但默认构建器未消费。建议通过 builder 或明确的配置接入。

## 建议优化

- 错误聚合与统计已具备，建议为失败案例打印更短摘要（文件名 + 首行错误）。
- 在 `execute_streaming_query` 中 fallback 后保留相同的 `QueryParam` 结构，仅切换 `stream=False`（已基本符合）。
- 帮助文本已充分，可以考虑在 `--help` 中同步列出 `extract` 的三种模式说明。

## 参考修复要点（代码走向）

- `get_cli_instance`
  - `async def get_cli_instance(working_dir: Optional[str] = None):`
  - `cfg = load_core_config(custom_defaults={"working_dir": working_dir}) if working_dir else load_core_config()`
  - 若启用 Qwen：构造 `builder(config)` 返回 `LightRAG(working_dir=config.working_dir, embedding_func=wrapped_qwen, llm_model_func=默认或配置, ...)`
  - `return await get_lightrag_instance(name="cli", config=cfg, builder=builder, auto_init=True)`
- 调用侧：
  - `lightrag = await get_cli_instance()` / `self.lightrag = await get_cli_instance(self.working_dir)`
- 插入文本：
  - `await self.lightrag.ainsert(extracted.full_text, ids=pdf_path.stem, file_paths=str(pdf_path))`
- 安全输出：
  - 不打印 API Key；移除 `Args` 中的硬编码默认 Key。
- 清理未使用导入，确保 `pre-commit` 通过。

## 验证用例（修复后应全部通过）

- 基本信息
  - `python -m cli modes` 与 `python -m cli version` 输出正常。
- 提取离线（dry-run）
  - `python -m cli extract --input-dir rag_pdfs --output-dir ./rag_lightrag_storage --max-concurrent 2 --mode dry-run`
  - 期望：统计完成、退出码 0；目录存在。
- 插入模式（insert）
  - 同上命令改 `--mode insert`；期望：无异常，统计成功。
- Qwen 集成（可选，需网络与 Key）
  - 设置 `QWEN_API_KEY/QWEN_EMBEDDING_HOST/QWEN_EMBEDDING_MODEL` 后运行 `--mode full`；期望：无异常、产生向量/状态数据。
- 质量检查
  - `pre-commit run -a` 全通过。

> 注：以上问题和方案均基于当前仓库的 `lightrag.core` 能力与 `instance_manager` 默认构建器实现推断，修复时请确保与既有配置系统一致。
