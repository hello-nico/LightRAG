# LightRAG CLI 重构和 PDF 处理功能实现计划

## 分析现状

当前 CLI 在 `cli.py` 中存在以下问题：

1. 依赖 `examples/rag_pdf_processor.py` 而不是使用全局实例管理器
2. 缺少独立的 PDF 处理命令
3. 没有充分利用已有的全局配置系统

## 实现计划

### 1. 重构现有 CLI 查询功能

- 将 `cli.py` 中的 `execute_query` 和 `execute_streaming_query` 函数改为直接使用 `lightrag.core` 模块
- 移除对 `examples.rag_pdf_processor` 的依赖
- 使用 `get_lightrag_instance()` 和 `load_core_config()` 获取实例
- 保持现有的 6 种查询模式和流式输出功能

### 2. 添加 PDF 处理命令

新增 `extract` 命令：

```bash
python -m cli extract --input_dir ./pdfs --output_dir ./output
```

其中output是lightrag的working_dir，pdfs是pdf文件目录。
功能包括：

- 扫描指定目录的 PDF 文件
- 使用 `lightrag.tools.pdf_reader.PDFExtractor` 提取内容
- 使用全局配置系统创建和管理 LightRAG 实例
- 支持并发处理多个 PDF 文件
- 提供处理进度和结果统计

### 3. 配置增强

- 支持通过命令行参数和 .env 文件配置 PDF 处理参数
- 集成 Qwen embedding 配置
- 支持自定义工作目录、存储配置等

### 4. 具体实现步骤

1. **修改 cli.py 主文件**
   - 导入 `lightrag.core` 模块
   - 重构查询函数使用全局实例管理器
   - 添加 `extract` 命令和相关函数
   - 添加 PDF 处理配置参数

2. **创建 PDF 处理模块**
   - 在 `cli.py` 中添加 `PDFProcessor` 类
   - 集成 `PDFExtractor` 和 Qwen embedding
   - 实现并发处理和进度显示

3. **保持向后兼容性**
   - 现有查询命令保持不变
   - 新增功能不影响现有使用方式

### 5. 使用示例

```bash
# 查询（使用全局实例）
python -m cli query -q local "什么是RAG"

# PDF 处理
python -m cli extract --input_dir ./pdfs --output_dir ./output

# 带配置的 PDF 处理
python -m cli extract --input_dir ./pdfs --max_concurrent 5 --model qwen-embedding
```

## 技术要点

1. 使用 `lightrag.core.get_lightrag_instance()` 管理实例生命周期
2. 使用 `lightrag.core.load_core_config()` 加载配置
3. 集成现有的 `PDFExtractor` 和 Qwen embedding 配置
4. 支持异步并发处理
5. 保持与现有 API 服务器的配置兼容性

## 文件修改清单

### 主要修改文件

- `cli.py` - 重构查询功能，添加 PDF 处理命令

### 依赖文件

- `lightrag/core/__init__.py` - 全局实例管理器
- `lightrag/core/config.py` - 核心配置系统
- `lightrag/core/instance_manager.py` - 实例管理器
- `lightrag/tools/pdf_reader.py` - PDF 提取器
- `examples/qwen_embedding_config.py` - Qwen embedding 配置

### 配置文件

- `.env` - 环境变量配置
- `examples/prompts/rag_prompts_example.json` - 提示词配置

## 实现优先级

1. **高优先级**: 重构现有查询功能使用全局实例管理器
2. **中优先级**: 添加基础的 PDF 处理命令
3. **低优先级**: 添加高级配置选项和优化功能

## 验证计划

1. 测试现有查询功能是否正常工作
2. 测试 PDF 处理功能是否正确提取和处理文件
3. 测试并发处理和错误处理
4. 验证配置系统是否正常加载
5. 确保向后兼容性

---

## 评估与调整（补充）

结论：总体方向正确，但需强化可执行性与离线可用性。

- 新增离线友好模式：为 `extract` 增加 `--mode {dry-run|insert|full}`。默认 `dry-run` 仅做提取与统计，无需网络/Key；`insert` 仅将文本插入；`full` 执行完整流程（需网络与 API Key）。
- 解除对 `examples/rag_pdf_processor` 的耦合：CLI 仅复用 `lightrag.core` 与 `lightrag.tools.pdf_reader.PDFExtractor`，Qwen 作为可选构建器。
- 明确并发、进度与错误聚合：使用 `asyncio.Semaphore(max_concurrent)` + `gather`，输出成功/失败统计。

## 可执行落地步骤（带验收标准）

1) 重构查询命令到核心实例

- 修改 `cli.py`：移除 `examples.rag_pdf_processor` 引用；新增 `get_cli_instance()`，通过 `lightrag.core.load_core_config/get_lightrag_instance` 获取或创建实例（首次调用初始化）。保留 6 种模式与 `--stream/--no-stream`。
- 验收：
  - `python -m cli modes` 正常列出 6 种模式
  - `python -m cli version` 输出版本与说明

2) 新增 `extract` 子命令（异步并发）

- 命令：
  - `python -m cli extract --input-dir ./rag_pdfs --output-dir ./rag_storage --max-concurrent 4 --mode dry-run`
- 选项：
  - `--input-dir` 默认 `./rag_pdfs`
  - `--output-dir` 默认读取 `WORKING_DIR` 或 `./rag_storage`
  - `--max-concurrent` 默认 4
  - `--mode`：`dry-run|insert|full`（默认 `dry-run`）
- 实现：
  - 使用 `PDFExtractor` 提取文本与元数据；并发受 `Semaphore` 限制；统计成功/失败；`insert/full` 时调用 `ainsert`。
- 验收：清晰的进度、成功/失败统计与非 0 失败计数时的错误列表。

3) 可选 Qwen 构建器

- 若检测到 `QWEN_*` 环境变量（`QWEN_API_KEY` 等），则用 `lightrag.llm.qwen.qwen_embedding_func` 包装后作为 embedding；否则沿用核心默认回退。
- 验收：未设置 Qwen 环境时命令可用；设置后 `--mode full` 正常执行。

4) 配置优先级与覆盖

- CLI 参数 > `.env` > 核心默认；`--output-dir` 覆盖核心配置中的 `working_dir`。
- 验收：更改 CLI 参数能反映到运行目录与实例配置。

5) 质量保障

- 通过 `pre-commit` 与 `ruff`（项目已配置）。
- 验收：`pre-commit run -a` 全部通过。

## 测试用例与验收标准（可直接执行）

环境准备：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[api]
pre-commit install
```

1. PDF 提取器离线单测（需仓库内任意 PDF 文件）

```bash
python - << 'PY'
from lightrag.tools.pdf_reader import PDFExtractor
from pathlib import Path
p = Path('rag_pdfs')
pdfs = [x for x in p.glob('*.pdf')]
assert pdfs, '请在 rag_pdfs 放置至少一个 PDF 用例'
ok = len(PDFExtractor().extract(str(pdfs[0])).full_text) > 0
print('OK' if ok else 'FAIL')
PY
```

期望：打印 `OK`。

2. CLI 基本回归（离线）

```bash
python -m cli modes
python -m cli version
```

期望：列出 6 种模式；显示版本与说明。

3. 提取命令（离线 dry-run）

```bash
python -m cli extract --input-dir rag_pdfs --output-dir ./rag_lightrag_storage --max-concurrent 2 --mode dry-run
```

期望：输出“Found N PDFs… Processed: N succeeded / 0 failed”，进程退出码 0；`./rag_lightrag_storage` 存在。

4. 可选 Qwen 集成（需网络与 Key）

```bash
export QWEN_API_KEY=your_key
export QWEN_EMBEDDING_MODEL=Qwen3-Embedding-0.6B
export QWEN_EMBEDDING_HOST=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
python -m cli extract --input-dir rag_pdfs --output-dir ./rag_lightrag_storage --max-concurrent 2 --mode full
```

期望：无错误结束，工作目录生成存储数据（向量库/状态文件），统计成功数>0。

5. 代码质量

```bash
pre-commit run -a
```

期望：所有钩子通过。
