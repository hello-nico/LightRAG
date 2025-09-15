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
