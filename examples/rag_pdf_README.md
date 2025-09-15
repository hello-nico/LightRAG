# RAG PDF处理器

基于LightRAG框架的PDF处理器，专门用于处理RAG相关学术论文，使用RAG domain prompts和Qwen embedding模型。

## 功能特点

- **智能PDF提取**: 基于pdfplumber的高质量PDF文本提取
- **引用去除**: 自动识别并去除论文中的引用部分
- **RAG专用**: 针对RAG领域优化的实体提取和知识图谱构建
- **Qwen集成**: 支持Qwen embedding模型
- **批量处理**: 支持并发处理多个PDF文件
- **灵活配置**: 支持多种配置方式和自定义设置

## 文件结构

```
examples/
├── pdf_reader.py                 # PDF提取器（增强版）
├── rag_pdf_processor.py          # RAG PDF处理器主文件
├── qwen_embedding_config.py      # Qwen embedding配置文件
├── rag_pdf_example.py            # 使用示例
└── README.md                     # 说明文档
```

## 快速开始

### 1. 准备环境

确保已安装必要的依赖：
```bash
pip install lightrag pdfplumber numpy
```

### 2. 配置Qwen Embedding服务

启动Qwen embedding服务API，或者使用现有的embedding服务。

### 3. 基础使用

```python
import asyncio
from rag_pdf_processor import RAGPDFProcessor, RAGPDFProcessorConfig

async def main():
    # 创建配置
    config = RAGPDFProcessorConfig(
        pdf_dir="rag_pdfs",
        output_dir="./rag_kg_output",
        rag_prompts_path="datasets/rag_domain_prompts.json",
        embedding_base_url="http://localhost:8000",
        embedding_api_key="your-api-key",
        embedding_model="qwen-embedding",
        max_concurrent_pdfs=2,
        language="English"
    )
    
    # 创建处理器
    processor = RAGPDFProcessor(config)
    
    try:
        # 初始化
        await processor.initialize()
        
        # 处理所有PDF
        results = await processor.process_all_pdfs()
        print(f"处理结果: {results}")
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 详细配置

### Qwen Embedding配置

使用预定义配置：
```python
from qwen_embedding_config import get_qwen_config

# 使用默认配置
qwen_config = get_qwen_config("default")

# 使用生产配置
qwen_config = get_qwen_config("production")
```

创建自定义配置：
```python
from qwen_embedding_config import create_custom_config

custom_config = create_custom_config(
    base_url="http://your-api:8080",
    api_key="your-api-key",
    model="qwen-7b-embedding",
    timeout=45
)
```

### 处理器配置

完整的处理器配置选项：
```python
config = RAGPDFProcessorConfig(
    # 基础配置
    pdf_dir="rag_pdfs",                    # PDF目录
    output_dir="./rag_kg_output",         # 输出目录
    rag_prompts_path="datasets/rag_domain_prompts.json",  # RAG prompts路径
    
    # Embedding配置
    embedding_base_url="http://localhost:8000",  # Embedding API URL
    embedding_api_key="your-api-key",            # API密钥
    embedding_model="qwen-embedding",             # 模型名称
    
    # LightRAG配置
    working_dir="./rag_lightrag_storage",    # LightRAG工作目录
    llm_model_func=None,                     # LLM模型函数
    enable_logging=True,                     # 启用日志
    log_level="INFO",                        # 日志级别
    
    # 处理配置
    max_concurrent_pdfs=3,                  # 最大并发PDF数
    chunk_size=8000,                         # 分块大小
    max_gleaning=3,                         # 最大提取轮数
    
    # RAG专用配置
    language="English",                      # 输出语言
    enable_chunking=False                   # 是否启用分块
)
```

## 核心功能

### 1. PDF文本提取

使用增强的PDFExtractor类：
- 智能标题提取
- 页眉页脚过滤
- 段落合并
- 引用部分去除

### 2. 引用去除

自动识别并去除论文中的引用部分：
- 支持中英文引用标题
- 智能边界检测
- 避免误删正文内容

### 3. RAG领域优化

- 使用专门的RAG实体类型（Paper, Researcher, Method等）
- 针对学术语言的提示词优化
- 专业的知识图谱构建

### 4. 批量处理

支持并发处理多个PDF文件：
- 可配置并发数量
- 错误处理和重试
- 进度跟踪和报告

## 使用示例

### 处理单个PDF

```python
result = await processor.process_pdf("path/to/paper.pdf")
print(f"处理结果: {result}")
```

### 批量处理

```python
results = await processor.process_all_pdfs()
print(f"总计: {results['total_pdfs']}, 成功: {results['successful']}, 失败: {results['failed']}")
```

### 知识图谱查询

```python
# 本地查询
local_result = await processor.query_knowledge_graph("What is RAG?", mode="local")

# 全局查询
global_result = await processor.query_knowledge_graph("RAG architectures", mode="global")

# 混合查询
hybrid_result = await processor.query_knowledge_graph("RAG evaluation", mode="hybrid")
```

## 输出结果

处理完成后，输出目录结构：
```
rag_kg_output/
├── processing_results.json       # 处理结果汇总
└── rag_lightrag_storage/         # LightRAG生成的知识图谱数据
    ├── entities/
    ├── relationships/
    └── ...
```

## 运行示例

```bash
# 运行基础示例
python examples/rag_pdf_example.py

# 运行批量处理
python examples/rag_pdf_processor.py
```

## 环境变量

可以通过环境变量配置：
- `QWEN_EMBEDDING_BASE_URL`: Embedding API URL
- `QWEN_EMBEDDING_API_KEY`: API密钥
- `QWEN_EMBEDDING_MODEL`: 模型名称

## 故障排除

### 常见问题

1. **PDF提取失败**
   - 检查PDF文件是否损坏
   - 确保有足够的权限访问文件

2. **Embedding服务连接失败**
   - 检查API URL是否正确
   - 确认API服务正在运行
   - 验证API密钥是否正确

3. **LightRAG初始化失败**
   - 检查工作目录权限
   - 确保所有依赖已安装
   - 验证配置参数

4. **内存不足**
   - 减少并发处理数量
   - 调整分块大小
   - 增加系统内存

### 日志调试

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展开发

### 添加新的实体类型

在`datasets/rag_domain_prompts.json`中添加新的实体类型定义。

### 自定义PDF处理

继承PDFExtractor类，重写相关方法：
```python
class CustomPDFExtractor(PDFExtractor):
    def extract_text_from_pdf(self, pdf_path):
        # 自定义提取逻辑
        pass
```

### 自定义Embedding服务

实现自定义的embedding函数：
```python
async def custom_embedding_func(texts):
    # 自定义embedding逻辑
    return embeddings
```

## 许可证

本项目遵循MIT许可证。