# RAG PDF处理器配置说明

## 环境配置

基于您的环境配置，我已经调整了RAG PDF处理器的配置文件：

### 配置文件位置

- 环境配置：`examples/env`
- Qwen Embedding配置：`examples/qwen_embedding_config.py`
- RAG PDF示例：`examples/rag_pdf_example.py`
- 配置测试：`examples/test_rag_config.py`

### 已配置的服务

#### 1. LLM服务

- **模型**: deepseek-v3-250324
- **API地址**: <http://10.0.62.214:15000/k-llm>
- **API密钥**: sk-4e76fcdf3f95467198edabdc0d6627f6

#### 2. Embedding服务

- **模型**: Qwen3-Embedding-0.6B
- **API地址**: <http://10.0.62.206:51200/v1>
- **API密钥**: sk-uHj8K2mNpL5vR9xQ4tY7wB3cA6nE0iF1gD8sZ2yX4jM9kP3h
- **向量维度**: 1024

#### 3. 向量数据库

- **URL**: <http://10.0.62.214:6333/>
- **API密钥**: abc123

### 使用方法

#### 1. 测试配置

```bash
cd examples
python test_rag_config.py
```

#### 2. 运行基础示例

```bash
cd examples
python rag_pdf_example.py
```

#### 3. 在代码中使用

```python
from rag_pdf_processor import RAGPDFProcessor, RAGPDFProcessorConfig
from qwen_embedding_config import get_qwen_config, env_llm_model_func

# 获取配置
qwen_config = get_qwen_config("env-config")

# 创建处理器配置
config = RAGPDFProcessorConfig(
    pdf_dir="rag_pdfs",
    output_dir="./rag_kg_output",
    rag_prompts_path="datasets/rag_domain_prompts.json",
    embedding_base_url=f"http://{qwen_config.base_url}",
    embedding_api_key=qwen_config.api_key,
    embedding_model=qwen_config.model,
    llm_model_func=env_llm_model_func,
    max_concurrent_pdfs=2,
    language="English"
)

# 创建处理器
processor = RAGPDFProcessor(config)
await processor.initialize()
```

### 主要修改

1. **qwen_embedding_config.py**:
   - 添加了`env-config`配置，使用您的实际环境参数
   - 添加了`env_llm_model_func`函数，配置LLM服务

2. **rag_pdf_example.py**:
   - 所有示例函数都使用`env-config`配置
   - 添加了LLM模型函数配置

3. **test_rag_config.py**:
   - 新建测试脚本，验证所有配置是否正确

### 注意事项

1. **PDF文件目录**: 确保将PDF文件放在`rag_pdfs`目录中
2. **Prompts文件**: 确保`datasets/rag_domain_prompts.json`文件存在
3. **网络连接**: 确保可以访问配置的API地址
4. **API密钥**: 确保所有API密钥都是有效的

### 故障排除

如果遇到问题，请检查：

1. **网络连接**: 使用`curl`或`ping`测试API地址是否可达
2. **API密钥**: 确认API密钥是否正确
3. **服务状态**: 确认所有服务都在运行
4. **文件路径**: 确认所有文件路径都存在

### 自定义配置

如果需要修改配置，可以：

1. 修改`examples/env`文件
2. 在`qwen_embedding_config.py`中添加新的配置
3. 在`rag_pdf_example.py`中使用不同的配置名称

### 运行示例

运行基础示例：

```bash
cd examples
python rag_pdf_example.py
```

运行测试脚本：

```bash
cd examples
python test_rag_config.py
```

这将测试所有配置是否正确工作。
