# LightRAG实例替换计划

## 当前状况分析

- `lightrag_server.py` 在第585行自己创建新的 LightRAG 实例
- `examples/rag_pdf_processor.py` 中的 `initialize_rag()` 函数返回一个已配置好的 `RAGPDFProcessor` 实例，该实例包含 `lightrag_instance` 属性
- 目标是让服务器使用 `rag_pdf_processor.py` 中的 LightRAG 实例

## 实施步骤

### 1. 修改 `lightrag_server.py` 导入部分

- 添加从 `examples.rag_pdf_processor` 导入 `initialize_rag` 函数
- 添加相应的依赖导入

### 2. 修改 `create_app` 函数中的 LightRAG 实例创建逻辑

- 在第584行左右，替换原有的 `LightRAG` 实例创建代码
- 调用 `await initialize_rag()` 获取 `RAGPDFProcessor` 实例
- 从 `processor.lightrag_instance` 获取 LightRAG 实例

### 3. 处理配置参数的传递

- 确保 `initialize_rag()` 函数能接收服务器配置参数
- 可能需要修改 `initialize_rag()` 函数以接受服务器的 `args` 参数
- 或者创建一个新的初始化函数，结合服务器配置和 RAG PDF 处理器的配置

### 4. 确保初始化顺序正确

- 需要在 `lifespan` 函数之前初始化 LightRAG 实例
- 确保存储初始化和管道状态初始化的正确性

### 5. 测试和验证

- 验证服务器能正常启动
- 确认 RAG 功能正常工作
- 测试 API 端点的可用性

## 需要注意的问题

1. **配置合并**：需要将服务器配置与 RAG PDF 处理器配置合并
2. **初始化时机**：确保在正确的时机初始化 LightRAG 实例
3. **错误处理**：添加适当的错误处理机制
4. **依赖管理**：确保所有必要的依赖都能正确导入

## 相关代码位置

- `lightrag_server.py` 第584-621行：LightRAG 实例创建
- `examples/rag_pdf_processor.py` 第352-373行：`initialize_rag` 函数
- `examples/rag_pdf_processor.py` 第139-190行：`_initialize_lightrag` 方法
- `examples/rag_pdf_processor.py` 第174行：LightRAG 实例创建

## 配置对比

### 服务器配置（lightrag_server.py）

- working_dir: args.working_dir
- workspace: args.workspace
- llm_model_func: create_llm_model_func(args.llm_binding)
- embedding_func: create_optimized_embedding_function(...)
- kv_storage: args.kv_storage
- vector_storage: args.vector_storage
- graph_storage: args.graph_storage
- doc_status_storage: args.doc_status_storage

### RAG PDF 处理器配置（rag_pdf_processor.py）

- working_dir: "./rag_lightrag_storage"
- llm_model_func: env_llm_model_func
- embedding_func: qwen_embedding_func 包装
- kv_storage: 从环境变量获取或默认值
- vector_storage: 从环境变量获取或默认值
- graph_storage: 从环境变量获取或默认值
- doc_status_storage: 从环境变量获取或默认值
