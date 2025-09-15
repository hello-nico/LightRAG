# LightRAG 查询模式分析

## 概述

LightRAG 提供了6种查询模式，每种模式适用于不同类型的问题和检索需求。这些模式可以根据问题的复杂度、信息类型和检索策略进行选择。

## 查询模式详解

### 1. naive（朴素模式）

**特点**：

- 仅使用向量检索，不涉及知识图谱
- 基于文本相似度进行检索
- 速度最快，最简单

**适用问题类型**：

- 简单的事实性问题
- 基于关键词匹配的问题
- 快速信息检索
- 对话式问答

**示例问题**：

- "什么是RAG？"
- "机器学习的基本概念是什么？"
- "这篇文章讲了什么？"

### 2. local（本地模式）

**特点**：

- 基于实体检索，关注上下文相关的详细信息
- 检索与查询相关的实体及其关系
- 适合深入挖掘特定主题的细节

**适用问题类型**：

- 需要详细解释的问题
- 关于特定概念的深入理解
- 需要具体案例和应用场景的问题
- 技术实现细节相关的问题

**示例问题**：

- "LightRAG的具体实现原理是什么？"
- "某个算法的详细步骤是怎样的？"
- "这个技术在实际项目中如何应用？"

### 3. global（全局模式）

**特点**：

- 基于关系检索，关注全局结构和概念间联系
- 检索实体间的关系和整体架构
- 适合理解系统性的知识

**适用问题类型**：

- 需要理解整体架构的问题
- 概念间关系和对比的问题
- 系统性和全局性的问题
- 理论框架相关的问题

**示例问题**：

- "RAG系统的各个组件如何协同工作？"
- "不同检索方法的优缺点比较？"
- "这个领域的主要研究方向有哪些？"

### 4. hybrid（混合模式）

**特点**：

- 结合本地和全局检索方法
- 先进行本地检索，再进行全局检索
- 平衡了细节和全局视角

**适用问题类型**：

- 需要同时考虑细节和全局的问题
- 复杂的多层次问题
- 需要全面分析的问题
- 综合性的技术评估

**示例问题**：

- "分析这个技术的优缺点及应用场景"
- "比较不同解决方案的差异和适用性"
- "评估某个方法的效果和局限性"

### 5. mix（混合增强模式）

**特点**：

- 融合知识图谱和向量检索
- 结合实体、关系和文本块检索
- 最全面的检索模式

**适用问题类型**：

- 最复杂的综合性问题
- 需要多维度信息支持的问题
- 跨领域的复杂查询
- 需要深度推理和分析的问题

**示例问题**：

- "基于这个架构设计一个完整的解决方案"
- "分析这个技术在不同场景下的应用效果"
- "综合评估并提出改进建议"

### 6. bypass（绕过模式）

**特点**：

- 完全绕过检索，直接使用LLM
- 仅依赖对话历史和LLM自身知识
- 速度最快，但可能缺乏特定领域的准确信息

**适用问题类型**：

- 一般性知识问题
- 创意性问题和头脑风暴
- 代码生成和调试
- 不需要特定领域知识的通用问题

**示例问题**：

- "写一个Python脚本"
- "给我一些创意想法"
- "如何学习编程？"

## 查询参数配置

### 核心参数

- `mode`: 查询模式（默认"mix"）
- `top_k`: 检索的实体/关系数量（默认60）
- `chunk_top_k`: 检索的文本块数量（默认10）
- `response_type`: 响应格式（默认"Multiple Paragraphs"）

### Token控制参数

- `max_entity_tokens`: 实体上下文最大token数（默认6000）
- `max_relation_tokens`: 关系上下文最大token数（默认8000）
- `max_total_tokens`: 总token预算（默认30000）

### 其他参数

- `only_need_context`: 仅返回检索上下文
- `only_need_prompt`: 仅返回生成的提示词
- `stream`: 是否启用流式输出
- `enable_rerank`: 是否启用重排序

## 选择建议

### 按问题复杂度选择

1. **简单问题** → naive
2. **细节导向问题** → local
3. **全局导向问题** → global
4. **平衡性问题** → hybrid
5. **复杂综合性问题** → mix
6. **通用知识问题** → bypass

### 按检索需求选择

- **速度优先** → naive / bypass
- **准确性优先** → mix / hybrid
- **细节深入** → local
- **全局理解** → global
- **全面分析** → mix

## 使用示例

### Python代码示例

```python
from lightrag import LightRAG, QueryParam

# 初始化RAG
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=your_llm_function,
    embedding_func=your_embedding_function
)

# 不同模式查询示例
query = "LightRAG的特点是什么？"

# 朴素模式
result = rag.query(query, param=QueryParam(mode="naive"))

# 本地模式
result = rag.query(query, param=QueryParam(mode="local"))

# 全局模式
result = rag.query(query, param=QueryParam(mode="global"))

# 混合模式
result = rag.query(query, param=QueryParam(mode="hybrid"))

# 混合增强模式
result = rag.query(query, param=QueryParam(mode="mix"))

# 绕过模式
result = rag.query(query, param=QueryParam(mode="bypass"))

# 带参数的查询
result = rag.query(query, param=QueryParam(
    mode="mix",
    top_k=30,
    response_type="Bullet Points",
    max_total_tokens=20000
))
```

### API调用示例

```python
import requests

# 通过API查询
response = requests.post("http://localhost:9621/api/query", json={
    "query": "LightRAG的特点是什么？",
    "mode": "mix",
    "top_k": 30,
    "response_type": "Multiple Paragraphs"
})
```

## 实际Demo代码

```python
#!/usr/bin/env python3
"""
LightRAG查询模式演示代码
Query Modes Demo for LightRAG
"""

import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag.utils import EmbeddingFunc

async def query_modes_demo():
    """
    演示LightRAG的6种查询模式
    """
    
    # 初始化LightRAG（需要先有数据）
    rag = LightRAG(
        working_dir="./demo_rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: gpt_4o_mini_complete(texts)
        )
    )
    
    # 测试查询
    query = "LightRAG系统的主要特点是什么？"
    
    # 1. Naive模式
    print("=== Naive Mode ===")
    result = await rag.aquery(
        query, 
        param=QueryParam(mode="naive")
    )
    print(f"Result: {result}\n")
    
    # 2. Local模式
    print("=== Local Mode ===")
    result = await rag.aquery(
        query, 
        param=QueryParam(mode="local")
    )
    print(f"Result: {result}\n")
    
    # 3. Global模式
    print("=== Global Mode ===")
    result = await rag.aquery(
        query, 
        param=QueryParam(mode="global")
    )
    print(f"Result: {result}\n")
    
    # 4. Hybrid模式
    print("=== Hybrid Mode ===")
    result = await rag.aquery(
        query, 
        param=QueryParam(mode="hybrid")
    )
    print(f"Result: {result}\n")
    
    # 5. Mix模式
    print("=== Mix Mode ===")
    result = await rag.aquery(
        query, 
        param=QueryParam(mode="mix")
    )
    print(f"Result: {result}\n")
    
    # 6. Bypass模式
    print("=== Bypass Mode ===")
    result = await rag.aquery(
        query, 
        param=QueryParam(mode="bypass")
    )
    print(f"Result: {result}\n")
    
    # 带参数的查询示例
    print("=== Configured Query ===")
    result = await rag.aquery(
        query, 
        param=QueryParam(
            mode="mix",
            top_k=20,
            chunk_top_k=5,
            response_type="Bullet Points",
            max_entity_tokens=3000,
            max_relation_tokens=4000,
            max_total_tokens=15000
        )
    )
    print(f"Result: {result}\n")

if __name__ == "__main__":
    asyncio.run(query_modes_demo())
```

## 总结

LightRAG的查询模式提供了灵活的信息检索策略，从最简单的向量检索到最复杂的混合增强检索，可以满足不同类型问题的需求。选择合适的查询模式需要考虑问题的复杂度、信息需求和性能要求。通常建议：

1. **简单问题**使用naive模式
2. **中等复杂度**使用local或global模式
3. **复杂问题**使用hybrid或mix模式
4. **通用知识**使用bypass模式

通过合理配置查询参数，可以在准确性、速度和资源消耗之间找到最佳平衡点。
