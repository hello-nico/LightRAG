# 论文知识图谱提取模板

这是一个专门为学术论文设计的知识图谱实体和关系提取提示词模板集合，基于LightRAG框架开发。

## 文件结构

```
examples/
├── paper_kg_prompts.py      # 论文知识图谱提示词模板
├── paper_kg_example.py      # 使用示例代码
├── paper_kg_config.py       # 集成配置文件
└── README.md               # 本说明文件
```

## 核心特性

### 1. 论文专用实体类型

- **Paper**: 论文本身（标题、发表信息）
- **Author**: 论文作者（姓名、所属机构）
- **Institution**: 研究机构（大学、公司、实验室）
- **Method**: 研究方法（算法、模型、框架）
- **Dataset**: 数据集（名称、来源、规模）
- **Theory**: 理论框架（概念、原理）
- **Metric**: 评价指标（准确率、F1分数等）
- **Task**: 研究任务（问题类型、应用场景）
- **Field**: 研究领域（学科方向、子领域）
- **Conference**: 学术会议（名称、年份）
- **Journal**: 学术期刊（名称、卷期）
- **Technology**: 技术工具（框架、库、平台）

### 2. 学术关系类型

- **authorship**: 作者关系（作者→论文）
- **affiliation**: 所属关系（作者→机构）
- **evaluation**: 评估关系（方法→指标）
- **application**: 应用关系（方法→任务）
- **comparison**: 比较关系（方法→方法）
- **citation**: 引用关系（论文→论文）
- **implementation**: 实现关系（方法→技术）
- **training**: 训练关系（模型→数据集）
- **publication**: 发表关系（论文→会议/期刊）

### 3. 多语言支持

- 支持英文和中文论文
- 可扩展到其他语言
- 保持学术术语的准确性

## 快速开始

### 1. 基本使用

```python
from paper_kg_prompts import PAPER_KG_PROMPTS
from paper_kg_example import PaperKGExtractor

# 创建提取器
extractor = PaperKGExtractor(llm_func=your_llm_function)

# 格式化提示词
prompt = extractor.format_extraction_prompt(paper_text)

# 提取知识图谱
kg_data = await extractor.extract_paper_kg(paper_text)
```

### 2. 集成到LightRAG

```python
from paper_kg_config import create_paper_lightrag_config, integrate_paper_prompts_to_lightrag

# 创建论文专用配置
config = create_paper_lightrag_config(
    working_dir="./paper_kg_storage",
    llm_model_func=your_llm_function,
    language="English"
)

# 创建LightRAG实例
rag = LightRAG(**config)

# 集成论文提示词
paper_config = PaperKGConfig()
integrate_paper_prompts_to_lightrag(rag, paper_config)
```

## 示例用例

### 示例1: 机器学习论文

```python
paper_text = """
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
"""

# 提取知识图谱
kg_data = await extractor.extract_paper_kg(paper_text)

# 结果包含:
# - BERT (Method)
# - Jacob Devlin (Author)
# - Google AI (Institution)
# - GLUE Benchmark (Metric)
# - Transformer (Technology)
```

### 示例2: 中文AI论文

```python
paper_text = """
在本文中，清华大学人工智能研究院的张三和李四提出了一种名为"智言"的大语言模型，该模型在中文自然语言处理任务上表现优异。该模型在CLUE基准测试上达到了91.2%的准确率。
"""

# 使用中文配置
extractor = PaperKGExtractor(llm_func=your_llm_function, language="Chinese")
kg_data = await extractor.extract_paper_kg(paper_text)

# 结果包含:
# - 智言 (Method)
# - 张三 (Author)
# - 清华大学人工智能研究院 (Institution)
# - CLUE基准测试 (Metric)
```

## 配置选项

### 领域特定配置

```python
from paper_kg_config import get_domain_config

# 获取NLP领域配置
nlp_config = get_domain_config("natural_language_processing")

# 获取中文AI研究配置
chinese_config = get_domain_config("chinese_ai_research")
```

### 自定义实体类型

```python
from paper_kg_config import PaperKGConfig

config = PaperKGConfig()
config.entity_types = [
    "Paper", "Author", "Method", "Dataset", "Metric", 
    "Field", "Conference", "Technology"  # 自定义子集
]
```

## 提示词模板详解

### 1. 实体提取模板 (`paper_entity_extraction`)

专门针对学术论文的实体和关系提取，包含：

- 学术实体识别指南
- 结构化输出格式
- 质量控制标准
- 多语言支持
- 详细示例

### 2. 实体描述总结模板 (`paper_summarize_entity_descriptions`)

专门用于学术实体描述的总结，强调：

- 学术准确性
- 技术精度
- 矛盾处理
- 学术写作风格

### 3. 关键词提取模板 (`paper_keywords_extraction`)

针对学术文献检索的关键词提取：

- 高层次研究领域关键词
- 具体技术术语关键词
- 学术搜索优化

### 4. RAG响应模板 (`paper_rag_response`)

学术研究助手响应模板：

- 学术内容准确性
- 技术术语使用
- 学术引用格式
- 结构化回答

## 高级功能

### 1. 批量处理

```python
async def batch_extract_papers(paper_texts: List[str]) -> List[Dict]:
    """批量提取多篇论文的知识图谱"""
    extractor = PaperKGExtractor(llm_func=your_llm_function)
    
    tasks = [extractor.extract_paper_kg(text) for text in paper_texts]
    results = await asyncio.gather(*tasks)
    
    return results
```

### 2. 知识图谱合并

```python
def merge_paper_kgs(kg_list: List[Dict]) -> Dict:
    """合并多篇论文的知识图谱"""
    merged_entities = {}
    merged_relationships = {}
    
    for kg in kg_list:
        for entity in kg['entities']:
            if entity['name'] not in merged_entities:
                merged_entities[entity['name']] = entity
            else:
                # 合并描述
                existing = merged_entities[entity['name']]
                existing['description'] += f" {entity['description']}"
        
        # 合并关系...
    
    return {
        'entities': list(merged_entities.values()),
        'relationships': list(merged_relationships.values())
    }
```

### 3. 可视化支持

```python
def visualize_paper_kg(kg_data: Dict):
    """可视化论文知识图谱"""
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.Graph()
    
    # 添加节点
    for entity in kg_data['entities']:
        G.add_node(entity['name'], type=entity['type'])
    
    # 添加边
    for rel in kg_data['relationships']:
        G.add_edge(rel['source'], rel['target'], 
                  relationship=rel['keywords'])
    
    # 绘制图形
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000)
    plt.show()
```

## 最佳实践

### 1. 论文预处理

```python
def preprocess_paper_text(raw_text: str) -> str:
    """预处理论文文本"""
    # 移除多余的空白字符
    text = ' '.join(raw_text.split())
    
    # 保留重要结构
    # 如标题、作者、摘要等
    
    return text
```

### 2. 实体标准化

```python
def normalize_entity_name(name: str, entity_type: str) -> str:
    """标准化实体名称"""
    if entity_type == "Author":
        # 标准化作者姓名格式
        return name.strip().title()
    elif entity_type == "Method":
        # 标准化方法名称
        return name.strip()
    else:
        return name.strip()
```

### 3. 质量控制

```python
def validate_kg_extraction(kg_data: Dict) -> bool:
    """验证提取结果质量"""
    # 检查实体和关系数量
    if len(kg_data['entities']) == 0:
        return False
    
    # 检查关系合理性
    for rel in kg_data['relationships']:
        if rel['source'] == rel['target']:
            return False
    
    return True
```

## 扩展和定制

### 1. 添加新的实体类型

```python
# 在paper_kg_prompts.py中添加新的实体类型
PAPER_KG_PROMPTS["PAPER_ENTITY_TYPES"].extend([
    "Funding",  # 资金信息
    "Patent",    # 专利信息
    "Software"   # 软件工具
])
```

### 2. 自定义关系类型

```python
# 添加特定领域的关系类型
domain_specific_relationships = {
    "machine_learning": [
        "pretraining", "fine_tuning", "transfer_learning",
        "model_comparison", "architecture_variation"
    ],
    "computer_vision": [
        "image_preprocessing", "feature_extraction", 
        "model_architecture", "performance_evaluation"
    ]
}
```

### 3. 多语言扩展

```python
# 添加新语言支持
language_prompts = {
    "Japanese": {
        "entity_extraction": """---Goal---
日本語の学術論文からエンティティと関係を抽出...
""",
        "examples": [
            # 日语示例
        ]
    }
}
```

## 性能优化

### 1. 缓存机制

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_prompt_formatting(template_name: str, **kwargs) -> str:
    """缓存提示词格式化结果"""
    template = PAPER_KG_PROMPTS[template_name]
    return template.format(**kwargs)
```

### 2. 批量处理优化

```python
async def optimized_batch_extract(paper_texts: List[str], batch_size: int = 5):
    """优化的批量提取"""
    extractor = PaperKGExtractor(llm_func=your_llm_function)
    
    results = []
    for i in range(0, len(paper_texts), batch_size):
        batch = paper_texts[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            extractor.extract_paper_kg(text) for text in batch
        ])
        results.extend(batch_results)
    
    return results
```

## 常见问题

### Q1: 如何处理论文中的数学公式？

A1: 目前模板专注于文本内容，数学公式可以在预处理阶段转换为文本描述。

### Q2: 如何提高提取准确性？

A2:

- 使用更具体的领域配置
- 提供更多示例
- 优化提示词
- 使用post-processing验证

### Q3: 支持哪些语言？

A3: 目前支持英文和中文，可以扩展到其他语言。

### Q4: 如何处理论文引用？

A4: 引用关系可以通过`citation`关系类型表示，也可以作为独立的`Paper`实体。

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目基于LightRAG许可证，请参考原始项目许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues
- Email: [your-email@example.com]

---

**注意**: 这些模板专门为学术论文设计，对于其他类型的文档可能需要调整。
