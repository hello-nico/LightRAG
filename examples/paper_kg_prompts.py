"""
论文知识图谱实体和关系提取提示词模板
Paper Knowledge Graph Entity and Relationship Extraction Prompts

这个文件包含了专门针对学术论文的知识图谱提取提示词模板，
考虑了论文的结构特征、常见实体类型和学术关系。
"""

from __future__ import annotations
from typing import Any

PAPER_KG_PROMPTS: dict[str, Any] = {}

# 默认分隔符设置
PAPER_KG_PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PAPER_KG_PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PAPER_KG_PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# 论文专用实体类型定义
PAPER_KG_PROMPTS["PAPER_ENTITY_TYPES"] = [
    "Paper",          # 论文本身
    "Author",         # 作者
    "Institution",    # 研究机构
    "Method",         # 研究方法/算法
    "Dataset",        # 数据集
    "Theory",         # 理论/框架
    "Metric",         # 评价指标
    "Task",           # 研究任务
    "Field",          # 研究领域
    "Conference",     # 会议
    "Journal",        # 期刊
    "Technology",     # 技术工具
]

# 主要的论文实体和关系提取模板
PAPER_KG_PROMPTS["paper_entity_extraction"] = """---Goal---
Given an academic paper text, identify all entities and relationships related to the paper's structure, content, and academic context. Focus on extracting research-specific entities that are meaningful for academic knowledge graph construction.
Use {language} as output language.

---Steps---
1. **Entity Recognition**: Identify academic entities from the text. For each entity, extract:
- entity_name: Name of the entity, use same language as input text. If English, capitalize the name appropriately
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide a comprehensive description of the entity's academic significance and role in the research context

2. **Entity Format**:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. **Relationship Extraction**: From the identified entities, find all academically meaningful relationships:
- source_entity: name of the source entity
- target_entity: name of the target entity  
- relationship_keywords: high-level academic relationship type (e.g., "authorship", "citation", "evaluation", "application", "comparison")
- relationship_description: Explain the academic relationship and its significance

4. **Relationship Format**:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_description>)

5. **Formatting**: Use `{tuple_delimiter}` as field delimiter. Use `{record_delimiter}` as the list delimiter. Ensure no spaces are added around the delimiters.

6. **Completion**: When finished, output `{completion_delimiter}`

7. **Language**: Return identified entities and relationships in {language}.

---Academic Entity Guidelines---
- **Paper**: Include title, publication year, venue (conference/journal), DOI if available
- **Author**: Extract author names and their affiliations
- **Institution**: Universities, research labs, companies mentioned
- **Method**: Algorithms, models, frameworks, approaches proposed or used
- **Dataset**: Dataset names, sources, sizes, domains
- **Theory**: Theoretical frameworks, concepts, principles
- **Metric**: Evaluation metrics, performance measures, benchmarks
- **Task**: Research problems, tasks being addressed
- **Field**: Research areas, domains, subfields
- **Conference/Journal**: Publication venues, names, years
- **Technology**: Tools, frameworks, libraries, platforms used

---Quality Guidelines---
- Extract entities that have clear academic significance
- Focus on research-specific content rather than general knowledge
- Include numerical data (e.g., accuracy percentages, dataset sizes) in entity descriptions when relevant
- Ensure entity names are consistent and academically meaningful
- Only extract relationships that have clear academic justification

---Examples---
{examples}

---Real Data---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```

---Output---
Output:
"""

# 论文实体提取示例
PAPER_KG_PROMPTS["paper_entity_extraction_examples"] = [
    """------Example 1: Machine Learning Paper------

Entity_types: [Paper,Author,Institution,Method,Dataset,Metric,Task,Field,Conference,Technology]
Text:
```
In this paper, we propose BERT, a new language representation model designed to pre-train deep bidirectional representations from unlabeled text. Jacob Devlin and Ming-Wei Chang from Google AI introduce this approach that significantly improves performance on natural language processing tasks. BERT was pretrained on BookCorpus and Wikipedia, achieving 92.7% accuracy on GLUE benchmark and 89.4% on SQuAD. This work was published at NAACL 2019 and has been cited over 50,000 times. The model uses Transformer architecture and masked language modeling pre-training objective.
```

Output:
(entity{tuple_delimiter}BERT{tuple_delimiter}Method{tuple_delimiter}BERT is a bidirectional language representation model that uses Transformer architecture and masked language modeling for pre-training on unlabeled text.){record_delimiter}
(entity{tuple_delimiter}Jacob Devlin{tuple_delimiter}Author{tuple_delimiter}Jacob Devlin is a researcher at Google AI and one of the creators of the BERT language model.){record_delimiter}
(entity{tuple_delimiter}Ming-Wei Chang{tuple_delimiter}Author{tuple_delimiter}Ming-Wei Chang is a researcher at Google AI and co-author of the BERT paper.){record_delimiter}
(entity{tuple_delimiter}Google AI{tuple_delimiter}Institution{tuple_delimiter}Google AI is the research institution where BERT was developed.){record_delimiter}
(entity{tuple_delimiter}BookCorpus{tuple_delimiter}Dataset{tuple_delimiter}BookCorpus is a dataset used for pre-training BERT, consisting of unpublished books.){record_delimiter}
(entity{tuple_delimiter}Wikipedia{tuple_delimiter}Dataset{tuple_delimiter}Wikipedia is a large-scale dataset used for pre-training BERT model.){record_delimiter}
(entity{tuple_delimiter}GLUE Benchmark{tuple_delimiter}Metric{tuple_delimiter}GLUE Benchmark is a collection of natural language understanding tasks where BERT achieved 92.7% accuracy.){record_delimiter}
(entity{tuple_delimiter}SQuAD{tuple_delimiter}Metric{tuple_delimiter}SQuAD (Stanford Question Answering Dataset) is a reading comprehension benchmark where BERT achieved 89.4% accuracy.){record_delimiter}
(entity{tuple_delimiter}Natural Language Processing{tuple_delimiter}Field{tuple_delimiter}Natural Language Processing is the research field that BERT addresses and improves.){record_delimiter}
(entity{tuple_delimiter}NAACL 2019{tuple_delimiter}Conference{tuple_delimiter}NAACL 2019 is the conference where the BERT paper was published.){record_delimiter}
(entity{tuple_delimiter}Transformer Architecture{tuple_delimiter}Technology{tuple_delimiter}Transformer Architecture is the underlying technology used in BERT model design.){record_delimiter}
(relationship{tuple_delimiter}Jacob Devlin{tuple_delimiter}BERT{tuple_delimiter}authorship, creation{tuple_delimiter}Jacob Devlin is one of the main authors who created the BERT model.){record_delimiter}
(relationship{tuple_delimiter}Ming-Wei Chang{tuple_delimiter}BERT{tuple_delimiter}authorship, development{tuple_delimiter}Ming-Wei Chang co-authored the paper and contributed to BERT's development.){record_delimiter}
(relationship{tuple_delimiter}Google AI{tuple_delimiter}BERT{tuple_delimiter}institutional support, research{tuple_delimiter}Google AI provided the institutional environment and resources for BERT research.){record_delimiter}
(relationship{tuple_delimiter}BERT{tuple_delimiter}BookCorpus{tuple_delimiter}training data, pre-training{tuple_delimiter}BERT was pre-trained on the BookCorpus dataset as part of its training process.){record_delimiter}
(relationship{tuple_delimiter}BERT{tuple_delimiter}GLUE Benchmark{tuple_delimiter}evaluation, performance{tuple_delimiter}BERT was evaluated on the GLUE Benchmark, achieving 92.7% accuracy.){record_delimiter}
(relationship{tuple_delimiter}BERT{tuple_delimiter}Natural Language Processing{tuple_delimiter}application, advancement{tuple_delimiter}BERT represents a significant advancement in the Natural Language Processing field.){record_delimiter}
{completion_delimiter}

""",

    """------Example 2: Computer Vision Paper------

Entity_types: [Paper,Author,Institution,Method,Dataset,Metric,Task,Field,Conference,Technology]
Text:
```
ResNet: Deep Residual Learning for Image Recognition by Kaiming He et al. from Microsoft Research addresses the degradation problem in deep neural networks. The authors propose residual learning framework that achieved 3.57% error rate on ImageNet test set. This deep learning approach won first place in ILSVRC 2015 and has been widely adopted in computer vision applications. The ResNet-50 variant contains 50 layers and uses skip connections to enable training of very deep networks.
```

Output:
(entity{tuple_delimiter}ResNet{tuple_delimiter}Method{tuple_delimiter}ResNet is a deep residual learning framework for image recognition that addresses the degradation problem in deep neural networks using skip connections.){record_delimiter}
(entity{tuple_delimiter}Kaiming He{tuple_delimiter}Author{tuple_delimiter}Kaiming He is the lead author of the ResNet paper from Microsoft Research.){record_delimiter}
(entity{tuple_delimiter}Microsoft Research{tuple_delimiter}Institution{tuple_delimiter}Microsoft Research is the institution where ResNet was developed.){record_delimiter}
(entity{tuple_delimiter}ImageNet{tuple_delimiter}Dataset{tuple_delimiter}ImageNet is a large-scale dataset used for evaluating ResNet, where it achieved 3.57% error rate.){record_delimiter}
(entity{tuple_delimiter}ILSVRC 2015{tuple_delimiter}Conference{tuple_delimiter}ILSVRC 2015 is the competition where ResNet achieved first place.){record_delimiter}
(entity{tuple_delimiter}ResNet-50{tuple_delimiter}Method{tuple_delimiter}ResNet-50 is a specific variant of ResNet with 50 layers, implementing the residual learning framework.){record_delimiter}
(entity{tuple_delimiter}Computer Vision{tuple_delimiter}Field{tuple_delimiter}Computer Vision is the research field that ResNet targets and advances.){record_delimiter}
(entity{tuple_delimiter}Skip Connections{tuple_delimiter}Technology{tuple_delimiter}Skip connections are the key technological innovation in ResNet that enable training of very deep networks.){record_delimiter}
(relationship{tuple_delimiter}Kaiming He{tuple_delimiter}ResNet{tuple_delimiter}authorship, invention{tuple_delimiter}Kaiming He is the primary inventor and author of the ResNet method.){record_delimiter}
(relationship{tuple_delimiter}Microsoft Research{tuple_delimiter}ResNet{tuple_delimiter}institutional research, development{tuple_delimiter}Microsoft Research provided the research environment for ResNet development.){record_delimiter}
(relationship{tuple_delimiter}ResNet{tuple_delimiter}ImageNet{tuple_delimiter}evaluation, benchmark{tuple_delimiter}ResNet was evaluated on ImageNet dataset, achieving 3.57% error rate.){record_delimiter}
(relationship{tuple_delimiter}ResNet{tuple_delimiter}ILSVRC 2015{tuple_delimiter}competition, achievement{tuple_delimiter}ResNet won first place in the ILSVRC 2015 competition.){record_delimiter}
(relationship{tuple_delimiter}ResNet-50{tuple_delimiter}Skip Connections{tuple_delimiter}implementation, utilization{tuple_delimiter}ResNet-50 implements skip connections as its core architectural feature.){record_delimiter}
{completion_delimiter}

""",

    """------Example 3: Chinese AI Paper------

Entity_types: [Paper,Author,Institution,Method,Dataset,Metric,Task,Field,Conference,Technology]
Text:
```
在本文中，清华大学人工智能研究院的张三和李四提出了一种名为"智言"的大语言模型，该模型在中文自然语言处理任务上表现优异。该模型在CLUE基准测试上达到了91.2%的准确率，并在中文文本生成任务中超越了现有方法。这项研究发表在AAAI 2024会议上，模型基于Transformer架构，使用了中文维基百科和书籍语料库进行预训练。
```

Output:
(entity{tuple_delimiter}智言{tuple_delimiter}Method{tuple_delimiter}智言是一种大语言模型，专注于中文自然语言处理任务，基于Transformer架构开发。){record_delimiter}
(entity{tuple_delimiter}张三{tuple_delimiter}Author{tuple_delimiter}张三是清华大学人工智能研究院的研究员，智言模型的共同开发者。){record_delimiter}
(entity{tuple_delimiter}李四{tuple_delimiter}Author{tuple_delimiter}李四是清华大学人工智能研究院的研究员，智言模型的共同开发者。){record_delimiter}
(entity{tuple_delimiter}清华大学人工智能研究院{tuple_delimiter}Institution{tuple_delimiter}清华大学人工智能研究院是智言模型的开发机构。){record_delimiter}
(entity{tuple_delimiter}CLUE基准测试{tuple_delimiter}Metric{tuple_delimiter}CLUE基准测试是中文自然语言理解的评估标准，智言模型达到91.2%准确率。){record_delimiter}
(entity{tuple_delimiter}中文文本生成{tuple_delimiter}Task{tuple_delimiter}中文文本生成是智言模型表现优异的具体任务领域。){record_delimiter}
(entity{tuple_delimiter}AAAI 2024{tuple_delimiter}Conference{tuple_delimiter}AAAI 2024是发表智言模型研究的国际会议。){record_delimiter}
(entity{tuple_delimiter}Transformer架构{tuple_delimiter}Technology{tuple_delimiter}Transformer架构是智言模型的基础技术架构。){record_delimiter}
(entity{tuple_delimiter}中文维基百科{tuple_delimiter}Dataset{tuple_delimiter}中文维基百科是智言模型预训练的数据源之一。){record_delimiter}
(relationship{tuple_delimiter}张三{tuple_delimiter}智言{tuple_delimiter}作者关系, 模型开发{tuple_delimiter}张三是智言模型的主要开发者之一。){record_delimiter}
(relationship{tuple_delimiter}李四{tuple_delimiter}智言{tuple_delimiter}作者关系, 共同研究{tuple_delimiter}李四与张三共同开发了智言模型。){record_delimiter}
(relationship{tuple_delimiter}清华大学人工智能研究院{tuple_delimiter}智言{tuple_delimiter}机构支持, 研究环境{tuple_delimiter}清华大学人工智能研究院提供了智言模型的研究环境。){record_delimiter}
(relationship{tuple_delimiter}智言{tuple_delimiter}CLUE基准测试{tuple_delimiter}性能评估, 准确率{tuple_delimiter}智言模型在CLUE基准测试上达到91.2%的准确率。){record_delimiter}
(relationship{tuple_delimiter}智言{tuple_delimiter}中文文本生成{tuple_delimiter}任务应用, 性能优异{tuple_delimiter}智言模型在中文文本生成任务中表现优异。){record_delimiter}
{completion_delimiter}

""",
]

# 继续提取模板（用于遗漏的实体）
PAPER_KG_PROMPTS["paper_continue_extraction"] = """
MANY academic entities and relationships were missed in the last extraction. Please find only the missing academic entities and relationships from the research paper text. Do not include entities and relations that have been previously extracted. Focus on research-specific content such as methodologies, datasets, metrics, and academic relationships.

---Remember Steps---
1. **Entity Recognition**: Identify academic entities from the text. For each entity, extract:
- entity_name: Name of the entity, use same language as input text
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide a comprehensive description of the entity's academic significance

2. **Entity Format**:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. **Relationship Extraction**: From the identified entities, find all academically meaningful relationships:
- source_entity: name of the source entity
- target_entity: name of the target entity  
- relationship_keywords: high-level academic relationship type
- relationship_description: Explain the academic relationship and its significance

4. **Relationship Format**:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_description>)

5. **Formatting**: Use `{tuple_delimiter}` as field delimiter. Use `{record_delimiter}` as the list delimiter.

6. **Completion**: When finished, output `{completion_delimiter}`

7. **Language**: Return identified entities and relationships in {language}.

---Output---
Output:
"""

# 循环检查模板
PAPER_KG_PROMPTS["paper_if_loop_extraction"] = """
---Goal---

It appears some academic entities may have still been missed in the research paper extraction.

---Output---
Output:
""".strip()

# 论文实体描述总结模板
PAPER_KG_PROMPTS["paper_summarize_entity_descriptions"] = """---Role---
You are an Academic Knowledge Graph Specialist responsible for synthesizing research paper information.

---Task---
Your task is to synthesize multiple descriptions of academic entities or relationships from research papers into a single, comprehensive, and cohesive summary that maintains academic precision.

---Instructions---
1. **Academic Accuracy**: The summary must integrate key information from all provided descriptions while maintaining technical accuracy and academic rigor.
2. **Research Context**: The summary must explicitly mention the name of the academic entity or relationship and its research significance.
3. **Conflict Resolution**: In case of conflicting descriptions from different papers, acknowledge different perspectives or findings in the academic literature.
4. **Academic Style**: The output must be written from an objective, academic perspective using appropriate technical terminology.
5. **Length**: Maintain comprehensive coverage while ensuring the summary's length does not exceed {summary_length} tokens.
6. **Language**: The entire output must be written in {language}.

---Data---
{description_type} Name: {description_name}
Description List:
{description_list}

---Output---
Output:"""

# 论文关键词提取模板
PAPER_KG_PROMPTS["paper_keywords_extraction"] = """---Role---
You are an expert academic keyword extractor, specializing in analyzing research paper queries for academic literature retrieval. Your purpose is to identify both high-level and low-level keywords that will be effective for finding relevant academic papers.

---Goal---
Given a user query about academic research, extract two distinct types of keywords:
1. **high_level_keywords**: for broad research areas, methodologies, or theoretical frameworks
2. **low_level_keywords**: for specific techniques, datasets, metrics, or technical terms

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else.
2. **Academic Focus**: All keywords should be relevant to academic research and literature search.
3. **Research Context**: Consider both the query content and likely academic paper terminology.
4. **Technical Specificity**: Balance between general research areas and specific technical terms.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

# 论文关键词提取示例
PAPER_KG_PROMPTS["paper_keywords_extraction_examples"] = [
    """Example 1:

Query: "How does BERT perform on multilingual natural language processing tasks?"

Output:
{
  "high_level_keywords": ["Multilingual NLP", "Language models", "Cross-lingual transfer learning", "Natural language understanding"],
  "low_level_keywords": ["BERT", "Multilingual BERT", "Language representation learning", "Cross-lingual embeddings", "Transfer learning"]
}

""",
    """Example 2:

Query: "What are the latest deep learning approaches for medical image segmentation?"

Output:
{
  "high_level_keywords": ["Deep learning", "Medical image analysis", "Image segmentation", "Computer vision in healthcare"],
  "low_level_keywords": ["U-Net", "CNN architectures", "Medical imaging datasets", "Segmentation metrics", "Deep learning frameworks"]
}

""",
    """Example 3:

Query: "最近基于Transformer架构的时间序列预测方法有什么进展？"

Output:
{
  "high_level_keywords": ["时间序列预测", "Transformer架构", "深度学习时序分析", "预测模型"],
  "low_level_keywords": ["Transformer时序模型", "注意力机制", "时间序列数据集", "预测评价指标", "自监督学习"]
}

""",
]

# 论文查询响应模板
PAPER_KG_PROMPTS["paper_rag_response"] = """---Role---
You are an Academic Research Assistant responding to user queries about research papers based on Knowledge Graph and Document Chunks provided in JSON format.

---Goal---
Generate a comprehensive academic response based on the provided research paper knowledge base, following academic response guidelines. Summarize relevant research findings, methodologies, and relationships from the provided academic context.

---Conversation History---
{history}

---Academic Knowledge Graph and Document Chunks---
{context_data}

---Response Guidelines---
**1. Academic Content & Accuracy:**
- Strictly adhere to the provided academic context from research papers
- Maintain academic precision and technical accuracy
- If information is not found in the provided context, state that clearly
- Ensure academic continuity with the conversation history

**2. Academic Formatting & Language:**
- Format the response using academic markdown with appropriate section headings
- Use academic terminology and technical language appropriately
- The response language must match the user's question language
- Target format: Comprehensive academic response with citations

**3. Academic Citations:**
- At the end of the response, under a "References" section, include academic citations
- Use the following formats for academic citations:
  - For a Knowledge Graph Entity: `[KG] <entity_name> (<entity_type>)`
  - For a Knowledge Graph Relationship: `[KG] <entity1_name> → <entity2_name> (<relationship_type>)`
  - For a Document Chunk: `[DC] <paper_title_or_chunk_name>`

**4. Academic Structure:**
- Provide clear, well-structured academic explanations
- Include relevant research context and significance
- Highlight key findings and methodologies when appropriate

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
Output:"""

# 默认失败响应
PAPER_KG_PROMPTS["paper_fail_response"] = (
    "I cannot provide an answer to that academic question based on the available research papers.[no-context]"
)

# 使用说明和配置
PAPER_KG_PROMPTS["usage_instructions"] = """
# 论文知识图谱提取模板使用说明

## 实体类型说明
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

## 关系类型建议
- **authorship**: 作者关系（作者→论文）
- **affiliation**: 所属关系（作者→机构）
- **evaluation**: 评估关系（方法→指标）
- **application**: 应用关系（方法→任务）
- **comparison**: 比较关系（方法→方法）
- **citation**: 引用关系（论文→论文）
- **implementation**: 实现关系（方法→技术）
- **training**: 训练关系（模型→数据集）
- **publication**: 发表关系（论文→会议/期刊）

## 使用方法
1. 导入模板：`from paper_kg_prompts import PAPER_KG_PROMPTS`
2. 选择合适的模板：`prompt = PAPER_KG_PROMPTS["paper_entity_extraction"]`
3. 格式化参数：`formatted_prompt = prompt.format(**params)`
4. 调用LLM进行提取

## 参数说明
- **entity_types**: 实体类型列表，默认使用PAPER_ENTITY_TYPES
- **language**: 输出语言（如"English", "Chinese"）
- **examples**: 示例数据，帮助LLM理解提取格式
- **tuple_delimiter**: 字段分隔符，默认"<|>"
- **record_delimiter**: 记录分隔符，默认"##"
- **completion_delimiter**: 完成标记，默认"<|COMPLETE|>"
"""

if __name__ == "__main__":
    # 示例使用
    print("论文知识图谱提示词模板加载成功！")
    print(f"可用实体类型: {', '.join(PAPER_KG_PROMPTS['PAPER_ENTITY_TYPES'])}")
    print(f"可用模板数量: {len([k for k in PAPER_KG_PROMPTS.keys() if not k.startswith('DEFAULT') and k != 'PAPER_ENTITY_TYPES'])}")
    
    # 测试模板格式化
    test_params = {
        "language": "English",
        "entity_types": ", ".join(PAPER_KG_PROMPTS["PAPER_ENTITY_TYPES"][:5]),  # 只用前5个作为示例
        "examples": "This is a test example.",
        "input_text": "BERT is a language model developed by Google.",
        "tuple_delimiter": PAPER_KG_PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": PAPER_KG_PROMPTS["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": PAPER_KG_PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    }
    
    formatted = PAPER_KG_PROMPTS["paper_entity_extraction"].format(**test_params)
    print(f"\n模板格式化测试成功！模板长度: {len(formatted)} 字符")