"""
论文知识图谱提取示例
Paper Knowledge Graph Extraction Example

这个文件展示了如何使用论文专用的知识图谱提取模板来处理学术论文。
特别针对单篇论文的完整处理，不需要分块。
"""

import asyncio
import json
from typing import Dict, Any
from paper_kg_prompts import PAPER_KG_PROMPTS
from paper_kg_config import SinglePaperConfig


class PaperKGExtractor:
    """论文知识图谱提取器"""
    
    def __init__(self, llm_func=None, language="English", config=None):
        """
        初始化论文知识图谱提取器
        
        Args:
            llm_func: LLM调用函数
            language: 输出语言，默认为English
            config: 论文配置实例，如果为None则使用默认的单篇论文配置
        """
        self.llm_func = llm_func
        self.language = language
        self.config = config or SinglePaperConfig()
        self.config.language = language
        self.entity_types = self.config.entity_types
        
    def format_extraction_prompt(self, paper_text: str) -> str:
        """
        格式化论文实体提取提示词
        
        Args:
            paper_text: 论文文本内容
            
        Returns:
            格式化后的提示词
        """
        # 选择相关的示例（根据文本长度和复杂度）
        examples = "\n".join(PAPER_KG_PROMPTS["paper_entity_extraction_examples"][:2])
        
        # 格式化示例
        example_context = {
            "tuple_delimiter": PAPER_KG_PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PAPER_KG_PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PAPER_KG_PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "entity_types": ", ".join(self.entity_types),
            "language": self.language,
        }
        
        formatted_examples = examples.format(**example_context)
        
        # 格式化主提示词
        prompt_context = {
            "language": self.language,
            "entity_types": ", ".join(self.entity_types),
            "examples": formatted_examples,
            "input_text": paper_text,
            "tuple_delimiter": PAPER_KG_PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PAPER_KG_PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PAPER_KG_PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        }
        
        return PAPER_KG_PROMPTS["paper_entity_extraction"].format(**prompt_context)
    
    def preprocess_paper_text(self, paper_text: str) -> str:
        """
        预处理论文文本
        
        Args:
            paper_text: 原始论文文本
            
        Returns:
            预处理后的文本
        """
        if not self.config.preprocessing_enabled:
            return paper_text
        
        # 基本清理
        processed_text = paper_text.strip()
        
        # 移除多余的空白字符
        processed_text = ' '.join(processed_text.split())
        
        # 如果论文过长，进行智能截断
        if len(processed_text) > self.config.max_context_length:
            # 保留标题、摘要、引言和结论等重要部分
            # 这里简化处理，实际可以根据论文结构进行更智能的截断
            processed_text = processed_text[:self.config.max_context_length]
            processed_text += "\n\n[Note: Paper was truncated due to length constraints]"
        
        return processed_text

    def parse_extraction_result(self, result: str) -> Dict[str, Any]:
        """
        解析提取结果
        
        Args:
            result: LLM返回的提取结果
            
        Returns:
            解析后的实体和关系字典
        """
        entities = []
        relationships = []
        
        # 分割记录
        records = result.split(PAPER_KG_PROMPTS["DEFAULT_RECORD_DELIMITER"])
        
        for record in records:
            record = record.strip()
            if not record:
                continue
                
            # 分割字段
            fields = record.split(PAPER_KG_PROMPTS["DEFAULT_TUPLE_DELIMITER"])
            
            if len(fields) >= 4 and fields[0] == "entity":
                # 解析实体
                entities.append({
                    "name": fields[1],
                    "type": fields[2],
                    "description": fields[3] if len(fields) > 3 else ""
                })
            elif len(fields) >= 5 and fields[0] == "relationship":
                # 解析关系
                relationships.append({
                    "source": fields[1],
                    "target": fields[2],
                    "keywords": fields[3],
                    "description": fields[4] if len(fields) > 4 else ""
                })
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
    async def extract_paper_kg(self, paper_text: str) -> Dict[str, Any]:
        """
        提取论文知识图谱
        
        Args:
            paper_text: 论文文本内容
            
        Returns:
            提取的知识图谱数据
        """
        if not self.llm_func:
            raise ValueError("LLM function is required for extraction")
        
        # 预处理论文文本
        processed_text = self.preprocess_paper_text(paper_text)
        
        # 格式化提示词
        prompt = self.format_extraction_prompt(processed_text)
        
        # 调用LLM
        try:
            result = await self.llm_func(prompt)
            
            # 解析结果
            kg_data = self.parse_extraction_result(result)
            
            # 添加处理元数据
            kg_data["metadata"] = {
                "original_length": len(paper_text),
                "processed_length": len(processed_text),
                "enable_chunking": self.config.enable_chunking,
                "language": self.language,
                "preprocessing_enabled": self.config.preprocessing_enabled
            }
            
            return kg_data
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            return {"entities": [], "relationships": [], "metadata": {"error": str(e)}}
    
    def format_keywords_prompt(self, query: str) -> str:
        """
        格式化关键词提取提示词
        
        Args:
            query: 用户查询
            
        Returns:
            格式化后的提示词
        """
        examples = "\n".join(PAPER_KG_PROMPTS["paper_keywords_extraction_examples"])
        
        prompt_context = {
            "examples": examples,
            "query": query
        }
        
        return PAPER_KG_PROMPTS["paper_keywords_extraction"].format(**prompt_context)
    
    def format_rag_prompt(self, context_data: str, user_prompt: str, history: str = "") -> str:
        """
        格式化RAG响应提示词
        
        Args:
            context_data: 知识图谱和文档块数据
            user_prompt: 用户提示
            history: 对话历史
            
        Returns:
            格式化后的提示词
        """
        prompt_context = {
            "history": history,
            "context_data": context_data,
            "user_prompt": user_prompt,
            "response_type": "comprehensive academic response"
        }
        
        return PAPER_KG_PROMPTS["paper_rag_response"].format(**prompt_context)


# 示例使用
async def demo_paper_extraction():
    """演示单篇论文知识图谱提取"""
    
    # 模拟LLM函数（实际使用时替换为真实的LLM调用）
    async def mock_llm(prompt: str) -> str:
        """模拟LLM响应"""
        # 这里返回一个示例响应
        return """
(entity<|>Attention Is All You Need<|>Paper<|>Transformer architecture paper that introduced self-attention mechanism for sequence transduction models.)##(entity<|>Ashish Vaswani<|>Author<|>Lead author of the Transformer paper, researcher at Google Brain.)##(entity<|>Google Brain<|>Institution<|>Google research team that developed the Transformer architecture.)##(entity<|>Transformer<|>Method<|>Neural network architecture based on self-attention mechanism for processing sequential data.)##(entity<|>Self-Attention<|>Technology<|>Mechanism that allows models to weigh the importance of different input tokens.)##(relationship<|>Ashish Vaswani<|>Attention Is All You Need<|>authorship<|>Ashish Vaswani is the lead author of the Transformer paper.)##(relationship<|>Google Brain<|>Transformer<|>development<|>Google Brain developed the Transformer architecture.)##(relationship<|>Transformer<|>Self-Attention<|>implementation<|>Transformer architecture implements self-attention mechanism.)<|COMPLETE|>
        """
    
    # 创建单篇论文配置
    config = SinglePaperConfig()
    config.language = "English"
    
    # 创建提取器
    extractor = PaperKGExtractor(llm_func=mock_llm, language="English", config=config)
    
    # 示例论文文本
    sample_paper = """
    Attention Is All You Need
    
    The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
    
    - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
    - Google Brain, Google
    """
    
    # 提取知识图谱
    print("开始提取单篇论文知识图谱...")
    print(f"配置信息:")
    print(f"- 是否分块: {config.enable_chunking}")
    print(f"- 最大上下文长度: {config.max_context_length}")
    print(f"- 是否启用预处理: {config.preprocessing_enabled}")
    
    kg_data = await extractor.extract_paper_kg(sample_paper)
    
    # 显示结果
    print("\n=== 提取结果 ===")
    print(f"实体数量: {len(kg_data['entities'])}")
    print(f"关系数量: {len(kg_data['relationships'])}")
    
    if "metadata" in kg_data:
        print(f"原始文本长度: {kg_data['metadata']['original_length']}")
        print(f"处理后长度: {kg_data['metadata']['processed_length']}")
    
    print("\n=== 实体列表 ===")
    for entity in kg_data['entities']:
        print(f"- {entity['name']} ({entity['type']})")
        print(f"  描述: {entity['description'][:100]}...")
    
    print("\n=== 关系列表 ===")
    for rel in kg_data['relationships']:
        print(f"- {rel['source']} → {rel['target']}")
        print(f"  关系类型: {rel['keywords']}")
        print(f"  描述: {rel['description'][:100]}...")
    
    # 保存结果到JSON文件
    output_file = "single_paper_kg_extraction_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    print("\n=== 单篇论文处理优势 ===")
    print("✓ 保持论文完整性 - 不需要分块")
    print("✓ 更好的上下文理解 - 完整论文内容")
    print("✓ 简化处理流程 - 一次处理完成")
    print("✓ 更准确的实体关系提取 - 全文上下文")


def demo_prompt_formatting():
    """演示提示词格式化"""
    
    extractor = PaperKGExtractor()
    
    # 示例论文文本
    sample_text = """
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    
    We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
    """
    
    # 格式化提取提示词
    extraction_prompt = extractor.format_extraction_prompt(sample_text)
    
    print("=== 论文实体提取提示词 ===")
    print(extraction_prompt[:1000] + "..." if len(extraction_prompt) > 1000 else extraction_prompt)
    
    # 格式化关键词提取提示词
    keywords_prompt = extractor.format_keywords_prompt(
        "How does BERT compare to other language models?"
    )
    
    print("\n=== 关键词提取提示词 ===")
    print(keywords_prompt)
    
    # 格式化RAG响应提示词
    rag_prompt = extractor.format_rag_prompt(
        context_data="[KG] BERT (Method)\n[DC] BERT paper content...",
        user_prompt="What is BERT and how does it work?",
        history="Previous discussion about language models"
    )
    
    print("\n=== RAG响应提示词 ===")
    print(rag_prompt[:1000] + "..." if len(rag_prompt) > 1000 else rag_prompt)


if __name__ == "__main__":
    print("单篇论文知识图谱提取示例")
    print("=" * 50)
    
    # 演示提示词格式化
    print("\n1. 提示词格式化演示:")
    demo_prompt_formatting()
    
    # 演示单篇论文提取过程
    print("\n2. 单篇论文知识图谱提取演示:")
    asyncio.run(demo_paper_extraction())
    
    print("\n=== 总结 ===")
    print("单篇论文知识图谱提取的关键优势：")
    print("1. 完整性 - 保持论文的完整上下文")
    print("2. 准确性 - 基于全文的实体关系提取")
    print("3. 高效性 - 一次处理，无需分块")
    print("4. 一致性 - 避免分块带来的信息丢失")
    
    print("\n演示完成！")