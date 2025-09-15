"""
论文知识图谱集成配置
Paper Knowledge Graph Integration Configuration

这个文件展示了如何将论文专用的知识图谱提取模板集成到LightRAG系统中。
"""

from typing import Dict, Any, Optional
from paper_kg_prompts import PAPER_KG_PROMPTS


class PaperKGConfig:
    """论文知识图谱配置类"""
    
    def __init__(self):
        """初始化论文知识图谱配置"""
        self.entity_types = PAPER_KG_PROMPTS["PAPER_ENTITY_TYPES"]
        self.language = "English"  # 默认语言
        self.max_gleaning = 3  # 最大提取轮数
        self.enable_chunking = False  # 单篇论文不需要分块
        self.max_context_length = 8000  # 最大上下文长度，适应大多数LLM


class SinglePaperConfig(PaperKGConfig):
    """单篇论文专用配置类 - 不需要分块处理"""
    
    def __init__(self):
        """初始化单篇论文配置"""
        super().__init__()
        self.enable_chunking = False  # 明确禁用分块
        self.max_context_length = 8000  # 适应大多数LLM的上下文限制
        self.preprocessing_enabled = True  # 启用论文预处理
        self.preserve_structure = True  # 保留论文结构信息
        
    def get_prompt_template(self, template_name: str) -> str:
        """
        获取指定名称的提示词模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            提示词模板字符串
        """
        return PAPER_KG_PROMPTS.get(template_name, "")
    
    def get_default_prompts(self) -> Dict[str, str]:
        """
        获取默认的提示词集合
        
        Returns:
            默认提示词字典
        """
        return {
            "entity_extraction": PAPER_KG_PROMPTS["paper_entity_extraction"],
            "continue_extraction": PAPER_KG_PROMPTS["paper_continue_extraction"],
            "if_loop_extraction": PAPER_KG_PROMPTS["paper_if_loop_extraction"],
            "summarize_entity_descriptions": PAPER_KG_PROMPTS["paper_summarize_entity_descriptions"],
            "keywords_extraction": PAPER_KG_PROMPTS["paper_keywords_extraction"],
            "rag_response": PAPER_KG_PROMPTS["paper_rag_response"],
            "fail_response": PAPER_KG_PROMPTS["paper_fail_response"],
        }
    
    def get_format_params(self) -> Dict[str, Any]:
        """
        获取格式化参数
        
        Returns:
            格式化参数字典
        """
        return {
            "tuple_delimiter": PAPER_KG_PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PAPER_KG_PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PAPER_KG_PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "entity_types": ", ".join(self.entity_types),
            "language": self.language,
        }
    
    def configure_for_paper_domain(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        为论文领域配置基础设置
        
        Args:
            base_config: 基础配置字典
            
        Returns:
            配置后的字典
        """
        # 复制基础配置
        config = base_config.copy()
        
        # 设置论文专用的实体类型
        if "addon_params" not in config:
            config["addon_params"] = {}
        
        config["addon_params"]["entity_types"] = self.entity_types
        config["addon_params"]["language"] = self.language
        
        # 设置提取参数
        config["entity_extract_max_gleaning"] = self.max_gleaning
        
        # 设置文本处理参数 - 单篇论文不需要分块
        config["enable_chunking"] = self.enable_chunking
        config["max_context_length"] = self.max_context_length
        
        return config


def create_single_paper_config(
    working_dir: str = "./single_paper_kg_storage",
    llm_model_func: Optional[callable] = None,
    embedding_func: Optional[callable] = None,
    language: str = "English",
    enable_logging: bool = True,
) -> Dict[str, Any]:
    """
    创建单篇论文专用的LightRAG配置 - 不分块处理
    
    Args:
        working_dir: 工作目录
        llm_model_func: LLM模型函数
        embedding_func: 嵌入函数
        language: 输出语言
        enable_logging: 是否启用日志
        
    Returns:
        配置字典
    """
    # 创建单篇论文配置
    paper_config = SinglePaperConfig()
    paper_config.language = language
    
    # 基础配置
    config = {
        "llm_model_func": llm_model_func,
        "embedding_func": embedding_func,
        "working_dir": working_dir,
        "enable_logging": enable_logging,
        "log_level": "INFO" if enable_logging else "ERROR",
    }
    
    # 应用论文专用配置
    config = paper_config.configure_for_paper_domain(config)
    
    return config


def create_paper_lightrag_config(
    working_dir: str = "./paper_kg_storage",
    llm_model_func: Optional[callable] = None,
    embedding_func: Optional[callable] = None,
    language: str = "English",
    enable_logging: bool = True,
) -> Dict[str, Any]:
    """
    创建论文专用的LightRAG配置
    
    Args:
        working_dir: 工作目录
        llm_model_func: LLM模型函数
        embedding_func: 嵌入函数
        language: 输出语言
        enable_logging: 是否启用日志
        
    Returns:
        配置字典
    """
    # 创建论文配置
    paper_config = PaperKGConfig()
    paper_config.language = language
    
    # 基础配置
    config = {
        "llm_model_func": llm_model_func,
        "embedding_func": embedding_func,
        "working_dir": working_dir,
        "enable_logging": enable_logging,
        "log_level": "INFO" if enable_logging else "ERROR",
    }
    
    # 应用论文专用配置
    config = paper_config.configure_for_paper_domain(config)
    
    return config


def integrate_paper_prompts_to_lightrag(lightrag_instance, paper_config: PaperKGConfig):
    """
    将论文提示词集成到LightRAG实例中
    
    Args:
        lightrag_instance: LightRAG实例
        paper_config: 论文配置实例
    """
    # 替换提示词
    if hasattr(lightrag_instance, 'prompt'):
        lightrag_instance.prompt.update(paper_config.get_default_prompts())
    
    # 替换实体类型
    if hasattr(lightrag_instance, 'entity_types'):
        lightrag_instance.entity_types = paper_config.entity_types
    
    # 更新配置
    if hasattr(lightrag_instance, 'addon_params'):
        lightrag_instance.addon_params.update({
            "entity_types": paper_config.entity_types,
            "language": paper_config.language,
        })


# 预定义的论文领域配置
PAPER_DOMAIN_CONFIGS = {
    "machine_learning": {
        "entity_types": [
            "Paper", "Author", "Institution", "Method", "Dataset", "Metric", 
            "Task", "Field", "Conference", "Journal", "Technology", "Theory"
        ],
        "language": "English",
        "specific_keywords": [
            "neural network", "deep learning", "machine learning", "artificial intelligence",
            "supervised learning", "unsupervised learning", "reinforcement learning"
        ]
    },
    
    "computer_vision": {
        "entity_types": [
            "Paper", "Author", "Institution", "Method", "Dataset", "Metric", 
            "Task", "Field", "Conference", "Journal", "Technology"
        ],
        "language": "English",
        "specific_keywords": [
            "image processing", "object detection", "image segmentation", "computer vision",
            "convolutional neural network", "CNN", "image classification"
        ]
    },
    
    "natural_language_processing": {
        "entity_types": [
            "Paper", "Author", "Institution", "Method", "Dataset", "Metric", 
            "Task", "Field", "Conference", "Journal", "Technology", "Theory"
        ],
        "language": "English",
        "specific_keywords": [
            "NLP", "natural language processing", "language model", "transformer",
            "BERT", "GPT", "text classification", "machine translation"
        ]
    },
    
    "chinese_ai_research": {
        "entity_types": [
            "论文", "作者", "机构", "方法", "数据集", "指标", 
            "任务", "领域", "会议", "期刊", "技术", "理论"
        ],
        "language": "Chinese",
        "specific_keywords": [
            "人工智能", "机器学习", "深度学习", "自然语言处理",
            "计算机视觉", "大语言模型", "预训练", "微调"
        ]
    }
}


def get_domain_config(domain_name: str) -> Dict[str, Any]:
    """
    获取特定领域的配置
    
    Args:
        domain_name: 领域名称
        
    Returns:
        领域配置字典
    """
    return PAPER_DOMAIN_CONFIGS.get(domain_name, PAPER_DOMAIN_CONFIGS["machine_learning"])


# 使用示例和最佳实践
if __name__ == "__main__":
    print("论文知识图谱集成配置")
    print("=" * 50)
    
    # 1. 创建单篇论文配置（推荐用于单篇论文处理）
    print("\n1. 创建单篇论文配置（不分块）:")
    single_config = create_single_paper_config(
        working_dir="./single_paper_kg_storage",
        language="English"
    )
    
    print(f"工作目录: {single_config['working_dir']}")
    print(f"是否分块: {single_config['enable_chunking']}")
    print(f"最大上下文长度: {single_config['max_context_length']}")
    print(f"实体类型: {single_config['addon_params']['entity_types']}")
    
    # 2. 创建多篇论文配置（如果需要处理多篇论文）
    print("\n2. 创建多篇论文配置:")
    multi_config = create_paper_lightrag_config(
        working_dir="./multi_paper_kg_storage",
        language="English"
    )
    
    print(f"工作目录: {multi_config['working_dir']}")
    print(f"是否分块: {multi_config['enable_chunking']}")
    
    # 3. 获取特定领域配置
    print("\n3. 获取NLP领域配置:")
    nlp_config = get_domain_config("natural_language_processing")
    print(f"NLP实体类型: {nlp_config['entity_types']}")
    print(f"NLP关键词: {nlp_config['specific_keywords']}")
    
    # 4. 创建单篇论文配置实例
    print("\n4. 创建单篇论文配置实例:")
    single_paper_config = SinglePaperConfig()
    single_paper_config.language = "Chinese"
    
    format_params = single_paper_config.get_format_params()
    print(f"格式化参数: {format_params}")
    print(f"是否启用预处理: {single_paper_config.preprocessing_enabled}")
    print(f"是否保留结构: {single_paper_config.preserve_structure}")
    
    # 5. 获取提示词模板
    print("\n5. 获取提示词模板:")
    extraction_template = single_paper_config.get_prompt_template("paper_entity_extraction")
    print(f"提取模板长度: {len(extraction_template)} 字符")
    
    # 6. 显示可用模板
    print("\n6. 可用模板:")
    default_prompts = single_paper_config.get_default_prompts()
    for template_name in default_prompts.keys():
        print(f"  - {template_name}")
    
    print("\n配置示例完成！")
    print("\n推荐使用单篇论文配置，因为：")
    print("- 不需要分块处理，保持论文完整性")
    print("- 更好的上下文理解")
    print("- 简化处理流程")
    print("- 更适合LLM处理完整论文内容")