"""
Qwen Embedding配置文件
用于配置RAG PDF处理器的Qwen embedding模型
"""

from dataclasses import dataclass
from typing import Optional
import os
import asyncio
from lightrag.llm.openai import openai_complete_if_cache


@dataclass
class QwenEmbeddingConfig:
    """Qwen Embedding模型配置"""
    
    # 基础配置
    base_url: str = "http://localhost:8000"
    api_key: str = "your-api-key-here"
    model: str = "qwen-embedding"
    
    # 高级配置
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 32
    
    # HTTPS配置
    https: bool = False
    
    # Embedding配置
    embedding_dimension: int = 1536
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量获取配置
        self.base_url = os.getenv("QWEN_EMBEDDING_BASE_URL", self.base_url)
        self.api_key = os.getenv("QWEN_EMBEDDING_API_KEY", self.api_key)
        self.model = os.getenv("QWEN_EMBEDDING_MODEL", self.model)
        
        # 规范化URL格式
        if "://" in self.base_url:
            self.base_url = self.base_url.split("://")[1]
        self.base_url = self.base_url.rstrip("/")
    
    def validate(self):
        """验证配置"""
        if not self.base_url:
            raise ValueError("base_url不能为空")
        if not self.api_key:
            raise ValueError("api_key不能为空")
        if not self.model:
            raise ValueError("model不能为空")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "batch_size": self.batch_size,
            "https": self.https,
            "embedding_dimension": self.embedding_dimension
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'QwenEmbeddingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'QwenEmbeddingConfig':
        """从环境变量创建配置"""
        return cls()


# 预定义的Qwen embedding配置
QWEN_EMBEDDING_CONFIGS = {
    "default": QwenEmbeddingConfig(
        base_url="http://localhost:8000",
        api_key="your-api-key-here",
        model="qwen-embedding"
    ),
    
    "qwen-7b": QwenEmbeddingConfig(
        base_url="http://localhost:8000",
        api_key="your-api-key-here",
        model="qwen-7b-embedding"
    ),
    
    "qwen-14b": QwenEmbeddingConfig(
        base_url="http://localhost:8000",
        api_key="your-api-key-here",
        model="qwen-14b-embedding"
    ),
    
    "qwen-72b": QwenEmbeddingConfig(
        base_url="http://localhost:8000",
        api_key="your-api-key-here",
        model="qwen-72b-embedding"
    ),
    
    "production": QwenEmbeddingConfig(
        base_url="https://api.example.com",
        api_key="your-production-api-key",
        model="qwen-embedding",
        https=True,
        timeout=60,
        max_retries=5
    ),
    
    "env-config": QwenEmbeddingConfig(
        base_url="10.0.62.206:51200/v1",
        api_key="sk-uHj8K2mNpL5vR9xQ4tY7wB3cA6nE0iF1gD8sZ2yX4jM9kP3h",
        model="Qwen3-Embedding-0.6B",
        timeout=45,
        max_retries=5,
        batch_size=32,
        embedding_dimension=1024
    )
}


def get_qwen_config(config_name: str = "default") -> QwenEmbeddingConfig:
    """
    获取预定义的Qwen embedding配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        QwenEmbeddingConfig实例
    """
    if config_name not in QWEN_EMBEDDING_CONFIGS:
        available_configs = list(QWEN_EMBEDDING_CONFIGS.keys())
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {available_configs}")
    
    return QWEN_EMBEDDING_CONFIGS[config_name]


def create_custom_config(
    base_url: str,
    api_key: str,
    model: str = "qwen-embedding",
    **kwargs
) -> QwenEmbeddingConfig:
    """
    创建自定义Qwen embedding配置
    
    Args:
        base_url: API基础URL
        api_key: API密钥
        model: 模型名称
        **kwargs: 其他配置参数
        
    Returns:
        QwenEmbeddingConfig实例
    """
    config_dict = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        **kwargs
    }
    
    return QwenEmbeddingConfig.from_dict(config_dict)


# 使用示例
if __name__ == "__main__":
    # 1. 使用默认配置
    default_config = get_qwen_config("default")
    print(f"默认配置: {default_config.to_dict()}")
    
    # 2. 使用生产环境配置
    prod_config = get_qwen_config("production")
    print(f"生产配置: {prod_config.to_dict()}")
    
    # 3. 创建自定义配置
    custom_config = create_custom_config(
        base_url="http://custom-api:8080",
        api_key="custom-key",
        model="custom-embedding",
        timeout=45
    )
    print(f"自定义配置: {custom_config.to_dict()}")
    
    # 4. 从环境变量创建配置
    env_config = QwenEmbeddingConfig.from_env()
    print(f"环境变量配置: {env_config.to_dict()}")


# LLM配置函数
async def env_llm_model_func(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False, 
    **kwargs
) -> str:
    """
    基于环境配置的LLM模型函数
    
    Args:
        prompt: 用户输入的提示
        system_prompt: 系统提示
        history_messages: 历史消息列表
        keyword_extraction: 是否进行关键词提取
        **kwargs: 其他参数
        
    Returns:
        模型回复
    """
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-v3-250324"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY", "sk-4e76fcdf3f95467198edabdc0d6627f6"),  # 使用环境中的API密钥
        base_url=os.getenv("LLM_BINDING_HOST", "http://10.0.62.214:15000/k-llm"),  # 使用环境中的API地址
        **kwargs
    )