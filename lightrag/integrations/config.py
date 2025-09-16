"""
LightRAG 集成配置管理

此模块提供了集成系统的配置管理功能。
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """集成配置数据类"""

    # 默认配置
    default_provider: str = "deer_flow"
    similarity_threshold: float = 0.5
    default_mode: str = "mix"
    max_results: int = 10

    # DeerFlow 特定配置
    deer_flow: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "similarity_threshold": 0.5,
        "default_mode": "mix",
        "max_results": 10,
        "auto_initialize": True
    })

    # 其他集成配置（可扩展）
    integrations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 性能配置
    batch_size: int = 10
    timeout: float = 30.0
    max_retries: int = 3

    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1小时
    cache_max_size: int = 1000

    # 日志配置
    log_level: str = "INFO"
    enable_performance_logging: bool = True

    # API 配置
    api_base_path: str = "/retrieval"
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "default_provider": self.default_provider,
            "similarity_threshold": self.similarity_threshold,
            "default_mode": self.default_mode,
            "max_results": self.max_results,
            "deer_flow": self.deer_flow,
            "integrations": self.integrations,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enable_cache": self.enable_cache,
            "cache_ttl": self.cache_ttl,
            "cache_max_size": self.cache_max_size,
            "log_level": self.log_level,
            "enable_performance_logging": self.enable_performance_logging,
            "api_base_path": self.api_base_path,
            "enable_cors": self.enable_cors,
            "cors_origins": self.cors_origins,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrationConfig':
        """从字典创建配置"""
        config = cls()

        # 更新基本字段
        for key, value in data.items():
            if hasattr(config, key) and key not in ["deer_flow", "integrations"]:
                setattr(config, key, value)

        # 更新 DeerFlow 配置
        if "deer_flow" in data:
            config.deer_flow.update(data["deer_flow"])

        # 更新其他集成配置
        if "integrations" in data:
            config.integrations.update(data["integrations"])

        return config

    def get_deer_flow_config(self) -> Dict[str, Any]:
        """获取 DeerFlow 配置"""
        config = self.deer_flow.copy()
        # 确保基本配置值被继承
        config.setdefault("similarity_threshold", self.similarity_threshold)
        config.setdefault("default_mode", self.default_mode)
        config.setdefault("max_results", self.max_results)
        return config

    def get_integration_config(self, provider: str) -> Dict[str, Any]:
        """获取指定提供商的配置"""
        if provider == "deer_flow":
            return self.get_deer_flow_config()
        else:
            return self.integrations.get(provider, {})


class IntegrationConfigManager:
    """集成配置管理器"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.config = IntegrationConfig()
        self._load_config()

    def _get_default_config_path(self) -> Path:
        """获取默认配置文件路径"""
        # 首先检查环境变量
        env_path = os.getenv("LIGHTRAG_INTEGRATION_CONFIG")
        if env_path:
            return Path(env_path)

        # 然后检查当前目录
        current_dir = Path.cwd()
        config_file = current_dir / "integration_config.json"
        if config_file.exists():
            return config_file

        # 最后使用用户目录
        home_dir = Path.home()
        return home_dir / ".lightrag" / "integration_config.json"

    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            logger.info(f"Config file not found at {self.config_path}, using defaults")
            self._save_config()  # 保存默认配置
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.config = IntegrationConfig.from_dict(data)
            logger.info(f"Loaded integration config from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")

    def _save_config(self) -> None:
        """保存配置文件"""
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved integration config to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")

    def get_config(self) -> IntegrationConfig:
        """获取当前配置"""
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        更新配置

        Args:
            updates: 配置更新字典
        """
        self.config = IntegrationConfig.from_dict({**self.config.to_dict(), **updates})
        self._save_config()

    def update_deer_flow_config(self, updates: Dict[str, Any]) -> None:
        """
        更新 DeerFlow 配置

        Args:
            updates: DeerFlow 配置更新字典
        """
        self.config.deer_flow.update(updates)
        self._save_config()

    def add_integration(self, name: str, config: Dict[str, Any]) -> None:
        """
        添加新的集成配置

        Args:
            name: 集成名称
            config: 集成配置
        """
        self.config.integrations[name] = config
        self._save_config()

    def remove_integration(self, name: str) -> bool:
        """
        移除集成配置

        Args:
            name: 集成名称

        Returns:
            bool: 是否成功移除
        """
        if name in self.config.integrations:
            del self.config.integrations[name]
            self._save_config()
            return True
        return False

    def set_default_provider(self, provider: str) -> None:
        """
        设置默认提供商

        Args:
            provider: 提供商名称
        """
        self.config.default_provider = provider
        self._save_config()

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        获取提供商配置

        Args:
            provider: 提供商名称

        Returns:
            Dict[str, Any]: 提供商配置
        """
        return self.config.get_integration_config(provider)

    def list_providers(self) -> List[str]:
        """
        列出所有可用的提供商

        Returns:
            List[str]: 提供商列表
        """
        providers = ["deer_flow"]  # 内置提供商
        providers.extend(self.config.integrations.keys())
        return providers

    def validate_config(self) -> List[str]:
        """
        验证配置

        Returns:
            List[str]: 验证错误列表
        """
        errors = []

        # 验证默认提供商
        if self.config.default_provider not in self.list_providers():
            errors.append(f"Default provider '{self.config.default_provider}' is not available")

        # 验证相似度阈值
        if not (0 <= self.config.similarity_threshold <= 1):
            errors.append("similarity_threshold must be between 0 and 1")

        # 验证 DeerFlow 配置
        if self.config.deer_flow.get("enabled", False):
            deer_flow_config = self.config.deer_flow
            if "similarity_threshold" in deer_flow_config:
                threshold = deer_flow_config["similarity_threshold"]
                if not (0 <= threshold <= 1):
                    errors.append("deer_flow.similarity_threshold must be between 0 and 1")

            if "default_mode" in deer_flow_config:
                mode = deer_flow_config["default_mode"]
                valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
                if mode not in valid_modes:
                    errors.append(f"deer_flow.default_mode must be one of {valid_modes}")

            if "max_results" in deer_flow_config:
                max_results = deer_flow_config["max_results"]
                if not isinstance(max_results, int) or max_results <= 0:
                    errors.append("deer_flow.max_results must be a positive integer")

        # 验证其他配置
        if self.config.batch_size <= 0:
            errors.append("batch_size must be a positive integer")

        if self.config.timeout <= 0:
            errors.append("timeout must be a positive number")

        if self.config.max_retries < 0:
            errors.append("max_retries must be non-negative")

        if self.config.cache_ttl <= 0:
            errors.append("cache_ttl must be a positive integer")

        if self.config.cache_max_size <= 0:
            errors.append("cache_max_size must be a positive integer")

        return errors


# 全局配置管理器实例
_global_config_manager: Optional[IntegrationConfigManager] = None


def get_global_config_manager() -> IntegrationConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = IntegrationConfigManager()
    return _global_config_manager


def get_integration_config() -> IntegrationConfig:
    """获取全局集成配置"""
    return get_global_config_manager().get_config()


def create_deer_flow_integration_config(rag_instance: Any) -> Dict[str, Any]:
    """
    创建 DeerFlow 集成配置

    Args:
        rag_instance: LightRAG 实例

    Returns:
        Dict[str, Any]: 集成配置
    """
    config = get_integration_config()
    deer_flow_config = config.get_deer_flow_config()

    # 添加 LightRAG 实例
    deer_flow_config["rag_instance"] = rag_instance

    return deer_flow_config


def load_config_from_env() -> IntegrationConfig:
    """
    从环境变量加载配置

    Returns:
        IntegrationConfig: 配置对象
    """
    config = IntegrationConfig()

    # 从环境变量覆盖配置
    if "LIGHTRAG_DEFAULT_PROVIDER" in os.environ:
        config.default_provider = os.environ["LIGHTRAG_DEFAULT_PROVIDER"]

    if "LIGHTRAG_SIMILARITY_THRESHOLD" in os.environ:
        try:
            config.similarity_threshold = float(os.environ["LIGHTRAG_SIMILARITY_THRESHOLD"])
        except ValueError:
            logger.warning("Invalid LIGHTRAG_SIMILARITY_THRESHOLD value")

    if "LIGHTRAG_DEFAULT_MODE" in os.environ:
        config.default_mode = os.environ["LIGHTRAG_DEFAULT_MODE"]

    if "LIGHTRAG_MAX_RESULTS" in os.environ:
        try:
            config.max_results = int(os.environ["LIGHTRAG_MAX_RESULTS"])
        except ValueError:
            logger.warning("Invalid LIGHTRAG_MAX_RESULTS value")

    # DeerFlow 特定环境变量
    deer_flow_env_vars = {
        "LIGHTRAG_DEER_FLOW_ENABLED": "enabled",
        "LIGHTRAG_DEER_FLOW_SIMILARITY_THRESHOLD": "similarity_threshold",
        "LIGHTRAG_DEER_FLOW_DEFAULT_MODE": "default_mode",
        "LIGHTRAG_DEER_FLOW_MAX_RESULTS": "max_results",
        "LIGHTRAG_DEER_FLOW_AUTO_INITIALIZE": "auto_initialize",
    }

    for env_var, config_key in deer_flow_env_vars.items():
        if env_var in os.environ:
            value = os.environ[env_var]

            # 类型转换
            if config_key in ["similarity_threshold"]:
                try:
                    value = float(value)
                except ValueError:
                    logger.warning(f"Invalid {env_var} value")
                    continue
            elif config_key in ["max_results"]:
                try:
                    value = int(value)
                except ValueError:
                    logger.warning(f"Invalid {env_var} value")
                    continue
            elif config_key in ["enabled", "auto_initialize"]:
                value = value.lower() in ["true", "1", "yes", "on"]

            config.deer_flow[config_key] = value

    return config