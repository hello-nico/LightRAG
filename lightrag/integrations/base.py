"""
LightRAG 集成基础抽象类

此模块定义了所有检索集成必须实现的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

from .models import (
    Resource,
    Document,
    RetrievalResult,
    RetrievalRequest,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    ListResourcesRequest,
    ListResourcesResponse,
)

logger = logging.getLogger(__name__)


class BaseIntegration(ABC):
    """检索集成基础抽象类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化集成实例

        Args:
            config: 集成配置
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化集成实例

        在此方法中执行任何必要的初始化操作，如：
        - 建立数据库连接
        - 加载模型
        - 验证配置
        """
        pass

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        执行单次检索

        Args:
            request: 检索请求

        Returns:
            RetrievalResult: 检索结果
        """
        pass

    @abstractmethod
    async def batch_retrieve(self, request: BatchRetrievalRequest) -> BatchRetrievalResponse:
        """
        执行批量检索

        Args:
            request: 批量检索请求

        Returns:
            BatchRetrievalResponse: 批量检索结果
        """
        pass

    @abstractmethod
    async def list_resources(self, request: ListResourcesRequest) -> ListResourcesResponse:
        """
        列出可用资源

        Args:
            request: 列出资源请求

        Returns:
            ListResourcesResponse: 资源列表响应
        """
        pass

    @abstractmethod
    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """
        获取指定资源

        Args:
            resource_id: 资源ID

        Returns:
            Optional[Resource]: 资源对象，如果不存在则返回None
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康检查结果，包含状态和相关指标
        """
        pass

    async def validate_config(self) -> List[str]:
        """
        验证配置

        Returns:
            List[str]: 验证错误列表，如果为空则表示配置有效
        """
        errors = []
        return errors

    async def close(self) -> None:
        """
        关闭集成实例，释放资源

        在此方法中执行任何必要的清理操作
        """
        self.is_initialized = False
        logger.info(f"Integration {self.name} closed")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        if not self.is_initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            Any: 配置值
        """
        return self.config.get(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值
        """
        self.config[key] = value

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置

        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取集成元数据

        Returns:
            Dict[str, Any]: 包含版本、描述、能力等信息的元数据
        """
        return {
            "name": self.name,
            "version": getattr(self, "version", "1.0.0"),
            "description": getattr(self, "description", ""),
            "capabilities": getattr(self, "capabilities", []),
            "is_initialized": self.is_initialized,
            "config_keys": list(self.config.keys()),
        }


class IntegrationError(Exception):
    """集成错误基类"""
    pass


class ConfigurationError(IntegrationError):
    """配置错误"""
    pass


class RetrievalError(IntegrationError):
    """检索错误"""
    pass


class ResourceNotFoundError(IntegrationError):
    """资源未找到错误"""
    pass


class HealthCheckError(IntegrationError):
    """健康检查错误"""
    pass


class RateLimitError(IntegrationError):
    """速率限制错误"""
    pass


class AuthenticationError(IntegrationError):
    """认证错误"""
    pass


class IntegrationFactory:
    """集成工厂类"""

    _integrations: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, integration_class: type) -> None:
        """
        注册集成类

        Args:
            name: 集成名称
            integration_class: 集成类（必须是 BaseIntegration 的子类）
        """
        if not issubclass(integration_class, BaseIntegration):
            raise ValueError(f"Integration class {integration_class} must inherit from BaseIntegration")

        cls._integrations[name] = integration_class
        logger.info(f"Registered integration: {name} -> {integration_class}")

    @classmethod
    def create(cls, name: str, config: Dict[str, Any] = None) -> BaseIntegration:
        """
        创建集成实例

        Args:
            name: 集成名称
            config: 集成配置

        Returns:
            BaseIntegration: 集成实例

        Raises:
            ValueError: 如果集成名称未注册
        """
        if name not in cls._integrations:
            raise ValueError(f"Unknown integration: {name}. Available: {list(cls._integrations.keys())}")

        integration_class = cls._integrations[name]
        return integration_class(config)

    @classmethod
    def list_integrations(cls) -> List[str]:
        """
        列出所有已注册的集成

        Returns:
            List[str]: 集成名称列表
        """
        return list(cls._integrations.keys())

    @classmethod
    def get_integration_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        获取集成信息

        Args:
            name: 集成名称

        Returns:
            Optional[Dict[str, Any]]: 集成信息，如果不存在则返回None
        """
        if name not in cls._integrations:
            return None

        integration_class = cls._integrations[name]
        return {
            "name": name,
            "class": integration_class.__name__,
            "module": integration_class.__module__,
            "docstring": integration_class.__doc__,
            "version": getattr(integration_class, "version", "1.0.0"),
            "description": getattr(integration_class, "description", ""),
        }