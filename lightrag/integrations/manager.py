"""
LightRAG 集成管理器

此模块负责管理所有检索集成实例，提供统一的访问接口。
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
from dataclasses import dataclass

from .base import BaseIntegration, IntegrationFactory, IntegrationError
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


@dataclass
class IntegrationConfig:
    """集成配置"""
    name: str
    integration_type: str
    config: Dict[str, Any]
    is_default: bool = False
    is_enabled: bool = True


class IntegrationManager:
    """集成管理器"""

    def __init__(self, default_config: Dict[str, Any] = None):
        """
        初始化集成管理器

        Args:
            default_config: 默认配置
        """
        self.default_config = default_config or {}
        self.integrations: Dict[str, BaseIntegration] = {}
        self.configs: Dict[str, IntegrationConfig] = {}
        self.default_integration: Optional[str] = None
        self.is_initialized = False

    async def initialize(self) -> None:
        """初始化管理器"""
        if self.is_initialized:
            return

        logger.info("Initializing IntegrationManager")

        # 从配置创建集成实例
        for config_name, integration_config in self.configs.items():
            if integration_config.is_enabled:
                try:
                    await self._create_integration(config_name, integration_config)
                except Exception as e:
                    logger.error(f"Failed to initialize integration {config_name}: {e}")

        self.is_initialized = True
        logger.info(f"IntegrationManager initialized with {len(self.integrations)} integrations")

    async def _create_integration(self, config_name: str, config: IntegrationConfig) -> None:
        """创建集成实例"""
        try:
            integration = IntegrationFactory.create(config.integration_type, config.config)
            await integration.initialize()

            self.integrations[config_name] = integration

            if config.is_default:
                self.default_integration = config_name

            logger.info(f"Created integration: {config_name} -> {config.integration_type}")

        except Exception as e:
            logger.error(f"Failed to create integration {config_name}: {e}")
            raise

    def register_integration(self, config: IntegrationConfig) -> None:
        """
        注册集成配置

        Args:
            config: 集成配置
        """
        self.configs[config.name] = config
        logger.info(f"Registered integration config: {config.name}")

    def register_integration_from_dict(self, name: str, config_dict: Dict[str, Any]) -> None:
        """
        从字典注册集成配置

        Args:
            name: 集成名称
            config_dict: 配置字典
        """
        config = IntegrationConfig(
            name=name,
            integration_type=config_dict.get("type", "deer_flow"),
            config=config_dict.get("config", {}),
            is_default=config_dict.get("is_default", False),
            is_enabled=config_dict.get("is_enabled", True)
        )
        self.register_integration(config)

    async def get_integration(self, name: Optional[str] = None) -> BaseIntegration:
        """
        获取集成实例

        Args:
            name: 集成名称，如果为None则使用默认集成

        Returns:
            BaseIntegration: 集成实例

        Raises:
            IntegrationError: 如果集成不存在
        """
        if not self.is_initialized:
            await self.initialize()

        if name is None:
            name = self.default_integration

        if name is None:
            raise IntegrationError("No default integration configured")

        if name not in self.integrations:
            raise IntegrationError(f"Integration '{name}' not found")

        return self.integrations[name]

    async def retrieve(self, request: RetrievalRequest, integration_name: Optional[str] = None) -> RetrievalResult:
        """
        执行检索

        Args:
            request: 检索请求
            integration_name: 集成名称

        Returns:
            RetrievalResult: 检索结果
        """
        integration = await self.get_integration(integration_name)
        return await integration.retrieve(request)

    async def batch_retrieve(self, request: BatchRetrievalRequest, integration_name: Optional[str] = None) -> BatchRetrievalResponse:
        """
        执行批量检索

        Args:
            request: 批量检索请求
            integration_name: 集成名称

        Returns:
            BatchRetrievalResponse: 批量检索结果
        """
        integration = await self.get_integration(integration_name)
        return await integration.batch_retrieve(request)

    async def list_resources(self, request: ListResourcesRequest, integration_name: Optional[str] = None) -> ListResourcesResponse:
        """
        列出资源

        Args:
            request: 列出资源请求
            integration_name: 集成名称

        Returns:
            ListResourcesResponse: 资源列表
        """
        integration = await self.get_integration(integration_name)
        return await integration.list_resources(request)

    async def get_resource(self, resource_id: str, integration_name: Optional[str] = None) -> Optional[Resource]:
        """
        获取资源

        Args:
            resource_id: 资源ID
            integration_name: 集成名称

        Returns:
            Optional[Resource]: 资源对象
        """
        integration = await self.get_integration(integration_name)
        return await integration.get_resource(resource_id)

    async def health_check(self, integration_name: Optional[str] = None) -> Dict[str, Any]:
        """
        健康检查

        Args:
            integration_name: 集成名称，如果为None则检查所有集成

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        if integration_name:
            integration = await self.get_integration(integration_name)
            return await integration.health_check()
        else:
            # 检查所有集成
            results = {}
            for name, integration in self.integrations.items():
                try:
                    results[name] = await integration.health_check()
                except Exception as e:
                    results[name] = {
                        "status": "error",
                        "error": str(e)
                    }
            return results

    def list_integrations(self) -> List[str]:
        """
        列出所有可用的集成

        Returns:
            List[str]: 集成名称列表
        """
        return list(self.integrations.keys())

    def get_integration_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取集成信息

        Args:
            name: 集成名称，如果为None则返回所有集成的信息

        Returns:
            Dict[str, Any]: 集成信息
        """
        if name:
            if name not in self.integrations:
                return {}
            integration = self.integrations[name]
            config = self.configs.get(name, {})
            return {
                "name": name,
                "type": integration.name,
                "is_default": config.is_default,
                "is_enabled": config.is_enabled,
                "is_initialized": integration.is_initialized,
                "config_keys": list(integration.config.keys()),
                "metadata": integration.get_metadata()
            }
        else:
            # 返回所有集成的信息
            info = {}
            for name in self.integrations:
                info[name] = self.get_integration_info(name)
            return info

    async def reload_integration(self, name: str) -> None:
        """
        重新加载集成

        Args:
            name: 集成名称
        """
        if name not in self.configs:
            raise IntegrationError(f"Integration config '{name}' not found")

        # 关闭现有集成
        if name in self.integrations:
            await self.integrations[name].close()
            del self.integrations[name]

        # 重新创建集成
        config = self.configs[name]
        await self._create_integration(name, config)

    async def close(self) -> None:
        """关闭管理器，释放所有资源"""
        logger.info("Closing IntegrationManager")

        # 关闭所有集成
        close_tasks = []
        for integration in self.integrations.values():
            close_tasks.append(integration.close())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.integrations.clear()
        self.is_initialized = False
        logger.info("IntegrationManager closed")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        if not self.is_initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    def set_default_integration(self, name: str) -> None:
        """
        设置默认集成

        Args:
            name: 集成名称
        """
        if name not in self.configs:
            raise IntegrationError(f"Integration config '{name}' not found")

        # 取消之前的默认设置
        for config in self.configs.values():
            config.is_default = False

        # 设置新的默认集成
        self.configs[name].is_default = True
        self.default_integration = name
        logger.info(f"Set default integration to: {name}")

    def enable_integration(self, name: str) -> None:
        """
        启用集成

        Args:
            name: 集成名称
        """
        if name not in self.configs:
            raise IntegrationError(f"Integration config '{name}' not found")

        self.configs[name].is_enabled = True
        logger.info(f"Enabled integration: {name}")

    def disable_integration(self, name: str) -> None:
        """
        禁用集成

        Args:
            name: 集成名称
        """
        if name not in self.configs:
            raise IntegrationError(f"Integration config '{name}' not found")

        self.configs[name].is_enabled = False
        logger.info(f"Disabled integration: {name}")


# 全局集成管理器实例
_global_integration_manager: Optional[IntegrationManager] = None


def get_global_integration_manager() -> IntegrationManager:
    """获取全局集成管理器实例"""
    global _global_integration_manager
    if _global_integration_manager is None:
        _global_integration_manager = IntegrationManager()
    return _global_integration_manager


async def initialize_global_integration_manager(config: Dict[str, Any] = None) -> IntegrationManager:
    """
    初始化全局集成管理器

    Args:
        config: 配置字典

    Returns:
        IntegrationManager: 全局集成管理器实例
    """
    global _global_integration_manager
    _global_integration_manager = IntegrationManager(config)

    # 从配置注册集成
    if config and "integrations" in config:
        for name, integration_config in config["integrations"].items():
            _global_integration_manager.register_integration_from_dict(name, integration_config)

    await _global_integration_manager.initialize()
    return _global_integration_manager