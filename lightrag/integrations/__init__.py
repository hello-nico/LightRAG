"""
LightRAG 检索集成模块

此模块提供了统一的检索接口，用于与外部系统集成。
"""

from .models import (
    Resource, Document, Chunk, RetrievalResult, RetrievalRequest, BatchRetrievalRequest, 
    BatchRetrievalResponse, ListResourcesRequest, ListResourcesResponse, ResourceType)
from .context_parser import ContextParser
from .converters import ConversionService
from .base import BaseIntegration, IntegrationFactory
from .manager import IntegrationManager, get_global_integration_manager, initialize_global_integration_manager
from .deer_flow import DeerFlowIntegration
from .config import (
    IntegrationConfig,
    IntegrationConfigManager,
    get_global_config_manager,
    get_integration_config,
    create_deer_flow_integration_config,
    load_config_from_env,
)

# 在模块加载时就注册 DeerFlow 集成，确保 IntegrationFactory 知晓该类型
IntegrationFactory.register("deer_flow", DeerFlowIntegration)

__all__ = [
    "Resource",
    "Document",
    "Chunk",
    "RetrievalResult",
    "RetrievalRequest",
    "BatchRetrievalRequest",
    "BatchRetrievalResponse",
    "ListResourcesRequest",
    "ListResourcesResponse",
    "ResourceType",
    "ContextParser",
    "ConversionService",
    "BaseIntegration",
    "IntegrationFactory",
    "IntegrationManager",
    "get_global_integration_manager",
    "initialize_global_integration_manager",
    "DeerFlowIntegration",
    "IntegrationConfig",
    "IntegrationConfigManager",
    "get_global_config_manager",
    "get_integration_config",
    "create_deer_flow_integration_config",
    "load_config_from_env",
]