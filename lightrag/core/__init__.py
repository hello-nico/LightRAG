"""
LightRAG Core Module

This module provides unified configuration and instance management for LightRAG.
"""

from .config import load_core_config, get_default_config
from .instance_manager import (
    get_lightrag_instance,
    set_instance,
    destroy_instance,
    get_instance_names,
    initialize_lightrag_with_config,
    inject_prompts_from_config,
    get_global_manager,
)

__all__ = [
    "load_core_config",
    "get_default_config",
    "get_lightrag_instance",
    "set_instance",
    "destroy_instance",
    "get_instance_names",
    "initialize_lightrag_with_config",
    "inject_prompts_from_config",
    "get_global_manager",
]