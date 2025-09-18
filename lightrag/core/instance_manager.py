"""
LightRAG Instance Manager

This module provides global instance management for LightRAG with async support,
lifecycle management, and configuration injection.
"""

import asyncio
import logging
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from enum import Enum
import os
from dotenv import load_dotenv

from lightrag import LightRAG
from .config import LightRAGCoreConfig, load_core_config, load_prompts_from_file, merge_prompts_with_defaults

logger = logging.getLogger(__name__)


class InstanceStatus(Enum):
    """Instance lifecycle status"""
    CREATING = "creating"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"


@dataclass
class ManagedInstance:
    """Managed LightRAG instance with metadata"""
    instance: LightRAG
    config: LightRAGCoreConfig
    status: InstanceStatus = InstanceStatus.CREATING
    error: Optional[str] = None
    creation_time: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    last_used: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    init_task: Optional[asyncio.Task] = None
    cleanup_hooks: List[Callable] = field(default_factory=list)


class LightRAGInstanceManager:
    """
    Global LightRAG instance manager with async support and lifecycle management.

    This is a per-process singleton that manages multiple named LightRAG instances.
    In multi-process environments (like Gunicorn), each process will have its own manager.
    """

    def __init__(self):
        self._instances: Dict[str, ManagedInstance] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._initialized = False

    async def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a lock for the named instance"""
        async with self._global_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            return self._locks[name]

    async def get_instance(
        self,
        name: str = "default",
        config: Optional[LightRAGCoreConfig] = None,
        builder: Optional[Callable[[LightRAGCoreConfig], LightRAG]] = None,
        auto_init: bool = False,
        force_recreate: bool = False
    ) -> LightRAG:
        """
        Get or create a named LightRAG instance.

        Args:
            name: Instance name
            config: Configuration to use (if None, loads from environment)
            builder: Function to build LightRAG instance (if None, uses default builder)
            auto_init: Whether to automatically initialize the instance
            force_recreate: Whether to force recreation if instance exists

        Returns:
            LightRAG: The managed instance

        Raises:
            Exception: If instance creation or initialization fails
        """
        # Load config if not provided
        if config is None:
            config = load_core_config()
        config.instance_name = name

        # Get or create instance
        lock = await self.get_lock(name)

        async with lock:
            # Check if instance exists and is ready
            if name in self._instances and not force_recreate:
                managed_instance = self._instances[name]

                if managed_instance.status == InstanceStatus.READY:
                    managed_instance.last_used = asyncio.get_event_loop().time()
                    return managed_instance.instance

                elif managed_instance.status == InstanceStatus.ERROR:
                    if force_recreate:
                        await self._destroy_instance_unsafe(name)
                    else:
                        raise Exception(f"Instance '{name}' is in error state: {managed_instance.error}")

                elif managed_instance.status in [InstanceStatus.CREATING, InstanceStatus.INITIALIZING]:
                    # Wait for existing initialization
                    await self._wait_for_instance_ready(name)
                    managed_instance.last_used = asyncio.get_event_loop().time()
                    return managed_instance.instance

            # Create new instance
            return await self._create_instance(name, config, builder, auto_init)

    async def _create_instance(
        self,
        name: str,
        config: LightRAGCoreConfig,
        builder: Optional[Callable[[LightRAGCoreConfig], LightRAG]],
        auto_init: bool
    ) -> LightRAG:
        """Create a new LightRAG instance"""
        try:
            # Create managed instance wrapper
            managed_instance = ManagedInstance(
                instance=None,  # Will be set after creation
                config=config,
                status=InstanceStatus.CREATING
            )
            self._instances[name] = managed_instance

            # Build the instance
            if builder:
                instance = builder(config)
            else:
                instance = await self._default_builder(config)

            managed_instance.instance = instance
            managed_instance.status = InstanceStatus.INITIALIZING

            # Initialize if requested
            if auto_init:
                init_task = asyncio.create_task(self._initialize_instance(instance, name))
                managed_instance.init_task = init_task

                # Wait for initialization to complete
                await init_task

                if managed_instance.status == InstanceStatus.ERROR:
                    raise Exception(f"Instance initialization failed: {managed_instance.error}")
            else:
                managed_instance.status = InstanceStatus.READY

            logger.info(f"Successfully created LightRAG instance '{name}'")
            return instance

        except Exception as e:
            if name in self._instances:
                self._instances[name].status = InstanceStatus.ERROR
                self._instances[name].error = str(e)
            logger.error(f"Failed to create LightRAG instance '{name}': {e}")
            raise

    async def _default_builder(self, config: LightRAGCoreConfig) -> LightRAG:
        """Default builder for LightRAG instances"""
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.llm.qwen import qwen_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        import numpy as np
        
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete_if_cache(
                config.llm_model,
                prompt,
                system_prompt=system_prompt,
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                history_messages=history_messages,
                **kwargs)
            
        @wrap_embedding_func_with_attrs(embedding_dim=config.embedding_dimensions)
        async def embedding_func(texts: List[str]) -> np.ndarray:
            return await qwen_embed(
                texts=texts,
                model=config.embedding_model,
                base_url=config.embedding_base_url,
                api_key=config.embedding_api_key,
                dimensions=config.embedding_dimensions)

        # Build instance with basic configuration
        rag_config = {
            "working_dir": config.working_dir,
            "llm_model_func": llm_model_func,
            "embedding_func": embedding_func,
        }

        # Add optional configuration
        if config.workspace:
            rag_config["workspace"] = config.workspace

        if config.kv_storage:
            rag_config["kv_storage"] = config.kv_storage

        if config.vector_storage:
            rag_config["vector_storage"] = config.vector_storage

        if config.graph_storage:
            rag_config["graph_storage"] = config.graph_storage

        if config.doc_status_storage:
            rag_config["doc_status_storage"] = config.doc_status_storage

        # Add addon parameters
        addon_params = {
            "language": config.summary_language,
            "entity_types": config.entity_types,
        }

        # Add custom configuration
        addon_params.update(config.custom_config)
        rag_config["addon_params"] = addon_params

        return LightRAG(**rag_config)

    async def _initialize_instance(self, instance: LightRAG, name: str) -> None:
        """Initialize a LightRAG instance"""
        try:
            await instance.initialize_storages()
            self._instances[name].status = InstanceStatus.READY
            logger.info(f"LightRAG instance '{name}' initialized successfully")
        except Exception as e:
            self._instances[name].status = InstanceStatus.ERROR
            self._instances[name].error = str(e)
            logger.error(f"Failed to initialize LightRAG instance '{name}': {e}")
            raise

    async def _wait_for_instance_ready(self, name: str, timeout: float = 60.0) -> None:
        """Wait for an instance to become ready"""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if name not in self._instances:
                raise Exception(f"Instance '{name}' not found")

            managed_instance = self._instances[name]

            if managed_instance.status == InstanceStatus.READY:
                return

            if managed_instance.status == InstanceStatus.ERROR:
                raise Exception(f"Instance '{name}' failed to initialize: {managed_instance.error}")

            # Wait a bit before checking again
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Timeout waiting for instance '{name}' to become ready")

    async def _destroy_instance_unsafe(self, name: str) -> None:
        """Destroy an instance (caller must hold the instance lock)"""
        if name not in self._instances:
            return

        managed_instance = self._instances[name]
        managed_instance.status = InstanceStatus.DESTROYING

        try:
            # Cancel initialization task if running
            if managed_instance.init_task and not managed_instance.init_task.done():
                managed_instance.init_task.cancel()
                try:
                    await managed_instance.init_task
                except asyncio.CancelledError:
                    pass

            # Call cleanup hooks
            for hook in managed_instance.cleanup_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(managed_instance.instance)
                    else:
                        hook(managed_instance.instance)
                except Exception as e:
                    logger.warning(f"Cleanup hook failed for instance '{name}': {e}")

            # Finalize storages
            if hasattr(managed_instance.instance, 'finalize_storages'):
                try:
                    await managed_instance.instance.finalize_storages()
                except Exception as e:
                    logger.warning(f"Failed to finalize storages for instance '{name}': {e}")

            managed_instance.status = InstanceStatus.DESTROYED
            logger.info(f"LightRAG instance '{name}' destroyed")

        except Exception as e:
            logger.error(f"Error destroying instance '{name}': {e}")
            managed_instance.status = InstanceStatus.ERROR
            managed_instance.error = str(e)
        finally:
            del self._instances[name]

    async def set_instance(self, name: str, instance: LightRAG, config: Optional[LightRAGCoreConfig] = None) -> None:
        """
        Set an externally created LightRAG instance under management.

        Args:
            name: Instance name
            instance: LightRAG instance to manage
            config: Configuration for the instance (optional)
        """
        lock = await self.get_lock(name)

        async with lock:
            # Destroy existing instance if present
            if name in self._instances:
                await self._destroy_instance_unsafe(name)

            # Create managed instance
            if config is None:
                config = load_core_config()
            config.instance_name = name

            managed_instance = ManagedInstance(
                instance=instance,
                config=config,
                status=InstanceStatus.READY
            )

            self._instances[name] = managed_instance
            logger.info(f"Externally created LightRAG instance '{name}' registered")

    async def destroy_instance(self, name: str) -> None:
        """
        Destroy a managed LightRAG instance.

        Args:
            name: Instance name to destroy
        """
        lock = await self.get_lock(name)

        async with lock:
            await self._destroy_instance_unsafe(name)

    async def get_instance_names(self) -> List[str]:
        """Get list of all managed instance names"""
        async with self._global_lock:
            return list(self._instances.keys())

    async def get_instance_status(self, name: str) -> Optional[InstanceStatus]:
        """Get the status of a managed instance"""
        async with self._global_lock:
            if name in self._instances:
                return self._instances[name].status
            return None

    async def get_instance_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a managed instance"""
        async with self._global_lock:
            if name not in self._instances:
                return None

            managed_instance = self._instances[name]
            return {
                "name": name,
                "status": managed_instance.status.value,
                "config": managed_instance.config.to_dict(),
                "creation_time": managed_instance.creation_time,
                "last_used": managed_instance.last_used,
                "error": managed_instance.error
            }

    def add_cleanup_hook(self, name: str, hook: Callable) -> None:
        """Add a cleanup hook for an instance"""
        if name in self._instances:
            self._instances[name].cleanup_hooks.append(hook)


# Global instance manager
_global_manager: Optional[LightRAGInstanceManager] = None


def get_global_manager() -> LightRAGInstanceManager:
    """Get the global instance manager (creates if needed)"""
    global _global_manager
    if _global_manager is None:
        _global_manager = LightRAGInstanceManager()
    return _global_manager


# Convenience functions for global usage
async def get_lightrag_instance(
    name: str = "default",
    config: Optional[LightRAGCoreConfig] = None,
    builder: Optional[Callable[[LightRAGCoreConfig], LightRAG]] = None,
    auto_init: bool = False,
    force_recreate: bool = False
) -> LightRAG:
    """Get or create a named LightRAG instance using the global manager"""
    load_dotenv()
    # 默认使用WORKSPACE环境变量作为实例名称
    name = os.getenv("WORKSPACE", "") if name == "default" else name
    if not name:
        raise ValueError("WORKSPACE environment variable is not set")
    manager = get_global_manager()
    return await manager.get_instance(name, config, builder, auto_init, force_recreate)


async def set_instance(name: str, instance: LightRAG, config: Optional[LightRAGCoreConfig] = None) -> None:
    """Set an externally created LightRAG instance under management"""
    manager = get_global_manager()
    await manager.set_instance(name, instance, config)


async def destroy_instance(name: str) -> None:
    """Destroy a managed LightRAG instance"""
    manager = get_global_manager()
    await manager.destroy_instance(name)


async def get_instance_names() -> List[str]:
    """Get list of all managed instance names"""
    manager = get_global_manager()
    return await manager.get_instance_names()


async def initialize_lightrag_with_config(
    config: LightRAGCoreConfig,
    name: str = "default",
    builder: Optional[Callable[[LightRAGCoreConfig], LightRAG]] = None
) -> LightRAG:
    """
    Initialize a LightRAG instance with the given configuration.

    This is a convenience function that creates and initializes an instance in one call.
    """
    instance = await get_lightrag_instance(name, config, builder, auto_init=True)
    return instance


async def inject_prompts_from_config(instance: LightRAG, config: LightRAGCoreConfig) -> bool:
    """
    Inject prompts from configuration into a LightRAG instance.

    Args:
        instance: LightRAG instance to inject prompts into
        config: Configuration containing prompts path

    Returns:
        bool: True if prompts were injected, False otherwise
    """
    if not config.prompts_json_path:
        logger.info("No prompts JSON path configured, skipping prompt injection")
        return False

    try:
        # Load prompts from file
        prompts_config = load_prompts_from_file(config.prompts_json_path)

        # Merge with entity types from config
        merged_prompts = merge_prompts_with_defaults(prompts_config, config.entity_types)

        # Inject prompts into the instance
        from lightrag.prompt import PROMPTS
        PROMPTS.update(merged_prompts)

        logger.info(f"Successfully injected prompts from {config.prompts_json_path}")

        # Update entity types if specified in prompts
        if "entity_types" in merged_prompts:
            config.entity_types = merged_prompts["entity_types"]
            logger.info(f"Updated entity types from prompts: {config.entity_types}")
            # Sync entity types with instance's addon_params
            if hasattr(instance, 'addon_params'):
                instance.addon_params['entity_types'] = merged_prompts["entity_types"]
                logger.info("Synced entity types with instance addon_params")
            # Also sync language if available
            if "language" in merged_prompts and hasattr(instance, 'addon_params'):
                instance.addon_params['language'] = merged_prompts["language"]
                logger.info(f"Synced language with instance addon_params: {merged_prompts['language']}")

        return True

    except FileNotFoundError:
        logger.warning(f"Prompts file not found: {config.prompts_json_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to inject prompts from {config.prompts_json_path}: {e}")
        return False
