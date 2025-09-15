"""
LightRAG Core Configuration System

This module provides unified configuration loading and validation for LightRAG instances.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from lightrag.utils import get_env_value
from lightrag.constants import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
)

logger = logging.getLogger(__name__)


@dataclass
class LightRAGCoreConfig:
    """Core configuration dataclass for LightRAG"""

    # Instance management
    instance_name: str = "default"
    auto_init: bool = False

    # Entity extraction configuration
    entity_types: List[str] = field(default_factory=lambda: DEFAULT_ENTITY_TYPES.copy())

    # Prompt configuration
    prompts_json_path: Optional[str] = None

    # Language and workspace
    summary_language: str = DEFAULT_SUMMARY_LANGUAGE
    workspace: str = ""

    # Storage configuration (fallbacks from api/config)
    kv_storage: Optional[str] = None
    vector_storage: Optional[str] = None
    graph_storage: Optional[str] = None
    doc_status_storage: Optional[str] = None

    # Working directories
    working_dir: str = "./rag_storage"
    input_dir: str = "./inputs"

    # Additional configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                result[key] = value
        return result


def load_core_config(
    env_file: str = ".env",
    override: bool = False,
    custom_defaults: Optional[Dict[str, Any]] = None
) -> LightRAGCoreConfig:
    """
    Load core configuration from environment variables and .env file

    Args:
        env_file: Path to .env file
        override: Whether to override existing environment variables
        custom_defaults: Custom default values for configuration

    Returns:
        LightRAGCoreConfig: Loaded configuration object
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file, override=override)
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")

    # Start with default configuration
    config = LightRAGCoreConfig()

    # Apply custom defaults if provided
    if custom_defaults:
        for key, value in custom_defaults.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.custom_config[key] = value

    # Load instance configuration
    config.instance_name = get_env_value("LIGHTRAG_INSTANCE_NAME", config.instance_name)
    config.auto_init = get_env_value("LIGHTRAG_AUTO_INIT", config.auto_init, bool)

    # Load entity types
    entity_types_env = get_env_value("ENTITY_TYPES", None)
    if entity_types_env:
        try:
            if isinstance(entity_types_env, str):
                if entity_types_env.startswith('[') and entity_types_env.endswith(']'):
                    # Parse JSON array format
                    config.entity_types = json.loads(entity_types_env)
                else:
                    # Parse comma-separated format
                    config.entity_types = [et.strip() for et in entity_types_env.split(',')]
            elif isinstance(entity_types_env, list):
                config.entity_types = entity_types_env
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse ENTITY_TYPES: {e}, using defaults")

    # Load prompt configuration
    config.prompts_json_path = get_env_value("PROMPTS_JSON_PATH", config.prompts_json_path)

    # Load language and workspace
    config.summary_language = get_env_value("SUMMARY_LANGUAGE", config.summary_language)
    config.workspace = get_env_value("WORKSPACE", config.workspace)

    # Load storage configuration (optional, for CLI usage)
    config.kv_storage = get_env_value("LIGHTRAG_KV_STORAGE", config.kv_storage)
    config.vector_storage = get_env_value("LIGHTRAG_VECTOR_STORAGE", config.vector_storage)
    config.graph_storage = get_env_value("LIGHTRAG_GRAPH_STORAGE", config.graph_storage)
    config.doc_status_storage = get_env_value("LIGHTRAG_DOC_STATUS_STORAGE", config.doc_status_storage)

    # Load working directories
    config.working_dir = get_env_value("WORKING_DIR", config.working_dir)
    config.input_dir = get_env_value("INPUT_DIR", config.input_dir)

    # Convert relative paths to absolute paths
    config.working_dir = os.path.abspath(config.working_dir)
    config.input_dir = os.path.abspath(config.input_dir)

    # Load additional configuration from environment
    # These are prefixed with LIGHTRAG_ and will be stored in custom_config
    env_prefix = "LIGHTRAG_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix) and key not in [
            "LIGHTRAG_INSTANCE_NAME",
            "LIGHTRAG_AUTO_INIT",
            "LIGHTRAG_KV_STORAGE",
            "LIGHTRAG_VECTOR_STORAGE",
            "LIGHTRAG_GRAPH_STORAGE",
            "LIGHTRAG_DOC_STATUS_STORAGE"
        ]:
            config_key = key[len(env_prefix):].lower()

            # Try to parse as JSON first, fallback to string
            try:
                config.custom_config[config_key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                config.custom_config[config_key] = value

    logger.info(f"Loaded core configuration for instance '{config.instance_name}'")
    logger.debug(f"Core config: {config.to_dict()}")

    return config


def validate_config(config: LightRAGCoreConfig) -> List[str]:
    """
    Validate configuration and return list of validation errors

    Args:
        config: Configuration to validate

    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    errors = []

    # Validate instance name
    if not config.instance_name or not isinstance(config.instance_name, str):
        errors.append("Instance name must be a non-empty string")

    # Validate entity types
    if not isinstance(config.entity_types, list) or not config.entity_types:
        errors.append("Entity types must be a non-empty list")
    elif not all(isinstance(et, str) for et in config.entity_types):
        errors.append("All entity types must be strings")

    # Validate paths
    if config.prompts_json_path:
        prompts_path = Path(config.prompts_json_path)
        if not prompts_path.exists():
            errors.append(f"Prompts JSON file not found: {config.prompts_json_path}")

    # Validate working directories
    try:
        working_dir = Path(config.working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Invalid working directory '{config.working_dir}': {e}")

    try:
        input_dir = Path(config.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Invalid input directory '{config.input_dir}': {e}")

    return errors


def load_prompts_from_file(prompts_path: str) -> Dict[str, Any]:
    """
    Load prompts configuration from JSON file

    Args:
        prompts_path: Path to prompts JSON file

    Returns:
        Dict[str, Any]: Loaded prompts configuration

    Raises:
        FileNotFoundError: If prompts file doesn't exist
        json.JSONDecodeError: If prompts file is invalid JSON
    """
    prompts_file = Path(prompts_path)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts_config = json.load(f)

        logger.info(f"Loaded prompts configuration from {prompts_path}")
        return prompts_config

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in prompts file {prompts_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompts from {prompts_path}: {e}")
        raise


def merge_prompts_with_defaults(
    prompts_config: Dict[str, Any],
    default_entity_types: List[str] = None
) -> Dict[str, Any]:
    """
    Merge prompts configuration with defaults and handle entity types override

    Args:
        prompts_config: Loaded prompts configuration
        default_entity_types: Default entity types (if not in prompts_config)

    Returns:
        Dict[str, Any]: Merged prompts configuration
    """
    result = prompts_config.copy()

    # Handle entity types override
    if "entity_types" in prompts_config:
        if isinstance(prompts_config["entity_types"], list):
            logger.info("Using entity types from prompts configuration")
        else:
            logger.warning("Invalid entity_types format in prompts configuration, using defaults")
            result.pop("entity_types", None)
    elif default_entity_types:
        result["entity_types"] = default_entity_types

    return result


# Default configuration for common use cases
DEFAULT_CONFIGS = {
    "rag": LightRAGCoreConfig(
        instance_name="rag",
        entity_types=["person", "organization", "location", "event", "concept"],
        summary_language="English",
        working_dir="./rag_storage",
        input_dir="./rag_inputs"
    ),
    "paper": LightRAGCoreConfig(
        instance_name="paper",
        entity_types=["method", "algorithm", "dataset", "metric", "concept"],
        summary_language="English",
        working_dir="./paper_storage",
        input_dir="./paper_inputs"
    ),
    "general": LightRAGCoreConfig(
        instance_name="general",
        entity_types=["person", "organization", "location", "event", "concept"],
        summary_language="English",
        working_dir="./general_storage",
        input_dir="./general_inputs"
    )
}


def get_default_config(preset: str = "general") -> LightRAGCoreConfig:
    """
    Get a default configuration for common use cases

    Args:
        preset: Configuration preset name ("rag", "paper", "general")

    Returns:
        LightRAGCoreConfig: Default configuration for the preset
    """
    if preset not in DEFAULT_CONFIGS:
        logger.warning(f"Unknown preset '{preset}', using 'general' preset")
        preset = "general"

    return DEFAULT_CONFIGS[preset]