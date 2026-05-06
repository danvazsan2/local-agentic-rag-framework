"""
Configuration loader for YAML files.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from rag_framework.config.models import RAGConfig


class ConfigLoader:
    """
    Loads configuration from YAML files.
    """

    DEFAULT_CONFIG_PATHS = [
        "config/rag_config.yaml",
        "config/rag_config.yml",
        "rag_config.yaml",
        "config.yaml",
    ]

    @classmethod
    def load(
        cls,
        config_path: Optional[str] = None,
    ) -> RAGConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file. If None, searches default paths.

        Returns:
            RAGConfig object with loaded configuration.
        """
        config_data = {}

        # Try to load from YAML file
        yaml_path = cls._find_config_file(config_path)
        if yaml_path:
            config_data = cls._load_yaml(yaml_path)
            print(f"📄 Loaded configuration from: {yaml_path}")

        # Create config object from merged data
        return RAGConfig.from_dict(config_data)

    @classmethod
    def load_from_yaml(cls, config_path: str) -> RAGConfig:
        """Load configuration exclusively from YAML file."""
        return cls.load(config_path=config_path)

    @classmethod
    def save_yaml(cls, config: RAGConfig, path: str) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml"
            )

        config_dict = config.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        print(f"Configuration saved to: {path}")

    @classmethod
    def _find_config_file(cls, config_path: Optional[str]) -> Optional[Path]:
        """Find configuration file."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Search default paths
        for default_path in cls.DEFAULT_CONFIG_PATHS:
            path = Path(default_path)
            if path.exists():
                return path

        return None

    @classmethod
    def _load_yaml(cls, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _merge_configs(
        base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader._merge_configs(result[key], value)
            else:
                result[key] = value

        return result
