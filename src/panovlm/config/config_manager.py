"""Backward compatibility shim for the legacy config_manager module."""

from .manager import ExperimentConfig, UnifiedConfigManager as ConfigManager

__all__ = ["ExperimentConfig", "ConfigManager"]
