"""
Registry for MLP methods and their configurations.

This module provides a global registry pattern for registering and retrieving
method implementations.

Usage:
    from hebbian.core.registry import Registry

    @Registry.register_method("gd")
    class GDMLPMethod(Method):
        ...

    # Later:
    method_cls = Registry.get_method("gd")
"""

from typing import Type, Dict, List


class Registry:
    """Global registry for methods and configurations."""

    _methods: Dict[str, Type] = {}
    _configs: Dict[str, Type] = {}

    @classmethod
    def register_method(cls, name: str):
        """
        Decorator to register a method class.

        Args:
            name: The name to register the method under (e.g., "gd", "ntk", "hebbian")

        Returns:
            Decorator function that registers the class

        Example:
            @Registry.register_method("gd")
            @Registry.register_method("gd_mlp")  # alias
            class GDMLPMethod(Method):
                ...
        """
        def decorator(method_class: Type) -> Type:
            cls._methods[name] = method_class
            return method_class
        return decorator

    @classmethod
    def get_method(cls, name: str) -> Type:
        """
        Retrieve a registered method class by name.

        Args:
            name: The registered name of the method

        Returns:
            The registered method class

        Raises:
            KeyError: If the method name is not registered
        """
        if name not in cls._methods:
            available = list(cls._methods.keys())
            raise KeyError(
                f"Method '{name}' not registered. Available methods: {available}"
            )
        return cls._methods[name]

    @classmethod
    def list_methods(cls) -> List[str]:
        """Return a list of all registered method names."""
        return list(cls._methods.keys())

    @classmethod
    def register_config(cls, name: str):
        """
        Decorator to register a config class.

        Args:
            name: The name to register the config under

        Returns:
            Decorator function that registers the class
        """
        def decorator(config_class: Type) -> Type:
            cls._configs[name] = config_class
            return config_class
        return decorator

    @classmethod
    def get_config(cls, name: str) -> Type:
        """
        Retrieve a registered config class by name.

        Args:
            name: The registered name of the config

        Returns:
            The registered config class

        Raises:
            KeyError: If the config name is not registered
        """
        if name not in cls._configs:
            available = list(cls._configs.keys())
            raise KeyError(
                f"Config '{name}' not registered. Available configs: {available}"
            )
        return cls._configs[name]

    @classmethod
    def list_configs(cls) -> List[str]:
        """Return a list of all registered config names."""
        return list(cls._configs.keys())

    @classmethod
    def clear(cls):
        """Clear all registrations. Useful for testing."""
        cls._methods.clear()
        cls._configs.clear()
