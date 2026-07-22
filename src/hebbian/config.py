"""Dataclass configuration helpers used by the experiment entrypoints.

The public repository keeps this deliberately small: configs are ordinary
dataclasses with strict attribute names, recursive finalization, and Python
expression overrides on the command line.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Callable, TypeVar, Union, get_args, get_origin, get_type_hints


class InvalidConfigurationError(ValueError):
    """Raised when an unknown configuration attribute is assigned."""


def _is_config(value: Any) -> bool:
    return bool(getattr(value, "_is_hebbian_config", False))


def _finalize(value: Any, active: set[int] | None = None) -> None:
    if active is None:
        active = set()

    if _is_config(value):
        if getattr(value, "_finalized", False):
            return
        value_id = id(value)
        if value_id in active:
            raise ValueError("Circular reference detected while finalizing a config")
        active.add(value_id)
        for config_field in dataclasses.fields(value):
            _finalize(getattr(value, config_field.name), active)
        custom_finalize = getattr(value, "custom_finalize", None)
        if custom_finalize is not None:
            custom_finalize()
        object.__setattr__(value, "_finalized", True)
        active.remove(value_id)
        return

    if isinstance(value, dict):
        for item in value.values():
            _finalize(item, active)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finalize(item, active)


def _to_serializable(value: Any, *, json_compatible: bool) -> Any:
    if _is_config(value):
        return value.to_dict(json_compatible=json_compatible)
    if isinstance(value, dict):
        return {
            key: _to_serializable(item, json_compatible=json_compatible)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _to_serializable(item, json_compatible=json_compatible) for item in value
        ]
    if json_compatible and not isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    return value


def _config_setattr(self: Any, name: str, value: Any) -> None:
    valid_attributes = self.__dict__.get("_valid_config_attributes")
    if valid_attributes is not None and name not in valid_attributes:
        available = ", ".join(sorted(field.name for field in dataclasses.fields(self)))
        raise InvalidConfigurationError(
            f"Unknown parameter {name!r} for {type(self).__name__}. "
            f"Available parameters: {available}"
        )
    object.__setattr__(self, name, value)


def _config_finalize(self: Any) -> None:
    _finalize(self)


def _config_to_dict(
    self: Any,
    *,
    json_compatible: bool = False,
    yaml_compatible: bool | None = None,
) -> dict[str, Any]:
    if yaml_compatible is not None:
        json_compatible = yaml_compatible
    return {
        field.name: _to_serializable(
            getattr(self, field.name), json_compatible=json_compatible
        )
        for field in dataclasses.fields(self)
    }


def _config_save_json(self: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(self.to_dict(json_compatible=True), indent=2) + "\n",
        encoding="utf-8",
    )


def _config_replace(self: Any, field_name: str, class_name: str, **kwargs: Any) -> None:
    hints = get_type_hints(type(self))
    if field_name not in hints:
        raise InvalidConfigurationError(
            f"Unknown parameter {field_name!r} for {type(self).__name__}"
        )
    annotation = hints[field_name]
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, types.UnionType):
        candidates = [candidate for candidate in get_args(annotation) if candidate is not type(None)]
    else:
        candidates = [annotation]
    target = next(
        (candidate for candidate in candidates if getattr(candidate, "__name__", None) == class_name),
        None,
    )
    if target is None:
        allowed = ", ".join(getattr(candidate, "__name__", str(candidate)) for candidate in candidates)
        raise ValueError(f"{class_name!r} is not valid for {field_name!r}; choose from {allowed}")
    setattr(self, field_name, target(**kwargs))


T = TypeVar("T")
R = TypeVar("R")


def pydraclass(cls: type[T]) -> type[T]:
    """Turn ``cls`` into a strict, recursively finalized dataclass."""

    if "finalize" in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} must define custom_finalize(), not finalize(); "
            "finalize() is supplied by @pydraclass"
        )

    user_post_init = cls.__dict__.get("__post_init__")

    def combined_post_init(self: Any) -> None:
        valid = {field.name for field in dataclasses.fields(self)}
        for base in type(self).__mro__:
            valid.update(name for name in vars(base) if not name.startswith("__"))
        valid.update(
            {
                "_finalized",
                "_is_hebbian_config",
                "_valid_config_attributes",
            }
        )
        object.__setattr__(self, "_is_hebbian_config", True)
        object.__setattr__(self, "_valid_config_attributes", frozenset(valid))
        object.__setattr__(self, "_finalized", False)
        if user_post_init is not None:
            user_post_init(self)

    cls.__post_init__ = combined_post_init
    cls.__setattr__ = _config_setattr
    cls.finalize = _config_finalize
    cls.to_dict = _config_to_dict
    cls.save_json = _config_save_json
    cls.replace = _config_replace
    return dataclasses.dataclass(cls, repr=True)


def apply_overrides(config: Any, args: list[str], *, finalize: bool = True) -> bool:
    """Apply ``field=value`` Python expressions to a config instance."""

    show = False
    statements: list[str] = []
    for argument in args:
        if argument == "--show":
            show = True
        else:
            statements.append(f"config.{argument}")

    if statements:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - NumPy is a project dependency
            np = None
        try:
            import torch
        except ImportError:  # pragma: no cover - Torch is a project dependency
            torch = None
        namespace = {
            "config": config,
            "math": math,
            "np": np,
            "numpy": np,
            "torch": torch,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "len": len,
            "range": range,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
            "pow": pow,
        }
        namespace.update(getattr(config, "__dict__", {}))
        source = "\n".join(statements)
        try:
            exec(source, namespace)
        except Exception as exc:
            raise type(exc)(f"{exc}\n\nConfiguration overrides:\n{source}").with_traceback(
                exc.__traceback__
            )

    if finalize and _is_config(config):
        config.finalize()
    return show


def main(config_type: type[T]) -> Callable[[Callable[[T], R]], Callable[[list[str] | None], R | None]]:
    """Decorate a ``main(config)`` function with CLI override handling."""

    def decorator(function: Callable[[T], R]) -> Callable[[list[str] | None], R | None]:
        @functools.wraps(function)
        def wrapped(args: list[str] | None = None) -> R | None:
            config = config_type()
            show = apply_overrides(config, sys.argv[1:] if args is None else args)
            if show:
                print(json.dumps(config.to_dict(json_compatible=True), indent=2))
                return None
            return function(config)

        return wrapped

    return decorator


__all__ = [
    "InvalidConfigurationError",
    "apply_overrides",
    "main",
    "pydraclass",
]
