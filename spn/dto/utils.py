from types import MappingProxyType
from typing import Callable


def named_function(name: str):
    def decorator(function: Callable):
        setattr(function, "__name__", name)
        return function

    return decorator


def make_immutable(obj):
    if isinstance(obj, list):
        return tuple(make_immutable(v) for v in obj)
    if isinstance(obj, dict):
        return MappingProxyType({k: make_immutable(v) for k, v in obj.items()})
    return obj
