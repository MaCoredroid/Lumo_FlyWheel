from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DuplicateKeyError(ValueError):
    """Raised when a YAML mapping repeats the same key."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = getattr(key_node.start_mark, "line", None)
            if line is None:
                raise DuplicateKeyError(f"Duplicate YAML key {key!r}")
            raise DuplicateKeyError(f"Duplicate YAML key {key!r} at line {line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_file(path: str | Path) -> Any:
    try:
        return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
