from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import ruamel.yaml


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str, cli_overrides: Dict[str, Any] | None = None) -> SimpleNamespace:
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.load(handle) or {}
    if cli_overrides:
        config = _merge_dicts(config, {k: v for k, v in cli_overrides.items() if v is not None})
    return _to_namespace(config)
