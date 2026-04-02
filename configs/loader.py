from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DatasetConfig:
    name: str
    kind: str
    split: Dict[str, float]
    features: list[str]
    preprocessing: Dict[str, Any]
    sampling: Dict[str, Any]
    universe: Dict[str, Any] | None = None


@dataclass
class EncoderConfig:
    type: str
    input_dim: int
    embed_dim: int
    depth: int
    heads: int
    mlp_ratio: float
    dropout: float
    positional_encoding: str
    use_cls_token: bool = False
    use_projection_head: bool = False


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_config(path: str) -> DatasetConfig:
    cfg = load_yaml(path)
    return DatasetConfig(**cfg)


def load_encoder_config(path: str) -> EncoderConfig:
    cfg = load_yaml(path)
    return EncoderConfig(**cfg)