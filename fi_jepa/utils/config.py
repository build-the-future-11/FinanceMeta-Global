from __future__ import annotations
import yaml


def load_config(path: str):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config, path):

    with open(path, "w") as f:
        yaml.dump(config, f)