"""Config loading: .env, models.yaml, benchmark.yaml, brands/*.yaml."""
import os
from dataclasses import dataclass

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=None):
    """Load KEY=VALUE lines into os.environ without overwriting existing vars."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass
class ModelCfg:
    name: str
    provider: str  # dashscope | openai | openrouter | anthropic
    model: str
    enabled: bool = True
    price_in: float = 0.0   # USD per 1M input tokens
    price_out: float = 0.0  # USD per 1M output tokens
    refs: bool = False      # attach the configured reference images
    tools: tuple = ()       # e.g. ("zoom",): the model may request enlarged crops
    per_brand: bool = False  # one focused call per brand, detections unioned


def load_reference_sheet(path=None):
    """Path of the optional labeled reference sheet from the brands file."""
    if path is None:
        bench = load_bench()
        path = os.path.join(ROOT, bench["brands_file"])
    return yaml.safe_load(open(path)).get("reference_sheet")


def load_models(path=None):
    path = path or os.path.join(ROOT, "configs", "models.yaml")
    rows = yaml.safe_load(open(path))["models"]
    return [ModelCfg(**{**r, "tools": tuple(r.get("tools") or ())}) for r in rows]


def load_bench(path=None):
    path = path or os.path.join(ROOT, "configs", "benchmark.yaml")
    return yaml.safe_load(open(path))


def load_brands(path=None):
    if path is None:
        bench = load_bench()
        path = os.path.join(ROOT, bench["brands_file"])
    return yaml.safe_load(open(path))["brands"]
