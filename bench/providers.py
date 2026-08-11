"""API adapters: one OpenAI-compatible adapter (OpenAI, DashScope intl, OpenRouter),
one Anthropic adapter, plus tolerant parsing of the detection JSON contract."""
import base64
import json
import os
import time
from dataclasses import dataclass

import requests

OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", ["OPENAI_API_KEY"]),
    "dashscope": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                  ["QWEN_API_KEY", "DASHSCOPE_API_KEY"]),
    "openrouter": ("https://openrouter.ai/api/v1", ["OPENROUTER_API_KEY"]),
}
SIZES = {"small", "medium", "large"}
PLACEMENTS = {"foreground", "background"}
LOCATIONS = {"chest", "sleeve", "shorts", "headwear", "board", "backdrop", "other"}


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def _key_from(env_names):
    for name in env_names:
        v = os.environ.get(name, "").strip().strip("'\"")
        if v:
            return v
    raise RuntimeError(f"none of {env_names} set; copy .env.example to .env")


class OpenAICompatible:
    def __init__(self, provider, model, bench):
        base, envs = OPENAI_COMPAT[provider]
        self.provider_key = provider
        self.url = base + "/chat/completions"
        self.key = _key_from(envs)
        self.model = model
        self.detail_high = provider == "openai"
        # OpenAI's current chat-completions models (gpt-5.6-*) reject `max_tokens`
        # in favor of `max_completion_tokens`, and reject a custom `temperature`
        # (only the default of 1 is accepted). dashscope/openrouter still expect
        # the old `max_tokens` name and accept temperature=0 for determinism.
        self.tokens_param = "max_completion_tokens" if provider == "openai" else "max_tokens"
        self.fixed_temperature = provider != "openai"
        self.timeout = bench["timeout_s"]
        self.max_tokens = bench["max_tokens"]

    def call(self, text, images):
        content = []
        for img in images:
            url = {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}
            if self.detail_high:
                url["detail"] = "high"
            content.append({"type": "image_url", "image_url": url})
        content.append({"type": "text", "text": text})
        body = {"model": self.model,
                "messages": [{"role": "user", "content": content}],
                self.tokens_param: self.max_tokens}
        if self.fixed_temperature:
            body["temperature"] = 0
        t0 = time.time()
        r = requests.post(self.url, json=body,
                          headers={"Authorization": "Bearer " + self.key},
                          timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage") or {}
        return CallResult(d["choices"][0]["message"].get("content") or "",
                          usage.get("prompt_tokens", 0),
                          usage.get("completion_tokens", 0),
                          time.time() - t0)


class Anthropic:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model, bench):
        self.provider_key = "anthropic"
        self.key = _key_from(["ANTHROPIC_API_KEY"])
        self.model = model
        self.timeout = bench["timeout_s"]
        self.max_tokens = bench["max_tokens"]

    def call(self, text, images):
        content = [{"type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": base64.b64encode(img).decode()}}
                   for img in images]
        content.append({"type": "text", "text": text})
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": content}]}
        t0 = time.time()
        r = requests.post(self.URL, json=body,
                          headers={"x-api-key": self.key,
                                   "anthropic-version": "2023-06-01"},
                          timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage") or {}
        text_out = "".join(b.get("text", "") for b in d.get("content", [])
                           if b.get("type") == "text")
        return CallResult(text_out, usage.get("input_tokens", 0),
                          usage.get("output_tokens", 0), time.time() - t0)


def make_provider(mcfg, bench):
    if mcfg.provider == "anthropic":
        return Anthropic(mcfg.model, bench)
    if mcfg.provider in OPENAI_COMPAT:
        return OpenAICompatible(mcfg.provider, mcfg.model, bench)
    raise ValueError(f"unknown provider {mcfg.provider!r}")


def parse_detections(text):
    """Extract and normalize the detections list; None when unparseable."""
    if not text:
        return None

    # Try parsing the whole text first (e.g., bare JSON with whitespace)
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and isinstance(obj.get("detections"), list):
            dets = obj["detections"]
        else:
            obj = None
    except json.JSONDecodeError:
        obj = None

    # If that failed, iterate over all "{" positions to find the JSON object
    if obj is None:
        decoder = json.JSONDecoder()
        for i, char in enumerate(text):
            if char == "{":
                try:
                    obj, end_idx = decoder.raw_decode(text[i:])
                    if isinstance(obj, dict) and isinstance(obj.get("detections"), list):
                        dets = obj["detections"]
                        break
                    obj = None
                except json.JSONDecodeError:
                    obj = None
        if obj is None:
            return None

    out = []
    for d in dets:
        if not isinstance(d, dict) or not isinstance(d.get("brand"), str):
            continue
        box = d.get("box")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            box = [min(1000, max(0, round(float(v)))) for v in box]
        except (TypeError, ValueError):
            continue
        # Guard against unhashable values by checking isinstance(v, str) first
        size_val = d.get("size")
        size = size_val if isinstance(size_val, str) and size_val in SIZES else "small"
        placement_val = d.get("placement")
        placement = placement_val if isinstance(placement_val, str) and placement_val in PLACEMENTS else "foreground"
        location_val = d.get("location")
        location = location_val if isinstance(location_val, str) and location_val in LOCATIONS else "other"
        try:
            conf = min(3, max(1, int(d.get("conf", 2))))
        except (TypeError, ValueError):
            conf = 2
        out.append({"brand": d["brand"].strip().casefold(), "box": box, "size": size,
                    "placement": placement, "location": location, "conf": conf})
    return out
