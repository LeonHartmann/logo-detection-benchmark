# Logo Detection Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone repo that benchmarks 12 vision LLMs (4 API providers) on logo detection, bounding-box quality, and resolution robustness, with a browser labeling UI, resumable runner, scoring, and an HTML leaderboard.

**Architecture:** Small Python package `bench/` (config, resize, prompt, providers, run, score, report) driven by `python -m bench <cmd>`; a stdlib HTTP server serves two single-file browser UIs (labeling and review); YAML configs define models and brands; all images and raw results are gitignored, manifests/labels/scores are committed.

**Tech Stack:** Python 3.10+, runtime deps exactly `requests`, `Pillow`, `PyYAML`; dev dep `pytest`; ffmpeg + yt-dlp only for the Delay dataset build script.

**Spec:** `docs/superpowers/specs/2026-08-11-logo-detection-benchmark-design.md` (approved). The spec is the authority on scoring definitions and repo layout.

## Global Constraints

- Repo root: `/Users/leon/Coding/Research/logo-detection-benchmark`. All paths below are relative to it.
- Runtime dependencies are exactly: `requests`, `Pillow`, `PyYAML`. Nothing else. `pytest` goes in `requirements-dev.txt`.
- Boxes are ALWAYS `[x0, y0, x1, y1]` integers normalized 0-1000 over the full image, in labels, model outputs, and gallery metadata.
- Resolution rungs default: `[1080, 720, 480, 240, 144]` (image height). No upscaling: rungs above native height are skipped.
- Temperature 0 on every API call where the API accepts it; JPEG quality 85 for all derived images.
- Tests never hit the network. Live API calls happen only in Task 12 (smoke) and real runs.
- No em-dashes in user-facing copy (README, HTML UIs, report text). Use commas, colons, or parentheses.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Env var names: `QWEN_API_KEY` (fallback `DASHSCOPE_API_KEY`), `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`.

---

### Task 1: Scaffolding and config loading

**Files:**
- Create: `.gitignore`, `.env.example`, `requirements.txt`, `requirements-dev.txt`
- Create: `configs/models.yaml`, `configs/benchmark.yaml`, `brands/delay.yaml`
- Create: `bench/__init__.py` (empty), `bench/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `bench.config.ROOT` (str, repo root), `load_env()`, `ModelCfg` dataclass (`name, provider, model, enabled, price_in, price_out`), `load_models(path=None) -> list[ModelCfg]`, `load_bench(path=None) -> dict`, `load_brands(path=None) -> list[dict]`.

- [ ] **Step 1: Write scaffolding files**

`.gitignore`:
```
.env
__pycache__/
*.pyc
.pytest_cache/
data/images/
data/rungs/
data/refs/
results/raw/
results/gallery/
```

`.env.example`:
```
# Copy to .env and fill in. Only the providers you enable in configs/models.yaml are needed.
QWEN_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=
ANTHROPIC_API_KEY=
```

`requirements.txt`:
```
requests
Pillow
PyYAML
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest
```

`configs/benchmark.yaml`:
```yaml
rungs: [1080, 720, 480, 240, 144]
jpeg_quality: 85
max_tokens: 2000
timeout_s: 180
max_retries: 4
concurrency:
  dashscope: 8
  openai: 8
  openrouter: 8
  anthropic: 8
brands_file: brands/delay.yaml
```

`configs/models.yaml` (prices are USD per 1M tokens; values marked 0 get filled from live sources in Task 11):
```yaml
models:
  - {name: qwen3.8-max,     provider: dashscope,  model: qwen3.8-max,                        enabled: true, price_in: 0,    price_out: 0}
  - {name: qwen3-vl-plus,   provider: dashscope,  model: qwen3-vl-plus,                      enabled: true, price_in: 0,    price_out: 0}
  - {name: gpt-5.6-sol,     provider: openai,     model: gpt-5.6-sol,                        enabled: true, price_in: 5.0,  price_out: 30.0}
  - {name: gpt-5.6-terra,   provider: openai,     model: gpt-5.6-terra,                      enabled: true, price_in: 1.0,  price_out: 6.0}
  - {name: gpt-5.6-luna,    provider: openai,     model: gpt-5.6-luna,                       enabled: true, price_in: 0.1,  price_out: 0.6}
  - {name: claude-opus-5,   provider: anthropic,  model: claude-opus-5,                      enabled: true, price_in: 0,    price_out: 0}
  - {name: claude-sonnet-5, provider: anthropic,  model: claude-sonnet-5,                    enabled: true, price_in: 0,    price_out: 0}
  - {name: gemini-3.6-flash,provider: openrouter, model: google/gemini-3.6-flash,            enabled: true, price_in: 1.5,  price_out: 7.5}
  - {name: gemini-3.1-pro,  provider: openrouter, model: google/gemini-3.1-pro-preview,      enabled: true, price_in: 2.0,  price_out: 12.0}
  - {name: grok-4.5,        provider: openrouter, model: x-ai/grok-4.5,                      enabled: true, price_in: 0,    price_out: 0}
  - {name: kimi-k3,         provider: openrouter, model: moonshotai/kimi-k3,                 enabled: true, price_in: 3.0,  price_out: 15.0}
  - {name: qwen3-vl-235b,   provider: openrouter, model: qwen/qwen3-vl-235b-a22b-instruct,   enabled: true, price_in: 0.21, price_out: 1.9}
```

`brands/delay.yaml` (descriptions may be refined from `~/Coding/Rabona/delay-social-review/scripts/qwen_pipeline.py` prompt text, but these defaults stand on their own):
```yaml
brands:
  - name: adidas
    description: adidas wordmark, trefoil, or performance (three-bar triangle) logo on kit, caps, or boards
    refs: []
  - name: stripes
    description: the three parallel adidas stripes as a kit design element on sleeves, shorts, socks, or pant legs
    refs: []
  - name: dkh
    description: DKH wordmark logo (main shirt sponsor), usually centered on the chest
    refs: []
  - name: 11teamsports
    description: 11teamsports wordmark logo (the "11" followed by "teamsports"), on sleeves or boards
    refs: []
  - name: delay
    description: Delay Sports club crest, on chest, backdrops, scoreboard graphics, or corner logos
    refs: []
```

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
import os
import bench.config as cfg


def test_load_models_parses_rows():
    models = cfg.load_models()
    assert len(models) == 12
    m = {x.name: x for x in models}
    assert m["gpt-5.6-terra"].provider == "openai"
    assert m["gemini-3.1-pro"].model == "google/gemini-3.1-pro-preview"
    assert m["gpt-5.6-sol"].price_out == 30.0
    assert all(isinstance(x.enabled, bool) for x in models)


def test_load_bench_defaults():
    b = cfg.load_bench()
    assert b["rungs"] == [1080, 720, 480, 240, 144]
    assert b["jpeg_quality"] == 85
    assert b["concurrency"]["anthropic"] == 8


def test_load_brands():
    brands = cfg.load_brands()
    names = [b["name"] for b in brands]
    assert names == ["adidas", "stripes", "dkh", "11teamsports", "delay"]
    assert all("description" in b for b in brands)


def test_load_env_reads_and_does_not_overwrite(tmp_path):
    p = tmp_path / ".env"
    p.write_text('FOO_TEST_KEY="abc" \nBAR_TEST_KEY=def\n# comment\n')
    os.environ["BAR_TEST_KEY"] = "already"
    cfg.load_env(str(p))
    assert os.environ["FOO_TEST_KEY"] == "abc"
    assert os.environ["BAR_TEST_KEY"] == "already"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/leon/Coding/Research/logo-detection-benchmark && python -m pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: bench.config)

- [ ] **Step 4: Implement `bench/config.py`**

```python
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


def load_models(path=None):
    path = path or os.path.join(ROOT, "configs", "models.yaml")
    rows = yaml.safe_load(open(path))["models"]
    return [ModelCfg(**r) for r in rows]


def load_bench(path=None):
    path = path or os.path.join(ROOT, "configs", "benchmark.yaml")
    return yaml.safe_load(open(path))


def load_brands(path=None):
    if path is None:
        bench = load_bench()
        path = os.path.join(ROOT, bench["brands_file"])
    return yaml.safe_load(open(path))["brands"]
```

Also create empty `bench/__init__.py` and `tests/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v` (after `pip install -r requirements-dev.txt`)
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffolding, configs, and config loader"
```

---

### Task 2: Resolution ladder (`bench/resize.py`)

**Files:**
- Create: `bench/resize.py`
- Test: `tests/test_resize.py`

**Interfaces:**
- Consumes: `bench.config.ROOT`.
- Produces: `rungs_for(native_h: int, rungs: list[int]) -> list[int]`; `derive(src_path: str, image_id: str, rungs_dir: str, rungs: list[int], quality: int = 85) -> list[tuple[int, str]]` returning `(rung, out_path)` pairs; skips work if output exists.

- [ ] **Step 1: Write the failing test**

`tests/test_resize.py`:
```python
import os
from PIL import Image
from bench.resize import rungs_for, derive

RUNGS = [1080, 720, 480, 240, 144]


def test_rungs_for_skips_upscales():
    assert rungs_for(1706, RUNGS) == [1080, 720, 480, 240, 144]
    assert rungs_for(1080, RUNGS) == [1080, 720, 480, 240, 144]
    assert rungs_for(500, RUNGS) == [480, 240, 144]


def test_rungs_for_tiny_image_uses_native():
    assert rungs_for(100, RUNGS) == [100]


def test_derive_writes_correct_sizes(tmp_path):
    src = tmp_path / "frame.jpg"
    Image.new("RGB", (1920, 1080), "red").save(src, "JPEG")
    outs = derive(str(src), "frame.jpg", str(tmp_path / "rungs"), RUNGS)
    assert [r for r, _ in outs] == RUNGS
    for rung, path in outs:
        im = Image.open(path)
        assert im.height == rung
        assert im.width == round(1920 * rung / 1080)


def test_derive_is_idempotent(tmp_path):
    src = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 480), "blue").save(src, "JPEG")
    outs1 = derive(str(src), "frame.jpg", str(tmp_path / "rungs"), RUNGS)
    mtimes = [os.path.getmtime(p) for _, p in outs1]
    outs2 = derive(str(src), "frame.jpg", str(tmp_path / "rungs"), RUNGS)
    assert outs1 == outs2
    assert [os.path.getmtime(p) for _, p in outs2] == mtimes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_resize.py -v`
Expected: FAIL (no module bench.resize)

- [ ] **Step 3: Implement `bench/resize.py`**

```python
"""Derive fixed-height downscaled copies of each source image (the resolution ladder)."""
import os

from PIL import Image


def rungs_for(native_h, rungs):
    """Rungs applicable to an image: no upscaling; tiny images get their native height."""
    fit = [r for r in rungs if r <= native_h]
    return fit or [native_h]


def derive(src_path, image_id, rungs_dir, rungs, quality=85):
    im = None
    outs = []
    for rung in rungs_for(_native_height(src_path), rungs):
        out = os.path.join(rungs_dir, str(rung), image_id)
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if im is None:
                im = Image.open(src_path).convert("RGB")
            w = round(im.width * rung / im.height)
            im.resize((w, rung), Image.LANCZOS).save(out, "JPEG", quality=quality)
        outs.append((rung, out))
    return outs


def _native_height(src_path):
    with Image.open(src_path) as im:
        return im.height
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_resize.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: resolution ladder derivation"
```

---

### Task 3: Canonical prompt builder (`bench/prompt.py`)

**Files:**
- Create: `bench/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: brands list from `bench.config.load_brands()`.
- Produces: `build_prompt(brands: list[dict], n_refs: int = 0) -> str`; `load_refs(brands: list[dict], root: str) -> list[tuple[bytes, str]]` returning `(jpeg_bytes, brand_name)` pairs for configured reference images; module constant `RETRY_SUFFIX` (str) appended on the one re-ask after a parse failure.

- [ ] **Step 1: Write the failing test**

`tests/test_prompt.py`:
```python
from PIL import Image
from bench.prompt import build_prompt, load_refs, RETRY_SUFFIX

BRANDS = [
    {"name": "adidas", "description": "adidas logo", "refs": []},
    {"name": "delay", "description": "Delay Sports crest", "refs": []},
]


def test_prompt_contains_brands_and_contract():
    p = build_prompt(BRANDS)
    assert '"adidas"' in p and '"delay"' in p
    assert "0-1000" in p
    assert '"detections"' in p
    assert "small|medium|large" in p
    assert "foreground|background" in p
    assert "conf" in p


def test_prompt_mentions_refs_only_when_present():
    assert "REFERENCE" not in build_prompt(BRANDS, n_refs=0)
    p = build_prompt(BRANDS, n_refs=3)
    assert "first 3 images" in p.lower()
    assert "TARGET" in p


def test_load_refs_reads_configured_images(tmp_path):
    (tmp_path / "data" / "refs").mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(tmp_path / "data" / "refs" / "a.jpg")
    brands = [{"name": "adidas", "description": "x", "refs": ["data/refs/a.jpg"]},
              {"name": "delay", "description": "y", "refs": []}]
    refs = load_refs(brands, str(tmp_path))
    assert len(refs) == 1
    assert refs[0][1] == "adidas"
    assert refs[0][0][:2] == b"\xff\xd8"  # JPEG magic


def test_retry_suffix_demands_bare_json():
    assert "JSON" in RETRY_SUFFIX
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: FAIL (no module bench.prompt)

- [ ] **Step 3: Implement `bench/prompt.py`**

```python
"""The one canonical prompt every model receives. No per-model tuning (spec non-goal)."""
import os

SCHEMA = ('{"detections":[{"brand":"<name>","box":[x0,y0,x1,y1],'
          '"size":"small|medium|large","placement":"foreground|background",'
          '"location":"chest|sleeve|shorts|headwear|board|backdrop|other","conf":1|2|3}]}')

RETRY_SUFFIX = ("\n\nIMPORTANT: your previous answer was not parseable. "
                "Respond ONLY with the single-line JSON object, no prose, no code fences.")


def build_prompt(brands, n_refs=0):
    brand_lines = "\n".join(f'- "{b["name"]}": {b["description"]}' for b in brands)
    ref_part = ""
    if n_refs:
        ref_part = (f"The first {n_refs} images are REFERENCE crops showing what the brand "
                    "marks look like; they are NOT the image to analyze. The LAST image is "
                    "the TARGET image.\n\n")
    return (f"{ref_part}You are a sponsor-logo auditor. Find EVERY visible logo instance "
            f"of these brands in the TARGET image:\n{brand_lines}\n\n"
            "Rules:\n"
            "- Report each visible instance separately, including tiny, blurry, or partly "
            "occluded ones.\n"
            "- box: integers 0-1000 over the FULL target image, [x0,y0,x1,y1], tight around "
            "the mark.\n"
            "- size: small (you must squint to see it), medium, large (a dominant element "
            "of the frame).\n"
            "- placement: foreground (on a person or object that is the subject) or "
            "background (backdrops, boards, banners, out-of-focus areas).\n"
            "- location: chest|sleeve|shorts|headwear|board|backdrop|other.\n"
            "- conf: 1 unsure, 2 confident, 3 certain.\n"
            "- Only the listed brands. If none are visible, \"detections\" is [].\n\n"
            f"Respond ONLY with compact single-line JSON: {SCHEMA}")


def load_refs(brands, root):
    """(jpeg_bytes, brand_name) for every configured reference image, brand order."""
    refs = []
    for b in brands:
        for rel in b.get("refs") or []:
            with open(os.path.join(root, rel), "rb") as f:
                refs.append((f.read(), b["name"]))
    return refs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompt.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: canonical prompt builder with optional brand references"
```

---

### Task 4: Provider adapters and response parsing (`bench/providers.py`)

**Files:**
- Create: `bench/providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `ModelCfg` from Task 1.
- Produces:
  - `CallResult` dataclass: `text: str, input_tokens: int, output_tokens: int, latency_s: float`
  - `make_provider(mcfg: ModelCfg, bench: dict) -> provider` where provider has `.call(text: str, images: list[bytes]) -> CallResult` (images are JPEG bytes, refs first, target LAST) and `.provider_key: str` (concurrency group name)
  - `parse_detections(text: str) -> list[dict] | None` (None = unparseable; valid dets are normalized: box clamped ints 0-1000, size/placement/location defaulted to "small"/"foreground"/"other" when missing or out of vocabulary, conf int 1-3 default 2)
- Errors: adapters raise `requests.HTTPError` on non-2xx (retry lives in Task 5); missing API key raises `RuntimeError` at construction.

- [ ] **Step 1: Write the failing test**

`tests/test_providers.py`:
```python
import json
import pytest
import bench.providers as pv
from bench.config import ModelCfg

BENCH = {"timeout_s": 5, "max_tokens": 500}


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
    def json(self):
        return self.payload
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)


def test_openai_compat_payload_and_result(monkeypatch):
    captured = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, body=json, headers=headers)
        return FakeResp({"choices": [{"message": {"content": '{"detections":[]}'}}],
                         "usage": {"prompt_tokens": 1500, "completion_tokens": 20}})
    monkeypatch.setattr(pv.requests, "post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = pv.make_provider(ModelCfg("gpt-5.6-terra", "openai", "gpt-5.6-terra"), BENCH)
    r = p.call("find logos", [b"\xff\xd8fakejpeg"])
    assert "api.openai.com" in captured["url"]
    assert captured["body"]["temperature"] == 0
    content = captured["body"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["detail"] == "high"   # openai only
    assert content[-1]["type"] == "text"
    assert r.input_tokens == 1500 and r.text == '{"detections":[]}'


def test_dashscope_and_openrouter_reuse_openai_adapter(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    d = pv.make_provider(ModelCfg("qwen3.8-max", "dashscope", "qwen3.8-max"), BENCH)
    o = pv.make_provider(ModelCfg("kimi-k3", "openrouter", "moonshotai/kimi-k3"), BENCH)
    assert "dashscope-intl" in d.url and "openrouter.ai" in o.url
    assert d.provider_key == "dashscope" and o.provider_key == "openrouter"


def test_anthropic_payload(monkeypatch):
    captured = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, body=json, headers=headers)
        return FakeResp({"content": [{"type": "text", "text": '{"detections":[]}'}],
                         "usage": {"input_tokens": 900, "output_tokens": 15}})
    monkeypatch.setattr(pv.requests, "post", fake_post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = pv.make_provider(ModelCfg("claude-opus-5", "anthropic", "claude-opus-5"), BENCH)
    r = p.call("find logos", [b"\xff\xd8fakejpeg"])
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    blocks = captured["body"]["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["type"] == "base64"
    assert r.output_tokens == 15


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        pv.make_provider(ModelCfg("gpt-5.6-terra", "openai", "gpt-5.6-terra"), BENCH)


def test_parse_detections_clean_fenced_and_garbage():
    clean = '{"detections":[{"brand":"adidas","box":[1,2,3,4],"size":"small","placement":"foreground","location":"chest","conf":3}]}'
    assert len(pv.parse_detections(clean)) == 1
    fenced = "Here you go:\n```json\n" + clean + "\n```"
    assert len(pv.parse_detections(fenced)) == 1
    assert pv.parse_detections("I see two logos, one adidas") is None
    assert pv.parse_detections('{"detections": "not a list"}') is None


def test_parse_detections_normalizes():
    raw = json.dumps({"detections": [
        {"brand": "delay", "box": [-5, 200.7, 1400, 900]},
        {"brand": "adidas", "box": [1, 2, 3]},          # bad box: dropped
        {"box": [1, 2, 3, 4]},                          # no brand: dropped
    ]})
    dets = pv.parse_detections(raw)
    assert len(dets) == 1
    d = dets[0]
    assert d["box"] == [0, 201, 1000, 900]
    assert d["size"] == "small" and d["placement"] == "foreground"
    assert d["location"] == "other" and d["conf"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL (no module bench.providers)

- [ ] **Step 3: Implement `bench/providers.py`**

```python
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
                "temperature": 0, "max_tokens": self.max_tokens}
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
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    dets = obj.get("detections")
    if not isinstance(dets, list):
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
        size = d.get("size") if d.get("size") in SIZES else "small"
        placement = d.get("placement") if d.get("placement") in PLACEMENTS else "foreground"
        location = d.get("location") if d.get("location") in LOCATIONS else "other"
        try:
            conf = min(3, max(1, int(d.get("conf", 2))))
        except (TypeError, ValueError):
            conf = 2
        out.append({"brand": d["brand"], "box": box, "size": size,
                    "placement": placement, "location": location, "conf": conf})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: provider adapters (OpenAI-compatible + Anthropic) and detection parsing"
```

---

### Task 5: Resumable runner (`bench/run.py`)

**Files:**
- Create: `bench/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `make_provider`, `parse_detections`, `CallResult` (Task 4); `build_prompt`, `load_refs`, `RETRY_SUFFIX` (Task 3); `rungs_for` (Task 2); configs (Task 1).
- Produces: `run_benchmark(models, bench, brands, manifest, root, only_models=None, only_rungs=None, limit_images=None, provider_factory=make_provider) -> dict` (counts: done, skipped, failed). Writes one JSONL row per (image, rung) to `results/raw/<model.name>.jsonl`:
  `{"image", "rung", "model", "detections": list|None, "parse_ok": bool, "retried": bool, "latency_s", "input_tokens", "output_tokens", "error": str|None, "ts"}`
- Manifest format consumed (produced by Task 9): `{"images": [{"id": "<filename>.jpg", "native": [w, h], "stratum": "...", "source": {...}}]}`; rung image files at `data/rungs/<rung>/<id>`.
- `provider_factory` is injectable so tests never construct real adapters.

- [ ] **Step 1: Write the failing test**

`tests/test_run.py`:
```python
import json
import os
from PIL import Image
import bench.run as rn
from bench.config import ModelCfg
from bench.providers import CallResult

BENCH = {"rungs": [480, 240], "jpeg_quality": 85, "max_tokens": 100, "timeout_s": 5,
         "max_retries": 2, "concurrency": {"fake": 2}}
BRANDS = [{"name": "adidas", "description": "x", "refs": []}]
GOOD = '{"detections":[{"brand":"adidas","box":[10,10,20,20],"size":"small","placement":"foreground","location":"chest","conf":3}]}'


class FakeProvider:
    provider_key = "fake"
    def __init__(self, script):
        self.script, self.calls = list(script), 0
    def call(self, text, images):
        self.calls += 1
        reply = self.script.pop(0) if self.script else GOOD
        if isinstance(reply, Exception):
            raise reply
        return CallResult(reply, 100, 10, 0.01)


def setup_repo(tmp_path, n_images=2):
    (tmp_path / "data" / "images").mkdir(parents=True)
    images = []
    for i in range(n_images):
        name = f"img{i}.jpg"
        Image.new("RGB", (854, 480), "green").save(tmp_path / "data" / "images" / name)
        images.append({"id": name, "native": [854, 480], "stratum": "normal", "source": {}})
    return {"images": images}


def test_run_writes_rows_and_resumes(tmp_path):
    manifest = setup_repo(tmp_path)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider([])
    stats = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                             provider_factory=lambda m, b: fp)
    raw = tmp_path / "results" / "raw" / "fake-model.jsonl"
    rows = [json.loads(l) for l in open(raw)]
    assert len(rows) == 4  # 2 images x 2 rungs
    assert stats["done"] == 4
    assert all(r["parse_ok"] and r["detections"] for r in rows)
    assert {(r["image"], r["rung"]) for r in rows} == {
        ("img0.jpg", 480), ("img0.jpg", 240), ("img1.jpg", 480), ("img1.jpg", 240)}
    # resume: nothing new is called
    before = fp.calls
    stats2 = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                              provider_factory=lambda m, b: fp)
    assert fp.calls == before and stats2["skipped"] == 4


def test_parse_failure_triggers_single_retry(tmp_path):
    # concurrency 1 so the scripted replies map to work items deterministically
    bench_seq = {**BENCH, "concurrency": {"fake": 1}}
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider(["not json at all", GOOD, "garbage", "still garbage"])
    rn.run_benchmark(models, bench_seq, BRANDS, manifest, str(tmp_path),
                     provider_factory=lambda m, b: fp)
    rows = [json.loads(l) for l in
            open(tmp_path / "results" / "raw" / "fake-model.jsonl")]
    rows.sort(key=lambda r: r["rung"], reverse=True)
    assert rows[0]["retried"] is True and rows[0]["parse_ok"] is True   # 480: fail then good
    assert rows[1]["retried"] is True and rows[1]["parse_ok"] is False  # 240: fail twice
    assert rows[1]["detections"] is None


def test_api_error_is_recorded_after_retries(tmp_path):
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("fake-model", "fake", "fake-1")]
    fp = FakeProvider([RuntimeError("boom"), RuntimeError("boom"),
                       RuntimeError("boom"), RuntimeError("boom")])
    stats = rn.run_benchmark(models, BENCH, BRANDS, manifest, str(tmp_path),
                             only_rungs=[480], provider_factory=lambda m, b: fp,
                             backoff_base=0)
    rows = [json.loads(l) for l in
            open(tmp_path / "results" / "raw" / "fake-model.jsonl")]
    assert len(rows) == 1 and rows[0]["error"] and rows[0]["detections"] is None
    assert stats["failed"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run.py -v`
Expected: FAIL (no module bench.run)

- [ ] **Step 3: Implement `bench/run.py`**

```python
"""Fan out model x image x rung calls; resumable JSONL output per model."""
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from bench.prompt import build_prompt, load_refs, RETRY_SUFFIX
from bench.providers import make_provider, parse_detections
from bench.resize import derive, rungs_for


def _load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["image"], r["rung"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_benchmark(models, bench, brands, manifest, root, only_models=None,
                  only_rungs=None, limit_images=None, provider_factory=make_provider,
                  backoff_base=2.0):
    rungs = only_rungs or bench["rungs"]
    images = manifest["images"][:limit_images] if limit_images else manifest["images"]
    raw_dir = os.path.join(root, "results", "raw")
    rungs_dir = os.path.join(root, "data", "rungs")
    os.makedirs(raw_dir, exist_ok=True)

    refs = load_refs(brands, root)
    prompt = build_prompt(brands, n_refs=len(refs))
    ref_bytes = [b for b, _ in refs]

    for img in images:  # ensure ladder exists before any API call
        derive(os.path.join(root, "data", "images", img["id"]), img["id"],
               rungs_dir, rungs, bench["jpeg_quality"])

    active = [m for m in models
              if m.enabled and (not only_models or m.name in only_models)]
    sems = {}
    for m in active:
        prov = provider_factory(m, bench)
        key = prov.provider_key
        sems.setdefault(key, threading.Semaphore(bench["concurrency"].get(key, 4)))
        m._prov = prov  # attach for the worker

    stats = {"done": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    files = {m.name: open(os.path.join(raw_dir, m.name + ".jsonl"), "a") for m in active}
    work = []
    for m in active:
        done = _load_done(os.path.join(raw_dir, m.name + ".jsonl"))
        for img in images:
            for rung in rungs_for(img["native"][1], rungs):
                if (img["id"], rung) in done:
                    stats["skipped"] += 1
                else:
                    work.append((m, img, rung))

    def attempt(prov, text, images_payload):
        delay = backoff_base
        for i in range(bench["max_retries"]):
            try:
                return prov.call(text, images_payload), None
            except Exception as e:  # HTTP errors, timeouts
                if i == bench["max_retries"] - 1:
                    return None, f"{type(e).__name__}: {e}"
                time.sleep(delay)
                delay *= 2
        return None, "unreachable"

    def worker(item):
        m, img, rung = item
        prov = m._prov
        with open(os.path.join(root, "data", "rungs", str(rung), img["id"]), "rb") as f:
            target = f.read()
        payload = ref_bytes + [target]
        row = {"image": img["id"], "rung": rung, "model": m.name, "detections": None,
               "parse_ok": False, "retried": False, "latency_s": 0.0,
               "input_tokens": 0, "output_tokens": 0, "error": None,
               "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with sems[prov.provider_key]:
            res, err = attempt(prov, prompt, payload)
            if res is not None:
                dets = parse_detections(res.text)
                if dets is None:  # one re-ask with the bare-JSON reminder, recorded
                    row["retried"] = True
                    res2, err2 = attempt(prov, prompt + RETRY_SUFFIX, payload)
                    if res2 is not None:
                        dets = parse_detections(res2.text)
                        res = res2
                    err = err2
                row.update(latency_s=round(res.latency_s, 3),
                           input_tokens=res.input_tokens,
                           output_tokens=res.output_tokens)
                if dets is not None:
                    row.update(detections=dets, parse_ok=True)
            if err and row["detections"] is None:
                row["error"] = err
        with lock:
            files[m.name].write(json.dumps(row) + "\n")
            files[m.name].flush()
            stats["failed" if row["error"] else "done"] += 1
            n = stats["done"] + stats["failed"]
            if n % 25 == 0:
                print(f"  {n}/{len(work)} calls done")

    if work:
        with ThreadPoolExecutor(max_workers=sum(
                bench["concurrency"].get(k, 4) for k in sems)) as ex:
            list(ex.map(worker, work))
    for f in files.values():
        f.close()
    return stats
```

The re-ask deliberately uses `prompt + RETRY_SUFFIX` rather than the identical prompt: temperature-0 models given the same input tend to repeat the same unparseable answer.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run.py -v`
Expected: 3 PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: resumable multi-provider benchmark runner"
```

---

### Task 6: Scoring (`bench/score.py`)

**Files:**
- Create: `bench/score.py`
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: raw JSONL rows (Task 5 schema), labels files (`data/labels/<image_id>.json` with `{"image", "boxes": [{"brand", "box", "size", "placement", "location"}], "done": true}`), manifest, `list[ModelCfg]`.
- Produces:
  - `iou(a: list[int], b: list[int]) -> float`
  - `greedy_match(dets: list[dict], truths: list[dict], min_iou: float = 0.3) -> list[tuple[int, int, float]]` (det_idx, truth_idx, iou); same-brand only; dets sorted by conf desc.
  - `score_all(raw_by_model: dict[str, list[dict]], labels: dict[str, list[dict]], models: list[ModelCfg], rungs: list[int]) -> dict` producing the scores.json structure below.
- scores.json structure:
```json
{"generated": "<iso ts>", "n_images": 40,
 "models": {"<name>": {"rungs": {"1080": {
     "presence": {"adidas": {"p": 1.0, "r": 0.9, "f1": 0.95, "tp": 9, "fp": 0, "fn": 1}, "_macro_f1": 0.9},
     "boxes": {"hit03": 0.8, "hit05": 0.6, "mean_iou": 0.55, "n_truth": 25, "n_det": 27},
     "attrs": {"size_acc": 0.9, "placement_acc": 0.95, "n_matched": 20},
     "ops": {"lat_p50": 3.1, "lat_p95": 8.0, "cost_per_frame": 0.004,
             "parse_fail_rate": 0.0, "n_frames": 40}}},
   "retention": {"presence_f1": {"720": 0.98}, "hit03": {"720": 0.9}}}}}
```

- [ ] **Step 1: Write the failing test**

`tests/test_score.py`:
```python
import bench.score as sc
from bench.config import ModelCfg


def test_iou_basic():
    assert sc.iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert sc.iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0
    assert abs(sc.iou([0, 0, 100, 100], [50, 0, 150, 100]) - 1 / 3) < 1e-9


def test_greedy_match_same_brand_conf_order():
    truths = [{"brand": "adidas", "box": [0, 0, 100, 100]},
              {"brand": "delay", "box": [500, 500, 600, 600]}]
    dets = [{"brand": "adidas", "box": [510, 510, 590, 590], "conf": 3},  # wrong brand overlap
            {"brand": "adidas", "box": [10, 10, 90, 90], "conf": 2},
            {"brand": "adidas", "box": [5, 5, 95, 95], "conf": 1}]       # duplicate, unmatched
    matches = sc.greedy_match(dets, truths)
    assert matches == [(1, 0, sc.iou([10, 10, 90, 90], [0, 0, 100, 100]))]


def _mk_rows(model, image, rung, dets, parse_ok=True):
    return {"image": image, "rung": rung, "model": model,
            "detections": dets if parse_ok else None, "parse_ok": parse_ok,
            "retried": False, "latency_s": 2.0, "input_tokens": 1000,
            "output_tokens": 100, "error": None, "ts": "t"}


def test_score_all_end_to_end():
    labels = {
        "a.jpg": [{"brand": "adidas", "box": [0, 0, 100, 100], "size": "large",
                   "placement": "foreground", "location": "chest"}],
        "b.jpg": [],  # empty frame, FP control
    }
    det_good = {"brand": "adidas", "box": [10, 10, 90, 90], "size": "large",
                "placement": "foreground", "location": "chest", "conf": 3}
    det_fp = {"brand": "delay", "box": [0, 0, 50, 50], "size": "small",
              "placement": "background", "location": "board", "conf": 1}
    raw = {"m1": [
        _mk_rows("m1", "a.jpg", 480, [det_good]),
        _mk_rows("m1", "b.jpg", 480, [det_fp]),
        _mk_rows("m1", "a.jpg", 240, []),
        _mk_rows("m1", "b.jpg", 240, [], parse_ok=False),
    ]}
    models = [ModelCfg("m1", "openai", "m1", price_in=1.0, price_out=10.0)]
    s = sc.score_all(raw, labels, models, [480, 240])
    r480 = s["models"]["m1"]["rungs"]["480"]
    assert r480["presence"]["adidas"]["f1"] == 1.0
    assert r480["presence"]["delay"]["fp"] == 1          # hallucinated on empty frame
    assert r480["boxes"]["hit03"] == 1.0
    assert r480["attrs"]["size_acc"] == 1.0
    assert r480["ops"]["parse_fail_rate"] == 0.0
    assert abs(r480["ops"]["cost_per_frame"] - (1000 * 1.0 + 100 * 10.0) / 1e6) < 1e-9
    r240 = s["models"]["m1"]["rungs"]["240"]
    assert r240["presence"]["adidas"]["f1"] == 0.0       # missed at 240p
    assert r240["ops"]["parse_fail_rate"] == 0.5
    # retention vs highest rung (480)
    assert s["models"]["m1"]["retention"]["presence_f1"]["240"] == 0.0


def test_retention_guards_zero_division():
    labels = {"a.jpg": []}
    raw = {"m1": [_mk_rows("m1", "a.jpg", 480, []), _mk_rows("m1", "a.jpg", 240, [])]}
    s = sc.score_all(raw, labels, [ModelCfg("m1", "openai", "m1")], [480, 240])
    assert s["models"]["m1"]["retention"]["presence_f1"]["240"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_score.py -v`
Expected: FAIL (no module bench.score)

- [ ] **Step 3: Implement `bench/score.py`**

```python
"""Scoring: presence F1 per brand, greedy IoU box matching, attribute accuracy,
resolution retention, and ops metrics. Pure functions, no I/O except helpers."""
import datetime
import statistics


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def greedy_match(dets, truths, min_iou=0.3):
    """Match detections to truth boxes of the SAME brand, best conf first."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].get("conf", 2))
    used, matches = set(), []
    for di in order:
        best_ti, best = None, min_iou
        for ti, t in enumerate(truths):
            if ti in used or t["brand"] != dets[di]["brand"]:
                continue
            v = iou(dets[di]["box"], t["box"])
            if v >= best:
                best_ti, best = ti, v
        if best_ti is not None:
            used.add(best_ti)
            matches.append((di, best_ti, best))
    return sorted(matches)


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def _score_rung(rows, labels, brands, mcfg):
    pres = {b: {"tp": 0, "fp": 0, "fn": 0} for b in brands}
    hit03 = hit05 = n_truth = n_det = 0
    ious, size_ok, plc_ok, n_matched = [], 0, 0, 0
    lats, cost, fails = [], 0.0, 0
    for row in rows:
        truth = labels.get(row["image"], [])
        dets = row["detections"] if row["parse_ok"] else []
        if not row["parse_ok"]:
            fails += 1
        lats.append(row["latency_s"])
        cost += (row["input_tokens"] * mcfg.price_in
                 + row["output_tokens"] * mcfg.price_out) / 1e6
        t_brands = {t["brand"] for t in truth}
        d_brands = {d["brand"] for d in dets}
        for b in brands:
            if b in t_brands and b in d_brands:
                pres[b]["tp"] += 1
            elif b in d_brands:
                pres[b]["fp"] += 1
            elif b in t_brands:
                pres[b]["fn"] += 1
        n_truth += len(truth)
        n_det += len(dets)
        for di, ti, v in greedy_match(dets, truth):
            n_matched += 1
            ious.append(v)
            hit03 += 1
            hit05 += v >= 0.5
            size_ok += dets[di]["size"] == truth[ti].get("size")
            plc_ok += dets[di]["placement"] == truth[ti].get("placement")
    presence = {}
    macro = []
    for b in brands:
        p, r, f1 = _f1(pres[b]["tp"], pres[b]["fp"], pres[b]["fn"])
        presence[b] = {"p": p, "r": r, "f1": f1, **pres[b]}
        if pres[b]["tp"] + pres[b]["fn"] > 0 or pres[b]["fp"] > 0:
            macro.append(f1)
    presence["_macro_f1"] = round(sum(macro) / len(macro), 4) if macro else 0.0
    return {
        "presence": presence,
        "boxes": {"hit03": round(hit03 / n_truth, 4) if n_truth else None,
                  "hit05": round(hit05 / n_truth, 4) if n_truth else None,
                  "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
                  "n_truth": n_truth, "n_det": n_det},
        "attrs": {"size_acc": round(size_ok / n_matched, 4) if n_matched else None,
                  "placement_acc": round(plc_ok / n_matched, 4) if n_matched else None,
                  "n_matched": n_matched},
        "ops": {"lat_p50": round(statistics.median(lats), 2) if lats else None,
                "lat_p95": round(sorted(lats)[max(0, int(len(lats) * 0.95) - 1)], 2) if lats else None,
                "cost_per_frame": round(cost / len(rows), 6) if rows else None,
                "parse_fail_rate": round(fails / len(rows), 4) if rows else None,
                "n_frames": len(rows)},
    }


def score_all(raw_by_model, labels, models, rungs):
    brands = sorted({t["brand"] for boxes in labels.values() for t in boxes}
                    | {d["brand"] for rows in raw_by_model.values() for r in rows
                       for d in (r["detections"] or [])})
    out = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "n_images": len(labels), "models": {}}
    mcfgs = {m.name: m for m in models}
    for name, rows in raw_by_model.items():
        by_rung = {}
        for r in rows:
            by_rung.setdefault(r["rung"], []).append(r)
        rung_scores = {str(rg): _score_rung(by_rung[rg], labels, brands, mcfgs[name])
                       for rg in sorted(by_rung, reverse=True)}
        top = str(max(by_rung))
        retention = {"presence_f1": {}, "hit03": {}}
        for rg in sorted(by_rung, reverse=True):
            if str(rg) == top:
                continue
            for metric, path in (("presence_f1", ("presence", "_macro_f1")),
                                 ("hit03", ("boxes", "hit03"))):
                base = rung_scores[top][path[0]][path[1]]
                cur = rung_scores[str(rg)][path[0]][path[1]]
                retention[metric][str(rg)] = (round(cur / base, 4)
                                              if base else None)
        out["models"][name] = {"rungs": rung_scores, "retention": retention}
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_score.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: scoring (presence F1, IoU matching, attrs, retention, ops)"
```

---

### Task 7: CLI (`bench/cli.py`, `bench/__main__.py`)

**Files:**
- Create: `bench/cli.py`, `bench/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above; `bench.report.build_report` and `bench.report.apply_reviews` exist from Task 10 (until then, the `report` and `apply-reviews` subcommands import lazily so the CLI works beforehand).
- Produces: `python -m bench {ladder,run,score,report,serve,apply-reviews}`.
  - `ladder`: derive all rungs for all manifest images.
  - `run [--models a,b] [--rungs 480,144] [--limit N]`: run the benchmark.
  - `score`: read `results/raw/*.jsonl` + labels, write `results/scores.json`.
  - `report`: write `results/leaderboard.html` + gallery.
  - `serve [--port 8765]`: exec `server.py`.
  - `apply-reviews`: apply verdicts from `data/reviews.json` to labels.
- Produces helpers other tasks reuse: `load_manifest(root) -> dict`, `load_labels(root) -> dict[str, list[dict]]` (only labels with `"done": true`), `load_raw(root) -> dict[str, list[dict]]`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import json
import bench.cli as cli


def test_helpers_load(tmp_path):
    (tmp_path / "data" / "labels").mkdir(parents=True)
    (tmp_path / "results" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(
        {"images": [{"id": "a.jpg", "native": [854, 480], "stratum": "n", "source": {}}]}))
    (tmp_path / "data" / "labels" / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "boxes": [], "done": True}))
    (tmp_path / "data" / "labels" / "b.jpg.json").write_text(json.dumps(
        {"image": "b.jpg", "boxes": [], "done": False}))
    (tmp_path / "results" / "raw" / "m1.jsonl").write_text(
        json.dumps({"image": "a.jpg", "rung": 480, "model": "m1", "detections": [],
                    "parse_ok": True, "retried": False, "latency_s": 1,
                    "input_tokens": 1, "output_tokens": 1, "error": None, "ts": "t"}) + "\n")
    assert cli.load_manifest(str(tmp_path))["images"][0]["id"] == "a.jpg"
    labels = cli.load_labels(str(tmp_path))
    assert list(labels) == ["a.jpg"]          # not-done label excluded
    raw = cli.load_raw(str(tmp_path))
    assert list(raw) == ["m1"] and len(raw["m1"]) == 1


def test_parser_has_all_subcommands():
    parser = cli.build_parser()
    subs = parser._subparsers._group_actions[0].choices
    assert set(subs) == {"ladder", "run", "score", "report", "serve", "apply-reviews"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL (no module bench.cli)

- [ ] **Step 3: Implement `bench/cli.py` and `bench/__main__.py`**

`bench/cli.py`:
```python
"""python -m bench <ladder|run|score|report|serve|apply-reviews>"""
import argparse
import glob
import json
import os
import subprocess
import sys

from bench import config
from bench.resize import derive


def load_manifest(root):
    return json.load(open(os.path.join(root, "data", "manifest.json")))


def load_labels(root):
    labels = {}
    for p in sorted(glob.glob(os.path.join(root, "data", "labels", "*.json"))):
        d = json.load(open(p))
        if d.get("done"):
            labels[d["image"]] = d["boxes"]
    return labels


def load_raw(root):
    raw = {}
    for p in sorted(glob.glob(os.path.join(root, "results", "raw", "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        raw[name] = [json.loads(l) for l in open(p)]
    return raw


def build_parser():
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ladder")
    run_p = sub.add_parser("run")
    run_p.add_argument("--models", default=None, help="comma-separated model names")
    run_p.add_argument("--rungs", default=None, help="comma-separated rungs, e.g. 480,144")
    run_p.add_argument("--limit", type=int, default=None, help="first N images only")
    sub.add_parser("score")
    sub.add_parser("report")
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--port", type=int, default=8765)
    sub.add_parser("apply-reviews")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = config.ROOT
    config.load_env()
    bench = config.load_bench()
    if args.cmd == "ladder":
        manifest = load_manifest(root)
        for img in manifest["images"]:
            derive(os.path.join(root, "data", "images", img["id"]), img["id"],
                   os.path.join(root, "data", "rungs"), bench["rungs"],
                   bench["jpeg_quality"])
        print(f"ladder derived for {len(manifest['images'])} images")
    elif args.cmd == "run":
        from bench.run import run_benchmark
        stats = run_benchmark(
            config.load_models(), bench, config.load_brands(), load_manifest(root),
            root, only_models=args.models.split(",") if args.models else None,
            only_rungs=[int(r) for r in args.rungs.split(",")] if args.rungs else None,
            limit_images=args.limit)
        print(stats)
    elif args.cmd == "score":
        from bench.score import score_all
        scores = score_all(load_raw(root), load_labels(root),
                           config.load_models(), bench["rungs"])
        out = os.path.join(root, "results", "scores.json")
        json.dump(scores, open(out, "w"), indent=1)
        print("wrote", out)
    elif args.cmd == "report":
        from bench.report import build_report
        build_report(root)
    elif args.cmd == "serve":
        os.execv(sys.executable, [sys.executable,
                                  os.path.join(root, "server.py"), str(args.port)])
    elif args.cmd == "apply-reviews":
        from bench.report import apply_reviews
        apply_reviews(root)


if __name__ == "__main__":
    main()
```

`bench/__main__.py`:
```python
from bench.cli import main

main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: bench CLI with ladder/run/score/report/serve/apply-reviews"
```

---

### Task 8: Local server and labeling UI (`server.py`, `ui/label.html`)

**Files:**
- Create: `server.py`, `ui/label.html`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: manifest (Task 9 produces the real one; tests fabricate one), labels dir.
- Produces HTTP API (all JSON):
  - `GET /api/manifest` -> manifest.json content
  - `GET /api/labels/<image_id>` -> label file or `{"image": id, "boxes": [], "done": false}` when absent
  - `POST /api/labels/<image_id>` (body = label JSON) -> writes `data/labels/<image_id>.json`
  - `POST /api/review` (body = `{"entry": <gallery entry>, "verdict": "model_right"|"truth_wrong"|"both_wrong"|"skip"}`) -> appends to `data/reviews.json` (a JSON list)
  - Static files from repo root (`/ui/label.html`, `/data/images/...`, `/results/gallery/...`).

- [ ] **Step 1: Write the failing test**

`tests/test_server.py`:
```python
import json
import threading
import urllib.error
import urllib.request

import server as srv


def _start(tmp_path):
    (tmp_path / "data" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(
        {"images": [{"id": "a.jpg", "native": [854, 480], "stratum": "n", "source": {}}]}))
    httpd = srv.make_server(str(tmp_path), 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def _post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_manifest_labels_roundtrip_and_review(tmp_path):
    httpd, base = _start(tmp_path)
    try:
        assert _get(base + "/api/manifest")["images"][0]["id"] == "a.jpg"
        empty = _get(base + "/api/labels/a.jpg")
        assert empty == {"image": "a.jpg", "boxes": [], "done": False}
        label = {"image": "a.jpg", "labeler": "leon", "done": True,
                 "boxes": [{"brand": "adidas", "box": [1, 2, 3, 4], "size": "small",
                            "placement": "foreground", "location": "chest"}]}
        assert _post(base + "/api/labels/a.jpg", label)["ok"]
        on_disk = json.load(open(tmp_path / "data" / "labels" / "a.jpg.json"))
        assert on_disk == label
        assert _get(base + "/api/labels/a.jpg") == label
        _post(base + "/api/review", {"entry": {"model": "m1"}, "verdict": "model_right"})
        _post(base + "/api/review", {"entry": {"model": "m2"}, "verdict": "truth_wrong"})
        reviews = json.load(open(tmp_path / "data" / "reviews.json"))
        assert len(reviews) == 2 and reviews[1]["verdict"] == "truth_wrong"
    finally:
        httpd.shutdown()


def test_path_traversal_rejected(tmp_path):
    httpd, base = _start(tmp_path)
    try:
        try:
            _get(base + "/api/labels/..%2F..%2Fetc%2Fpasswd")
            assert False, "should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        httpd.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_server.py -v`
Expected: FAIL (no module server)

- [ ] **Step 3: Implement `server.py`**

```python
#!/usr/bin/env python3
"""Local server for the labeling and review UIs. Stdlib only, single file.

Usage: python server.py [port]   (default 8765)
Then open http://localhost:<port>/ui/label.html
"""
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def make_server(root, port):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _label_path(self, image_id):
            if not SAFE_ID.match(image_id):
                return None
            return os.path.join(root, "data", "labels", image_id + ".json")

        def do_GET(self):
            if self.path == "/api/manifest":
                return self._json(json.load(
                    open(os.path.join(root, "data", "manifest.json"))))
            if self.path.startswith("/api/labels/"):
                image_id = self.path[len("/api/labels/"):]
                p = self._label_path(image_id)
                if p is None:
                    return self._json({"error": "bad id"}, 400)
                if os.path.exists(p):
                    return self._json(json.load(open(p)))
                return self._json({"image": image_id, "boxes": [], "done": False})
            return super().do_GET()

        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path.startswith("/api/labels/"):
                image_id = self.path[len("/api/labels/"):]
                p = self._label_path(image_id)
                if p is None:
                    return self._json({"error": "bad id"}, 400)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(body, open(p, "w"), indent=1)
                return self._json({"ok": True})
            if self.path == "/api/review":
                p = os.path.join(root, "data", "reviews.json")
                reviews = json.load(open(p)) if os.path.exists(p) else []
                reviews.append(body)
                json.dump(reviews, open(p, "w"), indent=1)
                return self._json({"ok": True, "count": len(reviews)})
            return self._json({"error": "unknown endpoint"}, 404)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    repo = os.path.dirname(os.path.abspath(__file__))
    print(f"serving on http://localhost:{port}/ui/label.html")
    make_server(repo, port).serve_forever()
```

Note: URL-encoded traversal like `..%2F` arrives already decoded by the time `self.path` is read in some client/server combinations; `SAFE_ID` rejects anything containing `/`, `%`, or `..` because only `[A-Za-z0-9._-]+` matches and `..%2F..` contains `%` and `/` after decoding. Verify the traversal test passes; if `self.path` arrives percent-encoded, unquote it first with `urllib.parse.unquote` BEFORE the SAFE_ID check.

- [ ] **Step 4: Run server tests**

Run: `python -m pytest tests/test_server.py -v`
Expected: 2 PASS

- [ ] **Step 5: Write `ui/label.html`**

Single self-contained file, no external assets. Complete content:

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Logo labeling</title>
<style>
  body { margin: 0; font: 14px system-ui, sans-serif; background: #111; color: #eee; }
  #bar { padding: 8px 12px; background: #1b1b1b; display: flex; gap: 12px;
         align-items: center; flex-wrap: wrap; position: sticky; top: 0; }
  #bar select, #bar button { font: inherit; background: #2a2a2a; color: #eee;
         border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }
  #bar button.primary { background: #2563eb; border-color: #2563eb; }
  #stage { position: relative; margin: 12px auto; width: fit-content; }
  #img { display: block; max-width: 96vw; max-height: 82vh; }
  #cv { position: absolute; left: 0; top: 0; cursor: crosshair; }
  #meta { text-align: center; color: #999; padding-bottom: 20px; }
  kbd { background: #333; border-radius: 3px; padding: 0 4px; }
</style>
</head>
<body>
<div id="bar">
  <span id="pos"></span>
  <label>brand <select id="brand"></select></label>
  <label>size <select id="size">
    <option>small</option><option>medium</option><option>large</option></select></label>
  <label>placement <select id="placement">
    <option>foreground</option><option>background</option></select></label>
  <label>location <select id="location">
    <option>chest</option><option>sleeve</option><option>shorts</option>
    <option>headwear</option><option>board</option><option>backdrop</option>
    <option>other</option></select></label>
  <button id="del">delete selected (Backspace)</button>
  <button id="prev">prev (P)</button>
  <button id="next" class="primary">done, next (N)</button>
  <span id="status"></span>
</div>
<div id="stage"><img id="img"><canvas id="cv"></canvas></div>
<div id="meta">Drag to draw a box. Click a box to select it, then change its fields
  in the toolbar or delete it. <kbd>1</kbd>-<kbd>9</kbd> pick the brand.
  Frames with no logos: just press <kbd>N</kbd>.</div>
<script>
let manifest = [], idx = 0, boxes = [], selected = -1, brands = [];
let drag = null, dirty = false;
const img = document.getElementById('img'), cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const $ = id => document.getElementById(id);

async function init() {
  manifest = (await (await fetch('/api/manifest')).json()).images;
  // brand list comes from a brands.json the CLI writes next to the manifest,
  // falling back to the union of brands seen in existing labels
  try { brands = await (await fetch('/data/brands.json')).json(); }
  catch (e) { brands = []; }
  if (!brands.length) brands = ['adidas','stripes','dkh','11teamsports','delay'];
  $('brand').innerHTML = brands.map(b => `<option>${b}</option>`).join('');
  idx = +(localStorage.getItem('label_idx') || 0);
  await load();
}

async function load() {
  idx = Math.max(0, Math.min(idx, manifest.length - 1));
  localStorage.setItem('label_idx', idx);
  const m = manifest[idx];
  const lab = await (await fetch('/api/labels/' + m.id)).json();
  boxes = lab.boxes || []; selected = -1; dirty = false;
  img.onload = () => { sync(); draw(); };
  img.src = '/data/images/' + m.id;
  $('pos').textContent = `${idx + 1}/${manifest.length}  ${m.id}` +
                         (lab.done ? ' (done)' : '');
}

function sync() {
  cv.width = img.clientWidth; cv.height = img.clientHeight;
}

function px(box) { // 0-1000 -> canvas px
  return [box[0] / 1000 * cv.width, box[1] / 1000 * cv.height,
          box[2] / 1000 * cv.width, box[3] / 1000 * cv.height];
}

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  boxes.forEach((b, i) => {
    const [x0, y0, x1, y1] = px(b.box);
    ctx.lineWidth = i === selected ? 3 : 2;
    ctx.strokeStyle = i === selected ? '#fbbf24' : '#22c55e';
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = 'rgba(0,0,0,.7)'; ctx.font = '12px system-ui';
    const tag = `${b.brand} ${b.size[0]}/${b.placement[0]}/${b.location}`;
    ctx.fillRect(x0, Math.max(0, y0 - 16), ctx.measureText(tag).width + 8, 16);
    ctx.fillStyle = i === selected ? '#fbbf24' : '#22c55e';
    ctx.fillText(tag, x0 + 4, Math.max(12, y0 - 4));
  });
  if (drag) {
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = 1.5;
    ctx.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
  }
}

async function save(done) {
  const m = manifest[idx];
  await fetch('/api/labels/' + m.id, { method: 'POST',
    body: JSON.stringify({ image: m.id, labeler: 'leon', boxes, done: !!done }) });
  dirty = false;
  $('status').textContent = 'saved ' + new Date().toLocaleTimeString();
}

cv.addEventListener('mousedown', e => {
  const r = cv.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const hit = boxes.findIndex(b => {
    const [x0, y0, x1, y1] = px(b.box);
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
  });
  if (hit >= 0 && !e.shiftKey) {
    selected = hit;
    const b = boxes[hit];
    $('brand').value = b.brand; $('size').value = b.size;
    $('placement').value = b.placement; $('location').value = b.location;
    draw(); return;
  }
  drag = { x0: x, y0: y, x1: x, y1: y };
});
cv.addEventListener('mousemove', e => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  drag.x1 = e.clientX - r.left; drag.y1 = e.clientY - r.top; draw();
});
cv.addEventListener('mouseup', () => {
  if (!drag) return;
  const [a, b] = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1)];
  const [c, d] = [Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)];
  drag = null;
  if (c - a > 4 && d - b > 4) {
    boxes.push({ brand: $('brand').value, size: $('size').value,
      placement: $('placement').value, location: $('location').value,
      box: [Math.round(a / cv.width * 1000), Math.round(b / cv.height * 1000),
            Math.round(c / cv.width * 1000), Math.round(d / cv.height * 1000)] });
    selected = boxes.length - 1; dirty = true; save(false);
  }
  draw();
});
['brand', 'size', 'placement', 'location'].forEach(f =>
  $(f).addEventListener('change', () => {
    if (selected >= 0) {
      boxes[selected][f === 'brand' ? 'brand' : f] = $(f).value;
      dirty = true; save(false); draw();
    }
  }));
$('del').onclick = () => {
  if (selected >= 0) { boxes.splice(selected, 1); selected = -1; save(false); draw(); }
};
$('next').onclick = async () => { await save(true); idx++; load(); };
$('prev').onclick = async () => { if (dirty) await save(false); idx--; load(); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'n' || e.key === 'N') $('next').click();
  if (e.key === 'p' || e.key === 'P') $('prev').click();
  if (e.key === 'Backspace') { e.preventDefault(); $('del').click(); }
  if (e.key >= '1' && e.key <= String(Math.min(9, brands.length)))
    $('brand').value = brands[+e.key - 1];
});
window.addEventListener('resize', () => { sync(); draw(); });
init();
</script>
</body>
</html>
```

Also add to `bench/cli.py`'s `ladder` command (same task, small edit): after deriving rungs, write the brand names for the UI:

```python
        json.dump([b["name"] for b in config.load_brands()],
                  open(os.path.join(root, "data", "brands.json"), "w"))
```

And add `data/brands.json` to `.gitignore` (generated file).

- [ ] **Step 6: Manual verification of the labeling UI**

1. Create a throwaway manifest: put any 2 JPEGs in `data/images/`, write `data/manifest.json` listing them with their real `[w, h]`.
2. `python server.py` and open `http://localhost:8765/ui/label.html`.
3. Draw two boxes, change a brand, delete one, press N, go back with P.
4. Confirm `data/labels/<id>.json` contains normalized 0-1000 int boxes and `"done": true` after N.
5. Delete the throwaway images/labels/manifest afterwards.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: labeling server and browser UI"
```

---

### Task 9: Delay dataset build (`scripts/build_delay_dataset.py`)

**Files:**
- Create: `scripts/build_delay_dataset.py`
- No unit tests (external tools + external repo); verification steps below.

**Interfaces:**
- Consumes (read-only) from the Delay repo `/Users/leon/Coding/Rabona/delay-social-review`:
  - `data/qwen_bench/bench_set.json`: 140 entries `{"id", "corpus", "source", "cell", "path", "truth": {brand: count}}`
  - `data/qwen_res/vid/*.mp4`: already-downloaded 1080p videos named `<date>_<ytid>.mp4`
  - `data/qwen_scan/frames/igimg__*.jpg`: native-resolution IG statics
- Produces: `data/images/*.jpg` (40 files), `data/manifest.json` (Task 5/7 schema).
- YT frame id convention: sheet cell to video second is `((NNN - 1) * 6 + (K - 1)) * 5` for `ytsheet__<date>_<ytid>_sNNN__cK.jpg` (same formula as `qwen_res_ladder.py` in the Delay repo).

- [ ] **Step 1: Implement the script**

```python
#!/usr/bin/env python3
"""Build the 40-frame Delay dataset for the benchmark.

Selection (stratified from the Delay repo's 140-frame bench set):
  YT (24): 8 busy (truth >= 4 marks), 8 normal, 4 empty, 4 small-logo
           (frames whose truth includes dkh or 11teamsports)
  IG (16): 6 dkh/small-text statics, 2 busy, 4 normal, 4 empty
YT frames are re-extracted at 1080p from the source videos with ffmpeg
(only videos already present in the Delay repo are used; entries without
a local 1080p video are skipped in favor of the next candidate).
IG statics are copied at native resolution (must be >= 1080px tall,
smaller ones are skipped with a warning).
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys

from PIL import Image

SRC = "/Users/leon/Coding/Rabona/delay-social-review"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_IMG = os.path.join(ROOT, "data", "images")
SHEET_RE = re.compile(r"^ytsheet__(\d{8}_.+?)_s(\d+)__c(\d+)\.jpg$")
QUOTA = {"yt": {"busy": 8, "normal": 8, "empty": 4, "small": 4},
         "ig": {"small": 6, "busy": 2, "normal": 4, "empty": 4}}


def stratum(entry):
    t = entry.get("truth", {})
    n = sum(t.values())
    if "dkh" in t or "11teamsports" in t:
        return "small"
    if n >= 4:
        return "busy"
    if n == 0:
        return "empty"
    return "normal"


def yt_second(frame_id):
    m = SHEET_RE.match(frame_id)
    if not m:
        return None
    base, sheet, cell = m.group(1), int(m.group(2)), int(m.group(3))
    return base, ((sheet - 1) * 6 + (cell - 1)) * 5


def extract_yt(base, second, out_path):
    mp4 = os.path.join(SRC, "data", "qwen_res", "vid", base + ".mp4")
    if not os.path.exists(mp4):
        return False
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(second),
                        "-i", mp4, "-frames:v", "1", "-q:v", "2", "-y", out_path])
    return r.returncode == 0 and os.path.exists(out_path)


def main():
    rnd = random.Random(11)
    bench = json.load(open(os.path.join(SRC, "data", "qwen_bench", "bench_set.json")))
    rnd.shuffle(bench)
    os.makedirs(OUT_IMG, exist_ok=True)
    picked, images = {"yt": {}, "ig": {}}, []
    for corpus in ("yt", "ig"):
        picked[corpus] = {s: 0 for s in QUOTA[corpus]}
    for e in bench:
        corpus = "yt" if e["corpus"] == "ytsheet" else (
            "ig" if e["corpus"] == "igimg" else None)
        if corpus is None:
            continue
        s = stratum(e)
        if picked[corpus].get(s, 99) >= QUOTA[corpus].get(s, 0):
            continue
        if corpus == "yt":
            parsed = yt_second(e["id"])
            if not parsed:
                continue
            base, second = parsed
            new_id = f"yt__{base}_{second}s.jpg"
            out = os.path.join(OUT_IMG, new_id)
            if not os.path.exists(out) and not extract_yt(base, second, out):
                continue
            source = {"type": "yt", "video": base, "second": second,
                      "bench_id": e["id"]}
        else:
            src_frame = os.path.join(SRC, "data", "qwen_scan", "frames", e["id"])
            if not os.path.exists(src_frame):
                continue
            with Image.open(src_frame) as im:
                if im.height < 1080:
                    print("skip (too small):", e["id"], im.size)
                    continue
            new_id = e["id"].replace("igimg__", "ig__")
            out = os.path.join(OUT_IMG, new_id)
            if not os.path.exists(out):
                shutil.copy(src_frame, out)
            source = {"type": "ig", "bench_id": e["id"]}
        with Image.open(out) as im:
            native = [im.width, im.height]
        images.append({"id": new_id, "native": native, "stratum": s, "source": source})
        picked[corpus][s] += 1
        if sum(sum(v.values()) for v in picked.values()) == 40:
            break
    manifest = {"images": sorted(images, key=lambda x: x["id"])}
    json.dump(manifest, open(os.path.join(ROOT, "data", "manifest.json"), "w"), indent=1)
    print(f"{len(images)} images", {c: picked[c] for c in picked})
    if len(images) < 40:
        print("WARNING: quota not filled; consider fetching more videos "
              "(see qwen_res_ladder.py fetch stage in the Delay repo)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

Run: `python scripts/build_delay_dataset.py`
Expected: prints `40 images` with per-stratum counts. If fewer than 40 (not enough local 1080p videos), report the shortfall to Leon and ask whether to fetch more videos with yt-dlp (command pattern: `yt-dlp -f "bv*[height<=1080]" -o "<SRC>/data/qwen_res/vid/<date>_<ytid>.mp4" "https://youtu.be/<ytid>"`) or rebalance quotas toward IG.

Verify:
```bash
ls data/images | wc -l                                   # 40
python -c "import json; m=json.load(open('data/manifest.json')); \
print(len(m['images']), sorted({i['stratum'] for i in m['images']}))"
python -m bench ladder                                    # derives all rungs
ls data/rungs                                             # 1080 720 480 240 144
```

- [ ] **Step 3: Commit (manifest only; images are gitignored)**

```bash
git add scripts/build_delay_dataset.py data/manifest.json
git commit -m "feat: Delay dataset build script and 40-frame manifest"
```

---

### Task 10: Report, gallery, and review verdicts (`bench/report.py`, `ui/review.html`)

**Files:**
- Create: `bench/report.py`, `ui/review.html`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `results/scores.json` (Task 6 schema), raw rows (`bench.cli.load_raw`), labels (`bench.cli.load_labels`), manifest, rung images.
- Produces:
  - `build_report(root)` -> writes `results/leaderboard.html`, `results/gallery/*.jpg` crops, `results/gallery/manifest.json` (list of entries: `{"entry_id", "img" (relative path), "model", "image", "rung", "kind": "truth_missed"|"model_extra"|"presence_fp"|"presence_fn", "brand", "box": [x0,y0,x1,y1]|None}`).
  - `disagreements(dets, truth) -> list[dict]` pure helper: unmatched truth boxes (`truth_missed`), unmatched detections (`model_extra`).
  - `render_overlay(image_path, truth_boxes, det_boxes, out_path)`: draws truth green (#22c55e), model red (#ef4444), 3px, saves JPEG.
  - `apply_reviews(root)`: reads `data/reviews.json`; for `verdict == "model_right"` and kind `model_extra`, appends the entry's box (brand, box, defaults size "small", placement "foreground", location "other", provenance flag `"from_review": true`) to the image's label file; for `verdict == "truth_wrong"` and kind `truth_missed`, removes the matching truth box (same brand and box) from the label file. Prints a summary; re-run `bench score` afterwards.
- Leaderboard content (self-contained HTML, inline CSS/JS, no external assets, no em-dashes in copy):
  - Model x rung table: macro presence F1, per-brand F1 (details row), hit03, hit05, mean IoU, size/placement accuracy, cost per frame, lat p50, parse-fail rate. Sortable by clicking column headers (plain JS sort).
  - One inline SVG line chart per model: presence F1 across rungs (x axis: rung, y: F1).
  - Gallery section grouped by model linking the overlay JPEGs with kind/brand captions.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
import json
import os
from PIL import Image
import bench.report as rp


def test_disagreements_classifies():
    truth = [{"brand": "adidas", "box": [0, 0, 100, 100]}]
    dets = [{"brand": "adidas", "box": [500, 500, 600, 600], "conf": 3}]
    ds = rp.disagreements(dets, truth)
    kinds = sorted(d["kind"] for d in ds)
    assert kinds == ["model_extra", "truth_missed"]


def test_disagreements_empty_when_matched():
    truth = [{"brand": "adidas", "box": [0, 0, 100, 100]}]
    dets = [{"brand": "adidas", "box": [5, 5, 95, 95], "conf": 3}]
    assert rp.disagreements(dets, truth) == []


def test_render_overlay_writes_jpeg(tmp_path):
    src = tmp_path / "x.jpg"
    Image.new("RGB", (200, 100), "gray").save(src)
    out = tmp_path / "overlay.jpg"
    rp.render_overlay(str(src), [[0, 0, 500, 500]], [[500, 500, 1000, 1000]], str(out))
    assert out.exists() and Image.open(out).size == (200, 100)


def test_apply_reviews_edits_labels(tmp_path):
    labels_dir = tmp_path / "data" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "done": True,
         "boxes": [{"brand": "delay", "box": [1, 1, 9, 9], "size": "small",
                    "placement": "foreground", "location": "chest"}]}))
    reviews = [
        {"entry": {"kind": "truth_missed", "image": "a.jpg", "brand": "delay",
                   "box": [1, 1, 9, 9]}, "verdict": "truth_wrong"},
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "adidas",
                   "box": [100, 100, 200, 200]}, "verdict": "model_right"},
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "dkh",
                   "box": [5, 5, 6, 6]}, "verdict": "both_wrong"},
    ]
    (tmp_path / "data" / "reviews.json").write_text(json.dumps(reviews))
    rp.apply_reviews(str(tmp_path))
    lab = json.load(open(labels_dir / "a.jpg.json"))
    brands = [b["brand"] for b in lab["boxes"]]
    assert brands == ["adidas"]                     # delay removed, adidas added
    assert lab["boxes"][0]["from_review"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL (no module bench.report)

- [ ] **Step 3: Implement `bench/report.py`**

```python
"""Leaderboard HTML, disagreement gallery, and review verdict application."""
import html
import json
import os

from PIL import Image, ImageDraw

from bench.cli import load_labels, load_manifest, load_raw
from bench.score import greedy_match

GALLERY_RUNG = 480  # overlays are rendered on the 480 rung: big enough to see, small files


def disagreements(dets, truth):
    matches = greedy_match(dets, truth)
    mdi = {di for di, _, _ in matches}
    mti = {ti for _, ti, _ in matches}
    out = []
    for ti, t in enumerate(truth):
        if ti not in mti:
            out.append({"kind": "truth_missed", "brand": t["brand"], "box": t["box"]})
    for di, d in enumerate(dets):
        if di not in mdi:
            out.append({"kind": "model_extra", "brand": d["brand"], "box": d["box"]})
    return out


def render_overlay(image_path, truth_boxes, det_boxes, out_path):
    im = Image.open(image_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for box, color in [(b, "#22c55e") for b in truth_boxes] + \
                      [(b, "#ef4444") for b in det_boxes]:
        x0, y0, x1, y1 = (box[0] * im.width // 1000, box[1] * im.height // 1000,
                          box[2] * im.width // 1000, box[3] * im.height // 1000)
        dr.rectangle([x0, y0, x1, y1], outline=color, width=3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, "JPEG", quality=88)


def apply_reviews(root):
    rp = os.path.join(root, "data", "reviews.json")
    if not os.path.exists(rp):
        print("no reviews.json")
        return
    added = removed = 0
    for r in json.load(open(rp)):
        e, verdict = r.get("entry", {}), r.get("verdict")
        lp = os.path.join(root, "data", "labels", e.get("image", "") + ".json")
        if not os.path.exists(lp):
            continue
        lab = json.load(open(lp))
        if verdict == "truth_wrong" and e.get("kind") == "truth_missed":
            n0 = len(lab["boxes"])
            lab["boxes"] = [b for b in lab["boxes"]
                            if not (b["brand"] == e["brand"] and b["box"] == e["box"])]
            removed += n0 - len(lab["boxes"])
        elif verdict == "model_right" and e.get("kind") == "model_extra":
            lab["boxes"].append({"brand": e["brand"], "box": e["box"], "size": "small",
                                 "placement": "foreground", "location": "other",
                                 "from_review": True})
            added += 1
        else:
            continue
        json.dump(lab, open(lp, "w"), indent=1)
    print(f"applied reviews: +{added} boxes, -{removed} boxes; re-run: python -m bench score")


def build_report(root):
    scores = json.load(open(os.path.join(root, "results", "scores.json")))
    raw = load_raw(root)
    labels = load_labels(root)
    gallery_dir = os.path.join(root, "results", "gallery")
    entries = []
    for model, rows in raw.items():
        for row in rows:
            if row["rung"] != GALLERY_RUNG:
                continue
            truth = labels.get(row["image"], [])
            dets = row["detections"] or []
            ds = disagreements(dets, truth)
            if not ds:
                continue
            img_rel = f"gallery/{model}__{row['image']}"
            render_overlay(os.path.join(root, "data", "rungs",
                                        str(GALLERY_RUNG), row["image"]),
                           [t["box"] for t in truth], [d["box"] for d in dets],
                           os.path.join(root, "results", img_rel))
            for i, d in enumerate(ds):
                entries.append({"entry_id": f"{model}|{row['image']}|{i}",
                                "img": img_rel, "model": model,
                                "image": row["image"], "rung": GALLERY_RUNG, **d})
    os.makedirs(gallery_dir, exist_ok=True)
    json.dump(entries, open(os.path.join(gallery_dir, "manifest.json"), "w"), indent=1)
    html_out = _render_html(scores, entries)
    out = os.path.join(root, "results", "leaderboard.html")
    open(out, "w").write(html_out)
    print(f"wrote {out} and {len(entries)} gallery entries")


def _svg_curve(rung_scores):
    rungs = sorted((int(r) for r in rung_scores), reverse=True)
    if len(rungs) < 2:
        return ""
    pts = []
    for i, rg in enumerate(rungs):
        f1 = rung_scores[str(rg)]["presence"]["_macro_f1"] or 0
        pts.append(f"{20 + i * (260 / (len(rungs) - 1)):.0f},{110 - f1 * 100:.0f}")
    labels_x = " ".join(
        f'<text x="{20 + i * (260 / (len(rungs) - 1)):.0f}" y="124" '
        f'font-size="9" text-anchor="middle" fill="#888">{rg}</text>'
        for i, rg in enumerate(rungs))
    return (f'<svg width="300" height="130" viewBox="0 0 300 130">'
            f'<line x1="20" y1="10" x2="20" y2="110" stroke="#444"/>'
            f'<line x1="20" y1="110" x2="280" y2="110" stroke="#444"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="#2563eb" stroke-width="2"/>{labels_x}</svg>')


def _render_html(scores, entries):
    rows = []
    for model, s in sorted(scores["models"].items()):
        for rung, rs in s["rungs"].items():
            o, b, a = rs["ops"], rs["boxes"], rs["attrs"]
            per_brand = ", ".join(f'{k} {v["f1"]}' for k, v in rs["presence"].items()
                                  if not k.startswith("_"))
            rows.append(
                f"<tr><td>{html.escape(model)}</td><td>{rung}</td>"
                f"<td>{rs['presence']['_macro_f1']}</td>"
                f"<td title='{html.escape(per_brand)}'>{b['hit03']}</td>"
                f"<td>{b['hit05']}</td><td>{b['mean_iou']}</td>"
                f"<td>{a['size_acc']}</td><td>{a['placement_acc']}</td>"
                f"<td>{o['cost_per_frame']}</td><td>{o['lat_p50']}</td>"
                f"<td>{o['parse_fail_rate']}</td></tr>")
    curves = "".join(
        f'<div class="curve"><h3>{html.escape(m)}</h3>{_svg_curve(s["rungs"])}</div>'
        for m, s in sorted(scores["models"].items()))
    gallery = "".join(
        f'<figure><img src="{html.escape(e["img"])}" loading="lazy">'
        f'<figcaption>{html.escape(e["model"])}: {e["kind"]} '
        f'({html.escape(e["brand"])}) on {html.escape(e["image"])}</figcaption></figure>'
        for e in entries)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Logo detection leaderboard</title><style>
body{{font:14px system-ui;margin:20px;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #333;
padding:4px 8px;text-align:right}}th{{cursor:pointer;background:#1b1b1b}}
td:first-child,th:first-child{{text-align:left}}
.curves{{display:flex;flex-wrap:wrap;gap:16px}}.curve h3{{margin:4px 0;font-size:13px}}
figure{{display:inline-block;margin:8px;max-width:420px}}img{{max-width:100%}}
figcaption{{font-size:12px;color:#aaa}}
</style></head><body>
<h1>Logo detection leaderboard</h1>
<p>Truth boxes are green, model boxes are red in the gallery. Hover the hit@0.3
column for per-brand presence F1. Open ui/review.html via server.py to record
verdicts on disagreements.</p>
<table id="lb"><thead><tr><th>model</th><th>rung</th><th>presence F1</th>
<th>hit@0.3</th><th>hit@0.5</th><th>mean IoU</th><th>size acc</th>
<th>placement acc</th><th>$/frame</th><th>lat p50</th><th>parse fail</th></tr>
</thead><tbody>{"".join(rows)}</tbody></table>
<h2>Presence F1 by resolution</h2><div class="curves">{curves}</div>
<h2>Disagreement gallery ({len(entries)} entries)</h2>{gallery}
<script>
document.querySelectorAll('#lb th').forEach((th,i)=>th.onclick=()=>{{
const tb=document.querySelector('#lb tbody');
[...tb.rows].sort((a,b)=>{{const x=a.cells[i].innerText,y=b.cells[i].innerText;
const nx=parseFloat(x),ny=parseFloat(y);
return isNaN(nx)||isNaN(ny)?x.localeCompare(y):ny-nx;}})
.forEach(r=>tb.appendChild(r));}});
</script></body></html>"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v`
Expected: 4 PASS

- [ ] **Step 5: Write `ui/review.html`**

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Disagreement review</title>
<style>
  body { margin: 0; font: 14px system-ui; background: #111; color: #eee; }
  #bar { padding: 10px; background: #1b1b1b; position: sticky; top: 0;
         display: flex; gap: 10px; align-items: center; }
  button { font: inherit; padding: 6px 14px; border-radius: 4px;
           border: 1px solid #444; background: #2a2a2a; color: #eee; cursor: pointer; }
  #right { background: #16a34a; border-color: #16a34a; }
  #wrong { background: #dc2626; border-color: #dc2626; }
  #stage { text-align: center; padding: 16px; }
  img { max-width: 94vw; max-height: 78vh; }
  .cap { color: #aaa; margin: 8px; }
</style>
</head>
<body>
<div id="bar">
  <span id="pos"></span>
  <button id="right">model right (R): add to truth</button>
  <button id="wrong">truth wrong (W): remove from truth</button>
  <button id="both">both wrong (B)</button>
  <button id="skip">skip (S)</button>
</div>
<div id="stage"><div class="cap" id="cap"></div><img id="img"></div>
<script>
let entries = [], idx = 0;
const $ = id => document.getElementById(id);
async function init() {
  entries = await (await fetch('/results/gallery/manifest.json')).json();
  idx = +(localStorage.getItem('review_idx') || 0);
  show();
}
function show() {
  if (idx >= entries.length) {
    $('cap').textContent = 'All ' + entries.length + ' entries reviewed. Run: ' +
      'python -m bench apply-reviews, then python -m bench score';
    $('img').src = ''; $('pos').textContent = 'done'; return;
  }
  const e = entries[idx];
  localStorage.setItem('review_idx', idx);
  $('pos').textContent = (idx + 1) + '/' + entries.length;
  $('cap').textContent = e.model + ': ' + e.kind + ' (' + e.brand + ') on ' + e.image +
    '. Green boxes are truth, red boxes are the model.';
  $('img').src = '/results/' + e.img;
}
async function verdict(v) {
  await fetch('/api/review', { method: 'POST',
    body: JSON.stringify({ entry: entries[idx], verdict: v }) });
  idx++; show();
}
$('right').onclick = () => verdict('model_right');
$('wrong').onclick = () => verdict('truth_wrong');
$('both').onclick = () => verdict('both_wrong');
$('skip').onclick = () => verdict('skip');
document.addEventListener('keydown', e => {
  if (e.key === 'r') verdict('model_right');
  if (e.key === 'w') verdict('truth_wrong');
  if (e.key === 'b') verdict('both_wrong');
  if (e.key === 's') verdict('skip');
});
init();
</script>
</body>
</html>
```

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: leaderboard report, disagreement gallery, review verdicts"
```

---

### Task 11: Pricing fill and README

**Files:**
- Modify: `configs/models.yaml` (replace the `price_in: 0` / `price_out: 0` rows with real numbers)
- Create: `README.md`

- [ ] **Step 1: Fetch real prices for the zero-priced rows**

OpenRouter rows (grok-4.5) and cross-check of others:
```bash
curl -sS "https://openrouter.ai/api/v1/models" | python3 -c "
import json,sys
for m in json.load(sys.stdin)['data']:
    if m['id'] in ('x-ai/grok-4.5','google/gemini-3.6-flash','google/gemini-3.1-pro-preview','moonshotai/kimi-k3','qwen/qwen3-vl-235b-a22b-instruct'):
        p=m['pricing']; print(m['id'], float(p['prompt'])*1e6, float(p['completion'])*1e6)"
```

Anthropic rows (claude-opus-5, claude-sonnet-5): WebFetch `https://docs.anthropic.com/en/docs/about-claude/pricing` and read the per-MTok input/output prices for those two models.

DashScope rows (qwen3.8-max, qwen3-vl-plus): WebFetch `https://www.alibabacloud.com/help/en/model-studio/models` and read the international (Singapore) prices per 1M tokens.

Write all fetched numbers into `configs/models.yaml`. If a page is unreachable, use the OpenRouter price for the closest equivalent listing and add a YAML comment `# approx, source: openrouter` on that row.

- [ ] **Step 2: Write `README.md`**

Content requirements (write it out fully, no stubs; no em-dashes anywhere):
- What it is: benchmark for vision LLMs on brand-logo detection, box quality, and resolution robustness; results are per model per resolution rung.
- Quickstart, exactly these steps: clone; `pip install -r requirements.txt`; `cp .env.example .env` and fill keys; put images in `data/images/`; write `data/manifest.json` (document the schema with a 2-image example); edit `brands/<yours>.yaml` and point `configs/benchmark.yaml` at it; `python -m bench ladder`; `python server.py` and label at `http://localhost:8765/ui/label.html`; `python -m bench run`; `python -m bench score`; `python -m bench report`; open `results/leaderboard.html`.
- The review loop: serve, open `ui/review.html`, record verdicts, `python -m bench apply-reviews`, re-run score and report.
- Metrics glossary: presence F1, hit@0.3/0.5, mean IoU, size and placement accuracy, retention, cost per frame, parse-fail rate (one sentence each, matching the spec definitions).
- Note that images and raw results never leave the machine except as API calls to the configured providers, and are never committed.
- Delay dataset note: `scripts/build_delay_dataset.py` is specific to Leon's source repo and is a template for building your own dataset script.

- [ ] **Step 3: Verify and commit**

```bash
python -c "import yaml; ms=yaml.safe_load(open('configs/models.yaml'))['models']; \
zeros=[m['name'] for m in ms if not m['price_in']]; print('zero-priced:', zeros)"
```
Expected: `zero-priced: []`

```bash
git add -A && git commit -m "docs: README quickstart and real model pricing"
```

---

### Task 12: Live smoke test (2 frames, 2 rungs, 2 cheap models)

**Files:** none created (operational verification).

- [ ] **Step 1: Run a tiny live slice**

Requires `.env` populated (copy values from the Delay repo's `.env`; note the Delay repo spells the Anthropic key `ANTHROPHIC_API_KEY`, this repo expects `ANTHROPIC_API_KEY`).

```bash
python -m bench run --models qwen3.8-max,gpt-5.6-luna --rungs 480,144 --limit 2
```
Expected: `{'done': 8, 'skipped': 0, 'failed': 0}` (2 models x 2 images x 2 rungs).

- [ ] **Step 2: Inspect the rows**

```bash
python3 -c "
import json
for name in ('qwen3.8-max','gpt-5.6-luna'):
    rows=[json.loads(l) for l in open(f'results/raw/{name}.jsonl')]
    ok=sum(r['parse_ok'] for r in rows)
    print(name, len(rows), 'rows,', ok, 'parsed,',
          sum(len(r['detections'] or []) for r in rows), 'detections')"
```
Expected: 4 rows per model, all or nearly all parsed, plausible detection counts (the chosen frames may legitimately have zero logos; if BOTH models return zero detections on ALL frames, pick 2 busy-stratum frames with `--limit` removed and `--models` kept, and eyeball one raw `detections` list against the image).

- [ ] **Step 3: Resume check**

Re-run the same command; expected `{'done': 0, 'skipped': 8, 'failed': 0}`.

- [ ] **Step 4: Commit any fixes surfaced by the smoke test**

```bash
git add -A && git commit -m "fix: adjustments from live smoke test"
```
(Skip the commit if nothing changed.)

---

## After implementation

Operational sequence for Leon (not part of this plan's tasks):
1. `python scripts/build_delay_dataset.py` (done in Task 9), `python -m bench ladder`
2. `python server.py`, label all 40 frames in `ui/label.html`
3. `python -m bench run` (full 12-model run)
4. `python -m bench score && python -m bench report`
5. Review disagreements in `ui/review.html`, `python -m bench apply-reviews`, re-score
6. Read `results/leaderboard.html`, decide the model and minimum resolution for the Delay pipeline
