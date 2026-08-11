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
    # gpt-5.6-* rejects a custom temperature and `max_tokens`; the openai
    # adapter omits temperature (falls back to the model's default of 1) and
    # sends `max_completion_tokens` instead.
    assert "temperature" not in captured["body"]
    assert captured["body"]["max_completion_tokens"] == 500
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


def _fake_post_capturing(monkeypatch):
    captured = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, body=json, headers=headers)
        return FakeResp({"choices": [{"message": {"content": '{"detections":[]}'}}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    monkeypatch.setattr(pv.requests, "post", fake_post)
    return captured


def test_non_openai_body_keeps_max_tokens_and_temperature(monkeypatch):
    """Regression guard for the exact 400s the live smoke test hit: a future
    refactor that flips the openai/non-openai condition in OpenAICompatible
    must fail a test. dashscope (and openrouter, which shares the same
    branch) must keep sending the legacy `max_tokens` + `temperature: 0` --
    NOT the openai-only `max_completion_tokens` / no-temperature shape."""
    monkeypatch.setenv("QWEN_API_KEY", "k1")
    captured = _fake_post_capturing(monkeypatch)
    p = pv.make_provider(ModelCfg("qwen3.8-max", "dashscope", "qwen3.8-max"), BENCH)
    p.call("find logos", [b"\xff\xd8fakejpeg"])
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 500
    assert "max_completion_tokens" not in captured["body"]

    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    captured2 = _fake_post_capturing(monkeypatch)
    o = pv.make_provider(ModelCfg("kimi-k3", "openrouter", "moonshotai/kimi-k3"), BENCH)
    o.call("find logos", [b"\xff\xd8fakejpeg"])
    assert captured2["body"]["temperature"] == 0
    assert captured2["body"]["max_tokens"] == 500
    assert "max_completion_tokens" not in captured2["body"]


def test_openai_body_omits_temperature_and_max_tokens(monkeypatch):
    """Mirror of the above for the openai branch: must send
    `max_completion_tokens` and must NOT send `temperature` or the legacy
    `max_tokens` key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = _fake_post_capturing(monkeypatch)
    p = pv.make_provider(ModelCfg("gpt-5.6-terra", "openai", "gpt-5.6-terra"), BENCH)
    p.call("find logos", [b"\xff\xd8fakejpeg"])
    assert captured["body"]["max_completion_tokens"] == 500
    assert "temperature" not in captured["body"]
    assert "max_tokens" not in captured["body"]


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


def test_parse_detections_normalizes_brand_case_and_whitespace():
    """Model output brand strings should collapse to the same casefold/strip
    form the label UI's brand list already uses, so e.g. "  Adidas " from a
    model matches the truth label "adidas" instead of scoring as a miss."""
    raw = json.dumps({"detections": [
        {"brand": "  Adidas ", "box": [1, 2, 3, 4]},
    ]})
    dets = pv.parse_detections(raw)
    assert dets[0]["brand"] == "adidas"


def test_parse_detections_with_braces_in_prose():
    """Test extraction from prose with braces before and after JSON."""
    # Prose with brace before JSON
    with_brace_before = 'I notice the shirt uses a {sponsor} pattern.\n{"detections":[{"brand":"adidas","box":[1,2,3,4]}]}'
    dets = pv.parse_detections(with_brace_before)
    assert len(dets) == 1 and dets[0]["brand"] == "adidas"

    # Prose with brace after JSON
    with_brace_after = '{"detections":[{"brand":"nike","box":[5,6,7,8]}]}\nNote: see also {reference}.'
    dets = pv.parse_detections(with_brace_after)
    assert len(dets) == 1 and dets[0]["brand"] == "nike"

    # Multiple braces scattered throughout
    messy = 'Check this {field}:\n{"detections":[{"brand":"puma","box":[10,20,30,40]}]} but also {note}.'
    dets = pv.parse_detections(messy)
    assert len(dets) == 1 and dets[0]["brand"] == "puma"


def test_parse_detections_handles_unhashable_values():
    """Test that unhashable field values (e.g., list instead of string) default correctly."""
    raw = json.dumps({"detections": [
        {"brand": "adidas", "box": [1, 2, 3, 4], "size": ["small"]},      # list instead of string
        {"brand": "nike", "box": [5, 6, 7, 8], "placement": {"type": "bg"}},  # dict instead of string
        {"brand": "puma", "box": [10, 20, 30, 40], "location": 123},       # int instead of string
    ]})
    dets = pv.parse_detections(raw)
    assert len(dets) == 3
    assert dets[0]["size"] == "small"      # unhashable value defaults
    assert dets[1]["placement"] == "foreground"  # unhashable value defaults
    assert dets[2]["location"] == "other"  # unhashable value defaults


def test_adapter_non_2xx_raises_http_error(monkeypatch):
    """Test that adapters propagate HTTPError on non-2xx responses."""
    def fake_post_error(url, json=None, headers=None, timeout=None):
        return FakeResp({"error": "unauthorized"}, status=401)

    monkeypatch.setattr(pv.requests, "post", fake_post_error)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")

    p = pv.make_provider(ModelCfg("gpt-5.6-terra", "openai", "gpt-5.6-terra"), BENCH)
    with pytest.raises(Exception):  # requests.HTTPError is the base exception
        p.call("find logos", [b"\xff\xd8fakejpeg"])
