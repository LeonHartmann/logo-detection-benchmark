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
