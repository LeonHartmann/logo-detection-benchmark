import json
import os
from PIL import Image

import bench.prelabel as pl
from bench.config import ModelCfg
from bench.providers import CallResult

BENCH = {"rungs": [480, 240], "jpeg_quality": 85, "max_tokens": 100, "timeout_s": 5,
         "max_retries": 2, "concurrency": {"fake": 2}}
BRANDS = [{"name": "adidas", "description": "x", "refs": []}]
GOOD = ('{"detections":[{"brand":"adidas","box":[10,10,20,20],"size":"small",'
        '"placement":"foreground","location":"chest","conf":3},'
        '{"brand":"nike","box":[1,1,2,2],"size":"small",'
        '"placement":"foreground","location":"chest","conf":1}]}')


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
    (tmp_path / "data" / "labels").mkdir(parents=True)
    images = []
    for i in range(n_images):
        name = f"img{i}.jpg"
        Image.new("RGB", (854, 480), "green").save(tmp_path / "data" / "images" / name)
        images.append({"id": name, "native": [854, 480], "stratum": "normal", "source": {}})
    return {"images": images}


def write_label(tmp_path, image_id, boxes, done):
    (tmp_path / "data" / "labels" / f"{image_id}.json").write_text(json.dumps(
        {"image": image_id, "labeler": "leon", "boxes": boxes, "done": done}))


def test_writes_suggestions_and_drops_off_list_brands(tmp_path):
    manifest = setup_repo(tmp_path, n_images=1)
    models = [ModelCfg("qwen3-vl-plus", "fake", "fake-1")]
    fp = FakeProvider([])
    stats = pl.prelabel(models, BENCH, BRANDS, manifest, str(tmp_path),
                        provider_factory=lambda m, b: fp)

    label = json.load(open(tmp_path / "data" / "labels" / "img0.jpg.json"))
    assert label["image"] == "img0.jpg"
    assert label["labeler"] == "prelabel:qwen3-vl-plus"
    assert label["done"] is False
    assert label["boxes"] == [{"brand": "adidas", "box": [10, 10, 20, 20], "size": "small",
                               "placement": "foreground", "location": "chest",
                               "suggested": True}]
    assert stats == {"processed": 1, "skipped": 0, "failed": 0,
                     "input_tokens": 100, "output_tokens": 10}


def test_skip_policy_preserves_human_work_and_refreshes_pure_suggestions(tmp_path):
    manifest = setup_repo(tmp_path, n_images=3)
    write_label(tmp_path, "img0.jpg", [{"brand": "adidas", "box": [1, 1, 2, 2], "size": "small",
                                        "placement": "foreground", "location": "chest"}],
                done=True)
    write_label(tmp_path, "img1.jpg", [{"brand": "adidas", "box": [3, 3, 4, 4], "size": "small",
                                        "placement": "foreground", "location": "chest"}],
                done=False)
    write_label(tmp_path, "img2.jpg", [{"brand": "adidas", "box": [5, 5, 6, 6], "size": "small",
                                        "placement": "foreground", "location": "chest",
                                        "suggested": True}],
                done=False)
    before0 = (tmp_path / "data" / "labels" / "img0.jpg.json").read_text()
    before1 = (tmp_path / "data" / "labels" / "img1.jpg.json").read_text()

    models = [ModelCfg("qwen3-vl-plus", "fake", "fake-1")]
    fp = FakeProvider([])
    stats = pl.prelabel(models, BENCH, BRANDS, manifest, str(tmp_path),
                        provider_factory=lambda m, b: fp)

    assert fp.calls == 1  # only img2 (pure-suggestion file) triggers a call
    assert stats["processed"] == 1 and stats["skipped"] == 2 and stats["failed"] == 0

    # untouched byte-for-byte: done:true and human-boxed files are never rewritten
    assert (tmp_path / "data" / "labels" / "img0.jpg.json").read_text() == before0
    assert (tmp_path / "data" / "labels" / "img1.jpg.json").read_text() == before1

    label2 = json.load(open(tmp_path / "data" / "labels" / "img2.jpg.json"))
    assert label2["labeler"] == "prelabel:qwen3-vl-plus"
    assert label2["done"] is False
    assert label2["boxes"] == [{"brand": "adidas", "box": [10, 10, 20, 20], "size": "small",
                               "placement": "foreground", "location": "chest",
                               "suggested": True}]


def test_per_image_failure_does_not_abort_run(tmp_path, capsys):
    manifest = setup_repo(tmp_path, n_images=2)
    models = [ModelCfg("qwen3-vl-plus", "fake", "fake-1")]
    # img0 fails both tries; img1 succeeds on its first call.
    fp = FakeProvider([RuntimeError("boom"), RuntimeError("boom"), GOOD])
    stats = pl.prelabel(models, BENCH, BRANDS, manifest, str(tmp_path),
                        provider_factory=lambda m, b: fp, backoff_base=0)

    assert stats["processed"] == 1 and stats["failed"] == 1 and stats["skipped"] == 0
    assert not (tmp_path / "data" / "labels" / "img0.jpg.json").exists()
    assert (tmp_path / "data" / "labels" / "img1.jpg.json").exists()
    out = capsys.readouterr().out
    assert "img0.jpg" in out and "WARNING" in out


def test_pick_model_defaults_and_errors_with_available_names():
    models = [ModelCfg("qwen3-vl-plus", "fake", "fake-1", enabled=True),
             ModelCfg("other-model", "fake", "fake-2", enabled=True),
             ModelCfg("disabled-model", "fake", "fake-3", enabled=False)]
    assert pl.pick_model(models).name == "qwen3-vl-plus"
    assert pl.pick_model(models, "other-model").name == "other-model"
    try:
        pl.pick_model(models, "nope")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "nope" in str(e) and "other-model" in str(e) and "qwen3-vl-plus" in str(e)
        assert "disabled-model" not in str(e)

    models_no_default = [ModelCfg("other-model", "fake", "fake-2", enabled=True)]
    try:
        pl.pick_model(models_no_default)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "qwen3-vl-plus" in str(e) and "other-model" in str(e)
