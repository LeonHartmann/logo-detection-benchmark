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
