import json
import os
from PIL import Image
import bench.report as rp


def test_fmt_handles_none():
    assert rp._fmt(None) == "-"
    assert rp._fmt(0.5) == 0.5
    assert rp._fmt("text") == "text"
    assert rp._fmt(0) == 0


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
    applied_path = tmp_path / "data" / "reviews.applied.json"
    assert applied_path.exists()
    applied = json.load(open(applied_path))
    assert len(applied) == 3                        # all 3 entries archived
    assert applied[0]["applied"] is True            # truth_wrong applied
    assert applied[1]["applied"] is True            # model_right applied
    assert applied[2]["applied"] is False           # both_wrong not applied


def test_apply_reviews_idempotent(tmp_path):
    labels_dir = tmp_path / "data" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "done": True,
         "boxes": [{"brand": "delay", "box": [1, 1, 9, 9], "size": "small",
                    "placement": "foreground", "location": "chest"}]}))
    reviews = [
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "adidas",
                   "box": [100, 100, 200, 200]}, "verdict": "model_right"},
    ]
    (tmp_path / "data" / "reviews.json").write_text(json.dumps(reviews))
    rp.apply_reviews(str(tmp_path))
    lab1 = json.load(open(labels_dir / "a.jpg.json"))
    assert len(lab1["boxes"]) == 2                   # delay + adidas
    assert not os.path.exists(tmp_path / "data" / "reviews.json")
    applied_path = tmp_path / "data" / "reviews.applied.json"
    assert applied_path.exists()
    rp.apply_reviews(str(tmp_path))  # run again
    lab2 = json.load(open(labels_dir / "a.jpg.json"))
    assert lab2["boxes"] == lab1["boxes"]            # no change on second run


def test_apply_reviews_deduplicates_boxes(tmp_path):
    labels_dir = tmp_path / "data" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.jpg.json").write_text(json.dumps(
        {"image": "a.jpg", "done": True,
         "boxes": [{"brand": "adidas", "box": [100, 100, 200, 200], "size": "large",
                    "placement": "background", "location": "sleeve"}]}))
    reviews = [
        {"entry": {"kind": "model_extra", "image": "a.jpg", "brand": "adidas",
                   "box": [100, 100, 200, 200]}, "verdict": "model_right"},
    ]
    (tmp_path / "data" / "reviews.json").write_text(json.dumps(reviews))
    rp.apply_reviews(str(tmp_path))
    lab = json.load(open(labels_dir / "a.jpg.json"))
    assert len(lab["boxes"]) == 1                   # not duplicated
    assert lab["boxes"][0]["brand"] == "adidas"
    assert lab["boxes"][0]["box"] == [100, 100, 200, 200]
